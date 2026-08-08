#!/usr/bin/env python3
"""
Phase 0c smoke check: do the published paper notebooks still *construct*?

`check_consumers.py` verifies that every module and symbol a paper imports still
resolves. That is the fast 90%, it runs anywhere in seconds, and every paper
currently passes it. What it cannot catch is API drift: the module resolves, the
symbol exists, and the constructor has grown a required parameter since the
notebook was written. That is exactly how `Missing 'tau'` sat in
idpy/LBM/test.py for years while the suite looked green.

This closes that gap without paying for the science. Each notebook is executed
cell by cell in one namespace until either it errors -- which is a finding -- or
a cell exceeds a wall-clock budget, which means the simulation itself has been
reached. **Reaching compute is the successful outcome.** We are testing that the
objects can be built, not that the paper's results reproduce; reproducing them
is hours of GPU time and belongs to the reader, not to CI.

The paper inventory comes from `papers/idpy-papers.py`, which is the registry
the project already maintains, rather than from whatever happens to be cloned on
the machine running this. Those differ: a survey of one development machine
found six repositories when the registry lists seven.

Clones go to a scratch directory and are read-only as far as this script is
concerned. It never touches `papers/`, whose checkouts on a development machine
carry uncommitted work.

Usage:
    python3 scripts/smoke_papers.py --list
    python3 scripts/smoke_papers.py                     # all, into a temp dir
    python3 scripts/smoke_papers.py arXiv-2505.23647    # one paper
    python3 scripts/smoke_papers.py --dir /path/to/repo # an existing checkout
    python3 scripts/smoke_papers.py --budget 90         # slower machines
    python3 scripts/smoke_papers.py --lang OCL_T        # force the backend

A paper whose default backend is missing has told you nothing about whether it
still constructs, so one is retried on a backend this machine has. The notebooks
detect what is available and then override it with a hardcoded `preferred_lang`,
which makes the backend a default rather than a requirement. Without the retry,
"no pycuda here" gets filed as though it were drift -- and it masks whatever the
next cell would have said. Both real breakages found so far were behind that
wall.

The retry has to be right, and three ways of getting it wrong were found by
running it (Phase 0b, 2026-08-07). Each produced a confident FAIL line against
a paper that was fine, which is worse than no result at all:

  - it rewrote `preferred_lang` only, so a repository using `set_lang` ran on
    its original backend and was reported as broken for lacking it;
  - it substituted a token the notebook had never imported, inventing a
    NameError at the cell it had just edited;
  - it read the wall-clock budget expiring inside a ctypes call as an error,
    because ctypes reshapes the timeout into ArgumentError on the way out.

The lesson generalises past this script: a harness that adapts the thing it is
measuring must be able to say when its own adaptation failed. Silence there is
indistinguishable from a finding.

Exit 0 when every notebook attempted either constructs or reaches compute.
Exit 1 on any error, or when nothing was attempted at all.
"""

import argparse
import contextlib
import io
import json
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import traceback

REGISTRY = 'papers/idpy-papers.py'
DEFAULT_BUDGET = 60


class _Timeout(Exception):
    pass


# Whether SIGALRM fired during the cell currently being executed.
#
# Raising _Timeout from the handler is not enough on its own, because the
# exception does not always survive the frame it was raised into. When the
# budget expires inside a ctypes FFI call -- which is exactly where a CTypes
# simulation spends its time -- ctypes catches whatever Python exception
# surfaces in the argument-conversion path and re-raises it as
# `ctypes.ArgumentError: argument 16: _Timeout:`. That reads as a broken
# constructor call and got reported as API drift in a paper that had in fact
# reached compute, which is the successful outcome.
#
# The flag is the ground truth: if the alarm fired while this cell was
# running, the cell was still running when the budget expired, whatever the
# exception was reshaped into on the way out.
_alarm_fired = False


def _alarm(signum, frame):
    global _alarm_fired
    _alarm_fired = True
    raise _Timeout()


def registry_papers(root):
    """{paper id: git url} from papers/idpy-papers.py, the project's own list."""
    src = (root / REGISTRY).read_text(errors='ignore')
    out = {}
    for pid, body in re.findall(
            r"arxiv_papers\['([^']+)'\]\s*=\s*\\?\s*\{(.*?)\}", src, re.S):
        m = re.search(r'"git"\s*:\s*"([^"]+)"', body)
        if m:
            out[pid] = m.group(1)
    return out


LANG_TOKENS = ('CUDA_T', 'OCL_T', 'METAL_T', 'CTYPES_T')

# Any `<something>_lang ... = <TOKEN>` assignment, not only `preferred_lang`.
#
# The name is not a convention across the seven repositories: one of them
# writes `set_lang, set_device, set_kind = OCL_T, 0, 'gpu'`. Matching only
# `preferred_lang` meant the substitution silently did nothing there, the
# notebook ran on OpenCL anyway, and a machine without pyopencl reported
# "Selected lang = OCL_T but the 'pyopencl' module is not found" as though it
# were the paper's fault. A retry that quietly fails to retry is worse than no
# retry, because the result still looks like a measurement.
#
# `\w*_lang` and not `\w*lang`: the detection block these notebooks already
# carry assigns bare `lang = CUDA_T` inside its own if/elif arms, and
# rewriting those would corrupt the chain rather than retarget it. The
# trailing `lang = preferred_lang` is what actually decides the backend. The
# underscore is required and a prefix is not: one repository names the
# variable plain `_lang`, which `\w+_lang` would have silently missed --
# the same class of near-miss as matching only `preferred_lang` did.
_PREFERRED = re.compile(
    r'(\b\w*_lang\b[^=\n]*=\s*)(' + '|'.join(LANG_TOKENS) + r')\b')


def available_langs(root):
    """Language tokens this machine can actually run, best first."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from idpy.core import idpy_langs_sys              # noqa: PLC0415
    import idpy.core as ic                            # noqa: PLC0415
    out = []
    for tok in ('CUDA_T', 'OCL_T', 'METAL_T', 'CTYPES_T'):
        val = getattr(ic, tok, None)
        if val is not None and idpy_langs_sys.get(val):
            out.append(tok)
    return out


def _backend_absent(detail):
    """Is this failure 'the machine lacks a binding' rather than drift?"""
    if not detail:
        return False
    d = detail.lower()
    return ('is not found in your python environment' in d
            or any(f"no module named '{m}'" in d
                   for m in ('pycuda', 'pyopencl', 'pymetallic')))


def substitute_lang(src, lang):
    """
    Retarget a notebook's backend by rewriting its `*_lang` assignment.

    These notebooks already detect what is available -- `if idpy_langs_sys[CUDA_T]`
    and so on -- and then discard the answer with `lang = preferred_lang`, which
    is hardcoded in an earlier cell. So the backend is a default, not a
    requirement, and changing it is how the papers are meant to be portable.

    Any `<name>_lang` is rewritten, because the name is not consistent across
    the seven repositories -- `preferred_lang` in most, `set_lang` in one. Bare
    `lang` is left alone: the detection block assigns it inside its own if/elif
    arms, and rewriting those would corrupt the chain rather than retarget it.

    Rewriting the token is not sufficient by itself; see run_notebook, which
    seeds the four tokens into the namespace. A notebook imports only the ones
    it uses, so substituting a name it never imported would raise NameError
    from the very line the retry replaced.
    """
    return _PREFERRED.sub(lambda m: m.group(1) + lang, src)


def run_notebook(path, root, budget, lang=None):
    """
    (status, cell_index, n_cells, detail) for one notebook.

    status is REACHED_COMPUTE (good), COMPLETED (good), or ERROR.
    """
    global _alarm_fired
    doc = json.loads(pathlib.Path(path).read_text(errors='ignore'))
    here = pathlib.Path(path).resolve().parent

    # Both paths matter. The repository root replaces the notebook's own
    # sys.path.append("../../"), which assumes it is sitting inside a checkout
    # of idea.deploy. The notebook's directory replaces what Jupyter provides
    # implicitly via the cwd: every paper repo carries local modules of its own
    # (TolmanSimulations, LBM_proxy, InterfaceTuning) that a plain script will
    # not find. Omitting the second reports the paper's own modules as missing,
    # which looks exactly like drift and is not.
    for p in (str(root), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import os
    cwd = os.getcwd()
    os.chdir(here)

    ns = {'__name__': '__main__', 'get_ipython': lambda: None}

    # Seed the backend tokens when we are retargeting the backend.
    #
    # substitute_lang rewrites the right-hand side of a `*_lang = <TOKEN>`
    # assignment, but a notebook only imports the tokens it happens to use:
    # arXiv-2009.12522v1 does `from idpy.core import CUDA_T, OCL_T` and never
    # mentions CTYPES_T. Substituting a name the notebook never imported turns
    # the retry itself into `NameError: name 'CTYPES_T' is not defined` -- a
    # failure invented by the harness, reported at the very cell it replaced,
    # and hiding whatever the notebook would really have said next.
    #
    # Only the four tokens, and only when substituting. They are plain strings
    # ("pycuda", "ctypes", ...) and the notebook's own import rebinds them to
    # the same values wherever it does import them.
    if lang:
        from idpy.core import (CUDA_T, OCL_T, METAL_T,     # noqa: PLC0415
                               CTYPES_T)
        ns.update(CUDA_T=CUDA_T, OCL_T=OCL_T,
                  METAL_T=METAL_T, CTYPES_T=CTYPES_T)

    signal.signal(signal.SIGALRM, _alarm)
    cells = [c for c in doc.get('cells', []) if c.get('cell_type') == 'code']
    try:
        for i, cell in enumerate(cells):
            src = ''.join(cell.get('source', []))
            # IPython magics, shell escapes and help lines are notebook UI, not
            # science; they are not valid Python and their absence changes no
            # constructor call.
            src = '\n'.join(l for l in src.splitlines()
                            if not l.lstrip().startswith(('%', '!', '?')))
            if lang:
                src = substitute_lang(src, lang)
            if not src.strip():
                continue
            _alarm_fired = False
            signal.alarm(budget)
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(src, f'<cell {i}>', 'exec'), ns)
                signal.alarm(0)
            except _Timeout:
                signal.alarm(0)
                return ('REACHED_COMPUTE', i, len(cells), None)
            except SystemExit:
                signal.alarm(0)
                continue
            except BaseException as exc:                   # noqa: BLE001
                signal.alarm(0)
                # The budget expired inside this cell, so the cell was still
                # running: reached compute, whatever the exception was reshaped
                # into on its way out. ctypes rewrites a _Timeout raised during
                # argument conversion into ctypes.ArgumentError, which is
                # indistinguishable from a genuinely bad constructor call by
                # type alone -- and it reported one paper as broken when it had
                # simply started simulating.
                if _alarm_fired:
                    return ('REACHED_COMPUTE', i, len(cells), None)
                tail = traceback.format_exc().strip().splitlines()[-3:]
                return ('ERROR', i, len(cells),
                        f"{type(exc).__name__}: {exc}||" + '||'.join(tail))
        return ('COMPLETED', len(cells), len(cells), None)
    finally:
        signal.alarm(0)
        os.chdir(cwd)


def run_isolated(path, root, budget, lang=None):
    """
    run_notebook in a fresh interpreter, and that isolation is mandatory.

    Every paper repository carries local modules of its own, and several of them
    share names -- TolmanSimulations.py exists in at least two, LBM_proxy.py in
    at least two more, with different contents. Run in one process, sys.modules
    caches whichever was imported first and every later paper silently gets
    another paper's code. That reports ImportErrors naming a *different* repo's
    file, which reads exactly like drift in the paper under test and is not.

    Restoring sys.modules by hand would leave transitive state behind. A
    subprocess costs one interpreter start per notebook and cannot be wrong.
    """
    r = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()),
         '--run-one', str(path), '--budget', str(budget)]
        + (['--lang', lang] if lang else []),
        capture_output=True, text=True)
    line = (r.stdout or '').strip().splitlines()
    if not line:
        return ('ERROR', 0, 0, f"smoke subprocess produced no result"
                               f"||{(r.stderr or '').strip()[-200:]}")
    try:
        payload = json.loads(line[-1])
    except json.JSONDecodeError:
        return ('ERROR', 0, 0, f"unparseable subprocess result||{line[-1][:200]}")
    return (payload['status'], payload['at'], payload['total'],
            payload.get('detail'))


# An archived snapshot of an earlier arXiv version, e.g. arXiv-2009.12522v1/
# sitting inside arXiv-2009.12522/.
_ARCHIVED_DIR = re.compile(r'^arXiv-\d+\.\d+v\d+$')


def notebooks_in(d):
    """
    (notebooks to smoke, archived notebooks skipped).

    Archived version directories are not smoked, and the reason is the point
    of having them: an archive edited to work against current code is no
    longer an archive of anything. They record what was submitted, so drift
    inside one is expected rather than actionable, and reporting it every run
    trains the reader to ignore the output.

    They are returned rather than dropped. A check that silently narrows what
    it looks at reads as "everything passed" when it means "I stopped
    looking", so the caller prints what was skipped.
    """
    live, archived = [], []
    for p in sorted(pathlib.Path(d).rglob('*.ipynb')):
        if '.git' in p.parts or '.ipynb_checkpoints' in p.parts:
            continue
        (archived if any(_ARCHIVED_DIR.match(part) for part in p.parts)
         else live).append(p)
    return live, archived


def main(argv):
    root = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('papers', nargs='*', help='paper ids; default is all')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dir', action='append', default=[],
                    help='smoke an existing checkout instead of cloning')
    ap.add_argument('--budget', type=int, default=DEFAULT_BUDGET,
                    help=f'seconds per cell before calling it compute '
                         f'(default {DEFAULT_BUDGET})')
    ap.add_argument('--keep', action='store_true', help='keep the clones')
    ap.add_argument('--lang', choices=LANG_TOKENS,
                    help='force the backend by rewriting preferred_lang; '
                         'without it, a paper whose default backend is absent '
                         'is retried on one this machine has')
    ap.add_argument('--archived', action='store_true',
                    help='also smoke archived arXiv version directories, '
                         'which are skipped by default')
    ap.add_argument('--run-one', help=argparse.SUPPRESS)
    args = ap.parse_args(argv[1:])

    if args.run_one:
        # The isolated child: one notebook, one interpreter, JSON on stdout.
        status, at, total, detail = run_notebook(
            args.run_one, root, args.budget, args.lang)
        print(json.dumps({'status': status, 'at': at, 'total': total,
                          'detail': detail}))
        return 0

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    known = registry_papers(root)
    if args.list:
        for pid, url in sorted(known.items()):
            print(f"{pid}\t{url}")
        return 0

    targets = []                       # (label, directory)
    tmp = None
    for d in args.dir:
        targets.append((pathlib.Path(d).name, pathlib.Path(d).resolve()))

    wanted = args.papers or (sorted(known) if not args.dir else [])
    unknown = [p for p in wanted if p not in known]
    if unknown:
        print(f"not in {REGISTRY}: {', '.join(unknown)}")
        return 1

    if wanted:
        tmp = tempfile.mkdtemp(prefix='idpy-smoke-')
        print(f"cloning {len(wanted)} paper repositories into {tmp}\n")
        for pid in wanted:
            dest = pathlib.Path(tmp) / pid
            r = subprocess.run(
                ['git', 'clone', '-q', '--depth', '1', known[pid], str(dest)],
                capture_output=True, text=True)
            if r.returncode:
                print(f"  [clone failed] {pid}: {r.stderr.strip()[:80]}")
                continue
            targets.append((pid, dest))

    if not targets:
        print("nothing was attempted, which is not a pass")
        return 1

    failures, attempted, skipped = [], 0, []
    for label, d in targets:
        nbs, archived = notebooks_in(d)
        if args.archived:
            nbs, archived = sorted(nbs + archived), []
        skipped += [(label, p.relative_to(d)) for p in archived]
        if not nbs:
            print(f"  [no notebook] {label}")
            continue
        for nb in nbs:
            attempted += 1
            used_lang = args.lang
            status, at, total, detail = run_isolated(
                nb, root, args.budget, used_lang)

            # A paper whose default backend is absent has told us nothing about
            # whether it still constructs. These notebooks detect what is
            # available and then override it with a hardcoded `preferred_lang`,
            # so the backend is a default rather than a requirement: retry on
            # one this machine has, and say so. Reporting the first failure
            # instead would file "no pycuda here" as though it were drift.
            if status == 'ERROR' and not args.lang and _backend_absent(detail):
                for alt in [l for l in available_langs(root) if l != 'CUDA_T']:
                    status, at, total, detail = run_isolated(
                        nb, root, args.budget, alt)
                    used_lang = alt
                    if status != 'ERROR' or not _backend_absent(detail):
                        break
            tag = {'REACHED_COMPUTE': 'ok  ', 'COMPLETED': 'ok  '}.get(
                status, 'FAIL')
            where = f"cell {at}/{total}"
            note = ('constructs, reached compute'
                    if status == 'REACHED_COMPUTE' else
                    'ran to the end' if status == 'COMPLETED' else detail)
            # Path relative to the repo, not the basename: arXiv-2009.12522
            # carries two notebooks called ShanChenPressureTensorIsotropy.ipynb
            # in different directories, and reporting both as the same name
            # makes one look like a duplicate result of the other.
            rel = nb.relative_to(d)
            on = f"  [{used_lang}]" if used_lang else ""
            print(f"  [{tag}] {label}/{rel}  {where}{on}")
            if status == 'ERROR':
                for line in (detail or '').split('||'):
                    print(f"           {line}")
                failures.append((label, nb.name, detail))
            else:
                print(f"           {note}")

    if tmp and not args.keep:
        subprocess.run(['rm', '-rf', tmp])

    print()
    # Say what was not looked at, every run. A narrowed scope that goes
    # unstated reads as a wider pass than it is.
    if skipped:
        print(f"{len(skipped)} archived notebook(s) skipped "
              f"(--archived to include):")
        for label, rel in skipped:
            print(f"  {label}/{rel}")
        print("An archive edited to work against current code is no longer an "
              "archive; drift inside one is expected, not actionable.\n")

    if failures:
        print(f"{len(failures)} of {attempted} notebook(s) do not construct.\n"
              "A paper that does not construct cannot be pinned to a release "
              "without freezing the breakage into the record: fix first, then "
              "pin.")
        return 1
    print(f"all {attempted} notebook(s) construct against this tree.")
    print("Construction only -- results are not reproduced here, by design.")
    print("This is a prefix check: a notebook that reaches compute is not "
          "executed past that cell,\nso later cells are unverified. See "
          "CONTRIBUTING.md.")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
