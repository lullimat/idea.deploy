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


def _alarm(signum, frame):
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


def run_notebook(path, root, budget):
    """
    (status, cell_index, n_cells, detail) for one notebook.

    status is REACHED_COMPUTE (good), COMPLETED (good), or ERROR.
    """
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
            if not src.strip():
                continue
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
                tail = traceback.format_exc().strip().splitlines()[-3:]
                return ('ERROR', i, len(cells),
                        f"{type(exc).__name__}: {exc}||" + '||'.join(tail))
        return ('COMPLETED', len(cells), len(cells), None)
    finally:
        signal.alarm(0)
        os.chdir(cwd)


def run_isolated(path, root, budget):
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
         '--run-one', str(path), '--budget', str(budget)],
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


def notebooks_in(d):
    return sorted(p for p in pathlib.Path(d).rglob('*.ipynb')
                  if '.git' not in p.parts and '.ipynb_checkpoints' not in p.parts)


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
    ap.add_argument('--run-one', help=argparse.SUPPRESS)
    args = ap.parse_args(argv[1:])

    if args.run_one:
        # The isolated child: one notebook, one interpreter, JSON on stdout.
        status, at, total, detail = run_notebook(
            args.run_one, root, args.budget)
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

    failures, attempted = [], 0
    for label, d in targets:
        nbs = notebooks_in(d)
        if not nbs:
            print(f"  [no notebook] {label}")
            continue
        for nb in nbs:
            attempted += 1
            status, at, total, detail = run_isolated(nb, root, args.budget)
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
            print(f"  [{tag}] {label}/{rel}  {where}")
            if status == 'ERROR':
                for line in (detail or '').split('||'):
                    print(f"           {line}")
                failures.append((label, nb.name, detail))
            else:
                print(f"           {note}")

    if tmp and not args.keep:
        subprocess.run(['rm', '-rf', tmp])

    print()
    if failures:
        print(f"{len(failures)} of {attempted} notebook(s) do not construct.\n"
              "A paper that does not construct cannot be pinned to a release "
              "without freezing the breakage into the record: fix first, then "
              "pin.")
        return 1
    print(f"all {attempted} notebook(s) construct against this tree.")
    print("Construction only -- results are not reproduced here, by design.")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
