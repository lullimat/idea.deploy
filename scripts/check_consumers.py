#!/usr/bin/env python3
"""
Verify that every idpy module imported by a paper or collab still imports.

The reproducibility promise in the README is currently unverified: papers/ and
collabs/ import idpy by module path, and nothing checks that those paths still
resolve. Two failure modes have already occurred in this repository, both
silently:

  - API drift. idpy/LBM/test.py sat broken for a long time because
    ShanChenMultiPhase grew required parameters. Paper notebooks calling the
    same constructor had been kept current; the repository's own test had not.
  - Import-path change. This is what Phase 0b (the src/ restructure) does by
    construction, to every consumer at once.

This script catches the second class immediately and cheaply, without executing
a single notebook: it collects every `import idpy...` across the consumers and
tries to import each one. That is exactly the set a 0b shim layer must cover, so
the output doubles as the specification for those shims -- derived from what is
actually imported rather than from what someone remembers.

It deliberately does NOT run the notebooks. Executing a paper's simulation is
minutes-to-hours of GPU time and needs the full environment; that belongs in the
periodic smoke job Phase 0c describes. Import resolution is the fast 90%.

Most of that surface is untracked -- collabs/ is 1 tracked file out of 80,
papers/ is 2 out of 19, because the arXiv-* checkouts are separate gitignored
repositories. A fresh clone therefore cannot see what it must not break, and
neither can CI. So the surface is frozen into two committed fixtures, captured
from a tree where those files do exist, and verified anywhere:

    scripts/consumer-surface.txt   47 modules
    scripts/consumer-symbols.txt   277 (module, symbol) pairs

Both, because they catch different things: a shim that forwards the module but
drops a symbol passes the first and fails the second.

Usage:
    python3 scripts/check_consumers.py            # papers/, collabs/, tutorials/
    python3 scripts/check_consumers.py papers     # one area
    python3 scripts/check_consumers.py --list     # just print the module set

    python3 scripts/check_consumers.py --freeze          # regenerate fixture 1
    python3 scripts/check_consumers.py --freeze-symbols  # regenerate fixture 2
    python3 scripts/check_consumers.py --check-surface   # verify 1, anywhere
    python3 scripts/check_consumers.py --check-symbols   # verify 2, anywhere

Freeze only where papers/ and collabs/ are populated. Both refuse an empty
surface, but neither can detect a partial one.

Exit 0 when everything checked resolves, 1 otherwise -- including when nothing
was checked at all, which must never look like a pass.
"""

import ast
import importlib
import importlib.util
import json
import pathlib
import re
import sys

AREAS = ('papers', 'collabs', 'tutorials')

# Notebooks vendored inside a checkout of somebody else's environment are not
# consumers of this framework; skip the noise rather than reporting on it.
SKIP_PARTS = {
    '.git', '.ipynb_checkpoints', '__pycache__', 'node_modules',
    '.venv', 'venv', 'site-packages', 'py-env', '.mplcache',
}

IMPORT_RE = re.compile(r'\b(?:from|import)\s+(idpy(?:\.[A-Za-z_][A-Za-z0-9_]*)*)')


def _skip(path):
    return any(part in SKIP_PARTS for part in path.parts)


def modules_in_python(path):
    """Imports from a .py file, via AST so commented-out lines do not count."""
    try:
        tree = ast.parse(path.read_text(errors='ignore'))
    except (SyntaxError, ValueError, OSError):
        # OSError covers what a live research tree actually contains: dangling
        # editor lock symlinks (.#file.py), unreadable mounts, broken links.
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] == 'idpy':
                    found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0 \
                    and node.module.split('.')[0] == 'idpy':
                found.add(node.module)
    return found


def modules_in_notebook(path):
    """
    Imports from a .ipynb, read as JSON.

    Regex rather than AST on the source: notebook cells frequently do not parse
    standalone (magics, partial cells, shell escapes), and a parse failure would
    silently drop every import in the file.
    """
    try:
        doc = json.loads(path.read_text(errors='ignore'))
    except (json.JSONDecodeError, ValueError, OSError):
        return set()
    found = set()
    for cell in doc.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith('#'):
                continue
            found.update(IMPORT_RE.findall(line))
    return found


def collect(root, areas):
    """{module: {consumer paths}} across the requested areas."""
    usage = {}
    for area in areas:
        base = root / area
        if not base.is_dir():
            continue
        for path in base.rglob('*'):
            if path.suffix not in ('.py', '.ipynb') or _skip(path):
                continue
            if not path.is_file():          # dangling symlinks
                continue
            mods = (modules_in_python(path) if path.suffix == '.py'
                    else modules_in_notebook(path))
            for m in mods:
                usage.setdefault(m, set()).add(
                    str(path.relative_to(root))
                )
    return usage


FROM_RE = re.compile(
    r'^\s*from\s+(idpy(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+(.+?)\s*$')


def _symbols_from_tree(tree):
    """{(module, symbol)} for every `from idpy... import name` in an AST."""
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.module or node.level != 0:
            continue
        if node.module.split('.')[0] != 'idpy':
            continue
        for alias in node.names:
            found.add((node.module, alias.name))
    return found


def _symbols_from_text(text):
    """Regex fallback, for notebook cells that do not parse standalone."""
    found = set()
    for line in text.splitlines():
        m = FROM_RE.match(line)
        if not m:
            continue
        module, names = m.group(1), m.group(2)
        # Trailing comment, and the open paren of a parenthesised list.
        names = names.split('#')[0].strip().lstrip('(').rstrip(')').strip()
        for part in names.split(','):
            name = part.strip().split(' as ')[0].strip()
            if name:
                found.add((module, name))
    return found


def symbols_in_python(path):
    try:
        return _symbols_from_tree(ast.parse(path.read_text(errors='ignore')))
    except (SyntaxError, ValueError, OSError):
        return set()


def symbols_in_notebook(path):
    """
    Per cell: AST first, regex only where the cell will not parse.

    Cell-at-a-time rather than the whole notebook joined, because one cell
    containing a magic or a shell escape would otherwise cost every import in
    the file. AST first because it will not be fooled by a commented-out import
    or a string that looks like one.
    """
    try:
        doc = json.loads(path.read_text(errors='ignore'))
    except (json.JSONDecodeError, ValueError, OSError):
        return set()
    found = set()
    for cell in doc.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        try:
            found |= _symbols_from_tree(ast.parse(src))
        except (SyntaxError, ValueError):
            found |= _symbols_from_text(src)
    return found


def collect_symbols(root, areas):
    """{(module, symbol): {consumer paths}} across the requested areas."""
    usage = {}
    for area in areas:
        base = root / area
        if not base.is_dir():
            continue
        for path in base.rglob('*'):
            if path.suffix not in ('.py', '.ipynb') or _skip(path):
                continue
            if not path.is_file():
                continue
            syms = (symbols_in_python(path) if path.suffix == '.py'
                    else symbols_in_notebook(path))
            for s in syms:
                usage.setdefault(s, set()).add(str(path.relative_to(root)))
    return usage


def freeze_symbols(usage, path):
    """
    Record the symbol-level consumer surface.

    The module list is a lower bound on the shim surface. STRATEGY.md's
    migration mapping is written directory-to-directory, but several rows are
    rename-and-split -- idpy/IdpyCode -> src/idpy/core/ implies IdpyCode.py ->
    kernel.py, IdpyMemory.py -> memory.py. Where a symbol crosses module
    boundaries, a shim has to re-export it by name; a shim that forwards the
    module alone will import cleanly and then fail on attribute access.

    A star import is the worst case and is recorded as the symbol `*`: it
    consumes whatever the module exports, so the shim must reproduce the whole
    public surface rather than an enumerated list.
    """
    # Mark what is already broken, so the fixture records the tree as it is
    # rather than as it should be. Grandfathering follows check_layering.py's
    # KNOWN allowlist: a pre-existing breakage must stay visible without
    # failing every future run, and nothing new may join it silently.
    _missing, _, _ = _unreachable({k: len(v) for k, v in usage.items()})
    already = {(module, symbol) for module, symbol, _n, _why in _missing}
    by_module = {}
    for (module, symbol), consumers in usage.items():
        mark = '!' if (module, symbol) in already else ''
        by_module.setdefault(module, []).append(
            (symbol, len(consumers), mark))
    lines = [
        "# Symbol-level consumer surface: every `from idpy... import name` in",
        "# papers/, collabs/ and tutorials/, frozen from a tree where those",
        "# checkouts are present.",
        "# Regenerate with: python3 scripts/check_consumers.py --freeze-symbols",
        "# Verify anywhere with: python3 scripts/check_consumers.py --check-symbols",
        "#",
        "# Phase 0b must keep every one of these reachable at its old path. A",
        "# shim that forwards the module but drops a symbol passes",
        "# --check-surface and fails here, which is the point of having both.",
        "#",
        "# `*` means a star import: the shim must reproduce the module's whole",
        "# public surface, not an enumerated list.",
        "",
    ]
    lines.append("# A leading `!` marks a symbol that was ALREADY unreachable "
                 "when this was")
    lines.append("# frozen -- drift that predates the restructure. Those are "
                 "reported but do")
    lines.append("# not fail the check; anything else that breaks does.")
    lines.append("")
    for module in sorted(by_module):
        for symbol, n, mark in sorted(by_module[module]):
            lines.append(f"{mark}{module}\t{symbol}\t{n}")
    path.write_text('\n'.join(lines) + '\n')
    print(f"froze {len(usage)} (module, symbol) pairs across "
          f"{len(by_module)} modules to {path.name}"
          + (f"; {len(already)} already unreachable and grandfathered"
             if already else ""))


def load_symbols(path):
    """({(module, symbol): count}, {already-broken pairs}) from the fixture."""
    counts, known = {}, set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        grandfathered = line.startswith('!')
        parts = line.lstrip('!').split('\t')
        if len(parts) < 2:
            continue
        key = (parts[0], parts[1])
        counts[key] = int(parts[2]) if len(parts) > 2 else 0
        if grandfathered:
            known.add(key)
    return counts, known


def _unreachable(counts):
    """
    (missing, unavailable, stars) for a {(module, symbol): count} mapping.

    Module-absent and binding-absent are separated exactly as in report(): a
    machine always lacks some backend, and reporting that as a break would make
    this cry wolf everywhere.
    """
    missing, unavailable, stars = [], {}, 0
    modules = sorted({m for m, _ in counts})
    loaded = {}
    for module in modules:
        try:
            loaded[module] = importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name and not exc.name.startswith('idpy'):
                unavailable[module] = exc.name
            else:
                loaded[module] = None
        except Exception:                                  # noqa: BLE001
            loaded[module] = None

    for (module, symbol), n in sorted(counts.items()):
        if module in unavailable:
            continue
        mod = loaded.get(module)
        if mod is None:
            missing.append((module, symbol, n, 'module does not import'))
            continue
        if symbol == '*':
            stars += 1
            continue
        if not hasattr(mod, symbol):
            # `from pkg import submodule` is valid even though the parent has
            # no such attribute until the submodule is imported, so hasattr
            # alone would report working code as broken.
            try:
                importlib.import_module(f"{module}.{symbol}")
                continue
            except ImportError:
                pass
            missing.append((module, symbol, n, 'not exported by the module'))

    return missing, unavailable, stars


def check_symbols(counts, known=frozenset()):
    """Verify each recorded symbol is reachable at its recorded module path."""
    missing, unavailable, stars = _unreachable(counts)
    fresh = [m for m in missing if (m[0], m[1]) not in known]
    stale = [m for m in missing if (m[0], m[1]) in known]

    checked = len(counts) - sum(1 for m, _ in counts if m in unavailable)
    modules = {m for m, _ in counts}
    print(f"{len(counts)} (module, symbol) pairs across {len(modules)} "
          f"modules; {stars} star imports")
    if unavailable:
        print(f"{len(unavailable)} module(s) skipped for absent backend "
              f"bindings: {', '.join(sorted(unavailable))}")
    if stale:
        print(f"\n{len(stale)} grandfathered breakage(s), predating this work:")
        for module, symbol, n, why in stale:
            print(f"  [known] {module}.{symbol} -- {why}; "
                  f"{n} consumer{'s' if n != 1 else ''}")

    if fresh:
        print(f"\n{len(fresh)} symbol(s) are no longer reachable:\n")
        for module, symbol, n, why in fresh:
            print(f"  {module}.{symbol}")
            print(f"    {why}; {n} consumer{'s' if n != 1 else ''}")
        print("\nEach is imported by a published paper or an active "
              "collaboration.\nIf a deliberate move caused this, the shim for "
              "that path is missing a symbol.")
        return 1

    if not checked:
        print("\nnothing was checked, which is not a pass")
        return 1
    print(f"\nall {checked - len(stale)} reachable symbols resolve at their "
          f"recorded paths.")
    return 0


def freeze(usage, path):
    """
    Record the consumer surface as a committed fixture.

    Necessary because the consumers are mostly untracked: collabs/ is 1 tracked
    file out of 80, papers/ is 2 out of 19 (the arXiv-* checkouts are separate
    gitignored repositories). A fresh clone therefore cannot see the surface it
    must not break, and neither can CI.

    Freezing it here, from the tree where those files do exist, turns "does this
    restructure break a live simulation?" into a question a machine without
    those files can answer.
    """
    lines = [
        "# Consumer surface: idpy modules imported by papers/, collabs/ and",
        "# tutorials/, frozen from a tree where those checkouts are present.",
        "# Regenerate with: python3 scripts/check_consumers.py --freeze",
        "# Verify anywhere with: python3 scripts/check_consumers.py --check-surface",
        "#",
        "# Phase 0b must keep every one of these importable, via shims if the",
        "# module has moved. The counts say which breakages would hurt most.",
        "",
    ]
    lines += [f"{m}\t{len(usage[m])}" for m in sorted(usage)]
    path.write_text('\n'.join(lines) + '\n')
    print(f"froze {len(usage)} modules to {path.name}")


def load_surface(path):
    """{module: recorded consumer count} from the fixture."""
    counts = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        module, _, count = line.partition('\t')
        counts[module] = int(count) if count else 0
    return counts


def main(argv):
    root = pathlib.Path(__file__).resolve().parent.parent
    # Kept for the tutorials/ and collabs/ helper modules, which are read from
    # the repository root. It no longer makes idpy importable: as of Phase 0b
    # the package lives under src/, so `import idpy` resolves to the installed
    # distribution or to nothing at all. That is deliberate -- this script
    # checks what a consumer would get, and a consumer gets the installed
    # package. Run it in an environment where idpy is installed; otherwise
    # every module reports ModuleNotFoundError and the run looks like a total
    # breakage rather than a missing install.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    args = [a for a in argv[1:] if not a.startswith('-')]
    areas = tuple(args) if args else AREAS
    flags = set(a for a in argv[1:] if a.startswith('-'))
    list_only = '--list' in flags
    surface = root / 'scripts' / 'consumer-surface.txt'
    symbols_fixture = root / 'scripts' / 'consumer-symbols.txt'

    if '--check-symbols' in flags:
        if not symbols_fixture.is_file():
            print(f"no fixture at {symbols_fixture} -- run --freeze-symbols on "
                  f"a tree that has papers/ and collabs/ populated")
            return 1
        counts, known = load_symbols(symbols_fixture)
        return check_symbols(counts, known)

    if '--freeze-symbols' in flags:
        sym = collect_symbols(root, areas)
        if not sym:
            print("refusing to freeze an empty symbol surface -- run this "
                  "where papers/ and collabs/ are populated")
            return 1
        freeze_symbols(sym, symbols_fixture)
        return 0

    if '--check-surface' in flags:
        # The mode a fresh clone runs: verify the frozen surface without
        # needing the (mostly untracked) consumers on disk.
        if not surface.is_file():
            print(f"no fixture at {surface} -- run --freeze on a tree that has "
                  f"papers/ and collabs/ populated")
            return 1
        counts = load_surface(surface)
        usage = {m: set() for m in counts}
        print(f"{len(usage)} modules in the frozen consumer surface "
              f"({surface.name})\n")
        return report(usage, counts, len(usage))

    usage = collect(root, areas)
    if '--freeze' in flags:
        if not usage:
            print("refusing to freeze an empty surface -- run this where "
                  "papers/ and collabs/ are populated")
            return 1
        freeze(usage, surface)
        return 0

    if not usage:
        # Non-zero deliberately. Most of the consumer surface is untracked --
        # collabs/ is 1 tracked file out of 80, papers/ is 2 out of 19 -- so a
        # fresh clone finds almost nothing here and would otherwise report a
        # silent pass having checked nothing. Use --check-surface against the
        # frozen fixture when the consumers are not on disk.
        print(f"no idpy imports found under {', '.join(areas)}/ "
              f"-- nothing was checked, which is not a pass")
        return 1

    if list_only:
        for m in sorted(usage):
            print(m)
        return 0

    print(f"{len(usage)} distinct idpy modules imported across "
          f"{', '.join(areas)}/\n")
    return report(usage, {m: len(v) for m, v in usage.items()})


def report(usage, counts, _=None):
    """
    Two failures look alike and mean opposite things.

      module absent from the tree      -> a consumer is genuinely broken. This
                                          is what the 0b restructure causes.
      module present, binding missing  -> the machine lacks pycuda/pyopencl/
                                          pymetallic. An environment fact, true
                                          of every machine for some backend, and
                                          not a break.

    find_spec answers the first; the name on the ModuleNotFoundError answers the
    second. Reporting them together would make this cry wolf everywhere.
    """
    broken, unavailable = [], []
    for module in sorted(usage):
        try:
            if importlib.util.find_spec(module) is None:
                broken.append((module, ModuleNotFoundError(
                    f"no module named {module!r} in this tree"),
                    sorted(usage[module])))
                continue
        except (ImportError, ValueError) as exc:
            broken.append((module, exc, sorted(usage[module])))
            continue
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name and not exc.name.startswith('idpy'):
                unavailable.append((module, exc.name))
            else:
                broken.append((module, exc, sorted(usage[module])))
        except Exception as exc:                       # noqa: BLE001
            broken.append((module, exc, sorted(usage[module])))

    width = max(len(m) for m in usage)
    skipped = {m: dep for m, dep in unavailable}
    for module in sorted(usage):
        n = counts.get(module, len(usage[module]))
        if any(module == b[0] for b in broken):
            tag = 'FAIL'
        elif module in skipped:
            tag = 'skip'
        else:
            tag = 'ok  '
        note = f"   (needs {skipped[module]})" if module in skipped else ''
        print(f"  [{tag}] {module:<{width}}  "
              f"{n} consumer{'s' if n != 1 else ''}{note}")

    if broken:
        print(f"\n{len(broken)} module(s) no longer import:\n")
        for module, exc, consumers in broken:
            print(f"  {module}")
            print(f"    {type(exc).__name__}: {exc}")
            for c in consumers[:4]:
                print(f"    used by: {c}")
            if len(consumers) > 4:
                print(f"    ... and {len(consumers) - 4} more")
            if not consumers:
                print(f"    used by: {counts.get(module, '?')} consumers "
                      f"(frozen surface; the files are not in this tree)")
        print("\nA published paper or an active collaboration imports each of "
              "these.\nIf this broke via a deliberate move, the shim for that "
              "path is missing.")
        return 1

    print(f"\nall {len(usage) - len(unavailable)} resolvable modules import; "
          f"{len(unavailable)} skipped for absent backend bindings.")
    print("No consumer is broken by the current tree.")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
