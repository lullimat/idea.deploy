#!/usr/bin/env python3
"""
Enforce the architectural invariant: idpy core never imports idpy physics.

    STRATEGY.md: "`idpy.core` never imports from `idpy.physics`. Ever."

The restructure that creates those package names (Phase 0b) has not happened
yet, so this checks the invariant against the *current* layout using the
migration mapping in STRATEGY.md §3. Checking it now is the point: the
restructure is only cheap while the layering is already clean, and layering rots
silently. Every accidental core->physics import added between now and then is
one more thing to unpick later, at a moment when everything else is also moving.

There are two known violations, listed in KNOWN below. They are deliberately
grandfathered rather than fixed here -- unpicking them is refactoring work that
belongs with Phase 0b -- but nothing *new* can be added. The allowlist is the
debt, written down where it will be seen.

Static analysis only: this walks import statements without importing anything,
so it needs no idpy environment, no GPU and no third-party packages, and can run
as the first step of CI before dependencies are installed.

Exit 0 clean, 1 on a new violation.
"""

import ast
import pathlib
import sys

# STRATEGY.md §3 migration mapping, in current-layout terms.
CORE = ('IdpyCode', 'CUDA', 'OpenCL', 'CTypes', 'Metal', 'Utils')
PHYSICS = {'LBM', 'IdpyStencils', 'SpinNetworks', 'PRNGS'}

# Grandfathered, to be removed by Phase 0b. Both are function-local imports, so
# idpy.Utils still *loads* cleanly without physics -- the cycle is a runtime
# dependency rather than an import-time one, which is why the restructure stays
# mechanical despite them.
KNOWN = {
    ('idpy/Utils/IdpySymbolic.py', 'idpy.IdpyStencils.IdpyConvolution'),
}


def imported_modules(tree):
    """Yield (lineno, dotted module name) for every import in a parsed file."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import: it cannot name a sibling package
            # of idpy, so it can never cross the core/physics line.
            if node.module and node.level == 0:
                yield node.lineno, node.module


def violations(root):
    found = []
    for area in CORE:
        for path in sorted((root / 'idpy' / area).rglob('*.py')):
            if path.name.endswith('~'):
                continue
            try:
                tree = ast.parse(path.read_text(errors='ignore'))
            except SyntaxError:
                # A file that does not parse is not this check's problem; the
                # test suites will fail on it far more informatively.
                continue
            rel = path.relative_to(root).as_posix()
            for lineno, module in imported_modules(tree):
                parts = module.split('.')
                if len(parts) > 1 and parts[0] == 'idpy' and parts[1] in PHYSICS:
                    if (rel, module) in KNOWN:
                        continue
                    found.append((rel, lineno, module))
    return found


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    found = violations(root)

    if not found:
        print(f"layering OK: no new core -> physics imports "
              f"({len(KNOWN)} grandfathered)")
        return 0

    print("core -> physics imports are not allowed "
          "(STRATEGY.md: 'idpy.core never imports from idpy.physics'):\n")
    for rel, lineno, module in found:
        print(f"  {rel}:{lineno}: imports {module}")
    print("\nMove the shared piece down into core, invert the dependency, or -- "
          "if it is genuinely unavoidable -- add it to KNOWN in this script "
          "with a reason, so the debt is recorded rather than absorbed.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
