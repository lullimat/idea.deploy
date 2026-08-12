#!/usr/bin/env python3
"""
Enforce the architectural invariant: idpy core never imports idpy physics.

    STRATEGY.md: "`idpy.core` never imports from `idpy.physics`. Ever."

Phase 0b built those package names, so this now checks the invariant directly
against src/idpy/core/ rather than against a mapping of the old flat layout.

The scanned-file count is not decoration. Before the restructure this script
walked root/idpy/<area>; the moment Phase 0b moved everything under src/, those
directories stopped existing, rglob returned nothing, and the script printed
"layering OK" and exited 0 having examined zero files. It stayed green through
the single commit most capable of breaking the invariant it protects. A run
that checked nothing must fail, on the same principle CONTRIBUTING.md states
for the print-style suites: "0 checks ran" must never look like "all checks
passed".

One known violation is listed in KNOWN below. It is a function-local import, so
idpy.core.utils still *loads* cleanly without physics -- the dependency is a
runtime one rather than an import-time one, which is why the restructure stayed
mechanical despite it. It is grandfathered rather than fixed here; nothing new
may join it.

Static analysis only: this walks import statements without importing anything,
so it needs no idpy environment, no GPU and no third-party packages, and can run
as the first step of CI before dependencies are installed.

Exit 0 clean, 1 on a new violation or on an empty scan.
"""

import ast
import pathlib
import sys

# The layers, as they now exist on disk.
CORE = pathlib.Path('src') / 'idpy' / 'core'
PHYSICS_ROOT = 'idpy.physics'

# The old dotted paths still resolve through the generated compatibility shims,
# so `from idpy.LBM...` inside core would violate the invariant just as surely
# as the new spelling while looking like neither. Both are checked.
PHYSICS_LEGACY = {'LBM', 'IdpyStencils', 'SpinNetworks', 'PRNGS'}

# Grandfathered. Phase 0b was the mechanical move; inverting this dependency is
# refactoring work and belongs with the backend-protocol pass, not with a
# restructure whose whole claim is that it changed no behaviour.
KNOWN = {
    ('src/idpy/core/utils/IdpySymbolic.py', 'idpy.physics.stencils.IdpyConvolution'),
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


def crosses_layer(module):
    """Does this dotted import name reach into the physics layer?"""
    if module == PHYSICS_ROOT or module.startswith(PHYSICS_ROOT + '.'):
        return True
    parts = module.split('.')
    return len(parts) > 1 and parts[0] == 'idpy' and parts[1] in PHYSICS_LEGACY


def violations(root):
    """(found, n_scanned) -- the count is what makes an empty scan detectable."""
    found, scanned = [], 0
    for path in sorted((root / CORE).rglob('*.py')):
        if path.name.endswith('~'):
            continue
        try:
            tree = ast.parse(path.read_text(errors='ignore'))
        except SyntaxError:
            # A file that does not parse is not this check's problem; the
            # test suites will fail on it far more informatively.
            continue
        scanned += 1
        rel = path.relative_to(root).as_posix()
        for lineno, module in imported_modules(tree):
            if crosses_layer(module) and (rel, module) not in KNOWN:
                found.append((rel, lineno, module))
    return found, scanned


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    found, scanned = violations(root)

    if not scanned:
        print(f"no Python files under {CORE.as_posix()}/ -- nothing was "
              f"checked, which is not a pass.\n"
              f"The core layer has moved or been renamed; update CORE in this "
              f"script to point at it.")
        return 1

    if not found:
        print(f"layering OK: no new core -> physics imports in {scanned} file(s) "
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
