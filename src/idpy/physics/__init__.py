"""
idpy.physics -- the physics layer: models built on top of idpy.core.

The one architectural invariant is directional:

    idpy.core never imports from idpy.physics. Ever.

Enforced by scripts/check_layering.py, which runs first in CI because it is
static analysis over import statements and needs no environment at all.

This package is deliberately empty of code. Importing idpy.physics must not
drag in lbm, stencils, spin_networks or prngs: they carry heavy optional
dependencies (sympy solving, networkx, scikit-learn) and a consumer that wants
one has no reason to pay for the other three.
"""
