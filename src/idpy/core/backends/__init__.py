"""
idpy.core.backends -- one subpackage per lowering target.

Each backend is a *package*, not a module, and that split is load-bearing
rather than stylistic:

    cuda/__init__.py    defines CUDA_T only -- cheap, always importable
    cuda/backend.py     imports pycuda -- present only where the hardware is

idpy.core's __init__ imports the four tokens eagerly (it needs them to build
idpy_langs_sys) and the four implementations only behind an AreModulesThere
guard. Collapsing a backend into a single module would put `import pycuda` on
the eager path and make `import idpy.core` fail on every machine without a GPU
binding -- which is most of them, including CI.

STRATEGY.md's migration table maps each backend directory to a single file
(core/backends/cuda.py). That row cannot be implemented as written, for the
reason above; the package form is what preserves its intent.

Importing this package imports no backend. Ask idpy.core for the one you want.
"""
