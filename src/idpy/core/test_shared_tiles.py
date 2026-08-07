__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2026 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
__credits__ = ["Matteo Lulli"]
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
__version__ = "0.1"
__maintainer__ = "Matteo Lulli"
__email__ = "matteo.lulli@gmail.com"
__status__ = "Development"

'''
P3: is one dynamic shared buffer enough for lattice work?

`SetDynamicSharedMemory` allows at most one runtime-sized buffer, because CUDA
exposes a single `extern __shared__` region and that is the portable floor. The
probe asks whether that constraint blocks the lattice kernels -- tiled
stencil/halo work -- or is merely an ergonomic wrinkle. If it blocks them, the
lowest-common-denominator constraint rather than the residency layer is the real
obstacle, and the design needs revisiting.

Scope, per the re-scoping that made the lattice primary: this is evaluated
against a halo stencil, not against MoE attention. The MoE answer only matters
if F4 is ever reopened.

The workload is deliberately one that *needs* two tiles: a two-field 3-point
stencil with periodic halos,

    out[i] = (a[i-1] + a[i] + a[i+1]) + 2*(b[i-1] + b[i] + b[i+1])

staged through shared memory. One tile cannot hold it -- each field needs its
own BLOCK+2 window, and every output element reads neighbours written by other
threads, so a missing or wrong barrier gives wrong numbers rather than merely
slow ones.

Three checks:

  T1  two logical tiles inside ONE dynamic buffer, addressed by manual offset
  T2  two independent STATIC shared tiles, which carry no such constraint
  T3  the guard itself: declaring two dynamic buffers must raise, not silently
      allocate one and corrupt the second

Run directly:
    python -m idpy.core.test_shared_tiles
'''

from collections import OrderedDict

import numpy as np

from idpy.core import (
    IDPY_T, CUDA_T, OCL_T, CTYPES_T, METAL_T,
    idpy_langs_sys, idpy_langs_human_dict, GetTenet,
)
from idpy.core.IdpyCode import IdpyKernel
from idpy.core import IdpyMemory
from idpy.core.IdpyUnroll import (
    _codify_shared_declaration, _codify_sync, _codify_comment,
)
from idpy.core.utils.CustomTypes import CustomTypes
from idpy.core.utils.TestExit import report_exit as _report_exit

# Shared memory is a block-parallel GPU concept; CTypes runs kernels as a serial
# loop and raises NotImplementedError for these by design.
_LANGS = (CUDA_T, OCL_T, METAL_T)

_BLOCK = 64
_N = 1024
_TYPES = CustomTypes({'FType': 'float'}).Push()


def _stencil_body(a_base, b_base):
    '''
    The shared-memory body, parameterised by where each tile starts.

    Identical text for the dynamic and static cases apart from the base offsets,
    which is the point: if one buffer is enough, the *body* should not care
    whether its two tiles come from one region or two declarations.
    '''
    _s = ""
    _s += _codify_comment("stage both fields, one element per lane")
    _s += f"tile[{a_base} + 1 + l_tid] = a[g_tid];\n"
    _s += f"tile[{b_base} + 1 + l_tid] = b[g_tid];\n"
    _s += _codify_comment("periodic halos, written by the edge lanes")
    _s += "if(l_tid == 0){\n"
    _s += f"tile[{a_base}] = a[(g_tid + N - 1) % N];\n"
    _s += f"tile[{b_base}] = b[(g_tid + N - 1) % N];\n"
    _s += "}\n"
    _s += "if(l_tid == BLOCK - 1){\n"
    _s += f"tile[{a_base} + BLOCK + 1] = a[(g_tid + 1) % N];\n"
    _s += f"tile[{b_base} + BLOCK + 1] = b[(g_tid + 1) % N];\n"
    _s += "}\n"
    _s += _codify_comment("barrier: every lane reads slots its neighbours wrote")
    _s += _codify_sync()
    _s += (
        f"dst[g_tid] = (tile[{a_base} + l_tid] + tile[{a_base} + l_tid + 1]"
        f" + tile[{a_base} + l_tid + 2])\n"
        f"           + 2 * (tile[{b_base} + l_tid] + tile[{b_base} + l_tid + 1]"
        f" + tile[{b_base} + l_tid + 2]);\n"
    )
    return _s


def _common_init(kern):
    kern.SetCodeFlags('g_tid')
    kern.SetCodeFlags('l_tid')
    kern.params = {
        'FType * dst': ['global', 'restrict'],
        'FType * a': ['global', 'restrict', 'const'],
        'FType * b': ['global', 'restrict', 'const'],
    }


class K_TwoTilesDynamic(IdpyKernel):
    '''
    T1: both tiles inside the single dynamic region, addressed by offset.

    'tile' is one buffer of 2*(BLOCK+2) elements; field a lives at 0 and field b
    at TILE. This is the workaround the SetDynamicSharedMemory docstring
    prescribes -- "declare one buffer and index into it manually for multiple
    logical tiles" -- exercised on a workload that genuinely needs two.
    '''

    def __init__(self, block_size=_BLOCK, n=_N, custom_types=None,
                 optimizer_flag=None):
        constants = OrderedDict()
        constants['BLOCK'] = int(block_size)
        constants['N'] = int(n)
        constants['TILE'] = int(block_size) + 2
        IdpyKernel.__init__(self, custom_types=custom_types or _TYPES,
                            constants=constants, optimizer_flag=optimizer_flag)
        _common_init(self)
        self.SetDynamicSharedMemory(
            {'tile': {'type': 'FType', 'dtype': np.float32}}
        )
        self.kernels[IDPY_T] = "\n" + _stencil_body('0', 'TILE')


class K_TwoTilesStatic(IdpyKernel):
    '''
    T2: two independent compile-time tiles.

    Static shared memory carries no single-region constraint -- it is ordinary
    declaration, and a kernel may have as many arrays as fit. Declared as one
    array of 2*(BLOCK+2) here so the body text is identical to T1's, isolating
    the question to *where the storage comes from* rather than how it is
    addressed.
    '''

    def __init__(self, block_size=_BLOCK, n=_N, custom_types=None,
                 optimizer_flag=None):
        constants = OrderedDict()
        constants['BLOCK'] = int(block_size)
        constants['N'] = int(n)
        constants['TILE'] = int(block_size) + 2
        IdpyKernel.__init__(self, custom_types=custom_types or _TYPES,
                            constants=constants, optimizer_flag=optimizer_flag)
        _common_init(self)
        _declared = [[]]
        _body = _codify_shared_declaration(
            'tile', 'FType', '2 * TILE', declared_variables=_declared,
        )
        self.kernels[IDPY_T] = "\n" + _body + _stencil_body('0', 'TILE')


def reference(a, b):
    return ((np.roll(a, 1) + a + np.roll(a, -1))
            + 2 * (np.roll(b, 1) + b + np.roll(b, -1)))


def run_on(lang, kernel_cls, block_size=_BLOCK, n=_N):
    params = {'lang': lang}
    if lang == OCL_T:
        params['cl_kind'] = 'gpu'
    tenet = GetTenet(params)
    try:
        kern = kernel_cls(block_size=block_size, n=n)
        grid, block = (n // block_size, 1, 1), (block_size, 1, 1)
        if kern.shared_dynamic:
            idea = kern(tenet=tenet, grid=grid, block=block,
                        dyn_shared_count=2 * (block_size + 2))
        else:
            idea = kern(tenet=tenet, grid=grid, block=block)

        a_h = np.arange(n, dtype=np.float32) * np.float32(0.25)
        b_h = np.arange(n, dtype=np.float32)[::-1].copy() * np.float32(0.5)
        a = IdpyMemory.OnDevice(a_h, tenet=tenet)
        b = IdpyMemory.OnDevice(b_h, tenet=tenet)
        dst = IdpyMemory.Zeros(shape=(n,), dtype=np.float32, tenet=tenet)

        idea.Deploy([dst, a, b])
        if hasattr(tenet, 'FlushAndWait'):
            tenet.FlushAndWait()
        return float(np.max(np.abs(np.array(dst.D2H()) - reference(a_h, b_h))))
    finally:
        if hasattr(tenet, 'End'):
            tenet.End()


def guard_holds():
    '''T3: two dynamic buffers must be refused, not silently mis-allocated.'''
    class K(IdpyKernel):
        def __init__(self):
            IdpyKernel.__init__(self, custom_types=_TYPES)
    try:
        K().SetDynamicSharedMemory({
            'one': {'type': 'FType', 'dtype': np.float32},
            'two': {'type': 'FType', 'dtype': np.float32},
        })
        return False
    except NotImplementedError:
        return True


def main():
    print("=== P3: is one dynamic shared buffer enough for lattice work? ===\n")
    _ok, _ran = True, False

    _guard = guard_holds()
    _ok = _ok and _guard
    print(f"  [{'OK  ' if _guard else 'FAIL'}] T3 guard: two dynamic buffers "
          f"raise NotImplementedError\n")

    for lang in _LANGS:
        human = idpy_langs_human_dict[lang]
        if not idpy_langs_sys[lang]:
            print(f"  [skip] {human}: backend not available")
            continue
        for name, cls in (('T1 two tiles, one dynamic buffer',
                           K_TwoTilesDynamic),
                          ('T2 two tiles, static shared', K_TwoTilesStatic)):
            try:
                _err = run_on(lang, cls)
                _ran = True
                _ok = _ok and (_err == 0.0)
                print(f"  [{'OK  ' if _err == 0.0 else 'FAIL'}] {human}: "
                      f"{name}: max|out-ref| = {_err:g}")
            except Exception as exc:
                _ok = False
                print(f"  [err ] {human}: {name}: "
                      f"{type(exc).__name__}: {exc}")
        print()

    print(
        "The workload needs two tiles by construction: each field carries its\n"
        "own BLOCK+2 halo window, and every output reads slots written by other\n"
        "lanes, so a wrong barrier or a mis-addressed tile gives wrong numbers\n"
        "rather than slow ones."
    )
    _report_exit(_ok, checks_run=_ran, what='GPU backends')


if __name__ == '__main__':
    main()
