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
Cross-architecture demonstration / test of the shared-memory metalanguage.

Exercises the portable IDPY_T tokens added on top of the existing kernel
metalanguage:

    idpy_shared         block/threadgroup-shared address qualifier
                        (CUDA __shared__ | OpenCL __local | Metal threadgroup)
    idpy_sync           collective barrier + shared(local) memory fence
                        (CUDA __syncthreads() | OpenCL barrier(...) |
                         Metal threadgroup_barrier(...))

The demo kernel 'K_SharedNeighborSum' does the canonical collective pattern:

    idpy_shared FType tile[BLOCK];   // one tile per CUDA block / OpenCL
                                     // work-group / Metal threadgroup
    tile[l_tid] = src[g_tid];        // every thread writes its own slot
    idpy_sync;                       // <-- barrier: without it, the next line
                                     //     races on a neighbour's store
    dst[g_tid] = tile[l_tid] + tile[(l_tid + 1) % BLOCK];

Each output therefore reads a slot written by a *different* thread, so a wrong
or missing barrier yields wrong results -- making this a real test of the
synchronization primitive, not just of code generation.

Run directly:
    python -m idpy.core.test_shared
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
    _codify_shared_declaration,
    _codify_sync,
    _codify_assignment,
    _codify_comment,
    _array_value,
)
from idpy.core.utils.CustomTypes import CustomTypes
from idpy.core.utils.TestExit import report_exit as _report_exit


class K_SharedNeighborSum(IdpyKernel):
    '''
    dst[i] = src[i] + src[block_base + (i_local + 1) % BLOCK]

    computed by staging a BLOCK-sized shared tile, synchronizing, then reading
    the neighbouring lane's slot. One source body (IDPY_T) compiles to CUDA,
    OpenCL and Metal via the idpy_shared / idpy_sync tokens.
    '''

    def __init__(self, block_size=64, n=1024,
                 custom_types=None, optimizer_flag=None):
        if n % block_size != 0:
            raise ValueError(
                "n must be a multiple of block_size so that grid*block == n "
                "and every thread participates uniformly in idpy_sync"
            )
        if custom_types is None:
            # float keeps parity across Metal (no fp64) and OpenCL fp32-only GPUs
            custom_types = CustomTypes({'FType': 'float'}).Push()

        constants = OrderedDict()
        constants['BLOCK'] = int(block_size)
        constants['N'] = int(n)

        IdpyKernel.__init__(
            self, custom_types=custom_types, constants=constants,
            optimizer_flag=optimizer_flag,
        )
        # g_tid: global lane index; l_tid: index within the block/threadgroup
        self.SetCodeFlags('g_tid')
        self.SetCodeFlags('l_tid')

        self.params = {
            'FType * dst': ['global', 'restrict'],
            'FType * src': ['global', 'restrict', 'const'],
        }

        declared_variables = [[]]
        body = ""
        body += _codify_comment("collectively stage a BLOCK-sized shared tile")
        body += _codify_shared_declaration(
            'tile', 'FType', 'BLOCK', declared_variables=declared_variables,
        )
        body += _codify_assignment(
            _array_value('tile', 'l_tid'), _array_value('src', 'g_tid'),
        )
        body += _codify_comment("barrier: publish every lane's store to the tile")
        body += _codify_sync()
        body += _codify_comment("read the slot written by the neighbouring lane")
        body += _codify_assignment(
            _array_value('dst', 'g_tid'),
            _array_value('tile', 'l_tid') + " + "
            + _array_value('tile', '(l_tid + 1) % BLOCK'),
        )
        self.kernels[IDPY_T] = "\n" + body

    def dump_code(self, lang=IDPY_T):
        return self.Code(lang)


class K_SharedNeighborSumDynamic(IdpyKernel):
    '''
    Same computation as K_SharedNeighborSum, but the tile lives in *runtime-sized*
    (dynamic) shared memory declared via SetDynamicSharedMemory rather than a
    compile-time 'idpy_shared FType tile[BLOCK];'. The size is fixed per launch:
        CUDA   -> extern __shared__ FType tile[];   + launch shared= bytes
        OpenCL -> __local FType * tile              + cl.LocalMemory(bytes)
        Metal  -> threadgroup FType * tile [[threadgroup(0)]]
                                                    + set_threadgroup_memory_length
    '''

    def __init__(self, block_size=64, n=1024,
                 custom_types=None, optimizer_flag=None):
        if n % block_size != 0:
            raise ValueError("n must be a multiple of block_size")
        if custom_types is None:
            custom_types = CustomTypes({'FType': 'float'}).Push()

        constants = OrderedDict()
        constants['BLOCK'] = int(block_size)
        constants['N'] = int(n)

        IdpyKernel.__init__(
            self, custom_types=custom_types, constants=constants,
            optimizer_flag=optimizer_flag,
        )
        self.SetCodeFlags('g_tid')
        self.SetCodeFlags('l_tid')
        # Runtime-sized shared buffer 'tile'; np.float32 matches FType for sizing
        self.SetDynamicSharedMemory({'tile': {'type': 'FType', 'dtype': np.float32}})

        self.params = {
            'FType * dst': ['global', 'restrict'],
            'FType * src': ['global', 'restrict', 'const'],
        }

        # No in-body tile declaration: 'tile' is a kernel param (OpenCL/Metal)
        # or an extern __shared__ region (CUDA), injected by the framework.
        body = ""
        body += _codify_comment("collectively stage the dynamic shared tile")
        body += _codify_assignment(
            _array_value('tile', 'l_tid'), _array_value('src', 'g_tid'),
        )
        body += _codify_comment("barrier: publish every lane's store to the tile")
        body += _codify_sync()
        body += _codify_comment("read the slot written by the neighbouring lane")
        body += _codify_assignment(
            _array_value('dst', 'g_tid'),
            _array_value('tile', 'l_tid') + " + "
            + _array_value('tile', '(l_tid + 1) % BLOCK'),
        )
        self.kernels[IDPY_T] = "\n" + body

    def dump_code(self, lang=IDPY_T):
        return self.Code(lang)


def reference(src, block_size):
    '''Host reference: per-block circular neighbour sum.'''
    n = src.shape[0]
    out = np.empty_like(src)
    for b in range(n // block_size):
        blk = src[b * block_size:(b + 1) * block_size]
        out[b * block_size:(b + 1) * block_size] = blk + np.roll(blk, -1)
    return out


def run_on(lang, block_size=64, n=1024, tenet_params=None, kernel_cls=None):
    if kernel_cls is None:
        kernel_cls = K_SharedNeighborSum
    tenet = GetTenet(tenet_params if tenet_params is not None else {'lang': lang})

    kern = kernel_cls(block_size=block_size, n=n)
    grid = (n // block_size, 1, 1)
    block = (block_size, 1, 1)
    # Dynamic-shared kernels size the tile per launch; default (one element per
    # thread == BLOCK here) is used, but pass it explicitly to exercise the API.
    if kern.shared_dynamic:
        idea = kern(tenet=tenet, grid=grid, block=block,
                    dyn_shared_count=block_size)
    else:
        idea = kern(tenet=tenet, grid=grid, block=block)

    src_h = np.arange(n, dtype=np.float32)
    src = IdpyMemory.OnDevice(src_h, tenet=tenet)
    dst = IdpyMemory.Zeros(shape=(n,), dtype=np.float32, tenet=tenet)

    idea.Deploy([dst, src])
    tenet.FlushAndWait() if hasattr(tenet, 'FlushAndWait') else None
    out = dst.D2H()

    ref = reference(src_h, block_size)
    ok = np.allclose(out, ref, rtol=0, atol=0)
    max_abs = float(np.max(np.abs(out - ref)))
    tenet.End() if hasattr(tenet, 'End') else None
    return ok, max_abs, out, ref


def dump_sources(kernel_cls, block_size=8, n=32):
    kern = kernel_cls(block_size=block_size, n=n)
    for lang in (CUDA_T, OCL_T, METAL_T):
        print("=" * 78)
        print(kernel_cls.__name__, "->", idpy_langs_human_dict[lang])
        print("=" * 78)
        print(kern.dump_code(lang))
        print()


def _check_backends(kernel_cls, block_size, n):
    _ok, _ran = True, False
    # Shared memory is a block-parallel GPU concept: CUDA / OpenCL / Metal only.
    for lang in (CUDA_T, OCL_T, METAL_T):
        human = idpy_langs_human_dict[lang]
        if not idpy_langs_sys[lang]:
            print(f"  [skip] {human}: backend not available on this machine")
            continue
        params = {'lang': lang}
        if lang == OCL_T:
            params['cl_kind'] = 'gpu'
        try:
            ok, max_abs, _, _ = run_on(
                lang, block_size, n, tenet_params=params, kernel_cls=kernel_cls
            )
            _ran = True
            _ok = _ok and ok
            status = "OK  " if ok else "FAIL"
            print(f"  [{status}] {human}: max|out-ref| = {max_abs:g}")
        except Exception as exc:
            _ok = False
            print(f"  [err ] {human}: {type(exc).__name__}: {exc}")
    return _ok, _ran


def main():
    print("=== STATIC shared memory (idpy_shared FType tile[BLOCK]) ===\n")
    print("Generated sources (small sizes):\n")
    dump_sources(K_SharedNeighborSum)
    print("Deploying and checking against the host reference:\n")
    block_size, n = 64, 1024
    _ok_a, _ran_a = _check_backends(K_SharedNeighborSum, block_size, n)

    print("\n\n=== DYNAMIC shared memory (runtime-sized tile) ===\n")
    print("Generated sources (small sizes):\n")
    dump_sources(K_SharedNeighborSumDynamic)
    print("Deploying and checking against the host reference:\n")
    _ok_b, _ran_b = _check_backends(K_SharedNeighborSumDynamic, block_size, n)

    print(
        "\nNote: the CTYPES backend runs kernels as a serial loop over the "
        "global\nthread id and has no block-local id / shared memory; such "
        "kernels raise\nNotImplementedError there by design."
    )
    _report_exit(_ok_a and _ok_b, checks_run=(_ran_a or _ran_b),
                 what='GPU backends')


if __name__ == '__main__':
    main()
