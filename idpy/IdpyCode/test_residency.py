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
Phase 2b acceptance test: sub-range views and the async path on IdpyArray*.

Motivation is finding F2 in docs/residency-probes.md -- before this work the
memory layer could not express the one operation a residency policy needs:

    write bytes into slot k of a device buffer, asynchronously,
    while the GPU reads slots j != k

H2D/D2H are whole-array and silently discarded 'async_' on every backend. This
test exercises the primitives added to close that gap:

    SubView(start, stop)                 aliasing sub-range, no copy
    H2DSub(ary, start, async_=...)       partial write
    D2HSub(start, stop, async_=...)      partial read
    Sync(idpy_stream=None)               wait on outstanding async work

Three checks, in increasing strength:

  T1  disjoint sub-range writes land where they should and nowhere else
  T2  SubView aliases the parent (write through the view, observe the parent)
  T3  an async partial write into the upper half stays correct while a kernel
      is concurrently reading and writing the lower half

T3 is the acceptance criterion from idea.deploy-extension.md Phase 2b. It
verifies *correctness under concurrent issue*, not that overlap actually
happened -- measuring achieved overlap is P1's job, and on CUDA genuine overlap
additionally requires page-locked host memory (see IdpyArrayCUDA.H2DSub).

Run directly:
    python -m idpy.IdpyCode.test_residency
'''

from collections import OrderedDict

import numpy as np

from idpy.IdpyCode import (
    IDPY_T, CUDA_T, OCL_T, CTYPES_T, METAL_T,
    idpy_langs_sys, idpy_langs_human_dict, GetTenet,
)
from idpy.IdpyCode.IdpyCode import IdpyKernel
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyCode.IdpyUnroll import (
    _codify_assignment, _codify_comment, _array_value,
)
from idpy.Utils.CustomTypes import CustomTypes

# Backends carrying the Phase 2b primitives. Metal joined once its
# drain-on-every-host-touch was replaced by range-scoped waiting (finding F2).
# CTypes is the remaining lowering and is trivially unified.
_RESIDENCY_LANGS = (CUDA_T, OCL_T, METAL_T)


class K_ScaleLowerHalf(IdpyKernel):
    '''
    buf[i] *= FACTOR for the whole launch range, which the harness sizes to the
    lower half of the buffer. The upper half is deliberately untouched by the
    GPU so the host can write it concurrently.
    '''

    def __init__(self, factor=3.0, custom_types=None, optimizer_flag=None):
        if custom_types is None:
            custom_types = CustomTypes({'FType': 'float'}).Push()

        constants = OrderedDict()
        constants['FACTOR'] = float(factor)

        IdpyKernel.__init__(
            self, custom_types=custom_types, constants=constants,
            optimizer_flag=optimizer_flag,
        )
        self.SetCodeFlags('g_tid')

        self.params = {'FType * buf': ['global', 'restrict']}

        body = ""
        body += _codify_comment("scale only the lanes in the launch range")
        body += _codify_assignment(
            _array_value('buf', 'g_tid'),
            _array_value('buf', 'g_tid') + " * FACTOR",
        )
        self.kernels[IDPY_T] = "\n" + body

    def dump_code(self, lang=IDPY_T):
        return self.Code(lang)


def _tenet_params(lang):
    params = {'lang': lang}
    if lang == OCL_T:
        params['cl_kind'] = 'gpu'
    return params


def t1_disjoint_writes(tenet, n=1024, n_slots=8):
    '''Write a distinct constant into each slot; nothing else may move.'''
    slot = n // n_slots
    buf = IdpyMemory.Zeros(shape=(n,), dtype=np.float32, tenet=tenet)
    ref = np.zeros((n,), dtype=np.float32)

    # write the even slots only, so the odd ones prove non-interference
    for k in range(0, n_slots, 2):
        payload = np.full((slot,), float(k + 1), dtype=np.float32)
        buf.H2DSub(payload, start=k * slot)
        ref[k * slot:(k + 1) * slot] = payload

    out = buf.D2H()
    return float(np.max(np.abs(out - ref)))


def t2_subview_aliases(tenet, n=1024):
    '''A SubView must share storage with its parent, not copy it.'''
    half = n // 2
    host = np.arange(n, dtype=np.float32)
    buf = IdpyMemory.OnDevice(host, tenet=tenet)

    view = buf.SubView(half, n)
    # read through the view
    seen = view.D2H()
    err_read = float(np.max(np.abs(seen - host[half:])))

    # write through the view, observe it in the parent
    payload = np.full((half,), -7.0, dtype=np.float32)
    view.H2DSub(payload, start=0)
    ref = host.copy()
    ref[half:] = payload
    err_alias = float(np.max(np.abs(buf.D2H() - ref)))

    return max(err_read, err_alias)


def t3_async_write_during_kernel(tenet, lang, n=1 << 16, factor=3.0):
    '''
    The acceptance criterion: async host write into slots [n/2, n) issued with a
    kernel scaling slots [0, n/2) still in flight. Both halves must be correct.

    Scope, stated precisely because the distinction matters: on a single
    in-order queue/stream the runtime is entitled to order the copy after the
    kernel, so this establishes that the async partial write is *correct when
    issued against in-flight work* -- it does not establish that the two
    actually overlapped. Proving overlap needs a second queue (OpenCL) or a
    non-default stream (CUDA), plus page-locked host memory on CUDA. That is
    P1's measurement, and the stronger two-queue variant is follow-up work.
    '''
    half = n // 2
    block_size = 128
    host = np.arange(n, dtype=np.float32)
    buf = IdpyMemory.OnDevice(host, tenet=tenet)

    kern = K_ScaleLowerHalf(factor=factor)
    # launch covers the lower half only
    grid = ((half + block_size - 1) // block_size, 1, 1)
    block = (block_size, 1, 1)
    idea = kern(tenet=tenet, grid=grid, block=block)

    payload = np.full((half,), 99.0, dtype=np.float32)

    idea.Deploy([buf])
    # issued without waiting on the kernel: the ranges are disjoint
    buf.H2DSub(payload, start=half, async_=True)
    buf.Sync()
    if hasattr(tenet, 'FlushAndWait'):
        tenet.FlushAndWait()

    ref = host.copy()
    ref[:half] *= factor
    ref[half:] = payload

    return float(np.max(np.abs(buf.D2H() - ref)))


def run_on(lang, n=1024):
    tenet = GetTenet(_tenet_params(lang))
    results = OrderedDict()
    try:
        results['T1 disjoint sub-range writes'] = t1_disjoint_writes(tenet, n=n)
        results['T2 SubView aliases parent'] = t2_subview_aliases(tenet, n=n)
        results['T3 async write, kernel in flight (correctness, not overlap)'] = \
            t3_async_write_during_kernel(tenet, lang)
    finally:
        if hasattr(tenet, 'End'):
            tenet.End()
    return results


def main():
    print("=== Phase 2b: sub-range views and the async path ===\n")
    for lang in _RESIDENCY_LANGS:
        human = idpy_langs_human_dict[lang]
        if not idpy_langs_sys[lang]:
            print(f"  [skip] {human}: backend not available on this machine")
            continue
        try:
            for name, err in run_on(lang).items():
                status = "OK  " if err == 0.0 else "FAIL"
                print(f"  [{status}] {human}: {name}: max|out-ref| = {err:g}")
        except Exception as exc:
            print(f"  [err ] {human}: {type(exc).__name__}: {exc}")
        print()

    print(
        "CTypes does not carry these primitives yet (trivially unified). On\n"
        "Metal the primitives are host stores into unified memory, so 'async_'\n"
        "is inert there; what used to make the pattern impossible was the\n"
        "drain on every host touch, now replaced by range-scoped waiting.\n"
        "Whether that wait is actually skipped is measured in test_overlap."
    )


if __name__ == '__main__':
    main()
