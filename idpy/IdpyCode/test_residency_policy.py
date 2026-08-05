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
Phase 2 acceptance: a lattice larger than the resident set, swept correctly.

The design's criterion is "a working-set-larger-than-RAM lattice test passes on
CPU". Larger-than-RAM is not something a test suite can honestly arrange, and it
is not the property that matters: what matters is that the dataset exceeds the
*device-resident* set, forcing eviction and write-back. On CTypes device memory
IS RAM, so the resident cache is exactly the constraint under test. This sweeps
a lattice 8x the size of the cache that holds it.

The workload is a periodic 3-point stencil, out[i] = in[i-1] + in[i] + in[i+1],
computed block by block. That choice is deliberate and answers an open question
from the design: does the lattice case have any *reuse* to evict against, or is
it a pure streaming sweep where every policy performs identically?

It has reuse, and halos are where it comes from. Computing block b needs blocks
b-1, b and b+1; the next step needs b, b+1 and b+2. Two of every three acquires
are already resident, so the steady-state hit rate is 2/3 and the eviction
policy is genuinely exercised -- LRU must retain b and b+1 while discarding
b-1. A cache that evicted wrongly would still run, and would produce wrong
numbers, which is what makes this a test rather than a demonstration.

Checks:

  P1  correctness      swept output equals a whole-lattice numpy reference,
                       exactly, with the dataset 8x the resident set
  P2  reuse            hit rate matches the 2/3 the access pattern predicts
  P3  pinning          asking for more blocks in one step than the cache can
                       hold raises, instead of evicting a block still in use
  P4  write-back       results reach the backing file, not just the cache
  P5  policy agnostic  LRU and FIFO agree on the answer

Compute runs host-side through the residency primitives, so this file has no
per-backend branches and the same program runs on all four. Handing the slot
views to a kernel instead is possible -- they are ordinary Idpy arrays -- but a
halo spanning three separate slots is a layout problem belonging to the real
lattice work, not to the policy layer under test here.

Run directly:
    python -m idpy.IdpyCode.test_residency_policy
'''

import os
import tempfile
from collections import OrderedDict

import numpy as np

from idpy.IdpyCode import (
    CUDA_T, OCL_T, CTYPES_T, METAL_T,
    idpy_langs_sys, idpy_langs_human_dict, GetTenet,
)
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyCode import IdpyResidency
from idpy.Utils.TestExit import report_exit as _report_exit

_POLICY_LANGS = (CTYPES_T, CUDA_T, OCL_T, METAL_T)

_N_BLOCKS = 32
_BLOCK_ELEMS = 262144      # 1 MiB per block at fp32
_N_SLOTS = 4               # 32 MiB lattice over a 4 MiB resident set: 8x
_DTYPE = np.float32


def expected_traffic(n_blocks):
    '''
    Exact hit/miss counts for this sweep, not an approximation.

    Step 0 acquires b-1, b, b+1 with nothing resident: 3 misses. Every later
    step re-acquires two blocks it already holds and pulls in one new one, so
    long as eviction discards the block that just fell out of the window. With
    4 slots and a 3-block window there is always exactly one spare, so:

        misses = 3 + (n_blocks - 1) = n_blocks + 2
        hits   = 2 * (n_blocks - 1)

    The asymptotic hit rate is 2/3; at finite n_blocks it is slightly lower
    because of the cold start. Checking the exact integers rather than the
    ratio is what makes this a test of the eviction policy: a cache that kept
    the wrong block would still produce correct numbers (the data is reloaded)
    but would miss more often, and only the exact count catches that.
    '''
    return 2 * (n_blocks - 1), n_blocks + 2


def _tenet_params(lang):
    params = {'lang': lang}
    if lang == OCL_T:
        params['cl_kind'] = 'gpu'
    return params


def reference(a):
    '''Whole-lattice periodic 3-point stencil.'''
    return np.roll(a, 1) + a + np.roll(a, -1)


def sweep(tenet, lattice, n_slots=_N_SLOTS, policy='lru', tmpdir=None):
    '''
    Blocked stencil sweep through a ResidentCache over a memory-mapped lattice.

    Returns (result, in_cache, out_cache, out_store).
    '''
    _in_path = os.path.join(tmpdir, 'lattice_in_%s.bin' % policy)
    _out_path = os.path.join(tmpdir, 'lattice_out_%s.bin' % policy)

    in_store = IdpyResidency.MemMapStore.Create(_in_path, lattice, _BLOCK_ELEMS)
    out_store = IdpyResidency.MemMapStore.Create(
        _out_path, np.zeros_like(lattice), _BLOCK_ELEMS
    )

    in_cache = IdpyResidency.Cache(
        tenet=tenet, store=in_store, n_slots=n_slots, policy=policy,
    )
    out_cache = IdpyResidency.Cache(
        tenet=tenet, store=out_store, n_slots=2, policy=policy,
    )

    _nb = in_store.n_blocks
    for b in range(_nb):
        _left = in_cache.Acquire((b - 1) % _nb)
        _cur = in_cache.Acquire(b)
        _right = in_cache.Acquire((b + 1) % _nb)

        # halo assembly: one element from each neighbouring block
        _padded = np.empty((_BLOCK_ELEMS + 2,), dtype=_DTYPE)
        _padded[0] = _left.D2HSub(_BLOCK_ELEMS - 1, _BLOCK_ELEMS)[0]
        _padded[1:-1] = _cur.D2HSub(0, _BLOCK_ELEMS)
        _padded[-1] = _right.D2HSub(0, 1)[0]

        _res = _padded[:-2] + _padded[1:-1] + _padded[2:]

        _dst = out_cache.Acquire(b)
        _dst.H2DSub(_res, start=0)
        out_cache.MarkDirty(b)

        in_cache.EndStep()
        out_cache.EndStep()

    out_cache.Flush()
    out_store.Flush()

    # read the answer back from the FILE, not from the cache, so the check
    # covers write-back rather than whatever happens to still be resident
    result = np.array(
        np.memmap(_out_path, dtype=_DTYPE, mode='r', shape=(lattice.size,))
    )
    return result, in_cache, out_cache, out_store


def run_on(lang, tmpdir):
    tenet = GetTenet(_tenet_params(lang))
    out = OrderedDict()
    try:
        _n = _N_BLOCKS * _BLOCK_ELEMS
        lattice = np.arange(_n, dtype=_DTYPE) * np.float32(0.5)
        ref = reference(lattice)

        # -- P1/P2/P4: sweep with LRU
        got, in_cache, out_cache, out_store = sweep(
            tenet, lattice, policy='lru', tmpdir=tmpdir,
        )
        out['P1 err'] = float(np.max(np.abs(got - ref)))
        _exp_hits, _exp_misses = expected_traffic(_N_BLOCKS)
        out['P2 hit_rate'] = in_cache.HitRate()
        out['P2 hits'] = in_cache.stats['hits']
        out['P2 misses'] = in_cache.stats['misses']
        out['P2 exact'] = (
            in_cache.stats['hits'] == _exp_hits
            and in_cache.stats['misses'] == _exp_misses
        )
        out['P2 expected'] = (_exp_hits, _exp_misses)
        out['P4 writebacks'] = out_cache.stats['writebacks']
        out['resident_MB'] = (
            in_cache.buffer.size * np.dtype(_DTYPE).itemsize / (1 << 20)
        )
        out['lattice_MB'] = _n * np.dtype(_DTYPE).itemsize / (1 << 20)
        out['evictions'] = in_cache.stats['evictions']

        # -- P3: pinning must refuse, not corrupt
        _store = IdpyResidency.ArrayStore(lattice, _BLOCK_ELEMS)
        _tiny = IdpyResidency.Cache(tenet=tenet, store=_store, n_slots=2)
        try:
            for b in range(3):
                _tiny.Acquire(b)
            out['P3 pinning'] = 'FAIL: allowed over-subscription'
        except RuntimeError:
            out['P3 pinning'] = 'raised'

        # -- P5: FIFO must agree with LRU on the answer
        got_fifo, _, _, _ = sweep(
            tenet, lattice, policy='fifo', tmpdir=tmpdir,
        )
        out['P5 lru_vs_fifo'] = float(np.max(np.abs(got_fifo - got)))
    finally:
        if hasattr(tenet, 'End'):
            tenet.End()
    return out


def main():
    print("=== Phase 2: residency policy over a lattice larger than the cache ===\n")
    _ok, _ran = True, False
    with tempfile.TemporaryDirectory() as tmpdir:
        for lang in _POLICY_LANGS:
            human = idpy_langs_human_dict[lang]
            if not idpy_langs_sys[lang]:
                print(f"  [skip] {human}: backend not available\n")
                continue
            try:
                r = run_on(lang, tmpdir)
            except Exception as exc:
                _ok = False
                print(f"  [err ] {human}: {type(exc).__name__}: {exc}\n")
                continue
            _ran = True

            ok = (
                r['P1 err'] == 0.0
                and r['P5 lru_vs_fifo'] == 0.0
                and r['P3 pinning'] == 'raised'
                and r['P4 writebacks'] > 0
                and r['P2 exact']
            )
            print(f"  {human}: lattice {r['lattice_MB']:.0f} MiB over a "
                  f"{r['resident_MB']:.0f} MiB resident set "
                  f"({r['lattice_MB'] / r['resident_MB']:.0f}x)")
            print(f"    P1 sweep vs reference   max|out-ref| = {r['P1 err']:g}")
            print(f"    P2 reuse                {r['P2 hits']} hits / "
                  f"{r['P2 misses']} misses, expected {r['P2 expected']}"
                  f"   -> {'exact' if r['P2 exact'] else 'MISMATCH'}")
            print(f"                            hit rate {r['P2 hit_rate']:.3f}, "
                  f"{r['evictions']} evictions")
            print(f"    P3 pinning guard        {r['P3 pinning']}")
            print(f"    P4 write-back           {r['P4 writebacks']} blocks "
                  f"reached the file")
            print(f"    P5 LRU vs FIFO          max|lru-fifo| = "
                  f"{r['P5 lru_vs_fifo']:g}")
            print(f"    -> {'OK' if ok else 'FAIL'}\n")
            _ok = _ok and ok

    print(
        "The policy layer has no per-backend branches: it is written entirely\n"
        "against SubView / H2DSub / D2HSub / Sync. The primitives differ per\n"
        "backend, the policy above them does not -- which is the asymmetry the\n"
        "design is built around."
    )

    _report_exit(_ok, checks_run=_ran, what='backends')


if __name__ == '__main__':
    main()
