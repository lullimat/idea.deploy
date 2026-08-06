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
P1: sustained storage->device bandwidth, and whether it overlaps with compute.

Phase 3 established that the direct paths are TAKEN -- 34 direct / 0 staged on
CUDA via cuFile, the same on Metal via MTLIOCommandQueue. It never established
that they are FASTER. Those are different claims and only one of them has been
checked.

There is a specific reason to doubt the second on hardware without GPUDirect:
KvikIO falls back to a POSIX read when GDS is unavailable, which is exactly why
it was the cheap row to start with. On such a machine "direct" may be a POSIX
read wearing a different hat, plausibly slower than the staged path once its
overhead is counted. That is not a defect -- it is the open question the
direct/staged counters were built to let us ask, and it is unanswerable without
measuring.

Three numbers per backend, reading blocks through a ResidentCache from a file
larger than the cache:

  B1  staged bandwidth    MemMapStore: storage -> page cache -> host -> device
  B2  direct bandwidth    CreateFileStore: cuFile / MTLIOCommandQueue
  B3  overlap             does a block load proceed WHILE a kernel runs on
                          already-resident slots?

B3 uses the same estimator as test_overlap: load alone, compute alone, both
together, overlap = (tK + tC - tB) / min(tK, tC), 1.0 concurrent and 0.0
serialized.

READ THE CACHE CAVEAT BEFORE QUOTING ANY NUMBER
-----------------------------------------------
The test file is far smaller than RAM, so after the first pass it is resident in
the OS page cache and B1 measures page-cache bandwidth, not the drive. That is
deliberate: making it cold would need a file larger than RAM or root to drop
caches, neither of which belongs in a test suite.

It stays informative because of what it does to the COMPARISON. With a warm
cache the staged path is reading from RAM, which is the most favourable case it
will ever see. So:

  B2 >> B1   the direct path is genuinely bypassing the host -- strong evidence
             GPUDirect is engaged, since it beats RAM
  B2 ~ B1    the mechanism engages but buys nothing here; on a machine without
             GDS this is the expected result and means cuFile fell back to POSIX
  B2 << B1   the direct path costs more than it saves at this block size

None of these is a pass or a failure. The exit status is gated on correctness
only -- bandwidth is a measurement, and a build must never fail for being run on
a slower disk.

Run directly:
    python -m idpy.IdpyCode.test_storage_bandwidth
'''

import gc
import os
import tempfile
from collections import OrderedDict
from time import perf_counter

import numpy as np

from idpy.IdpyCode import (
    IDPY_T, CUDA_T, OCL_T, CTYPES_T, METAL_T,
    idpy_langs_sys, idpy_langs_human_dict, GetTenet,
)
from idpy.IdpyCode.IdpyCode import IdpyKernel
from idpy.IdpyCode import IdpyMemory, IdpyResidency
from idpy.IdpyCode.IdpyUnroll import _codify_assignment, _array_value
from idpy.Utils.CustomTypes import CustomTypes
from idpy.Utils.TestExit import report_exit as _report_exit

_LANGS = (CUDA_T, OCL_T, METAL_T, CTYPES_T)

_BLOCK_ELEMS = 1 << 21          # 8 MiB blocks at fp32
_N_BLOCKS = 32                  # 256 MiB file
_N_SLOTS = 4                    # 32 MiB resident: the file is 8x the cache
_DTYPE = np.float32
_REPEATS = 3

_TYPES = CustomTypes({'FType': 'float'}).Push()


class K_Spin(IdpyKernel):
    '''
    Compute-bound filler, so B3 measures overlap rather than bandwidth
    contention. Same dependent-FMA-chain reasoning as test_overlap: a
    memory-bound kernel would fight the transfer for bandwidth and understate
    concurrency even where it is real.
    '''

    def __init__(self, iters=512, custom_types=None):
        constants = OrderedDict()
        constants['ITERS'] = int(iters)
        constants['CHAIN_A'] = np.float32(0.99999)
        constants['CHAIN_B'] = np.float32(0.00001)
        IdpyKernel.__init__(self, custom_types=custom_types or _TYPES,
                            constants=constants)
        self.SetCodeFlags('g_tid')
        self.params = {'FType * a': ['global', 'restrict']}
        self.kernels[IDPY_T] = (
            "\nFType v = a[g_tid];\n"
            "for(int i = 0; i < ITERS; i++){ v = v * CHAIN_A + CHAIN_B; }\n"
            "a[g_tid] = v;\n"
        )


def DropPageCache(path):
    '''
    Evict a file from the OS page cache, without root.

    This is the control the first version of this test was missing, and its
    absence produced a wrong conclusion: a warm B1 reads RAM while cuFile with
    GPUDirect bypasses the page cache by design and reads the drive, so
    comparing them measured RAM against disk and called the disk slow.

    posix_fadvise(POSIX_FADV_DONTNEED) is Linux-only -- which is where it
    matters, since that is where the discrete GPUs and cuFile are. Returns True
    if the cache was actually dropped, so callers can label their numbers
    honestly instead of assuming.
    '''
    if not hasattr(os, 'posix_fadvise'):
        return False
    try:
        _fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(_fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(_fd)
        return True
    except OSError:
        return False


def RawColdBandwidth(path, chunk=1 << 22):
    '''
    Sequential read bandwidth straight from the drive, with no idpy in the way.

    The ground truth this test lacked. Without it there is no way to tell a
    fast path from a path that is quietly reading RAM, and that ambiguity
    produced two wrong conclusions in a row: first that cuFile was a
    pessimization, then that the page-cache control had fixed the comparison.

    Any figure above this line is cache-assisted, by definition.
    '''
    if not DropPageCache(path):
        return None
    _t, _n = perf_counter(), 0
    with open(path, 'rb', buffering=0) as _fh:
        while True:
            _b = _fh.read(chunk)
            if not _b:
                break
            _n += len(_b)
    return _n / (perf_counter() - _t) / 1e9


def _tenet_params(lang):
    params = {'lang': lang}
    if lang == OCL_T:
        params['cl_kind'] = 'gpu'
    return params


def _sweep_cache(cache, n_blocks):
    '''Touch every block once, evicting as needed. Returns bytes moved.'''
    for b in range(n_blocks):
        cache.Acquire(b)
        cache.EndStep()
    return n_blocks * cache.block_elems * np.dtype(_DTYPE).itemsize


def _time_load(tenet, store_factory, path, lattice, n_blocks, repeats,
               cold=False):
    '''Minimum wall time over 'repeats' full sweeps, plus the store used.'''
    _best, _store, _cache = None, None, None
    for _ in range(repeats):
        if cold:
            '''
            Release the previous mapping BEFORE advising the kernel. numpy.memmap
            keeps the file mapped, and POSIX_FADV_DONTNEED cannot evict pages
            held by a live mapping -- so dropping without this is a silent no-op
            for the staged path, which is exactly how a "cold" staged number of
            6.5 GB/s came to sit next to a 1.6 GB/s drive.
            '''
            if _store is not None:
                _close = getattr(_store, 'Close', None)
                if callable(_close):
                    try:
                        _close()
                    except Exception:
                        pass
                _store = None
            _cache = None
            gc.collect()
            DropPageCache(path)
        _store = store_factory(path, lattice)
        _cache = IdpyResidency.Cache(tenet=tenet, store=_store,
                                     n_slots=_N_SLOTS)
        _t0 = perf_counter()
        _bytes = _sweep_cache(_cache, n_blocks)
        _dt = perf_counter() - _t0
        _best = _dt if _best is None else min(_best, _dt)
        _direct = _cache.stats['direct_reads']
    return _best, _bytes, _store, _direct


def measure(lang, tmpdir):
    tenet = GetTenet(_tenet_params(lang))
    out = OrderedDict()
    try:
        _n = _N_BLOCKS * _BLOCK_ELEMS
        lattice = np.arange(_n, dtype=_DTYPE) * np.float32(0.5)
        _path = os.path.join(tmpdir, 'bw_%s.bin' % lang)
        lattice.tofile(_path)

        # -- B1: staged. MemMapStore has no direct path by construction.
        _staged = lambda p, a: IdpyResidency.MemMapStore(
            p, a.size, _BLOCK_ELEMS, a.dtype)
        _t_staged, _bytes, _s1, _d1 = _time_load(
            tenet, _staged, _path, lattice, _N_BLOCKS, _REPEATS)

        # -- B2: whatever direct lowering this backend has, if any.
        _direct_f = lambda p, a: IdpyResidency.FileStoreClass(tenet)(
            p, a.size, _BLOCK_ELEMS, a.dtype)
        _t_direct, _, _s2, _d2 = _time_load(
            tenet, _direct_f, _path, lattice, _N_BLOCKS, _REPEATS)

        '''
        The comparison that actually answers the question: both routes with the
        page cache dropped first, so neither is reading RAM. Warm numbers are
        kept because they bound what the machinery can do when I/O is free.
        '''
        out['raw_cold_GBs'] = RawColdBandwidth(_path)
        _cold_ok = DropPageCache(_path)
        if _cold_ok:
            _t_cold_staged, _, _, _ = _time_load(
                tenet, _staged, _path, lattice, _N_BLOCKS, _REPEATS, cold=True)
            _t_cold_direct, _, _, _dc = _time_load(
                tenet, _direct_f, _path, lattice, _N_BLOCKS, _REPEATS, cold=True)
            out['cold_staged_GBs'] = _bytes / _t_cold_staged / 1e9
            out['cold_direct_GBs'] = _bytes / _t_cold_direct / 1e9
            out['cold_direct_reads'] = _dc
        else:
            out['cold_staged_GBs'] = None
            out['cold_direct_GBs'] = None
            out['cold_direct_reads'] = None

        out['MiB'] = _bytes / (1 << 20)
        out['staged_GBs'] = _bytes / _t_staged / 1e9
        out['direct_GBs'] = _bytes / _t_direct / 1e9
        out['direct_reads'] = _d2
        out['path'] = _s2.DirectPathName() or 'staged (no direct path)'
        out['speedup'] = out['direct_GBs'] / out['staged_GBs']

        # -- correctness: the two routes must agree, whatever their speed
        _c1 = IdpyResidency.Cache(tenet=tenet, store=_staged(_path, lattice),
                                  n_slots=_N_SLOTS)
        _c2 = IdpyResidency.Cache(tenet=tenet,
                                  store=_direct_f(_path, lattice),
                                  n_slots=_N_SLOTS)
        _err = 0.0
        for b in (0, _N_BLOCKS // 2, _N_BLOCKS - 1):
            _a = np.array(_c1.Acquire(b).D2HSub(0, _BLOCK_ELEMS))
            _b = np.array(_c2.Acquire(b).D2HSub(0, _BLOCK_ELEMS))
            _err = max(_err, float(np.max(np.abs(_a - _b))))
            _c1.EndStep(); _c2.EndStep()
        out['agree'] = _err

        # -- B3: overlap of a block load with a running kernel
        '''
        Overlap for BOTH routes on the same backend. Measuring only the direct
        one invites the comparison "direct overlaps better than staged" to be
        made across different backends, which is not a controlled statement --
        and concurrency, not throughput, is where a separate IO queue would be
        expected to pay.
        '''
        if lang == CTYPES_T:
            out['overlap'] = None
            out['overlap_staged'] = None
        else:
            _idea, _spin = _calibrate(tenet, lattice, _path, _staged)
            out['overlap'] = _overlap(tenet, _direct_f, _path, lattice,
                                      _idea, _spin)
            out['overlap_staged'] = _overlap(tenet, _staged, _path, lattice,
                                             _idea, _spin)
    finally:
        if hasattr(tenet, 'End'):
            tenet.End()
    return out


def _calibrate(tenet, lattice, path, store_factory, target_ms=20.0):
    '''
    Build the filler kernel once, sized to 'target_ms' on THIS machine.

    Once, not per route: the same kernel has to time both routes or the two
    overlap ratios are not comparable, which is the whole reason for measuring
    them together. A fixed iteration count cannot serve both machines either --
    it ran 14 ms on an M1 Max and 0.8 ms on an RTX 5060.
    '''
    _store = store_factory(path, lattice)
    _cache = IdpyResidency.Cache(tenet=tenet, store=_store, n_slots=_N_SLOTS)
    _cache.Acquire(0)
    _cache.EndStep()
    _slot = _cache.slots[0]

    _block = 256
    _grid = ((_BLOCK_ELEMS + _block - 1) // _block, 1, 1)

    def _run(iters):
        _idea = K_Spin(iters=iters)(tenet=tenet, grid=_grid,
                                    block=(_block, 1, 1))
        _idea.Deploy([_slot]); _slot.Sync()          # warm
        _t = perf_counter(); _idea.Deploy([_slot]); _slot.Sync()
        return _idea, perf_counter() - _t

    _probe_idea, _t_probe = _run(512)
    _iters = int(max(64, min(1 << 20,
                             round(512 * (target_ms * 1e-3) / max(_t_probe, 1e-9)))))
    _idea = K_Spin(iters=_iters)(tenet=tenet, grid=_grid, block=(_block, 1, 1))
    _idea.Deploy([_slot]); _slot.Sync()
    return _idea, _slot


def _overlap(tenet, store_factory, path, lattice, idea, spin_slot):
    '''
    Load a block into one slot while a kernel spins on another.

    Deliberately not the residency cache's own accounting: this issues the
    kernel, then the load, then waits for both, which is the only arrangement
    where a serialized implementation and a concurrent one give different wall
    times.
    '''
    _store = store_factory(path, lattice)
    _cache = IdpyResidency.Cache(tenet=tenet, store=_store, n_slots=_N_SLOTS)
    _cache.Acquire(0)               # slot 0 resident, kernel target
    _cache.EndStep()
    _victim = spin_slot          # calibration slot; a different cache
    _idea = idea

    _block = 256
    _grid = ((_BLOCK_ELEMS + _block - 1) // _block, 1, 1)


    def _sync():
        '''
        Sync through the array primitive, not the tenet. The CUDA Tenet has
        neither Finish nor finish, so poking it was a silent no-op there and the
        kernel timed 0.0 ms -- an async launch with nothing waiting on it. Every
        IdpyArray* has carried Sync() since Phase 2b; that is the portable
        spelling and it is what the residency layer itself uses.
        '''
        _victim.Sync()

    def _kernel_only():
        _t = perf_counter(); _idea.Deploy([_victim]); _sync()
        return perf_counter() - _t

    def _fill(slot, block):
        if not _cache.store.ReadBlockInto(block, _cache.slots[slot]):
            _cache.slots[slot].H2DSub(_cache.store.ReadBlock(block), start=0)

    # Several blocks, not one: a single 8 MiB read lands close enough to timer
    # noise that min(tK, tC) becomes the denominator of a ratio it cannot
    # support. Three loads put tC somewhere the estimator can see it.
    _loads = ((1, 5), (2, 6), (3, 7))

    def _load_only():
        _t = perf_counter()
        for _slot, _blk in _loads:
            _fill(_slot, _blk)
        return perf_counter() - _t

    def _both():
        _t = perf_counter()
        _idea.Deploy([_victim])
        for _slot, _blk in _loads:
            _fill(_slot, _blk)
        _sync()
        return perf_counter() - _t

    _tk = min(_kernel_only() for _ in range(_REPEATS))
    _tc = min(_load_only() for _ in range(_REPEATS))
    _tb = min(_both() for _ in range(_REPEATS))
    '''
    The ratio is only meaningful when both legs are comparable: it divides by
    min(tk, tc), so a leg near the timer floor produces a large number with no
    physical content. Report the raw times alongside and refuse the ratio when
    the smaller leg is under a millisecond or under 5% of the larger.
    '''
    _small, _large = min(_tk, _tc), max(_tk, _tc)
    _ratio, _reason = None, None
    if _small > 1e-3 and _small > 0.05 * _large:
        _ratio = (_tk + _tc - _tb) / _small
        # The estimator is bounded by construction: 0 serialized, 1 fully
        # concurrent. A value outside that band means tB beat one leg measured
        # alone, which is timing noise rather than physics -- report it as
        # unresolved instead of dressing noise as a result.
        if not (-0.2 <= _ratio <= 1.2):
            _ratio, _reason = None, 'out of band'
    else:
        _reason = 'legs not comparable'
    return {'kernel_ms': _tk * 1e3, 'load_ms': _tc * 1e3,
            'both_ms': _tb * 1e3, 'overlap': _ratio, 'reason': _reason}


def main():
    print("=== P1: storage->device bandwidth and overlap ===\n")
    _ok, _ran = True, False
    with tempfile.TemporaryDirectory() as tmpdir:
        for lang in _LANGS:
            human = idpy_langs_human_dict[lang]
            if not idpy_langs_sys[lang]:
                print(f"  [skip] {human}: backend not available\n")
                continue
            try:
                r = measure(lang, tmpdir)
            except Exception as exc:
                _ok = False
                print(f"  [err ] {human}: {type(exc).__name__}: {exc}\n")
                continue
            _ran = True
            _ok = _ok and (r['agree'] == 0.0)

            print(f"  {human}: {r['MiB']:.0f} MiB through a "
                  f"{_N_SLOTS * _BLOCK_ELEMS * 4 / (1 << 20):.0f} MiB cache")
            print(f"    B1 staged            {r['staged_GBs']:7.2f} GB/s")
            print(f"    B2 {r['path']:<18} {r['direct_GBs']:7.2f} GB/s"
                  f"   ({r['direct_reads']} direct reads)")
            print(f"    B2/B1                {r['speedup']:7.2f}x   "
                  f"(warm cache: B1 reads RAM, so this is NOT a fair race)")
            if r.get('raw_cold_GBs') is not None:
                print(f"    B0 raw cold read     {r['raw_cold_GBs']:7.2f} GB/s"
                      f"   <-- the drive; anything above this reads cache")
            if r['cold_staged_GBs'] is None:
                print(f"    B4 cold cache        unavailable "
                      f"(posix_fadvise is Linux-only)")
            else:
                print(f"    B4 cold staged       {r['cold_staged_GBs']:7.2f} GB/s")
                print(f"    B5 cold direct       {r['cold_direct_GBs']:7.2f} GB/s"
                      f"   ({r['cold_direct_reads']} direct reads)")
                print(f"    B5/B4                "
                      f"{r['cold_direct_GBs'] / r['cold_staged_GBs']:7.2f}x"
                      f"   <-- the fair comparison")
            _ovs = r.get('overlap_staged')
            if _ovs is not None and _ovs['overlap'] is not None:
                print(f"    B3 overlap, staged   {_ovs['overlap']:7.2f}"
                      f"   (kernel {_ovs['kernel_ms']:.1f} ms, load "
                      f"{_ovs['load_ms']:.1f} ms)")
            _ov = r['overlap']
            if _ov is None:
                print(f"    B3 overlap           n/a (serial backend: no "
                      f"concurrency to measure)")
            elif _ov['overlap'] is None:
                print(f"    B3 overlap           n/a "
                      f"({_ov.get('reason') or 'unresolved'}: kernel "
                      f"{_ov['kernel_ms']:.1f} ms, load "
                      f"{_ov['load_ms']:.1f} ms, both {_ov['both_ms']:.1f} ms)")
            else:
                print(f"    B3 overlap, direct   {_ov['overlap']:7.2f}"
                      f"   (kernel {_ov['kernel_ms']:.1f} ms, load "
                      f"{_ov['load_ms']:.1f} ms, both {_ov['both_ms']:.1f} ms)")
            print(f"    routes agree         max|staged-direct| = {r['agree']:g}"
                  f"   -> {'OK' if r['agree'] == 0.0 else 'FAIL'}\n")

    print(
        "B1 reads a warm page cache, so it is the staged path at its most\n"
        "favourable. B2 >> B1 means the direct path beats RAM and GPUDirect is\n"
        "genuinely engaged; B2 ~ B1 means the mechanism engages but buys nothing\n"
        "here, which is the expected result without GDS hardware. Neither is a\n"
        "failure: the exit status is gated on the two routes agreeing, never on\n"
        "throughput."
    )
    _report_exit(_ok, checks_run=_ran, what='backends')


if __name__ == '__main__':
    main()
