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

import fcntl
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

'''
Transfer-size sweep. A single block size samples one point on a bandwidth curve,
and if that point sits near the knee -- where per-transfer overhead still
competes with throughput -- run-to-run variance is large and a lone sample says
little. This is why the same cold CUDA measurement produced 0.36 GB/s once and
~1.4 the next time, and why a 4.36x ratio was recorded from it.

The plateau is the stable estimate. Where the curve reaches it is separately
useful: it is the block size the ResidentCache should be using.

Total bytes are held constant across sizes so each row is the same work split
differently, and the cache holds 4 blocks throughout, so the resident set grows
with the block size exactly as it would in use.
'''
_SWEEP_MiB = (0.25, 1, 4, 16, 64)
_SWEEP_TOTAL_MiB = 512
_SWEEP_MAX_BLOCKS = 64
_SWEEP_REPEATS = 2

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


def _PlainRead(path, chunk=1 << 22):
    _t, _n = perf_counter(), 0
    with open(path, 'rb', buffering=0) as _fh:
        while True:
            _b = _fh.read(chunk)
            if not _b:
                break
            _n += len(_b)
    return _n / (perf_counter() - _t) / 1e9


_F_NOCACHE = 48          # macOS fcntl; no O_DIRECT and no posix_fadvise


def DriveBandwidthNoCache(tmpdir, mib=256, chunk=1 << 22):
    '''
    Drive read bandwidth on macOS, which has neither posix_fadvise nor O_DIRECT.

    'purge' needs sudo, and F_NOCACHE on a read does not evict pages that are
    already resident -- it only changes future caching policy. The way through
    is to keep the pages out of the cache in the first place: write the scratch
    file with F_NOCACHE set, then read it back with F_NOCACHE set. Verified to
    give ~5 GB/s against ~11 GB/s for a cached read on the same file, so it is
    genuinely reaching the device.

    A separate scratch file rather than the test file, because the sweep's
    staged path reads through numpy.memmap and repopulates the cache on first
    touch. This measures the DEVICE; the sweep on macOS stays warm and is
    labelled as such.
    '''
    if not hasattr(fcntl, 'fcntl'):
        return None
    _path = os.path.join(tmpdir, 'drive_probe.bin')
    _blob = os.urandom(1 << 22)
    try:
        _fd = os.open(_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            fcntl.fcntl(_fd, _F_NOCACHE, 1)
            for _ in range(max(1, mib // 4)):
                os.write(_fd, _blob)
            os.fsync(_fd)
        finally:
            os.close(_fd)

        _fd = os.open(_path, os.O_RDONLY)
        try:
            fcntl.fcntl(_fd, _F_NOCACHE, 1)
            _t, _n = perf_counter(), 0
            while True:
                _b = os.read(_fd, chunk)
                if not _b:
                    break
                _n += len(_b)
        finally:
            os.close(_fd)
        return _n / (perf_counter() - _t) / 1e9
    except OSError:
        return None
    finally:
        try:
            os.remove(_path)
        except OSError:
            pass


def RawColdBandwidth(path, chunk=1 << 22):
    '''
    Sequential read bandwidth straight from the drive, with no idpy in the way.

    The ground truth this test lacked. Without it there is no way to tell a
    fast path from a path that is quietly reading RAM, and that ambiguity
    produced two wrong conclusions in a row: first that cuFile was a
    pessimization, then that the page-cache control had fixed the comparison.

    Any figure above this line is cache-assisted, by definition.
    '''
    if not hasattr(os, 'posix_fadvise'):
        return None
    _warm = _PlainRead(path)                 # ensure resident, and measure RAM
    if not DropPageCache(path):
        return None
    _cold = _PlainRead(path)
    """
    Self-validating: report BOTH, and let the caller see whether the eviction
    actually happened. A cold figure close to the warm one means the pages were
    not dropped -- which has now happened three times in this harness, always
    because something still held the file mapped, and each time it was noticed
    only by comparing against an external probe. The check belongs in the
    measurement, not in the reader.
    """
    return {'warm': _warm, 'cold': _cold, 'evicted': _cold < 0.5 * _warm}


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
    '''
    Wall time over 'repeats' full sweeps, plus the store used.

    Returns the best time AND the full list. Min alone was not enough: the cold
    CUDA staged leg read 0.36 GB/s in one run and ~1.4 in the next, a 4x spread
    on the same quantity, and reporting only the minimum turned a sample into an
    apparent result. Cold I/O has a long tail -- eviction cost, fault storms,
    drive state -- so the spread is the measurement, not an imperfection in it.
    '''
    _best, _store, _cache, _all = None, None, None, []
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
        _all.append(_dt)
        _best = _dt if _best is None else min(_best, _dt)
        _direct = _cache.stats['direct_reads']
    return _best, _bytes, _store, _direct, _all


def _ColdSweep(tenet, path, staged_f, direct_f, lattice):
    '''
    Cold bandwidth against transfer size, for both routes.

    Every point drops the page cache first, so every point reads the drive.
    Returns a list of (MiB, staged_GBs, direct_GBs, direct_reads).
    '''
    if not hasattr(os, 'posix_fadvise'):
        return None
    _item = np.dtype(_DTYPE).itemsize
    _rows = []
    for _mib in _SWEEP_MiB:
        _block = max(1, int(_mib * (1 << 20)) // _item)
        # Clamp to what the file actually holds. Belt and braces: the file is
        # sized for the sweep above, but a mismatch between the two should
        # shorten the measurement rather than raise IndexError mid-run.
        _avail = max(1, lattice.size // _block)
        '''
        Cap the block COUNT, not the total bytes. Cost is driven by per-block
        overhead times count, so holding total bytes constant makes the smallest
        row 2048 acquires -- which, with a command buffer per acquire on Metal,
        does not finish. Bandwidth is bytes over time, so each row only needs
        enough bytes to clear timer noise, not the same total as its neighbours.
        '''
        _nb = max(4, min(int(_SWEEP_TOTAL_MiB / _mib), _SWEEP_MAX_BLOCKS,
                         _avail))
        _bytes = _nb * _block * _item

        def _run(factory):
            _best, _direct = None, 0
            for _ in range(_SWEEP_REPEATS):
                gc.collect()
                DropPageCache(path)
                _store = factory(path, lattice, _block)
                _cache = IdpyResidency.Cache(tenet=tenet, store=_store,
                                             n_slots=_N_SLOTS)
                _t0 = perf_counter()
                for _b in range(_nb):
                    _cache.Acquire(_b)
                    _cache.EndStep()
                _dt = perf_counter() - _t0
                _direct = _cache.stats['direct_reads']
                _best = _dt if _best is None else min(_best, _dt)
                _close = getattr(_store, 'Close', None)
                if callable(_close):
                    try:
                        _close()
                    except Exception:
                        pass
            return _bytes / _best / 1e9, _direct

        _s_bw, _ = _run(lambda p, a, blk: IdpyResidency.MemMapStore(
            p, a.size, blk, a.dtype))
        _d_bw, _dr = _run(lambda p, a, blk: IdpyResidency.FileStoreClass(tenet)(
            p, a.size, blk, a.dtype))
        _rows.append((_mib, _s_bw, _d_bw, _dr))
    return _rows


def measure(lang, tmpdir):
    tenet = GetTenet(_tenet_params(lang))
    out = OrderedDict()
    try:
        '''
        The file must cover the LARGEST total any measurement asks for. Sizing
        it to the single-size run while the sweep requested twice that produced
        'IndexError: block 1024 out of range [0, 1024)' -- two constants that
        had to agree, in different places, with nothing enforcing it.
        '''
        _n = max(_N_BLOCKS * _BLOCK_ELEMS,
                 int(_SWEEP_TOTAL_MiB * (1 << 20)) // np.dtype(_DTYPE).itemsize)
        lattice = np.arange(_n, dtype=_DTYPE) * np.float32(0.5)
        _path = os.path.join(tmpdir, 'bw_%s.bin' % lang)
        lattice.tofile(_path)
        '''
        Ground truth FIRST, before any store exists. numpy.memmap keeps the file
        mapped and pinned pages cannot be evicted, so measuring the drive after
        the stores are built measures the page cache instead -- which is exactly
        how a 12 GB/s "drive" came to be printed next to a 1.6 GB/s one.
        '''
        with open(_path, 'rb') as _fh:
            os.fsync(_fh.fileno())
        out['raw'] = RawColdBandwidth(_path)
        if out['raw'] is None:
            # No posix_fadvise (macOS): measure the device directly instead, and
            # keep the sweep labelled warm rather than pretending otherwise.
            out['drive_nocache'] = DriveBandwidthNoCache(tmpdir)

        # -- B1: staged. MemMapStore has no direct path by construction.
        _staged = lambda p, a: IdpyResidency.MemMapStore(
            p, a.size, _BLOCK_ELEMS, a.dtype)
        _t_staged, _bytes, _s1, _d1, _a1 = _time_load(
            tenet, _staged, _path, lattice, _N_BLOCKS, _REPEATS)

        # -- B2: whatever direct lowering this backend has, if any.
        _direct_f = lambda p, a: IdpyResidency.FileStoreClass(tenet)(
            p, a.size, _BLOCK_ELEMS, a.dtype)
        _t_direct, _, _s2, _d2, _a2 = _time_load(
            tenet, _direct_f, _path, lattice, _N_BLOCKS, _REPEATS)

        '''
        The comparison that actually answers the question: both routes with the
        page cache dropped first, so neither is reading RAM. Warm numbers are
        kept because they bound what the machinery can do when I/O is free.
        '''
        # Read everything needed off the warm stores BEFORE releasing them --
        # the release is what makes the cold phase honest, and nulling them
        # while a later line still reads _s2.DirectPathName() is how the last
        # attempt broke.
        out['path'] = _s2.DirectPathName() or 'staged (no direct path)'
        _s1 = _s2 = None                 # release the warm stores' mappings
        gc.collect()
        out['sweep'] = _ColdSweep(tenet, _path, _staged, _direct_f, lattice)

        out['MiB'] = _bytes / (1 << 20)
        out['staged_GBs'] = _bytes / _t_staged / 1e9
        out['direct_GBs'] = _bytes / _t_direct / 1e9
        out['direct_reads'] = _d2
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
            _dn = r.get('drive_nocache')
            if _dn is not None:
                print(f"    B0 drive (F_NOCACHE) {_dn:7.2f} GB/s"
                      f"   <-- the device; the sweep below stays WARM here")
            _raw = r.get('raw')
            if _raw is not None:
                print(f"    B0 plain read        warm {_raw['warm']:6.2f} / "
                      f"cold {_raw['cold']:6.2f} GB/s"
                      + ("   <-- cold is the drive"
                         if _raw['evicted'] else
                         "   <-- NOT EVICTED: every figure below reads cache"))
            _sw = r.get('sweep')
            if _sw:
                print(f"    B6 cold sweep        block      staged    direct")
                for _mib, _sb, _db, _dr in _sw:
                    _lab = (f"{_mib:g} MiB" if _mib >= 1
                            else f"{int(_mib * 1024)} KiB")
                    print(f"                       {_lab:>8}   {_sb:7.2f}   "
                          f"{_db:7.2f} GB/s" + ("" if _dr else "  (staged)"))
                _pm, _ps, _pd, _ = _sw[-1]
                print(f"       plateau ({_pm:g} MiB)  staged {_ps:.2f} / "
                      f"direct {_pd:.2f} GB/s   ratio {_pd / _ps:.2f}x"
                      f"   <-- the stable estimate")
            if True:
                if not _sw:
                    print(f"    B4 cold cache        unavailable "
                          f"(posix_fadvise is Linux-only)")
            if False:
                print(f"    B4 cold staged       {r['cold_staged_GBs']:7.2f} GB/s")
                print(f"    B5 cold direct       {r['cold_direct_GBs']:7.2f} GB/s"
                      f"   ({r['cold_direct_reads']} direct reads)")
                _cs, _cd = r['cold_staged_spread'], r['cold_direct_spread']
                print(f"       spread staged     {_cs[0]:.2f} - {_cs[1]:.2f} GB/s"
                      f"   direct {_cd[0]:.2f} - {_cd[1]:.2f} GB/s")
                print(f"    B5/B4                "
                      f"{r['cold_direct_GBs'] / r['cold_staged_GBs']:7.2f}x"
                      f"   (best-of; see spread -- cold I/O has a long tail)")
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
