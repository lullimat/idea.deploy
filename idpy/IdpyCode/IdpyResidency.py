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
Residency policy: hold a working set larger than device memory.

Phase 2 of the residency layer. Phase 2b gave every backend the primitives to
move a *sub-range* of a device buffer without disturbing the rest
(SubView / H2DSub / D2HSub / Sync). This module is the policy built on top:
which blocks are resident right now, which one to evict to make room, and when
a modified block has to go back to the backing store.

The split is the point. The primitives are backend-specific and were the hard
part -- a staged async copy on CUDA, a second queue on OpenCL, range-scoped
waiting against in-flight command buffers on Metal, plain numpy on CTypes. The
policy above them is backend-*independent*: this file contains no per-language
branches at all, because it only ever calls the primitives.

That asymmetry is the publishable content of the design. Kernels are
syntactically different per backend; residency is semantically different per
backend; but the policy expressed on top is the same program everywhere.

Layout
------
    BackingStore          where the full dataset lives (bigger than the device)
      MemMapStore         a file, via numpy.memmap -- mmap + OS page cache
      ArrayStore          a host array; for tests and for in-RAM working sets
    ResidentCache         N device-resident slots + an eviction policy
    Cache(...)            constructor, tenet-parameterised

Placement note: the design sketch put this interface *on* Tenet. It is a free
module taking 'tenet=' instead, matching how IdpyMemory.Array/Zeros/OnDevice
already work, and keeping Tenet -- which every backend implements -- from
growing a dependency on policy code. The capability is still per-tenet; only the
spelling differs.
'''

from collections import OrderedDict

import numpy as np

from idpy.IdpyCode import IdpyMemory, CUDA_T, METAL_T
from idpy.Utils.IsModuleThere import IsModuleThere


class BackingStore:
    '''
    The full dataset, of which only a few blocks are device-resident at a time.

    Subclasses implement ReadBlock/WriteBlock. Blocks are fixed-size and
    contiguous; a partial trailing block is padded, so callers always see
    'block_elems' elements and must ignore the tail past 'n_elems'.
    '''

    def __init__(self, n_elems, block_elems, dtype):
        self.n_elems = int(n_elems)
        self.block_elems = int(block_elems)
        self.dtype = np.dtype(dtype)
        self.n_blocks = (self.n_elems + self.block_elems - 1) // self.block_elems
        self.bytes_read = 0
        self.bytes_written = 0

    def BlockSpan(self, block_id):
        '''Element range [start, stop) of a block, clipped to the dataset.'''
        if not 0 <= block_id < self.n_blocks:
            raise IndexError(
                "block %d out of range [0, %d)" % (block_id, self.n_blocks)
            )
        _start = block_id * self.block_elems
        return _start, min(_start + self.block_elems, self.n_elems)

    def ReadBlock(self, block_id):
        raise NotImplementedError

    def WriteBlock(self, block_id, data):
        raise NotImplementedError

    '''
    Storage -> device, Phase 3.

    ReadBlock hands back a host array, which the cache then copies into a device
    slot: storage -> page cache -> host buffer -> device. That is the universal
    path and it is always correct. It is also two copies more than the hardware
    requires, and on a residency workload -- where the whole point is that the
    dataset does not fit and is therefore read continuously -- those copies are
    the cost.

    A store that can write into device memory directly overrides ReadBlockInto.
    Returning False means "I cannot", and the cache falls back to the host path
    without caring why. That keeps the capability optional per store rather than
    per backend, and keeps the fallback the default rather than the exception.
    '''
    def ReadBlockInto(self, block_id, view):
        '''
        Fill a device slot with a block, bypassing host memory.

        'view' is an Idpy array covering exactly one block. Return True if the
        block was written, False to decline -- declining is not an error, it is
        how a store says it has no direct path on this configuration.
        '''
        return False

    def DirectPathName(self):
        '''Human-readable name of the direct path, or None when there is none.'''
        return None


class ArrayStore(BackingStore):
    '''Backed by a host array. The simplest store; useful for tests.'''

    def __init__(self, array, block_elems):
        array = np.ascontiguousarray(array)
        BackingStore.__init__(self, array.size, block_elems, array.dtype)
        self.array = array

    def ReadBlock(self, block_id):
        _start, _stop = self.BlockSpan(block_id)
        _out = np.zeros((self.block_elems,), dtype=self.dtype)
        _out[:_stop - _start] = self.array[_start:_stop]
        self.bytes_read += (_stop - _start) * self.dtype.itemsize
        return _out

    def WriteBlock(self, block_id, data):
        _start, _stop = self.BlockSpan(block_id)
        self.array[_start:_stop] = data[:_stop - _start]
        self.bytes_written += (_stop - _start) * self.dtype.itemsize


class MemMapStore(BackingStore):
    '''
    Backed by a file through numpy.memmap: mmap plus the OS page cache.

    This is the reference path the design calls for -- on a unified-memory or
    CPU target there is no copy to schedule, only which pages the OS keeps
    resident, and the kernel's own page cache does the caching underneath. The
    ResidentCache above it is then policy over a *device-side* working set,
    which stays meaningful even when the store itself is memory-mapped.
    '''

    def __init__(self, path, n_elems, block_elems, dtype, mode='r+'):
        BackingStore.__init__(self, n_elems, block_elems, dtype)
        self.path = str(path)
        self.map = np.memmap(self.path, dtype=self.dtype, mode=mode,
                             shape=(self.n_elems,))

    @classmethod
    def Create(cls, path, array, block_elems):
        '''Materialise 'array' into a file and return a store over it.'''
        array = np.ascontiguousarray(array)
        _map = np.memmap(str(path), dtype=array.dtype, mode='w+',
                         shape=(array.size,))
        _map[:] = array
        _map.flush()
        del _map
        return cls(path, array.size, block_elems, array.dtype)

    def ReadBlock(self, block_id):
        _start, _stop = self.BlockSpan(block_id)
        _out = np.zeros((self.block_elems,), dtype=self.dtype)
        _out[:_stop - _start] = self.map[_start:_stop]
        self.bytes_read += (_stop - _start) * self.dtype.itemsize
        return _out

    def WriteBlock(self, block_id, data):
        _start, _stop = self.BlockSpan(block_id)
        self.map[_start:_stop] = data[:_stop - _start]
        self.bytes_written += (_stop - _start) * self.dtype.itemsize

    def Flush(self):
        self.map.flush()

    def Close(self):
        self.Flush()
        self.map = None


class KvikIOStore(MemMapStore):
    '''
    A file store that can read straight into device memory on CUDA (Phase 3).

    Uses KvikIO, which wraps NVIDIA's cuFile / GPUDirect Storage. The reason it
    is the cheapest backend to start with is not performance: **KvikIO degrades
    to a POSIX read when GPUDirect is unavailable**, so the same code path is
    exercised, and stays correct, on a machine with no GDS hardware, no
    compatible filesystem and no nvidia-fs driver. A correctness path that does
    not depend on hardware you may not have is worth more early than a fast path
    that does.

    Inherits MemMapStore's host path unchanged, so the fallback is not a
    separate implementation that could drift from the direct one -- it is the
    same store answering a different way.

    Availability is decided per instance, not per import: kvikio present, a CUDA
    array on the other end, and a successful open. Any of those missing and
    ReadBlockInto declines, which the cache handles as an ordinary miss.
    '''

    def __init__(self, path, n_elems, block_elems, dtype, mode='r+'):
        MemMapStore.__init__(self, path, n_elems, block_elems, dtype, mode=mode)
        self._cufile = None
        self._direct = IsModuleThere('kvikio')
        '''
        Whether a direct read has actually SUCCEEDED, as opposed to whether
        kvikio merely imports. The two differ constantly: on a host with kvikio
        installed, a CTypes or OpenCL cache still takes the staged path, because
        the destination is not a CUDA array. Reporting availability as if it
        were use produced the line "kvikio/cuFile: 0 direct / 34 staged", which
        names a path that was never taken -- exactly the kind of label that lets
        a silently-degraded fast path look engaged.
        '''
        self._direct_used = False

    def _CuFile(self):
        if self._cufile is None:
            import kvikio
            self._cufile = kvikio.CuFile(self.path, 'r')
        return self._cufile

    def ReadBlockInto(self, block_id, view):
        '''
        Read a block straight into 'view'. Declines unless the destination is a
        CUDA array -- KvikIO writes through __cuda_array_interface__, which the
        host-backed arrays of the other backends do not provide, and which
        nothing else here should pretend to satisfy.
        '''
        if not self._direct:
            return False
        if getattr(view, 'lang', None) != CUDA_T:
            return False

        _start, _stop = self.BlockSpan(block_id)
        _itemsize = int(self.dtype.itemsize)
        _nbytes = (_stop - _start) * _itemsize
        try:
            _future = self._CuFile().pread(
                view, size=_nbytes, file_offset=_start * _itemsize,
            )
            _future.get()
        except Exception:
            '''
            Decline permanently rather than raising. A read that cannot use the
            direct path is a configuration fact, not a failure: the cache falls
            back and the run stays correct, only staged. Sticking the flag off
            avoids paying the exception on every subsequent miss.
            '''
            self._direct = False
            return False

        self.bytes_read += _nbytes
        self._direct_used = True
        return True

    def DirectPathName(self):
        '''
        The direct path actually in use, or None.

        Keyed on a successful read rather than on kvikio being importable, so a
        store that has only ever taken the staged route says so.
        '''
        return 'kvikio/cuFile' if self._direct_used else None

    def Close(self):
        if self._cufile is not None:
            try:
                self._cufile.close()
            except Exception:
                pass
            self._cufile = None
        MemMapStore.Close(self)


class MetalIOStore(MemMapStore):
    '''
    A file store that reads straight into a Metal buffer via MTLIOCommandQueue.

    The Metal counterpart of KvikIOStore, and the row that makes the storage
    claim portable rather than merely present. The binding lives in
    idpy/Metal/MetalIO.py: a Swift '@_cdecl' shim compiled by HostModule, which
    is the case that whole facility was built for -- Metal's storage API has no
    Python binding, pymetallic does not wrap it, and Swift is the only language
    that can see it.

    The load targets a byte offset inside an existing buffer, so it fills a
    SubView of the cache directly. That works because Phase 2b gave
    IdpyArrayMETAL an element offset against its parent Buffer; without that
    bookkeeping there would be nothing to point the read at.
    '''

    def __init__(self, path, n_elems, block_elems, dtype, mode='r+'):
        MemMapStore.__init__(self, path, n_elems, block_elems, dtype, mode=mode)
        self._io = None
        self._shim = None
        self._direct = True
        self._direct_used = False

    def _Open(self, view):
        '''Lazily build the queue+handle, using the device behind 'view'.'''
        if self._io is not None:
            return self._io
        from idpy.Metal.MetalIO import Shim
        self._shim = Shim()
        if self._shim is None:
            return None
        _device = getattr(getattr(view, 'tenet', None), 'device', None)
        if _device is None:
            return None
        self._io = self._shim['open'](
            _device._device_ptr, str(self.path).encode()
        )
        return self._io

    def ReadBlockInto(self, block_id, view):
        if not self._direct or getattr(view, 'lang', None) != METAL_T:
            return False
        try:
            if self._Open(view) is None:
                self._direct = False
                return False
            _start, _stop = self.BlockSpan(block_id)
            _itemsize = int(self.dtype.itemsize)
            _nbytes = (_stop - _start) * _itemsize
            _rc = self._shim['load'](
                self._io, view.data._buffer_ptr,
                int(view.offset) * _itemsize, _nbytes, _start * _itemsize,
            )
            if _rc != 1:
                self._direct = False
                return False
        except Exception:
            self._direct = False
            return False

        self.bytes_read += _nbytes
        self._direct_used = True
        return True

    def DirectPathName(self):
        return 'MTLIOCommandQueue' if self._direct_used else None

    def Close(self):
        if self._io is not None and self._shim is not None:
            try:
                self._shim['close'](self._io)
            except Exception:
                pass
            self._io = None
        MemMapStore.Close(self)


'''
Which file store suits a given backend.

The residency policy is backend-independent, but the store underneath it is
exactly where the backend shows through -- that is the asymmetry the design is
built around, so it belongs in one small factory rather than smeared through
callers. Every option subclasses MemMapStore, so an unmatched backend, a missing
binding or a failed open all land on the same staged path.
'''
_FILE_STORES = {}


def FileStoreClass(tenet=None):
    if tenet is None:
        return MemMapStore
    _lang = tenet.GetLang() if hasattr(tenet, 'GetLang') else None
    return _FILE_STORES.get(_lang, MemMapStore)


def CreateFileStore(path, array, block_elems, tenet=None):
    '''Materialise 'array' into a file and wrap it in the best store available.'''
    return FileStoreClass(tenet).Create(path, array, block_elems)


_FILE_STORES[CUDA_T] = KvikIOStore
_FILE_STORES[METAL_T] = MetalIOStore


class ResidentCache:
    '''
    A fixed number of device-resident slots over a larger backing store.

    One device allocation of n_slots * block_elems elements is made up front and
    never grows; each slot is a SubView of it, so a slot is an ordinary Idpy
    array that kernels can be handed directly.

    Usage:
        cache = Cache(tenet=t, store=store, n_slots=4)
        view = cache.Acquire(block_id)          # resident afterwards
        ...                                     # hand 'view' to a kernel
        cache.MarkDirty(block_id)               # if it was written
        cache.Flush()                           # push dirty blocks back

    Eviction is LRU by default. 'policy' may be 'lru' or 'fifo'; the
    turbo-fieldfare design uses LFU, which is a further option rather than a
    different structure -- only PickVictim changes.

    Pinning: a block acquired for the current step must not be evicted to make
    room for another block in the same step. Acquire pins, and the pins are
    released by EndStep(). Without that, a stencil needing three blocks with a
    two-slot cache would evict a block it is still using and quietly read
    garbage, rather than failing.
    '''

    _POLICIES = ('lru', 'fifo')

    def __init__(self, tenet=None, store=None, n_slots=4, policy='lru'):
        if tenet is None or store is None:
            raise ValueError("ResidentCache needs both tenet= and store=")
        if policy not in self._POLICIES:
            raise ValueError(
                "policy must be one of %s, got %r" % (self._POLICIES, policy)
            )
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")

        self.tenet, self.store, self.policy = tenet, store, policy
        self.n_slots = int(n_slots)
        self.block_elems = store.block_elems

        self.buffer = IdpyMemory.Zeros(
            shape=(self.n_slots * self.block_elems,),
            dtype=store.dtype, tenet=tenet,
        )
        self.slots = [
            self.buffer.SubView(i * self.block_elems,
                                (i + 1) * self.block_elems)
            for i in range(self.n_slots)
        ]

        self._resident = OrderedDict()   # block_id -> slot index, LRU order
        self._slot_block = [None] * self.n_slots
        self._free = list(range(self.n_slots))
        self._dirty = set()
        self._pinned = set()

        self.stats = OrderedDict(
            [('hits', 0), ('misses', 0), ('evictions', 0), ('writebacks', 0),
             ('direct_reads', 0), ('staged_reads', 0)]
        )

    # -- policy ------------------------------------------------------------
    def PickVictim(self):
        '''
        Choose a resident, unpinned block to evict.

        _resident is kept in touch order, so the oldest unpinned entry is the
        LRU victim; 'fifo' differs only in that Acquire does not re-order on a
        hit, which makes insertion order the same as touch order here.
        '''
        for _block_id in self._resident:
            if _block_id not in self._pinned:
                return _block_id
        raise RuntimeError(
            "every slot is pinned: the cache has %d slots but this step needs "
            "more. Increase n_slots or acquire fewer blocks per step."
            % self.n_slots
        )

    # -- residency ---------------------------------------------------------
    def IsResident(self, block_id):
        return block_id in self._resident

    def Acquire(self, block_id, pin=True):
        '''
        Ensure 'block_id' is device-resident and return its slot view.

        The returned view aliases the cache's own allocation, so it is valid
        only until that block is evicted -- do not hold it across EndStep().
        '''
        if block_id in self._resident:
            self.stats['hits'] += 1
            if self.policy == 'lru':
                self._resident.move_to_end(block_id)
            if pin:
                self._pinned.add(block_id)
            return self.slots[self._resident[block_id]]

        self.stats['misses'] += 1
        _slot = self._AllocateSlot()
        '''
        Try storage -> device first; fall back to storage -> host -> device.

        The counters distinguish the two so a run can say which path it actually
        took. A direct path that silently degrades to the host bounce would look
        exactly like a working one in every correctness test -- that is the same
        failure mode as an 'async' copy that is secretly synchronous, and it is
        worth being able to see rather than infer.
        '''
        if self.store.ReadBlockInto(block_id, self.slots[_slot]):
            self.stats['direct_reads'] += 1
        else:
            self.slots[_slot].H2DSub(self.store.ReadBlock(block_id), start=0)
            self.stats['staged_reads'] += 1
        self._resident[block_id] = _slot
        self._slot_block[_slot] = block_id
        if pin:
            self._pinned.add(block_id)
        return self.slots[_slot]

    def _AllocateSlot(self):
        if self._free:
            return self._free.pop()
        self.Evict(self.PickVictim())
        return self._free.pop()

    def MarkDirty(self, block_id):
        if block_id not in self._resident:
            raise KeyError("block %d is not resident" % block_id)
        self._dirty.add(block_id)

    def Evict(self, block_id):
        '''Write back if dirty, then release the slot.'''
        if block_id not in self._resident:
            return
        if block_id in self._pinned:
            raise RuntimeError("refusing to evict pinned block %d" % block_id)
        _slot = self._resident.pop(block_id)
        if block_id in self._dirty:
            self._WriteBack(block_id, _slot)
        self._slot_block[_slot] = None
        self._free.append(_slot)
        self.stats['evictions'] += 1

    def _WriteBack(self, block_id, slot):
        self.slots[slot].Sync()
        self.store.WriteBlock(block_id, self.slots[slot].D2HSub(
            0, self.block_elems
        ))
        self._dirty.discard(block_id)
        self.stats['writebacks'] += 1

    def EndStep(self):
        '''Release the pins taken during the current step.'''
        self._pinned.clear()

    def Flush(self):
        '''Push every dirty block back to the store; blocks stay resident.'''
        for _block_id in list(self._dirty):
            self._WriteBack(_block_id, self._resident[_block_id])

    def HitRate(self):
        _total = self.stats['hits'] + self.stats['misses']
        return (self.stats['hits'] / _total) if _total else 0.0

    def Report(self):
        _r = OrderedDict(self.stats)
        _r['hit_rate'] = self.HitRate()
        _r['bytes_read'] = self.store.bytes_read
        _r['bytes_written'] = self.store.bytes_written
        _r['direct_path'] = self.store.DirectPathName()
        return _r


def Cache(tenet=None, store=None, n_slots=4, policy='lru'):
    '''
    Build a ResidentCache. Free function taking 'tenet=' to match
    IdpyMemory.Array / Zeros / OnDevice rather than living on Tenet itself.
    '''
    return ResidentCache(tenet=tenet, store=store, n_slots=n_slots,
                         policy=policy)
