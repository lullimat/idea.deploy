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

from idpy.IdpyCode import IdpyMemory


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
            [('hits', 0), ('misses', 0), ('evictions', 0), ('writebacks', 0)]
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
        self.slots[_slot].H2DSub(self.store.ReadBlock(block_id), start=0)
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
        return _r


def Cache(tenet=None, store=None, n_slots=4, policy='lru'):
    '''
    Build a ResidentCache. Free function taking 'tenet=' to match
    IdpyMemory.Array / Zeros / OnDevice rather than living on Tenet itself.
    '''
    return ResidentCache(tenet=tenet, store=store, n_slots=n_slots,
                         policy=policy)
