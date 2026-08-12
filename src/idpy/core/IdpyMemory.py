__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2021 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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
Provides an interface class for the transparent use of numpy arrays extensions of
pyopencl and pycuda
'''

import numpy as np

from idpy.core import OCL_T, CUDA_T, CTYPES_T, METAL_T, idpy_tenet_types
from idpy.core import idpy_langs_list, idpy_langs_sys

if idpy_langs_sys[CUDA_T]:
    import pycuda as cu
    import pycuda.driver as cu_driver
    import pycuda.gpuarray as cu_array
    from idpy.core.backends.cuda.backend import Tenet as CUTenet

    class IdpyArrayCUDA(cu_array.GPUArray):
        def __init__(self, shape, dtype,
                     allocator = None, base = None,
                     gpudata = None, strides = None, order = 'C',
                     tenet = None):
            
            if allocator is None:
                allocator = (tenet.allocator if tenet is not None
                             else cu_driver.mem_alloc)
                
            super().__init__(shape = shape, dtype = dtype,
                             allocator = allocator, base = base,
                             gpudata = gpudata, strides = strides,
                             order = order)
            self.lang = CUDA_T
            self.tenet = tenet

        def H2D(self, ary, async_=None, idpy_stream=None):
            if async_:
                return super().set_async(ary = ary, stream = idpy_stream)
            return super().set(ary = ary)

        def D2H(self, ary = None, pagelocked = False, async_=None,
                idpy_stream=None):
            if async_:
                return super().get_async(stream = idpy_stream, ary = ary)
            return super().get(ary = ary, pagelocked = pagelocked)

        def SetConst(self, const = 0., stream = None):
            super().fill(value = const, stream = stream)

        '''
        Residency primitives (Phase 2b). See docs/residency-probes.md, F2.

        These exist to express the one operation the residency policy needs and
        H2D/D2H cannot:

            write bytes into slot k of a device buffer, asynchronously,
            while the GPU reads slots j != k

        H2D/D2H are whole-array and, before this change, silently discarded
        'async_'. Their default behaviour is unchanged: async_ is opt-in.
        '''
        def _ByteOffset(self, start):
            return int(start) * int(self.dtype.itemsize)

        def _CheckRange(self, start, stop):
            start, stop = int(start), int(stop)
            if start < 0 or stop > int(self.size) or start >= stop:
                raise ValueError(
                    "Sub-range [%d, %d) out of bounds for array of size %d"
                    % (start, stop, int(self.size))
                )
            return start, stop

        def SubView(self, start, stop):
            '''
            Contiguous element sub-range sharing this array's storage. No copy:
            writing the view writes the parent. 'base' keeps the parent alive so
            the borrowed device pointer cannot outlive its allocation.
            '''
            start, stop = self._CheckRange(start, stop)
            _sub = super().__getitem__(slice(start, stop))
            return IdpyArrayCUDA(
                shape = _sub.shape, dtype = _sub.dtype, tenet = self.tenet,
                gpudata = _sub.gpudata, base = _sub,
            )

        def H2DSub(self, ary, start = 0, async_ = False, idpy_stream = None):
            '''
            Write 'ary' into this array starting at element 'start'.

            NOTE on real overlap: CUDA only overlaps an async H2D with compute
            when the host buffer is page-locked. A pageable numpy array is
            accepted and correct here, but the copy will not actually overlap --
            use PinnedHost() for that. An API that claims async and behaves
            synchronously is precisely the failure mode F2 documents, so this is
            stated rather than left to be discovered.
            '''
            ary = np.ascontiguousarray(ary, dtype = self.dtype)
            start, _ = self._CheckRange(start, start + ary.size)
            _dest = int(self.gpudata) + self._ByteOffset(start)
            if async_:
                cu_driver.memcpy_htod_async(_dest, ary, stream = idpy_stream)
            else:
                cu_driver.memcpy_htod(_dest, ary)
            return ary

        def D2HSub(self, start, stop, ary = None, async_ = False,
                   idpy_stream = None):
            '''Read elements [start, stop) into 'ary' (allocated when None).'''
            start, stop = self._CheckRange(start, stop)
            if ary is None:
                ary = np.empty((stop - start,), dtype = self.dtype)
            _src = int(self.gpudata) + self._ByteOffset(start)
            if async_:
                cu_driver.memcpy_dtoh_async(ary, _src, stream = idpy_stream)
            else:
                cu_driver.memcpy_dtoh(ary, _src)
            return ary

        def Sync(self, idpy_stream = None):
            '''Wait on outstanding async work: one stream, or the whole context.'''
            if idpy_stream is not None:
                idpy_stream.synchronize()
            else:
                cu_driver.Context.synchronize()

    def _pinned_host_CUDA(shape, dtype):
        '''
        Page-locked host buffer. Required for H2DSub(async_=True) to genuinely
        overlap with compute instead of degrading to a synchronous copy.
        '''
        return cu_driver.pagelocked_empty(shape, dtype)

    def _on_device_CUDA(ary, tenet):
        _swap_array = IdpyArrayCUDA(shape = ary.shape,
                                    dtype = ary.dtype,
                                    tenet = tenet,
                                    allocator = tenet.allocator)
        _swap_array.H2D(ary)
        return _swap_array

    def _zeros_CUDA(shape, dtype, tenet):
        _swap_array = IdpyArrayCUDA(shape = shape,
                                    dtype = dtype,
                                    tenet = tenet,
                                    allocator = tenet.allocator)
        _swap_array.SetConst(0)
        return _swap_array

    def _range_CUDA(n, tenet, dtype = np.int32):
        _tmp_range = np.arange(n, dtype = dtype)
        _swap_array = _on_device_CUDA(_tmp_range, tenet = tenet)
        del _tmp_range
        return _swap_array

    def _const_CUDA(shape, dtype, const = 0., tenet = None):
        _swap_array = IdpyArrayCUDA(shape = shape,
                                    dtype = dtype,
                                    tenet = tenet,
                                    allocator = tenet.allocator)
        _swap_array.SetConst(const)
        return _swap_array

    def _sum_CUDA(a, dtype = None, stream = None):
        return cu_array.sum(a = a, dtype = dtype, stream = stream)

    def _max_CUDA(a, stream = None):
        return cu_array.max(a = a, stream = stream)

    def _min_CUDA(a, stream = None):
        return cu_array.min(a = a, stream = stream)
    

if idpy_langs_sys[OCL_T]:
    import pyopencl as cl
    import pyopencl.array as cl_array
    from idpy.core.backends.opencl.backend import Tenet as CLTenet

    class IdpyArrayOCL(cl_array.Array):
        def __init__(self, shape = None, queue = None, dtype = None,
                     order = 'C',
                     allocator = None, data = None,
                     offset = 0, strides = None,
                     events = None, **kwargs):
            # pyopencl ReductionKernel: Array(cq, shape, dtype, ...)
            if isinstance(shape, (cl.CommandQueue, cl.Context)):
                queue, shape = shape, queue
            # pyopencl builds derived arrays (slices, reshapes) via
            # self.__class__(..., _fast=True, ...). Without forwarding those
            # private kwargs, slicing an IdpyArrayOCL raises TypeError.
            super().__init__(cq = queue, shape = shape,
                             dtype = dtype, order = order,
                             allocator = allocator, data = data,
                             offset = offset, strides = strides,
                             events = events, **kwargs)

            self.lang, self.queue = OCL_T, queue
            # LBM-style ownership: queue is the OpenCL Tenet
            self.tenet = queue

        def H2D(self, ary, async_=None, idpy_stream=None):
            _queue = self.queue if idpy_stream is None else idpy_stream
            return super().set(ary = ary, queue = _queue,
                               async_ = bool(async_))

        def D2H(self, ary = None, async_=None, idpy_stream=None):
            _queue = self.queue if idpy_stream is None else idpy_stream
            return super().get(queue = _queue, ary = ary)

        def SetConst(self, const = 0., wait_for = None):
            super().fill(value = const, queue = self.queue, wait_for = wait_for)

        '''
        Residency primitives (Phase 2b) -- mirror of the CUDA surface above.
        See docs/residency-probes.md, F2. pyopencl's own Array.set already
        accepts 'async_'; H2D discarded it until now.
        '''
        def _ByteOffset(self, start):
            return int(start) * int(self.dtype.itemsize)

        def _CheckRange(self, start, stop):
            start, stop = int(start), int(stop)
            if start < 0 or stop > int(self.size) or start >= stop:
                raise ValueError(
                    "Sub-range [%d, %d) out of bounds for array of size %d"
                    % (start, stop, int(self.size))
                )
            return start, stop

        def SubView(self, start, stop):
            '''
            Contiguous element sub-range sharing this array's storage.

            Built directly from base_data + byte offset rather than by slicing,
            so it does not depend on pyopencl's internal derived-array protocol.
            '''
            start, stop = self._CheckRange(start, stop)
            return IdpyArrayOCL(
                shape = (stop - start,), dtype = self.dtype, queue = self.queue,
                data = self.base_data,
                offset = int(self.offset) + self._ByteOffset(start),
            )

        def H2DSub(self, ary, start = 0, async_ = False, idpy_stream = None):
            '''Write 'ary' into this array starting at element 'start'.'''
            ary = np.ascontiguousarray(ary, dtype = self.dtype)
            start, _ = self._CheckRange(start, start + ary.size)
            _queue = self.queue if idpy_stream is None else idpy_stream
            return cl.enqueue_copy(
                _queue, self.base_data, ary,
                dst_offset = int(self.offset) + self._ByteOffset(start),
                is_blocking = not async_,
            )

        def D2HSub(self, start, stop, ary = None, async_ = False,
                   idpy_stream = None):
            '''Read elements [start, stop) into 'ary' (allocated when None).'''
            start, stop = self._CheckRange(start, stop)
            if ary is None:
                ary = np.empty((stop - start,), dtype = self.dtype)
            _queue = self.queue if idpy_stream is None else idpy_stream
            _evt = cl.enqueue_copy(
                _queue, ary, self.base_data,
                src_offset = int(self.offset) + self._ByteOffset(start),
                is_blocking = not async_,
            )
            return ary if not async_ else (ary, _evt)

        def Sync(self, idpy_stream = None):
            '''Wait on outstanding async work on this array's queue.'''
            _queue = self.queue if idpy_stream is None else idpy_stream
            _queue.finish()

    def _on_device_OCL(ary, tenet):
        _swap_array = IdpyArrayOCL(shape = ary.shape,
                                   dtype = ary.dtype,
                                   queue = tenet,
                                   allocator = tenet.mem_pool)
        _swap_array.H2D(ary)
        return _swap_array

    def _zeros_OCL(shape, dtype, tenet):
        _swap_array = IdpyArrayOCL(shape = shape,
                                   dtype = dtype,
                                   queue = tenet,
                                   allocator = tenet.mem_pool)
        _swap_array.SetConst(0)
        return _swap_array

    def _range_OCL(n, tenet, dtype = np.int32):
        _tmp_range = np.arange(n, dtype = dtype)
        _swap_array = _on_device_OCL(_tmp_range, tenet = tenet)
        del _tmp_range
        return _swap_array

    def _const_OCL(shape, dtype, const = 0., tenet = None):
        _swap_array = IdpyArrayOCL(shape = shape,
                                   dtype = dtype,
                                   queue = tenet,
                                   allocator = tenet.mem_pool)
        _swap_array.SetConst(const)
        return _swap_array

    def _sum_OCL(a, dtype = None, queue = None, slice = None):
        return cl_array.sum(a = a, dtype = dtype,
                            queue = queue, slice = slice).get(queue = queue)

    def _max_OCL(a, queue = None):
        return cl_array.max(a = a, queue = queue)

    def _min_OCL(a, queue = None):
        return cl_array.min(a = a, queue = queue)

if idpy_langs_sys[CTYPES_T]:
    from idpy.core.backends.ctypes_backend.backend import Tenet as CTTenet

    class IdpyArrayCTYPES(np.ndarray):
        def __new__(subtype, shape, dtype, buffer=None, offset=0, strides=None,
                    order='C', tenet=None):
            
            obj = \
                super().__new__(
                    subtype, shape=shape, dtype=dtype, buffer=buffer, 
                    offset=offset, strides=strides, order=order
                )

            obj.lang = CTYPES_T
            obj.tenet = tenet
            return obj

        def __array_finalize__(self, obj):
            if obj is None: return
            self.lang = getattr(obj, 'lang', None)
            self.tenet = getattr(obj, 'tenet', None)
            '''
            H2D/D2H are deliberately NOT copied from 'obj'.

            They are class methods; copying them installed the *parent's bound
            method* as an instance attribute on every derived array, so a slice's
            .D2H() returned the whole parent rather than the slice. Harmless
            while arrays were only ever used whole, which is why it survived --
            it surfaces the moment sub-ranges exist. Leaving them off lets normal
            attribute lookup bind each method to the array it is called on.
            '''

        def H2D(self, ary, async_=None):
            return super().put(indices=np.arange(len(ary.ravel())), values=ary)

        def D2H(self, ary = None, async_=None):
            if ary is None:
                return super().copy()
            else:
                ary = super().copy()

        def SetConst(self, const=0.):
            super().fill(const)

        '''
        Residency primitives (Phase 2b), CTypes lowering.

        Host memory is the device here, so these are numpy operations and every
        synchronization concept is vacuous. They exist so that code written
        against the residency interface runs unchanged on the CPU -- which is
        what makes CTypes usable as the reference implementation for the
        policy layer, where the eviction logic can be validated with no vendor
        machinery in the way.
        '''
        def _CheckRange(self, start, stop):
            start, stop = int(start), int(stop)
            if start < 0 or stop > int(self.size) or start >= stop:
                raise ValueError(
                    "Sub-range [%d, %d) out of bounds for array of size %d"
                    % (start, stop, int(self.size))
                )
            return start, stop

        def SubView(self, start, stop):
            '''Aliasing sub-range: numpy slicing already shares storage.'''
            start, stop = self._CheckRange(start, stop)
            return self[start:stop]

        def H2DSub(self, ary, start = 0, async_ = False, idpy_stream = None):
            ary = np.ascontiguousarray(ary, dtype = self.dtype)
            start, stop = self._CheckRange(start, start + ary.size)
            np.copyto(np.asarray(self)[start:stop], ary)
            return ary

        def D2HSub(self, start, stop, ary = None, async_ = False,
                   idpy_stream = None):
            start, stop = self._CheckRange(start, stop)
            _src = np.asarray(self)[start:stop]
            if ary is None:
                return _src.copy()
            np.copyto(ary, _src)
            return ary

        def Sync(self, idpy_stream = None):
            '''Nothing executes asynchronously on this backend.'''
            return None


    def _on_device_CTYPES(ary, tenet=None):
        _swap_array = \
            IdpyArrayCTYPES(
                shape = ary.shape,
                dtype = ary.dtype,
                tenet = tenet,
                )

        _swap_array.H2D(ary)
        return _swap_array

    def _zeros_CTYPES(shape, dtype, tenet=None):
        _swap_array = \
            IdpyArrayCTYPES(
                shape = shape,
                dtype = dtype,
                tenet = tenet,
                )
        _swap_array.SetConst(0)
        return _swap_array

    def _range_CTYPES(n, dtype = np.int32, tenet=None):
        _tmp_range = np.arange(n, dtype = dtype)
        _swap_array = _on_device_CTYPES(_tmp_range, tenet=tenet)
        del _tmp_range
        return _swap_array

    def _const_CTYPES(shape, dtype, const = 0., tenet=None):
        _swap_array = \
            IdpyArrayCTYPES(
                shape = shape,
                dtype = dtype,
                tenet = tenet,
                )
        _swap_array.SetConst(const)
        return _swap_array

    def _sum_CTYPES(a, dtype = None, stream = None):
        return np.sum(a = a, dtype = dtype)

    def _max_CTYPES(a, stream = None):
        return np.amax(a = a)

    def _min_CTYPES(a, stream = None):
        return np.amin(a = a)

if idpy_langs_sys[METAL_T]:
    import pymetallic
    from idpy.core.backends.metal.backend import Tenet as MTTenet

    class IdpyArrayMETAL:
        '''
        Persistent Metal buffer + NumPy view of the same unified-memory storage.

        Write into ``.host`` for zero-copy CPU updates. ``H2D(ary)`` copies into
        that view without reallocating the Metal buffer. ``D2H()`` returns the
        shared view (mutating it mutates device-visible memory; unlike CUDA/OCL).
        After GPU kernels, wait on the command buffer before reading ``.host``.

        When allocated via ``tenet.mem_pool``, the underlying Buffer is returned
        to the free-list on teardown instead of being released immediately.
        '''
        def __init__(self, shape, dtype, tenet=None, data=None,
                     pooled=False, nbytes=None, host=None, offset=0):
            if tenet is None:
                raise Exception("Need to pass tenet = tenetObject")
            self.shape = shape if isinstance(shape, tuple) else (shape,)
            self.dtype = np.dtype(dtype)
            self.tenet = tenet
            self.lang = METAL_T
            self.size = int(np.prod(self.shape))
            self._pooled = bool(pooled)
            self._nbytes = (
                int(nbytes) if nbytes is not None
                else self.size * int(self.dtype.itemsize)
            )
            self._returned = False
            # Element offset of this array inside its Metal Buffer. Non-zero
            # only for SubView results; keeps byte spans comparable with the
            # parent for range-scoped waiting.
            self.offset = int(offset)
            if data is None:
                zeros = np.zeros(self.shape, dtype=self.dtype)
                self.data = pymetallic.Buffer.from_numpy(tenet.device, zeros)
            else:
                self.data = data
            # 'host' lets a caller supply a numpy view that is narrower than the
            # whole Buffer (SubView). to_numpy() validates shape against the
            # full allocation, so a sub-range cannot go through it.
            self.host = (
                host if host is not None
                else self.data.to_numpy(self.dtype, self.shape)
            )

        @property
        def ndim(self):
            return len(self.shape)

        '''
        Buffer identity and byte spans.

        Identity is the underlying pymetallic Buffer, not this wrapper, so a
        SubView and its parent resolve to the same key and their spans are
        directly comparable. Spans are byte offsets from the start of that
        Buffer, which is what makes overlap testing meaningful across views.
        '''
        def BufferKey(self):
            return id(self.data)

        def SpanBytes(self, start, stop):
            _item = int(self.dtype.itemsize)
            return ((self.offset + int(start)) * _item,
                    (self.offset + int(stop)) * _item)

        def BufferSpanBytes(self):
            return self.SpanBytes(0, self.size)

        def _sync_tenet(self, start=None, stop=None):
            '''
            Wait for GPU work that may touch [start, stop) of this array -- and
            only that work.

            Replaces the previous unconditional tenet.Finish() drain (finding
            F2), which serialized the host against every outstanding kernel and
            made "write slot k while the GPU reads slot j" impossible to express.

            Falls back to a full drain on any tenet that predates range
            tracking, so the conservative behaviour is what happens when
            anything is unknown.
            '''
            _wait = getattr(self.tenet, 'WaitForRange', None)
            if callable(_wait):
                _start, _stop = self.SpanBytes(
                    0 if start is None else start,
                    self.size if stop is None else stop,
                )
                return _wait(self.BufferKey(), _start, _stop)
            _finish = getattr(self.tenet, 'Finish', None)
            if callable(_finish):
                _finish()
            return True

        def H2D(self, ary, async_=None):
            self._sync_tenet()
            ary = np.asarray(ary, dtype=self.dtype).reshape(self.shape)
            np.copyto(self.host, ary)
            return None

        def D2H(self, ary=None, async_=None):
            self._sync_tenet()
            if ary is None:
                return self.host
            np.copyto(ary, self.host)
            return ary

        def SetConst(self, const=0., stream=None):
            self._sync_tenet()
            self.host.fill(const)

        '''
        Residency primitives (Phase 2b), Metal lowering.

        Unified memory makes these simpler than on CUDA/OpenCL: there is no
        staging copy, so a "partial write" is a host store straight into the
        shared buffer. What used to make it impossible was not the copy but the
        synchronization -- see _sync_tenet above.

        Consequence worth stating: 'async_' is accepted for signature parity and
        has no meaning here. There is no DMA to overlap with; the write IS the
        store, and once the range-scoped wait returns it is safe and immediate.
        '''
        def _CheckRange(self, start, stop):
            start, stop = int(start), int(stop)
            if start < 0 or stop > int(self.size) or start >= stop:
                raise ValueError(
                    "Sub-range [%d, %d) out of bounds for array of size %d"
                    % (start, stop, int(self.size))
                )
            return start, stop

        def SubView(self, start, stop):
            '''
            Contiguous element sub-range aliasing this array's storage: the same
            Metal Buffer, a numpy view of the same memory, and an offset so its
            spans stay expressed against the parent Buffer.
            '''
            start, stop = self._CheckRange(start, stop)
            return IdpyArrayMETAL(
                shape=(stop - start,), dtype=self.dtype, tenet=self.tenet,
                data=self.data, pooled=False,
                nbytes=(stop - start) * int(self.dtype.itemsize),
                host=self.host[start:stop], offset=self.offset + start,
            )

        def H2DSub(self, ary, start=0, async_=False, idpy_stream=None):
            '''Write 'ary' into this array starting at element 'start'.'''
            ary = np.ascontiguousarray(ary, dtype=self.dtype)
            start, stop = self._CheckRange(start, start + ary.size)
            self._sync_tenet(start, stop)
            np.copyto(self.host[start:stop], ary)
            return ary

        def D2HSub(self, start, stop, ary=None, async_=False, idpy_stream=None):
            '''Read elements [start, stop) into 'ary' (allocated when None).'''
            start, stop = self._CheckRange(start, stop)
            self._sync_tenet(start, stop)
            if ary is None:
                return self.host[start:stop].copy()
            np.copyto(ary, self.host[start:stop])
            return ary

        def Sync(self, idpy_stream=None):
            '''Wait for all outstanding GPU work on this tenet.'''
            _finish = getattr(self.tenet, 'Finish', None)
            if callable(_finish):
                _finish()

        def release_to_pool(self):
            if (
                self._pooled and not self._returned
                and self.tenet is not None
                and getattr(self.tenet, 'mem_pool', None) is not None
                and self.data is not None
            ):
                self.tenet.mem_pool.free_buffer(
                    self.data, self._nbytes, self.dtype.str,
                )
                self._returned = True
                self.data = None
                self.host = None

        def __del__(self):
            try:
                self.release_to_pool()
            except Exception:
                pass

    def _allocate_METAL(shape, dtype, tenet):
        pool = getattr(tenet, 'mem_pool', None)
        if pool is not None:
            buf, shape, dtype, nbytes = pool.allocate_buffer(shape, dtype)
            return IdpyArrayMETAL(
                shape=shape, dtype=dtype, tenet=tenet,
                data=buf, pooled=True, nbytes=nbytes,
            )
        return IdpyArrayMETAL(shape=shape, dtype=dtype, tenet=tenet)

    def _on_device_METAL(ary, tenet):
        ary = np.asarray(ary)
        _swap_array = _allocate_METAL(ary.shape, ary.dtype, tenet)
        _swap_array.H2D(ary)
        return _swap_array

    def _zeros_METAL(shape, dtype, tenet):
        _swap_array = _allocate_METAL(shape, dtype, tenet)
        _swap_array.SetConst(0)
        return _swap_array

    def _range_METAL(n, tenet, dtype=np.int32):
        _tmp_range = np.arange(n, dtype=dtype)
        _swap_array = _on_device_METAL(_tmp_range, tenet=tenet)
        del _tmp_range
        return _swap_array

    def _const_METAL(shape, dtype, const=0., tenet=None):
        _swap_array = _allocate_METAL(shape, dtype, tenet)
        _swap_array.SetConst(const)
        return _swap_array

    def _sum_METAL(a, dtype=None, stream=None):
        return np.sum(a.D2H(), dtype=dtype)

    def _max_METAL(a, stream=None):
        return np.amax(a.D2H())

    def _min_METAL(a, stream=None):
        return np.amin(a.D2H())

def PinnedHost(shape, dtype, tenet = None):
    '''
    Host staging buffer suitable for overlapped transfers.

    On CUDA this is page-locked memory, which is *required* for an async H2D to
    genuinely overlap with compute -- a pageable buffer silently degrades to a
    synchronous copy. Every other backend returns a plain numpy array, which is
    the correct answer there: OpenCL pins internally as needed, and Metal/CTypes
    are unified so no staging copy exists to overlap.

    Returning a real numpy-compatible buffer everywhere keeps caller code
    backend-agnostic.
    '''
    if tenet is not None and idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return _pinned_host_CUDA(shape, dtype)
    return np.empty(shape, dtype = dtype)


def SiblingStream(tenet):
    '''
    A second execution stream on the same device, for issuing a transfer
    concurrently with a kernel rather than behind it.

    CUDA   -> a new cu_driver.Stream(); pass it as 'idpy_stream' to Deploy or to
              the H2DSub/D2HSub family.
    OpenCL -> a sibling CommandQueue on the same context/device. Needed because
              a single in-order queue is entitled to serialize the two.
    Metal / CTypes -> None. Metal currently drains on every host touch (F2), so
              there is nothing to be concurrent with until that is reworked.

    Returns None when the backend has no sibling-stream concept, so callers can
    branch on it rather than on the language tag.
    '''
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return cu_driver.Stream()

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        '''
        from_parent() copies only context/device/properties, not the runtime
        attributes OpenCL.GetTenet() attaches afterwards (kind, device_name,
        mem_pool). Without them the sibling is not usable where the parent is --
        instantiating a kernel against it fails on 'kind'. Carry them over, but
        share the parent's memory pool rather than creating a second one: two
        pools on one context would fragment the same device memory.
        '''
        _sibling = CLTenet.from_parent(tenet)
        _sibling.SetKind(tenet.GetKind())
        _sibling.SetDeviceName(tenet.device_name)
        _sibling.mem_pool = tenet.mem_pool
        return _sibling

    return None


def Array(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return IdpyArrayCUDA(*args, **kwargs, tenet = tenet,
                             allocator = tenet.allocator)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return IdpyArrayOCL(*args, **kwargs, queue = tenet,
                            allocator = tenet.mem_pool)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return IdpyArrayCTYPES(*args, **kwargs, tenet=tenet)

    if idpy_langs_sys[METAL_T] and isinstance(tenet, MTTenet):
        return IdpyArrayMETAL(*args, **kwargs, tenet=tenet)

def OnDevice(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return _on_device_CUDA(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return _on_device_OCL(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return _on_device_CTYPES(*args, **kwargs, tenet=tenet)

    if idpy_langs_sys[METAL_T] and isinstance(tenet, MTTenet):
        return _on_device_METAL(*args, **kwargs, tenet=tenet)

def Zeros(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return _zeros_CUDA(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return _zeros_OCL(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return _zeros_CTYPES(*args, **kwargs, tenet=tenet)

    if idpy_langs_sys[METAL_T] and isinstance(tenet, MTTenet):
        return _zeros_METAL(*args, **kwargs, tenet=tenet)

def Range(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return _range_CUDA(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return _range_OCL(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return _range_CTYPES(*args, **kwargs, tenet=tenet)

    if idpy_langs_sys[METAL_T] and isinstance(tenet, MTTenet):
        return _range_METAL(*args, **kwargs, tenet=tenet)

def Const(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return _const_CUDA(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return _const_OCL(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return _const_CTYPES(*args, **kwargs, tenet=tenet)

    if idpy_langs_sys[METAL_T] and isinstance(tenet, MTTenet):
        return _const_METAL(*args, **kwargs, tenet=tenet)

'''
Very basic implementation: need to check the respective tenets
and possibly use more performant functions
'''
def D2D(ary_src, ary_dst, idpy_stream=None, async_src=False, async_dst=False):
    ary_dst.H2D(ary_src.D2H(async_=async_src), async_=async_dst)
    return idpy_stream

def Sum(ary, idpy_stream = None):
    if idpy_langs_sys[CUDA_T] and ary.lang == CUDA_T:
        return _sum_CUDA(a = ary, dtype = ary.dtype, stream = idpy_stream).get().item()

    if idpy_langs_sys[OCL_T] and ary.lang == OCL_T:
        return _sum_OCL(a = ary, dtype = ary.dtype, queue = ary.queue).item()

    if idpy_langs_sys[CTYPES_T] and ary.lang == CTYPES_T:
        return _sum_CTYPES(a = ary, dtype = ary.dtype, stream = idpy_stream).item()

    if idpy_langs_sys[METAL_T] and ary.lang == METAL_T:
        return _sum_METAL(a = ary, dtype = ary.dtype, stream = idpy_stream).item()

def Max(ary, idpy_stream = None):
    if idpy_langs_sys[CUDA_T] and ary.lang == CUDA_T:
        return _max_CUDA(a = ary, stream = idpy_stream).get().item()

    if idpy_langs_sys[OCL_T] and ary.lang == OCL_T:
        return _max_OCL(a = ary, queue = ary.queue).get(queue = ary.queue).item()

    if idpy_langs_sys[CTYPES_T] and ary.lang == CTYPES_T:
        return _max_CTYPES(a = ary, stream = idpy_stream).item()

    if idpy_langs_sys[METAL_T] and ary.lang == METAL_T:
        return _max_METAL(a = ary, stream = idpy_stream).item()

def Min(ary, idpy_stream = None):
    if idpy_langs_sys[CUDA_T] and ary.lang == CUDA_T:
        return _min_CUDA(a = ary, stream = idpy_stream).get().item()

    if idpy_langs_sys[OCL_T] and ary.lang == OCL_T:
        return _min_OCL(a = ary, queue = ary.queue).get(queue = ary.queue).item()

    if idpy_langs_sys[CTYPES_T] and ary.lang == CTYPES_T:
        return _min_CTYPES(a = ary, stream = idpy_stream).item()

    if idpy_langs_sys[METAL_T] and ary.lang == METAL_T:
        return _min_METAL(a = ary, stream = idpy_stream).item()
    
'''
need to define IdpySum:
a class that can used in IdpyLoop's
'''
