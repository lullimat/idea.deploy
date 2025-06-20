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

from idpy.IdpyCode import OCL_T, CUDA_T, CTYPES_T, idpy_tenet_types
from idpy.IdpyCode import idpy_langs_list, idpy_langs_sys

if idpy_langs_sys[CUDA_T]:
    import pycuda as cu
    import pycuda.driver as cu_driver
    import pycuda.gpuarray as cu_array
    from idpy.CUDA.CUDA import Tenet as CUTenet

    class IdpyArrayCUDA(cu_array.GPUArray):
        def __init__(self, shape, dtype,
                     allocator = None, base = None,
                     gpudata = None, strides = None, order = 'C'):
            
            if allocator is None:
                allocator = cu_driver.mem_alloc
                
            super().__init__(shape = shape, dtype = dtype,
                             allocator = allocator, base = base,
                             gpudata = gpudata, strides = strides,
                             order = order)
            self.lang = CUDA_T

        def H2D(self, ary, async_=None):
            return super().set(ary = ary)

        def D2H(self, ary = None, pagelocked = False, async_=None):
            return super().get(ary = ary, pagelocked = pagelocked)

        def SetConst(self, const = 0., stream = None):
            super().fill(value = const, stream = stream)

    def _on_device_CUDA(ary, allocator = None):
        _swap_array = IdpyArrayCUDA(shape = ary.shape,
                                    dtype = ary.dtype,
                                    allocator = allocator)
        _swap_array.H2D(ary)
        return _swap_array

    def _zeros_CUDA(shape, dtype, allocator = None):
        _swap_array = IdpyArrayCUDA(shape = shape,
                                    dtype = dtype,
                                    allocator = allocator)
        _swap_array.SetConst(0)
        return _swap_array

    def _range_CUDA(n, allocator = None, dtype = np.int32):
        _tmp_range = np.arange(n, dtype = dtype)
        _swap_array = _on_device_CUDA(_tmp_range, allocator = allocator)
        del _tmp_range
        return _swap_array

    def _const_CUDA(shape, dtype, const = 0., allocator = None):
        _swap_array = IdpyArrayCUDA(shape = shape,
                                    dtype = dtype,
                                    allocator = allocator)
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
    from idpy.OpenCL.OpenCL import Tenet as CLTenet

    class IdpyArrayOCL(cl_array.Array):
        def __init__(self, shape = None, queue = None, dtype = None,
                     order = 'C',
                     allocator = None, data = None,
                     offset = 0, strides = None,
                     events = None):
            super().__init__(cq = queue, shape = shape,
                             dtype = dtype, order = order,
                             allocator = allocator, data = data,
                             offset = offset, strides = strides,
                             events = events)

            self.lang, self.queue = OCL_T, queue

        def H2D(self, ary, async_=None):
            return super().set(ary = ary, queue = self.queue)

        def D2H(self, ary = None, async_=None):
            return super().get(queue = self.queue, ary = ary)

        def SetConst(self, const = 0., wait_for = None):
            super().fill(value = const, queue = self.queue, wait_for = wait_for)

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
    from idpy.CTypes.CTypes import Tenet as CTTenet

    class IdpyArrayCTYPES(np.ndarray):
        def __new__(subtype, shape, dtype, buffer=None, offset=0, strides=None, order='C'):
            
            obj = \
                super().__new__(
                    subtype, shape=shape, dtype=dtype, buffer=buffer, 
                    offset=offset, strides=strides, order=order
                )

            obj.lang = CTYPES_T
            return obj

        def __array_finalize__(self, obj):
            if obj is None: return
            self.lang = getattr(obj, 'lang', None)
            self.H2D = getattr(obj, 'H2D', None)
            self.D2H = getattr(obj, 'D2H', None)

        def H2D(self, ary, async_=None):
            return super().put(indices=np.arange(len(ary.ravel())), values=ary)

        def D2H(self, ary = None, async_=None):
            if ary is None:
                return super().copy()
            else:
                ary = super().copy()

        def SetConst(self, const=0.):
            super().fill(const)            


    def _on_device_CTYPES(ary):
        _swap_array = \
            IdpyArrayCTYPES(
                shape = ary.shape,
                dtype = ary.dtype
                )

        _swap_array.H2D(ary)
        return _swap_array

    def _zeros_CTYPES(shape, dtype):
        _swap_array = \
            IdpyArrayCTYPES(
                shape = shape,
                dtype = dtype
                )
        _swap_array.SetConst(0)
        return _swap_array

    def _range_CTYPES(n, dtype = np.int32):
        _tmp_range = np.arange(n, dtype = dtype)
        _swap_array = _on_device_CTYPES(_tmp_range)
        del _tmp_range
        return _swap_array

    def _const_CTYPES(shape, dtype, const = 0.):
        _swap_array = \
            IdpyArrayCTYPES(
                shape = shape,
                dtype = dtype
                )
        _swap_array.SetConst(const)
        return _swap_array

    def _sum_CTYPES(a, dtype = None, stream = None):
        return np.sum(a = a, dtype = dtype)

    def _max_CTYPES(a, stream = None):
        return np.amax(a = a)

    def _min_CTYPES(a, stream = None):
        return np.amin(a = a)

    
def Array(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return IdpyArrayCUDA(*args, **kwargs, allocator = tenet.allocator)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return IdpyArrayOCL(*args, **kwargs, queue = tenet,
                            allocator = tenet.mem_pool)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return IdpyArrayCTYPES(*args, **kwargs)

def OnDevice(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return _on_device_CUDA(*args, **kwargs, allocator = tenet.allocator)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return _on_device_OCL(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return _on_device_CTYPES(*args, **kwargs)        

def Zeros(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return _zeros_CUDA(*args, **kwargs, allocator = tenet.allocator)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return _zeros_OCL(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return _zeros_CTYPES(*args, **kwargs)

def Range(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return _range_CUDA(*args, **kwargs, allocator = tenet.allocator)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return _range_OCL(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return _range_CTYPES(*args, **kwargs)        

def Const(*args, **kwargs):
    if 'tenet' not in kwargs:
        raise Exception("Need to pass tenet = tenetObject")
    
    tenet = kwargs['tenet']
    del kwargs['tenet']
    
    if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
        return _const_CUDA(*args, **kwargs, allocator = tenet.allocator)

    if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
        return _const_OCL(*args, **kwargs, tenet = tenet)

    if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
        return _const_CTYPES(*args, **kwargs)

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

def Max(ary, idpy_stream = None):
    if idpy_langs_sys[CUDA_T] and ary.lang == CUDA_T:
        return _max_CUDA(a = ary, stream = idpy_stream).get().item()

    if idpy_langs_sys[OCL_T] and ary.lang == OCL_T:
        return _max_OCL(a = ary, queue = ary.queue).get(queue = ary.queue).item()

    if idpy_langs_sys[CTYPES_T] and ary.lang == CTYPES_T:
        return _max_CTYPES(a = ary, stream = idpy_stream).item() 

def Min(ary, idpy_stream = None):
    if idpy_langs_sys[CUDA_T] and ary.lang == CUDA_T:
        return _min_CUDA(a = ary, stream = idpy_stream).get().item()

    if idpy_langs_sys[OCL_T] and ary.lang == OCL_T:
        return _min_OCL(a = ary, queue = ary.queue).get(queue = ary.queue).item()

    if idpy_langs_sys[CTYPES_T] and ary.lang == CTYPES_T:
        return _min_CTYPES(a = ary, stream = idpy_stream).item() 
    
'''
need to define IdpySum:
a class that can used in IdpyLoop's
'''
