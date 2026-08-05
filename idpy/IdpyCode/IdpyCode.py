__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2022 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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
Provides a parent class for kernels and device functions meta-code
Philosophy/Hope: the idea is that language specifical built-in functions
should be called in separated verrsions of the same meta device functions
so that they can be selected at compile time, by simply selecting the language.
The most important example should be reading from the shared memory in CUDA
and some alternative actions in OpenCL and Metal. This should be fine
as long as the different meta-declarations are consistent
'''

import numpy as np
from collections import defaultdict
from pathlib import Path
import sys

from functools import reduce

from idpy.IdpyCode.IdpyConsts import AddrQualif, KernQualif, FuncQualif, SyncQualif

from idpy.IdpyCode import idpy_nvcc_path, idpy_langs_list
from idpy.IdpyCode import idpy_langs_dict_sym, idpy_langs_sys
from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T, CTYPES_T, METAL_T
from idpy.IdpyCode import idpy_opencl_macro_spacing
from idpy.IdpyCode import idpy_copyright

from idpy.IdpyCode.IdpyUnroll import _codify_comment
from idpy.Utils.SimpleTiming import SimpleTiming

# Max signed 64-bit; larger ints need an unsigned C literal suffix in macros.
_C_INT64_MAX = 0x7fffffffffffffff


def _format_c_macro_value(value, lang=None):
    '''
    Format a Python constant for #define / -D emission.
    Values above signed int64 max (e.g. ID_RANDMAX_MMIX = 2^64-1) need an
    explicit unsigned suffix or compilers warn / mis-parse the literal.
    '''
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, int):
        if value > _C_INT64_MAX:
            # Metal has no long long; unsigned long is 64-bit.
            if lang == METAL_T:
                return str(value) + 'UL'
            return str(value) + 'ULL'
        return str(value)
    return str(value)

if idpy_langs_sys[CUDA_T]:
    from idpy.IdpyCode.IdpyMemory import IdpyArrayCUDA
if idpy_langs_sys[OCL_T]:
    from idpy.IdpyCode.IdpyMemory import IdpyArrayOCL
if idpy_langs_sys[METAL_T]:
    from idpy.IdpyCode.IdpyMemory import IdpyArrayMETAL

# Need this to implement types checks
from idpy.Utils.CustomTypes import CustomTypes

if idpy_langs_sys[CUDA_T]:
    import pycuda as cu
    import pycuda.driver as cu_driver
    from pycuda.compiler import SourceModule as cu_SourceModule
    import pycuda.gpuarray as cu_array
    from idpy.CUDA.CUDA import Tenet as CUTenet

if idpy_langs_sys[OCL_T]:
    import pyopencl as cl
    import pyopencl.array as cl_array
    from idpy.OpenCL.OpenCL import Tenet as CLTenet
    from idpy.OpenCL.OpenCL import OpenCL

if idpy_langs_sys[CTYPES_T]:
    import ctypes
    from numpy import array as ct_array
    from idpy.CTypes.CTypes import Tenet as CTTenet
    from idpy.CTypes.CTypes import CTypes
    from idpy.CTypes.CTypes import CTYPES_N_THREAD

if idpy_langs_sys[METAL_T]:
    import pymetallic
    from idpy.Metal.Metal import Tenet as MTTenet
    from idpy.Metal.Metal import Metal

class IdpyKernel:
    '''
    class IdpyKernel:
    parent class for implementing the meta-code once and manage
    the different features of specific languages on demand
    It does not need to be aware of the possible types
    ---
    To be done:
    - Add method for managing 'special' declarations that can be language
    dependent: DONE just need to specify the language when writing the
    kernel
    - Need to discuss somewhere the difference between CUDA grid and OpenCL
    '''
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 gthread_id_code = 'g_tid', lthread_id_code = 'l_tid',
                 lthread_id_coords_code = 'l_tid_c', block_coords_code = 'bid_c',
                 optimizer_flag = None, declare_types = None, declare_macros = None,
                 headers_files = None, include_dirs = None,
                 definitions_files = None, objects_files = None):

        if type(custom_types) is not dict:
            raise Exception("custom_types param must be a dict")
        if headers_files is not None and type(headers_files) not in (list, tuple):
            raise Exception("headers_files param must be a list or tuple")
        if include_dirs is not None and type(include_dirs) is not list:
            raise Exception("include_dirs param must be a list")
        if definitions_files is not None and type(definitions_files) is not list:
            raise Exception("definitions_files param must be a list")
        if objects_files is not None and type(objects_files) is not list:
            raise Exception("objects_files param must be a list")
        
        self.code, self.name = "", self.__class__.__name__
        self.kernels, self.params, self.f_classes, self.functions = \
            {}, {}, f_classes, []
        
        self.custom_types, self.constants = custom_types, constants
        '''
        Need to check the type of optimizer_flag
        '''
        self.optimizer_flag = True if optimizer_flag is None else optimizer_flag
        self.declare_types = 'typedef' if declare_types is None else declare_types
        if self.declare_types not in ['typedef', 'macro']:
            raise Exception("declare_types must be either 'typedef' or 'macro'")

        self.declare_macros = 'header' if declare_macros is None else declare_macros
        if self.declare_macros not in ['header', 'macro']:
            raise Exception("declare_macros must be either 'header' or 'macro'")
        
        # Copy so callers' mutable defaults (e.g. headers_files=['math.h'])
        # are not emptied by lang-specific remove() in Code(). Prefer tuple
        # defaults so the shared default object cannot be mutated.
        self.headers_files = (
            list(headers_files) if headers_files is not None else None
        )
        self.declarations = {}

        '''
        The idea is to combine consts and types macros
        in the self.macros list
        '''
        self.macros_consts, self.macros = {}, None
        
        self.gthread_id_code, self.lthread_id_code = gthread_id_code, lthread_id_code
        self.lthread_id_coords_code, self.block_coords_code = \
            lthread_id_coords_code, block_coords_code

        '''
        Setting the default return type to 'int' which can be changed when inheriting
        '''
        self.return_type = 'int'

        self.kernels_qualifiers = KernQualif()
        self.AddrQ = AddrQualif()
        self.SyncQ = SyncQualif()

        '''
        List of variables and constants for metaprogramming
        '''
        self.declared_variables, self.declared_constants = [[]], [[]]

        '''
        Dynamic (runtime-sized) block/threadgroup-shared buffers.
        A list of dicts {'name': str, 'type': ctype_str, 'dtype': np.dtype};
        populated by SetDynamicSharedMemory(). Portability across CUDA/OpenCL/
        Metal follows the lowest common denominator (CUDA's single
        'extern __shared__' region), so at most one buffer is allowed.
        '''
        self.shared_dynamic = []

        # Code Flags
        self.code_flags = defaultdict(dict)
        self.InitCodeFlags()

    def SetDynamicSharedMemory(self, buffers = None):
        '''
        Declare runtime-sized shared memory that the kernel body can use by
        name (like a 'global' buffer but living in fast on-chip shared/local/
        threadgroup memory). The byte size is fixed per launch (at kernel
        instantiation), not at compile time -- mapping to:
            CUDA   : 'extern __shared__ T name[];' + launch 'shared=' bytes
            OpenCL : '__local T * name' kernel arg  + cl.LocalMemory(bytes)
            Metal  : 'threadgroup T * name [[threadgroup(0)]]'
                     + encoder.set_threadgroup_memory_length(bytes, 0)

        'buffers': a dict {name: {'type': ctype_str, 'dtype': np_dtype}} or a
        list of {'name','type','dtype'} dicts. numpy dtype is used to size the
        allocation from an element count. At most one buffer is supported
        (CUDA exposes a single dynamic shared region); index into it manually
        for multiple logical tiles.
        '''
        _norm = []
        if buffers is None:
            self.shared_dynamic = _norm
            return _norm
        if isinstance(buffers, dict):
            _items = [
                {'name': _n, 'type': _v['type'], 'dtype': np.dtype(_v['dtype'])}
                for _n, _v in buffers.items()
            ]
        else:
            _items = [
                {'name': _b['name'], 'type': _b['type'],
                 'dtype': np.dtype(_b['dtype'])}
                for _b in buffers
            ]
        if len(_items) > 1:
            raise NotImplementedError(
                "At most one dynamic shared buffer per kernel is portable: "
                "CUDA exposes a single 'extern __shared__' region. Declare one "
                "buffer and index into it manually for multiple logical tiles."
            )
        self.shared_dynamic = _items
        return _items

    def DynSharedBytes(self, block = None, dyn_shared_count = None,
                       dyn_shared_bytes = None):
        '''
        Resolve the dynamic-shared byte size for one launch. Explicit
        'dyn_shared_bytes' wins; else 'dyn_shared_count' elements times the
        buffer dtype's itemsize; else default to one element per thread
        (product of the block dimensions), the common tiling case.
        '''
        if not self.shared_dynamic:
            return 0
        if dyn_shared_bytes is not None:
            return int(dyn_shared_bytes)
        _itemsize = self.shared_dynamic[0]['dtype'].itemsize
        if dyn_shared_count is not None:
            return int(dyn_shared_count) * _itemsize
        if block is not None:
            _threads = int(reduce(lambda x, y: x * y, block))
            return _threads * _itemsize
        raise ValueError(
            "Cannot size dynamic shared memory: pass dyn_shared_bytes=, "
            "dyn_shared_count=, or a block to default to one element per thread"
        )

    def InitFunctions(self):
        '''
        need to manually insert a list of the needed functions:
        need to double check that duting the declaration the require functions
        are inserted
        '''
        if len(self.functions) == 0:
            for f_class in self.f_classes:
                self.functions.append(f_class(custom_types = self.custom_types))

    def SetDeclaredConstants(self):
        for const in self.constants:
            self.declared_constants[0] += [const]
        for param in self.params:
            if 'const' in self.params[param]:
                self.declared_constants[0] += [param.split(' ')[-1]]

    def SetDeclaredVariables(self):
        for param in self.params:
            if 'const' not in self.params[param]:
                '''
                The name of the variable is supposed to be last
                '''
                self.declared_variables[0] += [param.split(' ')[-1]]
                
    def SetMacros(self, lang = None):
        if lang == CUDA_T:
            self.macros = []
            # Constants
            if self.declare_macros == 'macro':
                for const in self.constants:
                    self.macros.append(
                        "-D " + const + "=" +
                        _format_c_macro_value(self.constants[const], lang)
                    )
                
            # Types
            if self.declare_types == 'macro':
                for c_type in self.custom_types:
                    self.macros.append("-D " + c_type + "=" + str(self.custom_types[c_type]))

            # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
            if self.optimizer_flag is False:
                self.macros.append("--device-debug")

            return self.macros

        if lang == OCL_T:
            self.macros = ''
            # Constants
            if self.declare_macros == 'macro':
                for const in self.constants:
                    self.macros += (
                        " -D " + const + "=" +
                        _format_c_macro_value(self.constants[const], lang)
                    )
                
            # Types
            # https://stackoverflow.com/questions/13531100/escaping-space-in-opencl-compiler-arguments
            if self.declare_types == 'macro':
                for c_type in self.custom_types:
                    self.macros += (" -D " + c_type + "=" + '\"' + str(self.custom_types[c_type]).replace(" ", idpy_opencl_macro_spacing) + '\"')

            if self.optimizer_flag is False:
                self.macros += " -cl-opt-disable"
                
            ##if self.macros == '':
            ##    self.macros = None
                
            return self.macros

        if lang == CTYPES_T:
            self.macros = ''
            # Constants
            if self.declare_macros == 'macro':            
                for const in self.constants:
                    self.macros += (
                        " -D " + const + "=" +
                        _format_c_macro_value(self.constants[const], lang)
                    )
                
            # Types
            # https://stackoverflow.com/questions/13531100/escaping-space-in-opencl-compiler-arguments
            if self.declare_types == 'macro':
                for c_type in self.custom_types:
                    self.macros += (" -D " + c_type + "=" + '\"' + str(self.custom_types[c_type]).replace(" ", idpy_opencl_macro_spacing) + '\"')

            if self.optimizer_flag is True:
                self.macros += " -O3"

            '''
            link agains the math library if math.h is included
            '''
            if self.headers_files is not None and 'math.h' in self.headers_files:
                self.macros += " -lm"
                
            if self.macros == '':
                self.macros = None
                
            return self.macros

        if lang == METAL_T:
            # Metal uses typedef/#define in source (header mode).
            # Compile-time optimization is controlled at Library() creation via
            # optimizer_flag → fast_math (MTLCompileOptions), not -D macros.
            self.macros = None
            return self.macros

    def GetCodeFlags(self):
        return self.code_flags

    def InitCodeFlags(self):
        self.UnsetCodeFlags(self.gthread_id_code)
        self.UnsetCodeFlags(self.lthread_id_code)
        self.UnsetCodeFlags(self.lthread_id_coords_code)
        self.UnsetCodeFlags(self.block_coords_code)

    def SetCodeFlags(self, key):
        self.code_flags[key] = True

    def UnsetCodeFlags(self, key):
        self.code_flags[key] = False

    def SetReturnType(self, type_str):
        self.return_type = type_str

    '''
    For the moment this method applies only to the 'global thread id'
    turning the parallel execution of threads into a loop over the 
    global thread id variable
    '''        
    def SetGlobalThreadId(self):
        _swap = {}
        _swap[CUDA_T] = ("""unsigned int """ + self.gthread_id_code + """ = """ + \
                         """\n
                         threadIdx.x + 
                         (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x + 
                         (blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x) * 
                         (blockDim.x * blockDim.y * blockDim.z);\n""")
        _swap[OCL_T] = ("""unsigned int """ + self.gthread_id_code + """ = """ + \
                        """\n
                        get_global_id(0) + 
                        (get_global_id(1) + get_global_id(2) * get_global_size(1)) * get_global_size(0);\n""")

        ## For CTypes we need to implement a loop, so we also need to close it at the end
        _swap[CTYPES_T] = \
            ("""for(unsigned int """ + self.gthread_id_code + """=0; """ + 
                self.gthread_id_code + """< """ + CTYPES_N_THREAD + """; """ + self.gthread_id_code + """++){\n""")

        # Metal: g_tid is a kernel parameter with [[thread_position_in_grid]]
        _swap[METAL_T] = ""

        return _swap

    def SetLocalThreadId(self):
        _swap = {}
        _swap[CUDA_T] = ("""unsigned int """ + self.lthread_id_code + """ = """ + \
                         """
                         threadIdx.x + 
                         (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x;\n""")
        _swap[OCL_T] = ("""unsigned int """ + self.lthread_id_code + """ = """ + \
                        """
                        get_local_id(0) +
                        (get_local_id(1) + get_local_id(2) * get_local_size(1)) * get_local_size(0);\n""")
        # Metal: l_tid is a kernel parameter with [[thread_position_in_threadgroup]]
        # when requested on its own. If the local thread coords are *also*
        # requested, that attribute is carried by the uint3 '<l_tid_c>_vec'
        # param (an attribute may appear once), so derive the linear id from it.
        if self.code_flags[self.lthread_id_coords_code]:
            _swap[METAL_T] = ("""unsigned int """ + self.lthread_id_code +
                              """ = """ + self.lthread_id_coords_code +
                              """_vec.x;\n""")
        else:
            _swap[METAL_T] = ""
        return _swap

    def SetLocalThreadCoords(self):
        _swap = {}
        _swap[CUDA_T] = ("""unsigned int """ + self.lthread_id_coords_code + """_x""" + """ = """ + \
                         """threadIdx.x;\n""" +
                         """unsigned int """ + self.lthread_id_coords_code + """_y""" + """ = """ + \
                         """threadIdx.y;\n""" +
                         """unsigned int """ + self.lthread_id_coords_code + """_z""" + """ = """ + \
                         """threadIdx.z;\n""")

        _swap[OCL_T] = ("""unsigned int """ + self.lthread_id_coords_code + """_x""" + """ = """ + \
                        """get_local_id(0);\n""" +
                        """unsigned int """ + self.lthread_id_coords_code + """_y""" + """ = """ + \
                        """get_local_id(1);\n""" +
                        """unsigned int """ + self.lthread_id_coords_code + """_z""" + """ = """ + \
                        """get_local_id(2);\n""")
        # Metal: coords come from the uint3 '<code>_vec [[thread_position_in_threadgroup]]'
        # kernel parameter injected by WriteCodeParams.
        _c = self.lthread_id_coords_code
        _swap[METAL_T] = ("""unsigned int """ + _c + """_x = """ + _c + """_vec.x;\n""" +
                          """unsigned int """ + _c + """_y = """ + _c + """_vec.y;\n""" +
                          """unsigned int """ + _c + """_z = """ + _c + """_vec.z;\n""")
        return _swap

    def SetLocalBlockCoords(self):
        _swap = {}
        _swap[CUDA_T] = ("""unsigned int """ + self.block_coords_code + """_x""" + """ = """ + \
                         """blockIdx.x;\n""" +
                         """unsigned int """ + self.block_coords_code + """_y""" + """ = """ + \
                         """blockIdx.y;\n""" +
                         """unsigned int """ + self.block_coords_code + """_z""" + """ = """ + \
                         """blockIdx.z;\n""")

        _swap[OCL_T] = ("""unsigned int """ + self.block_coords_code + """_x""" + """ = """ + \
                        """get_group_id(0);\n""" +
                        """unsigned int """ + self.block_coords_code + """_y""" + """ = """ + \
                        """get_group_id(1);\n""" +
                        """unsigned int """ + self.block_coords_code + """_z""" + """ = """ + \
                        """get_group_id(2);\n""")
        # Metal: block/threadgroup coords from the uint3
        # '<code>_vec [[threadgroup_position_in_grid]]' kernel parameter.
        _b = self.block_coords_code
        _swap[METAL_T] = ("""unsigned int """ + _b + """_x = """ + _b + """_vec.x;\n""" +
                          """unsigned int """ + _b + """_y = """ + _b + """_vec.y;\n""" +
                          """unsigned int """ + _b + """_z = """ + _b + """_vec.z;\n""")
        return _swap

    def WriteAsHeader(self, lang = None, prepend_path = None):
        if lang is None:
            raise Exception("'lang' param is not defined")

        _extension = '.cuh' if lang == CUDA_T else ('.hpp' if lang == OCL_T else '.h')
        _as_header_name = \
            self.__class__.__name__ + _extension
        _file_path = \
            Path(prepend_path if prepend_path is not None else '.') / _as_header_name

        with open(_file_path, 'w') as _header_file:
            for _line in idpy_copyright.splitlines():
                _header_file.write(_codify_comment(_line))
            _header_file.write(_codify_comment(""))
            _header_file.write(_codify_comment("This file was automatically generated from"))
            _header_file.write(_codify_comment("an instance of " + self.__class__.__name__))
            _header_file.write(_codify_comment("a child class of idpy.IdpyCode.IdpyKernel"))
            _header_file.write("\n")
            _header_file.write(self.Code(lang = lang))
        return _file_path

    def CleanAsHeader(self, prepend_path = None):
        _as_header_name = self.__class__.__name__ + '.h'
        _file_path = \
            Path(prepend_path if prepend_path is not None else '.') / _as_header_name
        if _file_path.is_file():
            pass

    def DeclareTypes(self):
        _swap = ''
        for c_type in self.custom_types:
            _swap  += 'typedef ' + str(self.custom_types[c_type]) + ' ' + c_type + ';\n'
        _swap += '\n'
        
        return _swap

    def DeclareMacros(self, lang = None):
        _swap = ''
        for c_macro in self.constants:
            _swap += (
                '#define ' + c_macro + ' ' +
                _format_c_macro_value(self.constants[c_macro], lang) + '\n'
            )
        _swap += '\n'
        
        return _swap    

    def IncludeHeaders(self):
        _swap = ''
        for _h_file in self.headers_files:
            _swap += '#include <' + _h_file + '>\n'
        _swap += '\n'
        return _swap

    def CollectiveMacros(self, lang = None):
        '''
        Emit the portable shared-memory / barrier metalanguage tokens as
        per-language #define's, so a single IDPY_T kernel body can harness
        block/threadgroup-shared memory and thread synchronization without
        hard-coding any backend intrinsic:

            idpy_shared         block/threadgroup-shared address qualifier
                                (CUDA __shared__ | OpenCL __local |
                                 Metal threadgroup | CTypes static)
            idpy_sync           collective barrier + shared(local) mem fence
                                (CUDA __syncthreads() | OpenCL
                                 barrier(CLK_LOCAL_MEM_FENCE) | Metal
                                 threadgroup_barrier(mem_flags::mem_threadgroup))
            idpy_sync_global    collective barrier + global(device) mem fence

        Same mechanism as the Metal 'precise::' math defines above: the right
        construct is chosen at compile time by 'lang'. Companion codegen helpers
        live in idpy.IdpyCode.IdpyUnroll (_codify_shared_declaration,
        _codify_sync, _codify_sync_global).
        '''
        _shared = self.AddrQ[lang]['shared']
        _sync = self.SyncQ[lang]['sync']
        _sync_global = self.SyncQ[lang]['sync_global']

        _macros = ''
        _macros += '#define idpy_shared ' + _shared + '\n'
        _macros += '#define idpy_sync ' + _sync + '\n'
        _macros += '#define idpy_sync_global ' + _sync_global + '\n\n'
        return _macros

    def DynSharedParams(self, lang = None, AddrQ = None):
        '''
        Kernel-parameter fragments for runtime-sized shared memory. OpenCL and
        Metal pass the region as a pointer argument (with '__local' /
        'threadgroup' address space); Metal also needs the '[[threadgroup(0)]]'
        attribute. CUDA/CTYPES return an empty list (CUDA declares it in-body).
        '''
        if not self.shared_dynamic or lang not in (OCL_T, METAL_T):
            return []
        _shared_q = AddrQ['shared']
        _params = []
        for _i, _buf in enumerate(self.shared_dynamic):
            _p = _shared_q + " " + _buf['type'] + " * " + _buf['name']
            if lang == METAL_T:
                _p += " [[threadgroup(" + str(_i) + ")]]"
            _params.append(_p)
        return _params

    def DynSharedCUDADeclaration(self):
        '''
        CUDA in-body declaration of the runtime-sized shared region:
            extern __shared__ T name[];
        The byte size is supplied at launch via the 'shared=' kernel argument.
        '''
        if not self.shared_dynamic:
            return ""
        _buf = self.shared_dynamic[0]
        return ("extern " + self.AddrQ[CUDA_T]['shared'] + " " +
                _buf['type'] + " " + _buf['name'] + "[];\n")

    def Code(self, lang = None):
        # Argument Qualifiers
        AddrQ = self.AddrQ[lang]
        self.ResetCode()
        # Inserting headers
        ## Checking for 'math.h'

        if lang == METAL_T:
            self.code += "#include <metal_stdlib>\n"
            self.code += "using namespace metal;\n"
            # Default Metal tanh/exp/log overflow to NaN for large |x| on Apple
            # GPUs (e.g. tanh(|x|≳44)); precise:: stays finite and saturates.
            self.code += "#define tanh(X) precise::tanh(X)\n"
            self.code += "#define exp(X) precise::exp(X)\n"
            self.code += "#define log(X) precise::log(X)\n"
            self.code += "#define log2(X) precise::log2(X)\n"
            self.code += "#define log10(X) precise::log10(X)\n\n"

        if self.headers_files is not None:
            # Work on a local copy — never mutate self.headers_files in place
            # (shared default lists like ['math.h'] must stay intact for CTYPES).
            _headers_for_code = list(self.headers_files)
            if lang == CUDA_T or lang == OCL_T or lang == METAL_T:
                if 'math.h' in _headers_for_code:
                    _headers_for_code.remove('math.h')
            _saved_headers = self.headers_files
            self.headers_files = _headers_for_code
            self.code += self.IncludeHeaders()
            self.headers_files = _saved_headers

        # Inserting portable collective-memory metalanguage macros
        # (idpy_shared / idpy_sync / idpy_sync_global). Emitted after any
        # 'using namespace metal;' so Metal's mem_flags/threadgroup_barrier
        # resolve, and before functions/kernel body so both can use them.
        self.code += self.CollectiveMacros(lang=lang)

        # Inserting macros
        if self.declare_macros == 'header':
            self.code += self.DeclareMacros(lang=lang)
        # Inserting types
        if self.declare_types == 'typedef':
            self.code += self.DeclareTypes()
        # Inserting Functions
        self.InitFunctions()
        for function in self.functions:
            self.code += function.Code(lang = lang)
            self.code += "\n"
        # Kernel Qualifier and Kernel name
        if lang != CTYPES_T:
            self.code += self.kernels_qualifiers[lang] + " " + self.name
        else:
            self.code += self.return_type + " " + self.name

        # Dynamic (runtime-sized) shared memory: OpenCL and Metal carry it as a
        # kernel parameter; CUDA declares 'extern __shared__' in the body and
        # CTYPES has no shared memory (guarded below).
        _dyn_shared_params = self.DynSharedParams(lang, AddrQ)

        # Kernel Parameters
        if lang == METAL_T:
            _metal_builtins = {
                'g_tid': (self.gthread_id_code
                          if self.code_flags[self.gthread_id_code] else None),
                'l_tid': (self.lthread_id_code
                          if self.code_flags[self.lthread_id_code] else None),
                'l_tid_c': (self.lthread_id_coords_code
                            if self.code_flags[self.lthread_id_coords_code] else None),
                'bid_c': (self.block_coords_code
                          if self.code_flags[self.block_coords_code] else None),
            }
            self.code += WriteCodeParams(
                self.params, AddrQ, metal_kernel=True,
                metal_builtins=_metal_builtins, extra_params=_dyn_shared_params
            )
        else:
            self.code += WriteCodeParams(
                self.params, AddrQ, extra_params=_dyn_shared_params
            )

        # Inserting kernel body
        self.code += """{\n"""
        ## Global thread id
        if self.code_flags[self.gthread_id_code]:
            self.code += self.SetGlobalThreadId()[lang]
        ## Block-local indexing (l_tid / l_tid_c / bid_c) is a block-parallel
        ## concept with no counterpart in the CTYPES serial-loop model; the
        ## same holds for idpy_shared / idpy_sync (block-collective) kernels.
        if lang == CTYPES_T and (
                self.code_flags[self.lthread_id_code] or
                self.code_flags[self.lthread_id_coords_code] or
                self.code_flags[self.block_coords_code] or
                self.shared_dynamic):
            raise NotImplementedError(
                "CTYPES backend has no block-local thread id / shared memory: "
                "its kernel body runs as a serial loop over the global thread "
                "id, so 'l_tid'/'l_tid_c'/'bid_c' and idpy_shared/idpy_sync "
                "kernels are unsupported. Target CUDA, OpenCL or Metal instead."
            )
        ## Local thread id
        if self.code_flags[self.lthread_id_code]:
            self.code += self.SetLocalThreadId()[lang]
        ## Local thread coords
        if self.code_flags[self.lthread_id_coords_code]:
            self.code += self.SetLocalThreadCoords()[lang]
        ## Block coords
        if self.code_flags[self.block_coords_code]:
            self.code += self.SetLocalBlockCoords()[lang]
        ## Dynamic shared memory: CUDA declares the runtime-sized region in the
        ## body as 'extern __shared__ T name[];' (OpenCL/Metal took it as a
        ## kernel parameter above).
        if lang == CUDA_T:
            self.code += self.DynSharedCUDADeclaration()

        ## Kernel Code
        if lang in self.kernels:
            self.code += self.kernels[lang]
        else:
            self.code += self.kernels[IDPY_T]

        ## if CTypes and global thread id: close loop
        if lang == CTYPES_T and self.code_flags[self.gthread_id_code]:
            self.code += """\n}\n"""

        ## Closing function
        if lang != CTYPES_T:
            self.code += """return;\n}\n"""
        elif lang == CTYPES_T and CTYPES_T not in self.kernels:
            self.code += """return 0;\n}\n"""
        elif lang == CTYPES_T and CTYPES_T in self.kernels:
            self.code += """\n}\n"""

        return self.code

    def __call__(self, tenet = None,
                 grid = None, block = None, **kwargs):

        # Resolve runtime-sized shared memory bytes once, from the *original*
        # block (OpenCL rewrites block to None for CPU devices below). 0 when
        # the kernel declares no dynamic shared buffer.
        _dyn_shared_bytes = self.DynSharedBytes(
            block=block,
            dyn_shared_count=kwargs.get('dyn_shared_count'),
            dyn_shared_bytes=kwargs.get('dyn_shared_bytes'),
        )

        if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):

            _kernel_module = cl.Program(tenet.context, self.Code(OCL_T)).build(self.SetMacros(OCL_T))
            _kernel_function = _kernel_module.__getattr__(self.name)
            '''               
            I need to rewrite block and grid to match the opencl style and non-C ordering
            '''
            grid = tuple(map(lambda x, y: x * y, block, grid))
            '''
            Still not completely sure why I need to fall back on PyOpenCL automatic choice
            of workgroup size when using CPUs, at least on MacOS
            '''
            block = block if tenet.kind == OpenCL.GPU_T else None
            ##block = None

            class Idea:
                def __init__(self, k_dict = None):
                    self.k_dict, self.lang = k_dict, OCL_T
                    self.st = SimpleTiming()

                def Deploy(self, args_list = None, idpy_stream = None):
                    _args_data = []
                    for arg in args_list:
                        if isinstance(arg, IdpyArrayOCL):
                            _args_data.append(arg.data)
                        else:
                            _args_data.append(arg)

                    # Runtime-sized __local buffer is a trailing kernel arg
                    if self.k_dict.get('dyn_shared_bytes'):
                        _args_data.append(
                            cl.LocalMemory(self.k_dict['dyn_shared_bytes'])
                        )

                    '''
                    print(self.k_dict['_kernel_function'].get_info(cl.kernel_info.FUNCTION_NAME))
                    print(self.k_dict['_kernel_function'].get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, self.k_dict['tenet'].device))
                    '''
                    self.k_dict['_kernel_function'].set_args(*_args_data)
                    return cl.enqueue_nd_range_kernel(self.k_dict['tenet'],
                                                      self.k_dict['_kernel_function'],
                                                      global_work_size = self.k_dict['grid'],
                                                      local_work_size = self.k_dict['block'],
                                                      wait_for = idpy_stream)

                def DeployProfiling(self, args_list = None, idpy_stream = None):
                    _args_data = []
                    for arg in args_list:
                        if isinstance(arg, IdpyArrayOCL):
                            _args_data.append(arg.data)
                        else:
                            _args_data.append(arg)

                    # Runtime-sized __local buffer is a trailing kernel arg
                    if self.k_dict.get('dyn_shared_bytes'):
                        _args_data.append(
                            cl.LocalMemory(self.k_dict['dyn_shared_bytes'])
                        )

                    # Apple OpenCL event timestamps are unreliable; use host
                    # wall clock (enqueue + wait), matching Metal DeployProfiling.
                    if sys.platform == "darwin":
                        self.st.Start()
                        self.k_dict['_kernel_function'].set_args(*_args_data)
                        _swap_event = cl.enqueue_nd_range_kernel(
                            self.k_dict['tenet'],
                            self.k_dict['_kernel_function'],
                            global_work_size=self.k_dict['grid'],
                            local_work_size=self.k_dict['block'],
                            wait_for=idpy_stream)
                        _swap_event.wait()
                        self.st.End()
                        _time_sec = self.st.GetElapsedTime()['time_s']
                        return _swap_event, _time_sec

                    self.k_dict['_kernel_function'].set_args(*_args_data)
                    _swap_event = cl.enqueue_nd_range_kernel(self.k_dict['tenet'],
                                                             self.k_dict['_kernel_function'],
                                                             global_work_size = self.k_dict['grid'],
                                                             local_work_size = self.k_dict['block'],
                                                             wait_for = idpy_stream)
                    _swap_event.wait()
                    _time_sec = (_swap_event.profile.end - _swap_event.profile.start) * 1e-9
                    return _swap_event, _time_sec
                

            return Idea({'tenet': tenet, 'grid': grid, 'block': block,
                         '_kernel_function': _kernel_function, '_kernel_name': self.name,
                         'dyn_shared_bytes': _dyn_shared_bytes})

        if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
            _kernel_module = cu_SourceModule(self.Code(CUDA_T), options = self.SetMacros(CUDA_T),
                                             nvcc = idpy_nvcc_path)
            _kernel_function = _kernel_module.get_function(self.name)

            class Idea:
                def __init__(self, k_dict = None):
                    self.k_dict, self.lang = k_dict, CUDA_T
                    
                def Deploy(self, args_list = None, idpy_stream = None):
                    return self.k_dict['_kernel_function'](*args_list,
                                                           grid = self.k_dict['grid'],
                                                           block = self.k_dict['block'],
                                                           shared = self.k_dict['dyn_shared_bytes'],
                                                           stream = idpy_stream)

                def DeployProfiling(self, args_list = None, idpy_stream = None):
                    _start, _end = cu_driver.Event(), cu_driver.Event()
                    '''
                    Unprofiled 'warm-up' call: can it be done better ?
                    '''
                    self.k_dict['_kernel_function'](*args_list,
                                                    grid = self.k_dict['grid'],
                                                    block = self.k_dict['block'],
                                                    shared = self.k_dict['dyn_shared_bytes'],
                                                    stream = idpy_stream)

                    _start.record(stream = idpy_stream)
                    self.k_dict['_kernel_function'](*args_list,
                                                    grid = self.k_dict['grid'],
                                                    block = self.k_dict['block'],
                                                    shared = self.k_dict['dyn_shared_bytes'],
                                                    stream = idpy_stream)
                    _end.record(stream = idpy_stream)
                    _end.synchronize()
                    _time_sec = _start.time_till(_end) * 1e-3
                    return None, _time_sec


            return Idea({'_kernel_function': _kernel_function, '_kernel_name': self.name,
                         'tenet': tenet, 'grid': grid, 'block': block,
                         'dyn_shared_bytes': _dyn_shared_bytes})

        if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):

            grid = tuple(map(lambda x, y: x * y, block, grid))
            n_threads = reduce(lambda x, y: x * y, grid)

            if self.code_flags[self.gthread_id_code]:
                self.constants[CTYPES_N_THREAD] = n_threads

            _kernel_module = \
                tenet.GetKernelModule(
                    params=self.params, 
                    code=self.Code(CTYPES_T), 
                    options=self.SetMacros(CTYPES_T)
                    )

            _kernel_function = _kernel_module.GetKernelFunction(self.name, self.custom_types)

            class Idea:
                def __init__(self, k_dict = None):
                    self.k_dict, self.lang = k_dict, CTYPES_T
                    self.st = SimpleTiming()
                    
                def Deploy(self, args_list = None, idpy_stream = None):
                    self.k_dict['_kernel_function'](*args_list)
                    return None

                def DeployProfiling(self, args_list = None, idpy_stream = None):                    
                    self.st.Start()
                    self.k_dict['_kernel_function'](*args_list)
                    self.st.End()
                    _time_sec = self.st.GetElapsedTime()['time_s']
                    return None, _time_sec
                
                
            return Idea({'_kernel_function': _kernel_function, '_kernel_name': self.name,
                         'tenet': tenet, 'grid': grid, 'block': block})

        if idpy_langs_sys[METAL_T] and isinstance(tenet, MTTenet):
            _source = self.Code(METAL_T)
            _library = pymetallic.Library(
                tenet.device, _source, fast_math=bool(self.optimizer_flag),
            )
            _function = _library.make_function(self.name)
            _pipeline = tenet.device.compute_pipeline_state(_function) \
                if hasattr(tenet.device, 'compute_pipeline_state') else \
                pymetallic.ComputePipelineState(tenet.device, _function)

            # CUDA (grid, block) -> Metal global threads = grid * block per dim
            _global = tuple(map(lambda x, y: x * y, block, grid))
            if len(_global) < 3:
                _global = _global + (1,) * (3 - len(_global))
            _block = block if len(block) == 3 else block + (1,) * (3 - len(block))

            class Idea:
                '''
                Metal Idea: command buffers are single-use after commit.
                Deploy encodes+commits asynchronously (like OCL enqueue / CUDA launch);
                sync via returned CommandBuffer.wait_until_completed, DeployProfiling,
                Tenet.Finish(), or IdpyLoop end-of-sequence wait.
                Same-queue CBs preserve GPU order without host wait between Deploys.
                IdpyLoop may Encode many consecutive kernels into one CB (flush before
                host IdpyMethods); standalone Deploy remains one-kernel-per-CB.
                '''
                def __init__(self, k_dict = None):
                    self.k_dict, self.lang = k_dict, METAL_T
                    self.st = SimpleTiming()

                def _bind_args(self, args_list):
                    _args_data = []
                    for arg in args_list:
                        if isinstance(arg, IdpyArrayMETAL):
                            _args_data.append(arg.data)
                        elif isinstance(arg, np.ndarray):
                            _args_data.append(
                                pymetallic.Buffer.from_numpy(
                                    self.k_dict['tenet'].device, arg
                                )
                            )
                        else:
                            # numpy scalar / Python scalar -> 1-element buffer
                            _arr = np.array([arg])
                            _args_data.append(
                                pymetallic.Buffer.from_numpy(
                                    self.k_dict['tenet'].device, _arr
                                )
                            )
                    return _args_data

                def Encode(self, encoder, args_list = None):
                    '''Encode one dispatch onto an open compute encoder (no commit).'''
                    _args_data = self._bind_args(args_list)
                    encoder.set_compute_pipeline_state(self.k_dict['_pipeline'])
                    for i, buf in enumerate(_args_data):
                        encoder.set_buffer(buf, 0, i)
                    # Runtime-sized threadgroup memory for the [[threadgroup(0)]] param
                    if self.k_dict.get('dyn_shared_bytes'):
                        encoder.set_threadgroup_memory_length(
                            self.k_dict['dyn_shared_bytes'], 0
                        )
                    encoder.dispatch_threads(
                        self.k_dict['global'], self.k_dict['block']
                    )

                def Deploy(self, args_list = None, idpy_stream = None):
                    # idpy_stream unused for ordering on the single Tenet queue
                    # (GPU serializes same-queue command buffers).
                    tenet = self.k_dict['tenet']
                    command_buffer = tenet.queue.make_command_buffer()
                    encoder = command_buffer.make_compute_command_encoder()
                    self.Encode(encoder, args_list)
                    encoder.end_encoding()
                    command_buffer.commit()
                    tenet.last_command_buffer = command_buffer
                    return command_buffer

                def DeployProfiling(self, args_list = None, idpy_stream = None):
                    self.st.Start()
                    command_buffer = self.Deploy(args_list, idpy_stream)
                    command_buffer.wait_until_completed()
                    tenet = self.k_dict['tenet']
                    if tenet.last_command_buffer is command_buffer:
                        tenet.last_command_buffer = None
                    self.st.End()
                    _time_sec = self.st.GetElapsedTime()['time_s']
                    return command_buffer, _time_sec

            return Idea({'_pipeline': _pipeline, '_kernel_name': self.name,
                         'tenet': tenet, 'grid': grid, 'block': _block,
                         'global': _global, 'dyn_shared_bytes': _dyn_shared_bytes})

    def ResetCode(self):
        self.code = ""
                        
class IdpyFunction:
    '''
    class MetaFunction:
    parent class for implementing the meta-code once and manage
    the different features of specific languages on demand
    It does not need to be aware of the possible types
    '''
    def __init__(self, custom_types = None, f_type = None):
        self.code, self.name = "", self.__class__.__name__
        self.functions, self.params, self.macros = {}, {}, {}
        self.functions_qualifiers = FuncQualif()

        '''
        Need to use custom_types for exit condition
        '''
        self.f_type, self.custom_types = f_type, custom_types
        self.declarations = {}

        self.AddrQ = AddrQualif()

    def Code(self, lang = None):
        if lang is None:
            raise Exception("Parameter lang must be in list: ", list(idpy_langs_dict_sym.values()))
        if lang not in idpy_langs_list:
            raise Exception("Parameter lang can only be: ", idpy_langs_list)
        
        AddrQ = self.AddrQ[lang]
        self.ResetCode()
        '''
        Some compilers need the function declaration first
        '''
        self.code += (self.functions_qualifiers[lang] + " " +
                      self.f_type + " " + self.name)
        self.code += WriteCodeParams(self.params, AddrQ)
        self.code += """;\n"""
        '''
        The the function definition
        '''
        self.code += (self.functions_qualifiers[lang] + " " +
                      self.f_type + " " + self.name)
        # Setting parameters
        self.code += WriteCodeParams(self.params, AddrQ)
        # Function body
        self.code += """{\n"""
        
        if lang in self.functions:
            self.code += self.functions[lang]
        else:
            self.code += self.functions[IDPY_T]
            
        self.code += """}\n"""
        return self.code

    def ResetCode(self):
        self.code = ""

def WriteCodeParams(params = None, AddrQ = None,
                    metal_kernel = False, g_tid_name = None,
                    metal_builtins = None, extra_params = None):
    # Back-compat: a bare g_tid_name folds into the builtins dict.
    if metal_builtins is None:
        metal_builtins = {}
    if extra_params is None:
        extra_params = []
    if g_tid_name is not None and 'g_tid' not in metal_builtins:
        metal_builtins = dict(metal_builtins)
        metal_builtins['g_tid'] = g_tid_name
    _code = ""
    _code += "("
    buffer_i = 0
    for param in params:
        restrict_flag = False
        qualifiers_prefix = ""
        for qualifier in params[param]:
            if qualifier == 'restrict':
                restrict_flag = True
            else:
                qualifiers_prefix += AddrQ[qualifier] + " "

        is_pointer = '*' in param
        if metal_kernel and not is_pointer:
            # Scalar kernel args: constant T & name [[buffer(i)]]
            _parts = param.rsplit(None, 1)
            _code += ("constant " + _parts[0] + " & " + _parts[1] +
                      " [[buffer(" + str(buffer_i) + ")]], ")
            buffer_i += 1
        elif metal_kernel:
            if restrict_flag:
                _splitted = param.split('*')
                _code += (qualifiers_prefix + _splitted[0] + '* ' +
                          (AddrQ['restrict'] + ' ' if AddrQ['restrict'] else '') +
                          _splitted[1].strip() +
                          " [[buffer(" + str(buffer_i) + ")]], ")
            else:
                _code += (qualifiers_prefix + param +
                          " [[buffer(" + str(buffer_i) + ")]], ")
            buffer_i += 1
        else:
            _code += qualifiers_prefix
            if restrict_flag:
                _splitted = param.split('*')
                _code += _splitted[0] + ' * ' + AddrQ['restrict'] + ' ' + _splitted[1] + ','
            else:
                _code += param + ","

    # Framework-injected extra params (already fully formatted, e.g. dynamic
    # shared memory pointers). For Metal these carry their own [[threadgroup(i)]]
    # attribute and must precede the position-attributed built-ins below.
    for _extra in extra_params:
        _code += _extra + ", "

    if metal_kernel:
        # Attributed built-in kernel params. A given Metal attribute may appear
        # at most once, so when both the linear local id (l_tid) and the local
        # coords (l_tid_c) are requested, only the uint3 coords vector carries
        # [[thread_position_in_threadgroup]] and l_tid is derived from its .x
        # in the kernel body (see SetLocalThreadId).
        _g = metal_builtins.get('g_tid')
        _l = metal_builtins.get('l_tid')
        _lc = metal_builtins.get('l_tid_c')
        _bc = metal_builtins.get('bid_c')

        if _g is not None:
            _code += "uint " + _g + " [[thread_position_in_grid]], "
        if _lc is not None:
            _code += ("uint3 " + _lc +
                      "_vec [[thread_position_in_threadgroup]], ")
        elif _l is not None:
            _code += "uint " + _l + " [[thread_position_in_threadgroup]], "
        if _bc is not None:
            _code += ("uint3 " + _bc +
                      "_vec [[threadgroup_position_in_grid]], ")

    # Eliminating last comma/space
    _code = _code.rstrip()
    if _code.endswith(","):
        _code = _code[:-1]
    _code += """)"""
    return _code

def _metal_idea_is_gpu_kernel(Idea):
    '''True for Metal GPU Ideas (have a pipeline); False for IdpyMethods.'''
    k_dict = getattr(Idea, 'k_dict', None)
    return isinstance(k_dict, dict) and ('_pipeline' in k_dict)


def _metal_flush_encode_batch(tenet, encoder, command_buffer):
    '''End encoding and commit; return (None, None) so callers clear batch state.'''
    if encoder is not None:
        encoder.end_encoding()
        command_buffer.commit()
        tenet.last_command_buffer = command_buffer
    return None, None


def _metal_wait_last_cb(meta_stream_slots):
    '''Wait the last non-None command buffer in an IdpyLoop meta_streams slot list.'''
    for slot in reversed(meta_stream_slots):
        if slot is not None and slot[0] is not None:
            slot[0].wait_until_completed()
            return


## Methods and Loops
class IdpyMethod:
    def __init__(self, tenet = None):
        self.tenet, self.lang = tenet, None
        if idpy_langs_sys[OCL_T] and isinstance(tenet, CLTenet):
            self.lang = OCL_T
        if idpy_langs_sys[CUDA_T] and isinstance(tenet, CUTenet):
            self.lang = CUDA_T
        if idpy_langs_sys[CTYPES_T] and isinstance(tenet, CTTenet):
            self.lang = CTYPES_T
        if idpy_langs_sys[METAL_T] and isinstance(tenet, MTTenet):
            self.lang = METAL_T

        '''
        Mocking the kernels variables
        '''
        self.k_dict = {'_kernel_name': self.__class__.__name__}
            
        '''
        the child class need to define the Deploy method
        def Deploy(self, args, idpy_stream = None)
        '''
    def PassIdpyStream(self, idpy_stream):
        if self.lang == OCL_T:
            if idpy_stream is None:
                return None
            else:
                return idpy_stream[0]

        if self.lang == CUDA_T or self.lang == CTYPES_T or self.lang == METAL_T:
            return None


class IdpyLoop:
    '''
    class IdpyLoop:
    the idea is to pass a list of arguments lists
    and a list of lists of tuples of IdpyKernels/IdpyMethods and arguments indices
    automatically creating streams and events in order to allow
    the concurrent execution of these lists
    '''
    def __init__(
            self, args_dicts = None, sequences = None, 
            idloop_k_type=np.int32, idloop_k_name='idloop_k'):
        '''
        Insert 'idpy_loop_counter' in 'args_dict'
        '''
        self.idloop_k_type = idloop_k_type
        self.idloop_k_offset = self.idloop_k_type(0)
        self.idloop_k_name = idloop_k_name
        self.args_dicts = args_dicts
        self.sequences = sequences
        self.meta_streams, self.langs = [], []
        self.first_run = True

    def SetMetaStreams(self, seq):
        if seq[0][0].lang == CUDA_T:
            if idpy_langs_sys[CUDA_T]:
                return cu_driver.Stream()
            else:
                raise Exception("CUDA not present on the system")

        if seq[0][0].lang == OCL_T:
            if idpy_langs_sys[OCL_T]:
                return [None for _ in range(len(seq))]
            else:
                raise Exception("OpenCL not present on the system")

        if seq[0][0].lang == CTYPES_T:
            if idpy_langs_sys[CTYPES_T]:
                return [None for _ in range(len(seq))]
            else:
                raise Exception("CTypes not present on the system")

        if seq[0][0].lang == METAL_T:
            if idpy_langs_sys[METAL_T]:
                return [None for _ in range(len(seq))]
            else:
                raise Exception("Metal not present on the system")

    def SetLang(self, seq):
        return seq[0][0].lang

    def SetArgs(self, seq_index, args_keys):
        if len(args_keys):
            return [self.args_dicts[seq_index][_] for _ in args_keys]
        else:
            raise Exception("List of arguments keys cannot be empty!")

    def PutArgs(self, seq_index, args_indices, args_list_swap):
        if len(args_indices):
            for i in range(len(args_indices)):
                self.args_dicts[seq_index][args_indices[i]] = args_list_swap[i]
        else:
            raise Exception("List of arguments keys cannot be empty!")    

    def Run(self, loop_range = None, profiling = False, idloop_k_offset=0):
        '''
        Begin by setting up meta_streams and langs
        Neet to do this only once to avoid re-allocating (CUDA) streams
        '''
        if self.first_run is True:
            for seq in self.sequences:
                self.meta_streams.append(self.SetMetaStreams(seq))
                self.langs.append(self.SetLang(seq))
            self.first_run = False

        idloop_k_offset = self.idloop_k_type(idloop_k_offset)

        for step_k, step in enumerate(loop_range):

            for seq_i in range(len(self.sequences)):
                if self.idloop_k_name in self.args_dicts[seq_i]:
                    self.args_dicts[seq_i][self.idloop_k_name] = \
                        idloop_k_offset + self.idloop_k_type(step_k) + \
                        self.idloop_k_offset

                seq_len = len(self.sequences[seq_i])                
                '''
                OpenCL
                '''                
                if self.langs[seq_i] == OCL_T:                
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        '''
                        Deploying
                        '''
                        _prev_evt = self.meta_streams[seq_i][(item_i - 1 + seq_len) % seq_len]
                        # print("ev pre:", Idea.k_dict['_kernel_name'], _prev_evt)

                        self.meta_streams[seq_i][item_i] = \
                            [Idea.Deploy(_args,
                                         idpy_stream = (None if _prev_evt is None or _prev_evt == [None]
                                                        else _prev_evt))]
                        # print("ev post:", self.meta_streams[seq_i][item_i])
                        # print()
                        self.PutArgs(seq_i, _indices, _args)

                '''
                CUDA
                '''
                if self.langs[seq_i] == CUDA_T:                
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        _stream = self.meta_streams[seq_i]                        
                        '''
                        Deploying
                        '''
                        Idea.Deploy(_args, idpy_stream = _stream)
                        self.PutArgs(seq_i, _indices, _args)


                '''
                CTYPES
                '''
                if self.langs[seq_i] == CTYPES_T:                
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        _stream = self.meta_streams[seq_i]                        
                        '''
                        Deploying
                        '''
                        Idea.Deploy(_args, idpy_stream = _stream)
                        self.PutArgs(seq_i, _indices, _args)

                '''
                Metal: batch consecutive GPU kernels into one CB; flush before IdpyMethod
                '''
                if self.langs[seq_i] == METAL_T:
                    _encoder, _cb, _tenet = None, None, None
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        if _metal_idea_is_gpu_kernel(Idea):
                            _tenet = Idea.k_dict['tenet']
                            if _encoder is None:
                                _cb = _tenet.queue.make_command_buffer()
                                _encoder = _cb.make_compute_command_encoder()
                            Idea.Encode(_encoder, _args)
                            self.meta_streams[seq_i][item_i] = [_cb]
                        else:
                            if _encoder is not None:
                                _encoder, _cb = _metal_flush_encode_batch(
                                    _tenet, _encoder, _cb)
                            self.meta_streams[seq_i][item_i] = \
                                [Idea.Deploy(_args, idpy_stream = None)]
                        self.PutArgs(seq_i, _indices, _args)
                    if _encoder is not None:
                        _metal_flush_encode_batch(_tenet, _encoder, _cb)

        self.idloop_k_offset += \
            self.idloop_k_type(loop_range[-1] - loop_range[0] + 1)
        ## print("self.idloop_k_offset", self.idloop_k_offset)                      

        '''
        Synchronizing with device: can this be done better? Are we waisting time?
        '''
        for seq_i in range(len(self.sequences)):
            seq_len = len(self.sequences[seq_i])                
            '''
            OpenCL
            '''                
            if self.langs[seq_i] == OCL_T:                
                '''
                Waiting
                '''
                if self.meta_streams[seq_i][-1][0] is not None:
                    self.meta_streams[seq_i][-1][0].wait()

            '''
            CUDA
            '''
            if self.langs[seq_i] == CUDA_T:
                _end = cu_driver.Event()
                '''
                Waiting
                '''
                _end.record(stream = self.meta_streams[seq_i])
                _end.synchronize()

            '''
            Metal: wait last non-None CB (sequences may end with IdpyMethod)
            '''
            if self.langs[seq_i] == METAL_T:
                _metal_wait_last_cb(self.meta_streams[seq_i])

'''
most likely to be deleted before merging to master
'''
def IdpyProfile(idea_object = None, args_list = [], idpy_stream = None):
    '''
    IdpyProfile: method that executes the Deploy method of an Idea object
    returning a tuple:
    first: an idpy_stream
    second: a dictionary containing 
    '''
    if idea_object.__class__.__name__ != 'Idea':
        raise Exception("First argument must be an instance of 'Idea' class")
    if len(args_list) == 0:
        raise Exception("args_list must not be an empty list")
    
    _lang = idea_object.lang
    _kernel_name = idea_object.k_dict['_kernel_name']
    
    if _lang == OCL_T:
        # Apple OpenCL event timestamps are unreliable; use host wall clock.
        if sys.platform == "darwin":
            _st = SimpleTiming()
            _st.Start()
            _idpy_stream_out = idea_object.Deploy(args_list, idpy_stream)
            _idpy_stream_out.wait()
            _st.End()
            _time_sec = _st.GetElapsedTime()['time_s']
            return _idpy_stream_out, _time_sec
        _idpy_stream_out = idea_object.Deploy(args_list, idpy_stream)
        _idpy_stream_out.wait()
        _time_sec = (_idpy_stream_out.profile.end - _idpy_stream_out.profile.start) * 1e-9
        return _idpy_stream_out, _time_sec
    
    if _lang == CUDA_T:
        _start = cu_driver.Event()
        _end = cu_driver.Event()
        _start.record(stream = idpy_stream)
        idea_object.Deploy(args_list, idpy_stream)
        _end.record(stream = idpy_stream)
        _end.synchronize()
        _time_sec = _start.time_till(_end)*1e-3
        return idpy_stream, _time_sec

    if _lang == CTYPES_T:
        _st = SimpleTiming()
        _st.Start()
        idea_object.Deploy(args_list, idpy_stream)
        _st.End()
        _time_sec = _st.GetElapsedTime()['time_s']
        return idpy_stream, _time_sec

    if _lang == METAL_T:
        _st = SimpleTiming()
        _st.Start()
        _idpy_stream_out = idea_object.Deploy(args_list, idpy_stream)
        _idpy_stream_out.wait_until_completed()
        _st.End()
        _time_sec = _st.GetElapsedTime()['time_s']
        return _idpy_stream_out, _time_sec
                        

class IdpyLoopProfile:
    '''
    class IdpyLoop:
    the idea is to pass a list of arguments lists
    and a list of lists of tuples of IdpyKernels/IdpyMethods and arguments indices
    automatically creating streams and events in order to allow
    the concurrent execution of these lists
    '''
    def __init__(self, args_dicts = None, sequences = None):
        self.args_dicts, self.sequences = args_dicts, sequences
        self.meta_streams, self.langs = [], []
        self.first_run = True

    def SetMetaStreams(self, seq):
        if seq[0][0].lang == CUDA_T:
            if idpy_langs_sys[CUDA_T]:
                return cu_driver.Stream()
            else:
                raise Exception("CUDA not present on the system")
        if seq[0][0].lang == OCL_T:
            if idpy_langs_sys[OCL_T]:
                return [None for _ in range(len(seq))]
            else:
                raise Exception("OpenCL not present on the system")
        if seq[0][0].lang == CTYPES_T:
            if idpy_langs_sys[CTYPES_T]:
                return [None for _ in range(len(seq))]
            else:
                raise Exception("OpenCL not present on the system")

        if seq[0][0].lang == METAL_T:
            if idpy_langs_sys[METAL_T]:
                return [None for _ in range(len(seq))]
            else:
                raise Exception("Metal not present on the system")

    def SetLang(self, seq):
        return seq[0][0].lang

    def SetArgs(self, seq_index, args_keys):
        if len(args_keys):
            return [self.args_dicts[seq_index][_] for _ in args_keys]
        else:
            raise Exception("List of arguments keys cannot be empty!")

    def PutArgs(self, seq_index, args_indices, args_list_swap):
        if len(args_indices):
            for i in range(len(args_indices)):
                self.args_dicts[seq_index][args_indices[i]] = args_list_swap[i]
        else:
            raise Exception("List of arguments keys cannot be empty!")    

    def Run(self, loop_range = None, profiling = False):
        '''
        Begin by setting up meta_streams and langs
        Neet to do this only once to avoid re-allocating (CUDA) streams
        '''
        if self.first_run is True:
            for seq in self.sequences:
                self.meta_streams.append(self.SetMetaStreams(seq))
                self.langs.append(self.SetLang(seq))
            self.first_run = False

        '''
        Set up dictionary for keeping timings
        '''
        _timing_dict = \
            defaultdict( # seq_i
                lambda: defaultdict(dict) # _kernel_name
            )
        
        for seq_i in range(len(self.sequences)):
            seq_len = len(self.sequences[seq_i])
            for item_i in range(seq_len):
                _item = self.sequences[seq_i][item_i]
                Idea = _item[0]
                if hasattr(Idea, 'k_dict'):
                    _timing_dict[seq_i][Idea.k_dict['_kernel_name']] = []

        '''
        Loop
        '''
        for step in loop_range:
            for seq_i in range(len(self.sequences)):
                seq_len = len(self.sequences[seq_i])
                '''
                OpenCL
                '''
                if self.langs[seq_i] == OCL_T:
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        '''
                        Deploying
                        '''
                        _prev_evt = self.meta_streams[seq_i][(item_i - 1 + seq_len) % seq_len]
                        _stream_swap, _time_swap = \
                            Idea.DeployProfiling(_args, idpy_stream = (None if _prev_evt is None
                                                                       else _prev_evt))
                        self.meta_streams[seq_i][item_i] = [_stream_swap]
                        self.PutArgs(seq_i, _indices, _args)
                        if hasattr(Idea, 'k_dict'):
                            _timing_dict[seq_i][Idea.k_dict['_kernel_name']] += [_time_swap]

                '''
                CUDA
                '''            
                if self.langs[seq_i] == CUDA_T:
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        _stream = self.meta_streams[seq_i]                        
                        '''
                        Deploying
                        '''
                        _stream_swap, _time_swap = \
                            Idea.DeployProfiling(_args, idpy_stream = _stream)
                        self.PutArgs(seq_i, _indices, _args)
                        _timing_dict[seq_i][Idea.k_dict['_kernel_name']] += [_time_swap]

                '''
                CTYPES
                '''            
                if self.langs[seq_i] == CTYPES_T:
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        _stream = self.meta_streams[seq_i]                        
                        '''
                        Deploying
                        '''
                        _stream_swap, _time_swap = \
                            Idea.DeployProfiling(_args, idpy_stream = _stream)
                        self.PutArgs(seq_i, _indices, _args)
                        _timing_dict[seq_i][Idea.k_dict['_kernel_name']] += [_time_swap]

                '''
                Metal (DeployProfiling waits; store CB handles like OpenCL)
                '''
                if self.langs[seq_i] == METAL_T:
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        _stream_swap, _time_swap = \
                            Idea.DeployProfiling(_args, idpy_stream = None)
                        self.meta_streams[seq_i][item_i] = [_stream_swap]
                        self.PutArgs(seq_i, _indices, _args)
                        if hasattr(Idea, 'k_dict'):
                            _timing_dict[seq_i][Idea.k_dict['_kernel_name']] += [_time_swap]

        '''
        Collecting profiling values
        '''
        for seq_i in range(len(self.sequences)):
            seq_len = len(self.sequences[seq_i])
            for item_i in range(seq_len):
                _item = self.sequences[seq_i][item_i]
                Idea = _item[0]
                if hasattr(Idea, 'k_dict'):
                    _timing_dict[seq_i][Idea.k_dict['_kernel_name']] = \
                        np.array(_timing_dict[seq_i][Idea.k_dict['_kernel_name']])
                '''
                Need to modify the Tenet class for passing the device name
                '''

                if 'device_name' not in _timing_dict[seq_i]:
                    _timing_dict[seq_i]['device_name'] = Idea.k_dict['tenet'].device_name

        return _timing_dict

                      
'''
changes: I should be able to pass the dictiionary with the arguments rather
than a list so that I can name the argument by name rather than by number
'''
class IdpyLoopList:
    '''
    class IdpyLoopNew:
    the idea is to pass a list of arguments lists
    and a list of lists of tuples of IdpyKernels/IdpyMethods and arguments indices
    automatically creating streams and events in order to allow
    the concurrent execution of these lists
    '''
    def __init__(self, args_lists = None, sequences = None):
        self.args_lists, self.sequences = args_lists, sequences
        self.meta_streams, self.langs = [], []
        self.first_run = True        

    def SetMetaStreams(self, seq):
        if seq[0][0].lang == CUDA_T:
            if idpy_langs_sys[CUDA_T]:
                return cu_driver.Stream()
            else:
                raise Exception("CUDA not present on the system")
        if seq[0][0].lang == OCL_T:
            if idpy_langs_sys[OCL_T]:
                return [None for _ in range(len(seq))]
            else:
                raise Exception("OpenCL not present on the system")
        if seq[0][0].lang == METAL_T:
            if idpy_langs_sys[METAL_T]:
                return [None for _ in range(len(seq))]
            else:
                raise Exception("Metal not present on the system")

    def SetLang(self, seq):
        return seq[0][0].lang

    def SetArgs(self, seq_index, args_indices):
        if len(args_indices):
            return [self.args_lists[seq_index][_] for _ in args_indices]
        else:
            return self.args_lists[seq_index]

    def PutArgs(self, seq_index, args_indices, args_list_swap):
        if len(args_indices):
            for i in range(len(args_indices)):
                self.args_lists[seq_index][args_indices[i]] = args_list_swap[i]

    def Run(self, loop_range = None):
        '''
        Begin by setting up meta_streams and langs
        Neet to do this only once to avoid re-allocating (CUDA) streams
        '''
        if self.first_run is True:
            for seq in self.sequences:
                self.meta_streams.append(self.SetMetaStreams(seq))
                self.langs.append(self.SetLang(seq))
            self.first_run = False

        for step in loop_range:
            for seq_i in range(len(self.sequences)):
                seq_len = len(self.sequences[seq_i])

                if self.langs[seq_i] == OCL_T:
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        '''
                        Deploying
                        '''
                        _prev_evt = self.meta_streams[seq_i][(item_i - 1 + seq_len) % seq_len]
                        self.meta_streams[seq_i][item_i] = \
                            [Idea.Deploy(_args,
                                         idpy_stream = (None if _prev_evt is None
                                                        else _prev_evt))]
                        self.PutArgs(seq_i, _indices, _args)

                if self.langs[seq_i] == CUDA_T:
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        _stream = self.meta_streams[seq_i]
                        
                        '''
                        Deploying
                        '''
                        Idea.Deploy(_args, idpy_stream = _stream)
                        self.PutArgs(seq_i, _indices, _args)

                if self.langs[seq_i] == METAL_T:
                    _encoder, _cb, _tenet = None, None, None
                    for item_i in range(seq_len):
                        _item = self.sequences[seq_i][item_i]
                        Idea, _indices = _item[0], _item[1]
                        _args = self.SetArgs(seq_i, _indices)
                        if _metal_idea_is_gpu_kernel(Idea):
                            _tenet = Idea.k_dict['tenet']
                            if _encoder is None:
                                _cb = _tenet.queue.make_command_buffer()
                                _encoder = _cb.make_compute_command_encoder()
                            Idea.Encode(_encoder, _args)
                            self.meta_streams[seq_i][item_i] = [_cb]
                        else:
                            if _encoder is not None:
                                _encoder, _cb = _metal_flush_encode_batch(
                                    _tenet, _encoder, _cb)
                            self.meta_streams[seq_i][item_i] = \
                                [Idea.Deploy(_args, idpy_stream = None)]
                        self.PutArgs(seq_i, _indices, _args)
                    if _encoder is not None:
                        _metal_flush_encode_batch(_tenet, _encoder, _cb)

