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

from functools import reduce

from idpy.IdpyCode.IdpyConsts import AddrQualif, KernQualif, FuncQualif

from idpy.IdpyCode import idpy_nvcc_path, idpy_langs_list
from idpy.IdpyCode import idpy_langs_dict_sym, idpy_langs_sys
from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T, CTYPES_T
from idpy.IdpyCode import idpy_opencl_macro_spacing
from idpy.IdpyCode import idpy_copyright

from idpy.IdpyCode.IdpyUnroll import _codify_comment
from idpy.Utils.SimpleTiming import SimpleTiming

if idpy_langs_sys[CUDA_T]:
    from idpy.IdpyCode.IdpyMemory import IdpyArrayCUDA
if idpy_langs_sys[OCL_T]:
    from idpy.IdpyCode.IdpyMemory import IdpyArrayOCL

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
        if headers_files is not None and type(headers_files) is not list:
            raise Exception("headers_files param must be a list")
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
        
        self.headers_files = headers_files
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

        '''
        List of variables and constants for metaprogramming
        '''
        self.declared_variables, self.declared_constants = [[]], [[]]

        # Code Flags
        self.code_flags = defaultdict(dict)
        self.InitCodeFlags()

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
                    self.macros.append("-D " + const + "=" + str(self.constants[const]))
                
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
                    self.macros += (" -D " + const + "=" + str(self.constants[const]))
                
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
                    self.macros += (" -D " + const + "=" + str(self.constants[const]))
                
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

    def DeclareMacros(self):
        _swap = ''
        for c_macro in self.constants:
            _swap  += '#define ' + c_macro + ' ' + str(self.constants[c_macro]) + '\n'
        _swap += '\n'
        
        return _swap    

    def IncludeHeaders(self):
        _swap = ''
        for _h_file in self.headers_files:
            _swap += '#include <' + _h_file + '>\n'
        _swap += '\n'
        return _swap

    def Code(self, lang = None):
        # Argument Qualifiers
        AddrQ = self.AddrQ[lang]
        self.ResetCode()
        # Inserting headers
        ## Checking for 'math.h'

        if self.headers_files is not None:
            _swap_headers_files = self.headers_files.copy()

            if lang == CUDA_T or lang == OCL_T:
                if 'math.h' in self.headers_files:
                    self.headers_files.remove('math.h')

            self.code += self.IncludeHeaders()
            self.headers_files = _swap_headers_files.copy()

        # Inserting macros
        if self.declare_macros == 'header':
            self.code += self.DeclareMacros()
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

        # Kernel Paremeters
        self.code += WriteCodeParams(self.params, AddrQ)

        # Inserting kernel body
        self.code += """{\n"""
        ## Global thread id
        if self.code_flags[self.gthread_id_code]:
            self.code += self.SetGlobalThreadId()[lang]
        ## Local thread id
        if self.code_flags[self.lthread_id_code]:
            self.code += self.SetLocalThreadId()[lang]
        ## Local thread coords
        if self.code_flags[self.lthread_id_coords_code]:
            self.code += self.SetLocalThreadCoords()[lang]
        ## Block coords
        if self.code_flags[self.block_coords_code]:
            self.code += self.SetLocalBlockCoords()[lang]

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

                def Deploy(self, args_list = None, idpy_stream = None):
                    _args_data = []
                    for arg in args_list:
                        if isinstance(arg, IdpyArrayOCL):
                            _args_data.append(arg.data)
                        else:
                            _args_data.append(arg)

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
                         '_kernel_function': _kernel_function, '_kernel_name': self.name})

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
                                                           stream = idpy_stream)

                def DeployProfiling(self, args_list = None, idpy_stream = None):
                    _start, _end = cu_driver.Event(), cu_driver.Event()
                    '''
                    Unprofiled 'warm-up' call: can it be done better ?
                    '''
                    self.k_dict['_kernel_function'](*args_list,
                                                    grid = self.k_dict['grid'],
                                                    block = self.k_dict['block'],
                                                    stream = idpy_stream)
                    
                    _start.record(stream = idpy_stream)
                    self.k_dict['_kernel_function'](*args_list,
                                                    grid = self.k_dict['grid'],
                                                    block = self.k_dict['block'],
                                                    stream = idpy_stream)
                    _end.record(stream = idpy_stream)
                    _end.synchronize()
                    _time_sec = _start.time_till(_end) * 1e-3
                    return None, _time_sec
                
                
            return Idea({'_kernel_function': _kernel_function, '_kernel_name': self.name,
                         'tenet': tenet, 'grid': grid, 'block': block})

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

def WriteCodeParams(params = None, AddrQ = None):
    _code = ""
    _code += "("
    for param in params:
        restrict_flag = False
        for qualifier in params[param]:
            if qualifier == 'restrict':
                restrict_flag = True
            else:
                _code += AddrQ[qualifier] + " "
        if restrict_flag:
            _splitted = param.split('*')
            _code += _splitted[0] + ' * ' + AddrQ['restrict'] + ' ' + _splitted[1] + ','
        else:
            _code += param + ","
    # Eliminating last comma
    _code = _code[:-1]
    _code += """)"""
    return _code

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

        if self.lang == CUDA_T or self.lang == CTYPES_T:
            return None


class IdpyLoop:
    '''
    class IdpyLoop:
    the idea is to pass a list of arguments lists
    and a list of lists of tuples of IdpyKernels/IdpyMethods and arguments indices
    automatically creating streams and events in order to allow
    the concurrent execution of these lists
    '''
    def __init__(self, args_dicts = None, sequences = None, idloop_k_type=np.int32):
        '''
        Insert 'idpy_loop_counter' in 'args_dict'
        '''
        self.idloop_k_type = idloop_k_type
        self.args_dicts = [{**args_dict, **{'idloop_k': idloop_k_type(0)}} for args_dict in args_dicts]
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
                self.args_dicts[seq_i]['idloop_k'] = idloop_k_offset + self.idloop_k_type(step_k)

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
                        self.meta_streams[seq_i][item_i] = \
                            [Idea.Deploy(_args,
                                         idpy_stream = (None if _prev_evt is None
                                                        else _prev_evt))]
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

