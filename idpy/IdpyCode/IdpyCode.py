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

from idpy.IdpyCode.IdpyConsts import AddrQualif, KernQualif, FuncQualif

from idpy.IdpyCode import idpy_nvcc_path, idpy_langs_list
from idpy.IdpyCode import idpy_langs_dict_sym, idpy_langs_sys
from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T
from idpy.IdpyCode import idpy_opencl_macro_spacing

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
                 optimizer_flag = None):
        self.code, self.name = "", self.__class__.__name__
        self.kernels, self.params, self.f_classes, self.functions = {}, {}, f_classes, []
        self.custom_types, self.constants = custom_types, constants
        '''
        Need to check the type of optimizer_flag
        '''
        self.optimizer_flag = True if optimizer_flag is None else optimizer_flag
        self.declarations = {}

        '''
        The idea is to combine consts and types macros
        in the self.macros list
        '''
        self.macros_consts, self.macros = {}, None
        
        self.gthread_id_code, self.lthread_id_code = gthread_id_code, lthread_id_code
        self.lthread_id_coords_code, self.block_coords_code = lthread_id_coords_code, block_coords_code

        self.kernels_qualifiers = KernQualif()
        self.AddrQ = AddrQualif()

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
        
    def SetMacros(self, lang = None):
        if lang == CUDA_T:
            self.macros = []
            # Constants
            for const in self.constants:
                self.macros.append("-D " + const + "=" + str(self.constants[const]))
            # Types
            for c_type in self.custom_types:
                self.macros.append("-D " + c_type + "=" + str(self.custom_types[c_type]))

            # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
            if self.optimizer_flag is False:
                self.macros.append("--device-debug")

            return self.macros

        if lang == OCL_T:
            self.macros = ''
            # Constants
            for const in self.constants:
                self.macros += (" -D " + const + "=" + str(self.constants[const]))
            # Types
            # https://stackoverflow.com/questions/13531100/escaping-space-in-opencl-compiler-arguments
            for c_type in self.custom_types:
                self.macros += (" -D " + c_type + "=" + '\"' + str(self.custom_types[c_type]).replace(" ", idpy_opencl_macro_spacing) + '\"')

            if self.optimizer_flag is False:
                self.macros += " -cl-opt-disable"
                
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

    def Code(self, lang = None):
        # Argument Qualifiers
        AddrQ = self.AddrQ[lang]
        self.ResetCode()
        # Inserting Functions
        self.InitFunctions()
        for function in self.functions:
            self.code += function.Code(lang = lang)
            self.code += "\n"
        # Kernel Qualifier and Kernel name
        self.code += self.kernels_qualifiers[lang] + " " + self.name
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

        # Closing function
        self.code += """return; }"""
        
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

        if self.lang == CUDA_T:
            return None


class IdpyLoop:
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

