__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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

# https://stackoverflow.com/questions/50499/how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing/50905#50905

import inspect, os, sys
from sys import platform
'''
find os
'''
idpy_os_found = None
if platform == "linux" or platform == "linux2":
    idpy_os_found = "linux"
elif platform == "darwin":
    idpy_os_found = "darwin"
elif platform == "win32":
    idpy_os_found == "win32"

idpy_opencl_macro_spacing = None
if idpy_os_found == "linux":
    idpy_opencl_macro_spacing = '\t'
if idpy_os_found == "darwin":
    idpy_opencl_macro_spacing = '\ '
if idpy_os_found == "win32":
    idpy_opencl_macro_spacing = '\ '    

_module_abs_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_idea_dot_deploy_path = os.path.dirname(os.path.abspath(_module_abs_path + "../../"))
'''
append to sys path in order to avoid relative imports
'''
sys.path.append(_idea_dot_deploy_path)

_idpy_env_path = _idea_dot_deploy_path + "/" + "idpy-env/"
_cuda_path_found = _idpy_env_path + "/" + "cuda_path_found"

'''
Reading system CUDA path
'''
_file_swap = open(_cuda_path_found, "r")
idpy_nvcc_path = _file_swap.readline().rstrip() + "/bin/nvcc"
_file_swap.close()
'''
Language Types and metaTypes
'''
CUDA_T, OCL_T, IDPY_T = "pycuda", "pyopencl", "idpy"
idpy_langs_dict = {'CUDA_T': CUDA_T, 'OCL_T': OCL_T}

idpy_langs_human_dict = {CUDA_T: "CUDA", OCL_T: "OpenCL"}
idpy_langs_dict_sym = {CUDA_T: "CUDA_T", OCL_T: "OCL_T"}
idpy_langs_list = list(idpy_langs_dict.values())

from idpy.Utils.IsModuleThere import AreModulesThere
idpy_langs_sys = AreModulesThere(modules_list = idpy_langs_list)

'''
Compilers warnings
'''
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

'''
Tenet types
'''
idpy_tenet_types = {}
if idpy_langs_sys[CUDA_T]:
    from idpy.CUDA.CUDA import CUDA
    from idpy.CUDA.CUDA import Tenet as CUTenet
    idpy_tenet_types[CUDA_T] = CUTenet

if idpy_langs_sys[OCL_T]:
    import pyopencl as cl
    from idpy.OpenCL.OpenCL import OpenCL
    from idpy.OpenCL.OpenCL import Tenet as CLTenet
    idpy_tenet_types[OCL_T] = CLTenet

'''
Methods: GetTenet
'''
def GetTenet(params_dict):
    '''
    GetTenet:
    it looks general enough to be further abstracted
    ''' 
    if 'lang' in params_dict and params_dict['lang'] == CUDA_T:
        if idpy_langs_sys[CUDA_T]:
            cu = CUDA()
            device = 0 if 'device' not in params_dict else params_dict['device']
            cu.SetDevice(device)
            print("CUDA: ", cu.GetDeviceName())
            return cu.GetTenet()
        else:
            raise Exception("Selected lang = CUDA_T but CUDA is not found on the system!")

    if 'lang' in params_dict and params_dict['lang'] == OCL_T:
        if idpy_langs_sys[OCL_T]:
            ocl = OpenCL()
            cl_type = 'gpu' if 'cl_kind' not in params_dict else params_dict['cl_kind']
            device = 0 if 'device' not in params_dict else params_dict['device']
            ocl.SetDevice(kind = cl_type, device = device)
            print("OpenCL: ", ocl.GetDeviceName())
            return ocl.GetTenet()
        else:
            raise Exception("Selected lang = OCL_T but openCL is not found on the system!")

'''
Method CheckOCLFP
what about unsigned 64 bits integers?
'''
from idpy.Utils.CustomTypes import CustomTypes
def CheckOCLFP(tenet, custom_types):
    if idpy_langs_sys[OCL_T] and isinstance(tenet, idpy_tenet_types[OCL_T]):
        if tenet.device.get_info(cl.device_info.DOUBLE_FP_CONFIG) == 0:
            print("\nThe device",
                  tenet.device.get_info(cl.device_info.NAME),
                  "does not support 64 bits floating-point variables")
            print("Changing all custom types from 64-bits to 32-bits")
            _swap_dict = {}
            for key, value in custom_types.Push().items():
                if value == 'double':
                    value = 'float'
                    
                _swap_dict[key] = value
                
            return CustomTypes(_swap_dict)
        else:
            return custom_types
    else:
        return custom_types

'''
Method GetParamsClean
'''
def GetParamsClean(kwargs, _a_params_dict, needed_params = None):
    '''
    GetParamsClean:
    it looks general enough to be further abstracted
    '''
    for var in needed_params:
        if var in kwargs:
            _a_params_dict[0][var] = kwargs[var]
            del kwargs[var]

    return kwargs
