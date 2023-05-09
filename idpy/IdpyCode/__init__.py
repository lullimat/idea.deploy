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

# https://stackoverflow.com/questions/50499/how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing/50905#50905

import inspect, os, sys

_module_abs_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_idea_dot_deploy_path = os.path.dirname(os.path.abspath(_module_abs_path + "../../"))
'''
append to sys path in order to avoid relative imports
'''
sys.path.append(_idea_dot_deploy_path)

from idpy import idpy_os_found

if False:
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

'''
Define some virtual environment variables
'''
import re
_VENV_ROOT_STR, _IDPY_ENV_F = "VENV_ROOT", ".idpy-env"
_id_env_file = open(_idea_dot_deploy_path + "/" + _IDPY_ENV_F)
_venv_root = None
for _line in _id_env_file.readlines():
    if re.search(_VENV_ROOT_STR, _line):
        _venv_root = _line.split("/")[1].strip()
        break
_id_env_file.close()
if _venv_root is None:
    raise \
        Exception(
            "Could not find string ", _VENV_ROOT_STR, 
            "in file", _idea_dot_deploy_path + "/" + _IDPY_ENV_F
            )

_idpy_env_path = _idea_dot_deploy_path + "/" + _venv_root + "/"
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

from idpy.OpenCL import OCL_T
from idpy.CUDA import CUDA_T
from idpy.CTypes import CTYPES_T
from idpy.Metal import METAL_T

IDPY_T = "idpy"

idpy_langs_dict = {'CUDA_T': CUDA_T, 'OCL_T': OCL_T, 'CTYPES_T': CTYPES_T, 'METAL_T': METAL_T}

idpy_langs_human_dict = {CUDA_T: "CUDA", OCL_T: "OpenCL", CTYPES_T: "ctypes", METAL_T: "Metal"}
idpy_langs_dict_sym = {CUDA_T: "CUDA_T", OCL_T: "OCL_T", CTYPES_T: "CTYPES_T", METAL_T: "METAL_T"}
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

if idpy_langs_sys[CTYPES_T]:
    from idpy.CTypes.CTypes import CTypes
    from idpy.CTypes.CTypes import Tenet as CTTenet
    idpy_tenet_types[CTYPES_T] = CTTenet

if idpy_langs_sys[METAL_T]:
    from idpy.Metal.Metal import Metal
    from idpy.Metal.Metal import Tenet as MTTenet
    idpy_tenet_types[METAL_T] = MTTenet    


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
            raise Exception("Selected lang = CUDA_T but the module pycuda is not found in your python environment!")

    if 'lang' in params_dict and params_dict['lang'] == OCL_T:
        if idpy_langs_sys[OCL_T]:
            ocl = OpenCL()
            cl_type = 'gpu' if 'cl_kind' not in params_dict else params_dict['cl_kind']
            cl_type = cl_type if cl_type in ocl.devices else 'cpu'
            device = 0 if 'device' not in params_dict else params_dict['device']
            
            ocl.SetDevice(kind = cl_type, device = device)
            
            print("OpenCL: ", ocl.GetDeviceName())
            return ocl.GetTenet()
        else:
            raise Exception("Selected lang = OCL_T but the 'pyopencl' module is not found in your python environment!")
        
    if 'lang' in params_dict and params_dict['lang'] == CTYPES_T:
        if idpy_langs_sys[CTYPES_T]:
            c_types = CTypes()
            print("CTypes: ", c_types.GetDeviceName())
            return c_types.GetTenet()
        else:
            raise Exception("Selected lang = CTYPES_T but the 'ctypes' module is not found in your python environment!")

'''
Method CheckOCLFP
what about unsigned 64 bits integers?
The question was good...see CRNGS
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

                if value == 'unsigned long':
                    value = 'unsigned int'
                    
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

from idpy.IdpyCode import CUDA_T, OCL_T, CTYPES_T, METAL_T, IDPY_T
from idpy.IdpyCode import idpy_langs_sys, idpy_langs_list

'''
Methods: IdpyHardware
'''

def IdpyHardware():
    if idpy_langs_sys[CUDA_T]:
        from idpy.CUDA.CUDA import CUDA
        print("CUDA Found!")
        cuda = CUDA()
        gpus_list = cuda.DiscoverGPUs()
        for gpu_i in gpus_list:
            print("\nCUDA GPU[" + str(gpu_i) + "]")
            for key in gpus_list[gpu_i]:
                print(key, ": ", gpus_list[gpu_i][key])
            print()
        del cuda
        print("=" * 80)
        print()

    if idpy_langs_sys[OCL_T]:
        from idpy.OpenCL.OpenCL import OpenCL
        print("OpenCL Found!")
        ocl = OpenCL()
        gpus_list = ocl.DiscoverGPUs()
        cpus_list = ocl.DiscoverCPUs()
        print("\nListing GPUs:")
        for gpu_i in gpus_list:
            print("OpenCL GPU[" + str(gpu_i) + "]")
            for key in gpus_list[gpu_i]:
                print(key, ": ", gpus_list[gpu_i][key])
            print()
        print("\nListing CPUs:")
        for cpu_i in cpus_list:
            print("OpenCL CPU[" + str(cpu_i) + "]")
            for key in cpus_list[cpu_i]:
                print(key, ": ", cpus_list[cpu_i][key])
            print()
        del ocl
        print("=" * 80)
        print()

    if idpy_langs_sys[CTYPES_T]:
        from idpy.CTypes.CTypes import CTypes
        print("CTypes Found!")
        c_types = CTypes()
        print("\nListing CPUs:")
        print(c_types.GetDeviceName())
        del c_types
        print()
        print("=" * 80)
        print()


    if idpy_langs_sys[METAL_T]:
        from idpy.Metal.Metal import Metal
        print("Metal Found!")
        metal = Metal()
        print("\nListing GPUs:")
        print(metal.GetDeviceName())
        del metal
        print()
        print("=" * 80)
        print()                

'''
Methods: GridAndBlocks
'''
def GridAndBlocks1D(_n_threads_min, _block_size = 128):
    _grid = ((_n_threads_min + _block_size - 1)//_block_size, 1, 1)
    _block = (_block_size, 1, 1)

    return _grid, _block

'''
Copyright string
'''
_license_path = _idea_dot_deploy_path + "/" + "LICENSE"
_file_swap = open(_license_path, "r")
idpy_copyright = _file_swap.read()
_file_swap.close()
