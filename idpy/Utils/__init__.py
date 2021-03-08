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
Basic imports
'''
import inspect, os, sys
from sys import platform

_module_abs_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_idea_dot_deploy_path = os.path.dirname(os.path.abspath(_module_abs_path + "../../"))
'''
append to sys path in order to avoid relative imports
'''
sys.path.append(_idea_dot_deploy_path)

from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T
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
