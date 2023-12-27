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
Provides a minimal interface for the use of pyopencl
methods names, if shared, match those of idpy.CUDA.CUDA class
'''

import ctypes
from collections import defaultdict
import cpuinfo
import psutil
import subprocess

from numpy.ctypeslib import ndpointer

from idpy.Utils.CTypesTypes import CTypesTypes

CTT = CTypesTypes()

from pathlib import Path
import hashlib

from . import CTYPES_T, idpy_ctypes_cache_dir, idpy_ctypes_compiler_string_h

CTYPES_N_THREAD = "N_THREADS"

'''
The main idea is to create the child classes
neede to make the Tenet class work homogenously
throughout different classes
'''

class Tenet:
    CPU_T = "cpu"

    def __init__(self, *args, **kwargs):
        self.device_name = None
    
    def End(self):
        pass

    def FreeMemoryDict(self, memory_dict = None):
        pass
    '''
    Need to get the name of the system's CPU
    '''
    def SetDeviceName(self, device_name):
        self.device_name = device_name

    def GetLang(self):
        return CTYPES_T

    def GetKind(self):
        return self.CPU_T        

    def GetKernelModule(self, params, code, options):
        return CTypesKernelModule(params, code, options)

class CTypesKernelModule:
    def __init__(self, params, code, options):
        self.params, self.code, self.options = params, code, options
        self.SetCompileStringHead()

        '''
        Encoding the code string for hashing
        '''
        self.code_utf = self.code.encode(encoding='UTF-8',errors='strict')
        self.compile_str_utf = self.compile_string_head.encode(encoding='UTF-8',errors='strict')

        self.code_hash = self.GetCodeHash()
        self.compile_str_hash = self.GetCompileStrHash()

        '''
        Since I am giving to the temporary source and object files the name of the hash, I only
        need to check for the file name
        '''
        ## Source code file
        self.code_file = \
            idpy_ctypes_cache_dir / (str(self.code_hash) + '.c')

        self.is_code_file = self.CheckCodeFile()
        if not self.is_code_file:
            with open(self.code_file, 'w') as _code_file:
                _code_file.write(self.code)

        ## Shared object: given that one can compile with different options it is important
        ## to add these options as another hashing to append to the module name: would the
        ## string be too long?
        self.so_file = \
            idpy_ctypes_cache_dir / (str(self.code_hash) + '_' + str(self.compile_str_hash) + '.so')

        self.is_so_file = self.CheckSOFile()
        self.compile_status = False
        if not self.is_so_file:
            self.compile_status = self.Compile()
        else:
            self.compile_status = True


    def GetKernelFunction(self, name, custom_types):
        self.kernel_function = ctypes.CDLL(self.so_file)        
        self.argtypes = ()

        for _param in self.params.keys():
            
            _param_nws = _param.split(" ")
            _is_pointer = '*' in _param_nws
            _type = _param_nws[0]
            
            if not _is_pointer:
                if _type not in list(custom_types.keys()) or _type in list(custom_types.values()):
                    self.argtypes += (CTT.C[_type], )
                else:
                    self.argtypes += (CTT.C[custom_types[_type]], )
            else:
                if _type not in list(custom_types.keys()) or _type in list(custom_types.values()):
                    self.argtypes += \
                        (
                            ndpointer(CTT.C[_type],
                            flags="C_CONTIGUOUS"), 
                        )
                else:
                    self.argtypes += \
                        (
                            ndpointer(CTT.C[custom_types[_type]],
                            flags="C_CONTIGUOUS"), 
                        )

        self.kernel_function.__getattr__(name).argtypes = self.argtypes
        return getattr(self.kernel_function, name)


    def SetCompileStringHead(self): 
        self.compile_string_head = \
            idpy_ctypes_compiler_string_h + self.options

    def Compile(self):
        self.compile_string =  \
            self.compile_string_head + " -o " + \
            str(self.so_file) + " " + str(self.code_file)

        _compile_tuple = tuple(self.compile_string.split(" "))
        return True if subprocess.run(_compile_tuple).returncode == 0 else False

    def GetCodeHash(self):
        return hashlib.md5(self.code_utf).hexdigest()

    def GetCompileStrHash(self):
        return hashlib.md5(self.compile_str_utf).hexdigest()        

    def CheckCacheDir(self):
        if not idpy_ctypes_cache_dir.is_dir():
            idpy_ctypes_cache_dir.mkdir()
            return False
        else:
            return True

    def CheckCodeFile(self):
        if self.CheckCacheDir():
            return self.code_file.is_file()
        else:
            return False

    def CheckSOFile(self):
        if self.CheckCacheDir():
            return self.so_file.is_file()
        else:
            return False        


class CTypes:
    '''
    class CTypes:
    -- GetDevice: returns the set device
    '''
    CPU_T = "cpu"
    
    def __init__(self):
        self.system_info = cpuinfo.get_cpu_info()
        self.devices = {}
        self.kind, self.device = None, None
        
        # Getting platform and devices infos
        pass
    
    def GetTenet(self):
        _tenet = Tenet()
        _tenet.SetDeviceName(self.GetDeviceName())
        return _tenet
            
    def GetDeviceName(self):            
        return ("Device: CPU " +
                str(cpuinfo.get_cpu_info()['brand_raw']) +
                " Memory:" +  str(psutil.virtual_memory()[0]))
