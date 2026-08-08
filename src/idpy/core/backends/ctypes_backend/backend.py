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
methods names, if shared, match those of idpy.core.backends.cuda.backend class
'''

import ctypes
from collections import defaultdict
import cpuinfo
import psutil
import subprocess

from numpy.ctypeslib import ndpointer

from idpy.core.utils.CTypesTypes import CTypesTypes
from idpy.core.utils.HostModule import HostModule, CToolchain

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

class CTypesKernelModule(HostModule):
    '''
    The CTYPES_T kernel module: HostModule bound to the C toolchain.

    The body of this class moved to idpy/Utils/HostModule.py (Phase 1). Nothing
    in it was ever CPU-compute-specific -- hash a source string, cache the
    build, hand back ctypes callables -- so it is now shared machinery with the
    compiler as a parameter, and CTYPES_T is simply its first consumer.

    Behaviour is unchanged, deliberately down to the details: the same cache
    directory, the same hash inputs (so warm caches stay valid), the same
    option-string concatenation without a separator, and the same
    ndpointer-based argtypes.

    The subclass is kept rather than replaced by an alias so that
    Tenet.GetKernelModule keeps its familiar name and any external caller
    referring to CTypesKernelModule still works.
    '''

    def __init__(self, params, code, options):
        HostModule.__init__(
            self, params, code, options,
            toolchain=CToolchain(idpy_ctypes_compiler_string_h),
            cache_dir=idpy_ctypes_cache_dir,
        )

    def SetCompileStringHead(self):
        '''Retained for compatibility; HostModule computes this at build time.'''
        self.compile_string_head = self.toolchain.Head(self.options)
        return self.compile_string_head


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
        info = cpuinfo.get_cpu_info()
        brand = info.get('brand_raw') or info.get('brand') or info.get('hardware') or 'CPU'
        return ("Device: CPU " + str(brand) +
                " Memory:" + str(psutil.virtual_memory()[0]))
