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
Provides a minimal interface for the use of pyopencl
methods names, if shared, match those of idpy.CUDA.CUDA class
'''

import ctypes
from collections import defaultdict
import cpuinfo
import psutil

from idpy.IdpyCode import idpy_ctypes_compiler_string_h

'''
The main idea is to create the child classes
neede to make the Tenet class work homogenously
throughout different classes
'''

class Tenet:
    def __init__(self, *args, **kwargs):
        self.device_name = None
    
    def End(self):
        pass
 
    '''
    Need to get the name of the system's CPU
    '''
    def SetDeviceName(self, device_name):
        self.device_name = device_name

    def GetLang(self):
        return CTYPES_T                   

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
