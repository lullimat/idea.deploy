"""
Provides a minimal interface for the use of pycuda
methods names, if shared, match those of idpy.OpenCL.OpenCL class
"""

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


import pycuda as cu
import pycuda.driver as cu_driver
from collections import defaultdict
from idpy.IdpyCode import CUDA_T

'''
The main idea is to create the child classes
neede to make the Tenet class work homogenously
throughout different classes
'''

class Tenet:
    GPU_T = "gpu"
    
    def __init__(self, cuda_context, device_name):
        self.cuda_context = cuda_context
        self.device_name = device_name

    def FreeMemoryDict(self, memory_dict = None):
        pass

    def AllocatedBytes(self):
        return self.mem_pool.active_bytes

    def GetKind(self):
        return self.GPU_T

    def GetDeviceName(self):
        return self.device_name['Name']

    def GetDeviceNumber(self):
        return self.device_name['Device']

    def GetDeviceMemory(self):
        return self.device_name['Memory']

    def GetDrvVersion(self):
        return str(self.device_name['DrvVersion'])

    def GetLang(self):
        return CUDA_T                      

    def End(self, memory_dict = None):
        self.mem_pool.free_held()
        self.mem_pool.stop_holding()
        return self.cuda_context.detach()

    def SetMemoryPool(self):
        self.mem_pool = cu.tools.DeviceMemoryPool()
        self.allocator = self.mem_pool.allocate

class CUDA:
    '''                                                                               
    class CUDA:
    -- GetContext: returns the OpenCL context of the selected device
    -- SetDevice: sets the device
    -- GetDevice: returns the set device
    -- ListDevices: list the detected devices
    '''

    def __init__(self):
        self.devices, self.device = [], None
        
        cu_driver.init()
        for gpu_i in range(cu_driver.Device.count()):
            self.devices.append(cu_driver.Device(gpu_i))
            
    def GetContext(self):
        if self.device is not None:
            return self.devices[self.device].make_context()

    def GetTenet(self):
        _tenet = Tenet(cuda_context = self.GetContext(),
                       device_name = self.GetDeviceName())
        _tenet.SetMemoryPool()
        
        return _tenet
            
    def SetDevice(self, device = 0):
        self.device = device
        
    def GetDevice(self):
        if self.device is not None:
            return self.devices[self.device]

    def GetDeviceName(self):
        _dict = self.DiscoverGPUs()
            
        return {'Name': _dict[self.device]['Name'],
                'Device': str(self.device),
                'Memory':  str(_dict[self.device]['Memory']), 
                'DrvVersion': _dict[self.device]['DrvVersion']}

            
    def DiscoverGPUs(self):
        gpus_dict = defaultdict(
            lambda: defaultdict(dict)
        )
        for gpu_i in range(len(self.devices)):
            swap_dev = self.devices[gpu_i]
            gpus_dict[gpu_i]['Name'] = swap_dev.name()
            gpus_dict[gpu_i]['Memory'] = swap_dev.total_memory()
            gpus_dict[gpu_i]['DrvVersion'] = cu_driver.get_version()

        return gpus_dict
