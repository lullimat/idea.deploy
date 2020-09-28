"""
Provides a minimal interface for the use of pycuda
methods names, if shared, match those of idpy.OpenCL.OpenCL class
"""

import pycuda as cu
import pycuda.driver as cu_driver
from collections import defaultdict

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

'''
The main idea is to create the child classes
neede to make the Tenet class work homogenously
throughout different classes
'''

class Tenet:
    def __init__(self, cuda_context):
        self.cuda_context = cuda_context

    def End(self):
        return self.cuda_context.detach()

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
        return Tenet(self.GetContext())
            
    def SetDevice(self, device = 0):
        self.device = device
        
    def GetDevice(self):
        if self.device is not None:
            return self.devices[self.device]

    def GetDeviceName(self):
        _dict = self.DiscoverGPUs()
            
        return ("Device: " + str(self.device) + " " +
                _dict[self.device]['Name'] +
                " Memory:" +  str(_dict[self.device]['Memory']))

            
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
