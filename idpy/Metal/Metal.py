"""
Provides a minimal interface for the use of pymetallic.
Method names, if shared, match those of idpy.CUDA.CUDA class.

Requires Apple Silicon (or Metal-capable macOS), Xcode/Swift toolchain,
and: pip install pymetallic
"""

__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2023 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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

import platform
from collections import defaultdict

import pymetallic

from . import METAL_T

'''
The main idea is to create the child classes
needed to make the Tenet class work homogenously
throughout different classes
'''

class Tenet:
    GPU_T = "gpu"

    def __init__(self, device, queue, device_name):
        self.device = device
        self.queue = queue
        self.device_name = device_name
        self.mem_pool = None
        self.allocator = None

    def FreeMemoryDict(self, memory_dict=None):
        pass

    def AllocatedBytes(self):
        print("Metal Tenet.AllocatedBytes: functionality is not available")
        return 0

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
        return METAL_T

    def End(self, memory_dict=None):
        self.queue = None
        self.device = None
        return None

    def SetMemoryPool(self):
        print("Metal Tenet.SetMemoryPool: functionality is not available")
        self.mem_pool = None
        self.allocator = None


class Metal:
    '''
    class Metal:
    -- GetTenet: returns a Tenet for the selected device
    -- SetDevice: sets the device
    -- GetDevice: returns the set device
    -- DiscoverGPUs: list the detected devices
    '''
    GPU_T = "gpu"

    def __init__(self):
        self.devices = list(pymetallic.Device.get_all_devices())
        if len(self.devices) == 0:
            default = pymetallic.Device.get_default_device()
            if default is not None:
                self.devices = [default]
        self.device = None

    def GetTenet(self):
        if self.device is None:
            self.SetDevice(0)
        device = self.GetDevice()
        queue = device.make_command_queue()
        _tenet = Tenet(device=device,
                       queue=queue,
                       device_name=self.GetDeviceName())
        _tenet.SetMemoryPool()
        return _tenet

    def SetDevice(self, device=0):
        self.device = device

    def GetDevice(self):
        if self.device is not None:
            return self.devices[self.device]

    def GetDeviceName(self):
        _dict = self.DiscoverGPUs()
        return {'Name': _dict[self.device]['Name'],
                'Device': str(self.device),
                'Memory': str(_dict[self.device]['Memory']),
                'DrvVersion': _dict[self.device]['DrvVersion']}

    def DiscoverGPUs(self):
        gpus_dict = defaultdict(lambda: defaultdict(dict))
        for gpu_i in range(len(self.devices)):
            swap_dev = self.devices[gpu_i]
            name = getattr(swap_dev, 'name', None)
            if callable(name):
                name = name()
            if name is None:
                name = str(swap_dev)
            # Prefer pymetallic Device.recommended_max_working_set_size
            mem = getattr(swap_dev, 'recommended_max_working_set_size', None)
            if callable(mem):
                mem = mem()
            if mem is None:
                mem = getattr(swap_dev, 'max_buffer_length', None)
                if callable(mem):
                    mem = mem()
            if mem is None:
                mem = 0
            gpus_dict[gpu_i]['Name'] = name
            gpus_dict[gpu_i]['Memory'] = mem
            gpus_dict[gpu_i]['DrvVersion'] = (
                platform.mac_ver()[0] if platform.system() == 'Darwin'
                else platform.platform()
            )
        return gpus_dict
