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
Provides a minimal interface for the use of pyopencl
methods names, if shared, match those of idpy.CUDA.CUDA class
'''

import pyopencl as cl
from collections import defaultdict

'''
The main idea is to create the child classes
neede to make the Tenet class work homogenously
throughout different classes
'''

# https://stackoverflow.com/questions/30105134/initialize-child-class-with-parent
class Tenet(cl.CommandQueue):
    @classmethod
    def from_parent(cls, parent):
        return cls(context = parent.context,
                   device = parent.device,
                   properties = parent.properties)

    def __init__(self, *args, **kwargs):
        super(Tenet, self).__init__(*args, **kwargs)

    def End(self):
        return super().finish()

    def SetKind(self, kind):
        self.kind = kind

class OpenCL:
    '''
    class OpenCL:
    -- GetContext: returns the OpenCL context of the selected device
    -- SetDevice: sets the device
    -- GetDevice: returns the set device
    -- ListDevices: list the detected devices
    '''
    CPU_T, GPU_T = "cpu", "gpu"
    
    def __init__(self):
        self.platforms = cl.get_platforms()
        self.cpus, self.gpus = [], []
        self.devices = {}
        self.kind, self.device = None, None
        
        # Getting platform and devices infos
        for platform in self.platforms:
            # CPUS
            if len(platform.get_devices(cl.device_type.CPU)):
                self.devices[self.CPU_T] = []
                for cpu in platform.get_devices(cl.device_type.CPU):
                    self.devices[self.CPU_T].append(cpu)
            # GPUS
            if len(platform.get_devices(cl.device_type.GPU)):
                self.devices[self.GPU_T] = []
                for gpu in platform.get_devices(cl.device_type.GPU):
                    self.devices[self.GPU_T].append(gpu)
    
    def GetContext(self):
        if self.kind is not None:
            return cl.Context([self.GetDevice()])

    def GetQueue(self):
        if self.kind is not None:
            return cl.CommandQueue(self.GetContext(), None,
                                   cl.command_queue_properties.PROFILING_ENABLE)

    def GetTenet(self):
        _tenet = Tenet.from_parent(self.GetQueue())
        _tenet.SetKind(self.kind)
        return _tenet
        
    def SetDevice(self, kind = GPU_T, device = 0):
        self.kind, self.device = kind, device
    
    def GetDevice(self):
        if self.kind is not None:
            return self.devices[self.kind][self.device]

    def GetDeviceName(self):
        _dict = None
        if self.kind == self.GPU_T:
            _dict = self.DiscoverGPUs()
        if self.kind == self.CPU_T:
            _dict = self.DiscoverCPUs()
            
        return ("Device: " + str(self.device) + " " +
                str(_dict[self.device]['Name']) +
                " Memory:" +  str(_dict[self.device]['Memory'])) 

    def DiscoverGPUs(self):
        gpus_dict = defaultdict(
            lambda: defaultdict(dict)
        )

        for platfrom in self.platforms:
            if self.GPU_T in self.devices:
                for gpu_i in range(len(self.devices[self.GPU_T])):
                    swap_dev = self.devices[self.GPU_T][gpu_i]
                    gpus_dict[gpu_i]['Name'] = swap_dev.get_info(cl.device_info.NAME)
                    gpus_dict[gpu_i]['Memory'] = swap_dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
                    gpus_dict[gpu_i]['Double'] = swap_dev.get_info(cl.device_info.DOUBLE_FP_CONFIG)
                    gpus_dict[gpu_i]['DrvVersion'] = swap_dev.get_info(cl.device_info.DRIVER_VERSION)

        return gpus_dict

    def DiscoverCPUs(self):
        cpus_dict = defaultdict(
            lambda: defaultdict(dict)
        )

        for platfrom in self.platforms:
            if self.CPU_T in self.devices:
                for cpu_i in range(len(self.devices[self.CPU_T])):
                    swap_dev = self.devices[self.CPU_T][cpu_i]
                    cpus_dict[cpu_i]['Name'] = swap_dev.get_info(cl.device_info.NAME)
                    cpus_dict[cpu_i]['Memory'] = swap_dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
                    cpus_dict[cpu_i]['Double'] = swap_dev.get_info(cl.device_info.DOUBLE_FP_CONFIG)
                    cpus_dict[cpu_i]['DrvVersion'] = swap_dev.get_info(cl.device_info.DRIVER_VERSION)

        return cpus_dict
