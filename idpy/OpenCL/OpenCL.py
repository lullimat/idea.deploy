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

import pyopencl as cl
from collections import defaultdict
from . import OCL_T

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

    def FreeMemoryDict(self, memory_dict = None):
        if memory_dict is not None and type(memory_dict) is dict:
            '''
            I should be checking it is IdpyMemory type
            '''
            for _ in memory_dict:
                if memory_dict[_] is not None:
                    memory_dict[_].data.release()
        self.mem_pool.free_held()

    def AllocatedBytes(self):
        return self.mem_pool.active_bytes
        
    def End(self):
        super().flush()
        return super().finish()

    def SetKind(self, kind):
        self.kind = kind

    def GetKind(self):
        return self.kind

    def SetDeviceName(self, device_name):
        self.device_name = device_name

    def SetMemoryPool(self, allocator = None):
        if allocator is None:
            self.mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self))
        else:
            self.mem_pool = cl.tools.MemoryPool(allocator(self))

    def GetDeviceName(self):
        return self.device_name['Name']

    def GetDeviceNumber(self):
        return self.device_name['Device']

    def GetDeviceMemory(self):
        return self.device_name['Memory']

    def GetDeviceFP64(self):
        return self.device_name['FP64']

    def GetLang(self):
        return OCL_T

    def GetDrvVersion(self):
        return str(self.device_name['DrvVersion'])

class TenetNew:
    def __init__(self, cl_context, device):
        self.context = cl_context
        self.device = device
        self.queues = []

    def SetDefaultQueue(self):
        self.queues = \
            [cl.CommandQueue(self.context, self.device,
                             cl.command_queue_properties.PROFILING_ENABLE)]

    def FreeMemoryDict(self, memory_dict = None):
        if memory_dict is not None and type(memory_dict) is dict:
            '''
            I should be checking it is IdpyMemory type
            '''
            for _ in memory_dict:
                if memory_dict[_] is not None:
                    memory_dict[_].data.release()
        self.mem_pool.free_held()

    def AllocatedBytes(self):
        return self.mem_pool.active_bytes
        
    def End(self):
        super().flush()
        return super().finish()

    def SetKind(self, kind):
        self.kind = kind

    def GetKind(self):
        return self.kind
        
    def SetDeviceName(self, device_name):
        self.device_name = device_name

    def SetMemoryPool(self, allocator = None):
        if allocator is None:
            self.mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self))
        else:
            self.mem_pool = cl.tools.MemoryPool(allocator(self))

            
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
            cpu_devices, gpu_devices = [], []
            try:
                cpu_devices = platform.get_devices(cl.device_type.CPU)
            except:
                print("There is some issue for pyopencl to list cpu's")

            try:
                gpu_devices = platform.get_devices(cl.device_type.GPU)
            except:
                print("There is some issue for pyopencl to list gpu's")


            if len(cpu_devices):
                self.devices[self.CPU_T] = []
                for cpu in platform.get_devices(cl.device_type.CPU):
                    self.devices[self.CPU_T].append(cpu)
            # GPUS
            if len(gpu_devices):
                self.devices[self.GPU_T] = []
                for gpu in platform.get_devices(cl.device_type.GPU):
                    self.devices[self.GPU_T].append(gpu)
    
    def GetContext(self):
        if self.kind is not None:
            return cl.Context([self.GetDevice()])
            # return cl.Context(self.GetDevice())

    def GetQueue(self):
        if self.kind is not None:
            return cl.CommandQueue(self.GetContext(), self.GetDevice(),
                                   cl.command_queue_properties.PROFILING_ENABLE)

    def GetTenet(self):
        _tenet = Tenet.from_parent(self.GetQueue())
        _tenet.SetKind(self.kind)
        _tenet.SetDeviceName(self.GetDeviceName())
        _tenet.SetMemoryPool()
        return _tenet
        
    def SetDevice(self, kind = GPU_T, device = 0):
        if kind not in self.devices:
            kind = self.CPU_T
            
        self.kind, self.device = kind, device
    
    def GetDevice(self):
        if self.kind is not None and self.kind in self.devices:
            return self.devices[self.kind][self.device]

    def GetDeviceName(self):
        _dict = None
        if self.kind == self.GPU_T:
            _dict = self.DiscoverGPUs()
        if self.kind == self.CPU_T:
            _dict = self.DiscoverCPUs()
            
        return {'Name': _dict[self.device]['Name'],
                'Device': str(self.device),
                'Memory':  str(_dict[self.device]['Memory']), 
                'Kind': str(self.kind), 
                'DrvVersion': _dict[self.device]['DrvVersion'], 
                'FP64': _dict[self.device]['Double']}

    def GetDeviceNameOld(self):
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
                    gpus_dict[gpu_i]['Memory'] = \
                        swap_dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
                    gpus_dict[gpu_i]['Double'] = \
                        swap_dev.get_info(cl.device_info.DOUBLE_FP_CONFIG)
                    gpus_dict[gpu_i]['DrvVersion'] = \
                        swap_dev.get_info(cl.device_info.DRIVER_VERSION)

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
                    cpus_dict[cpu_i]['Memory'] = \
                        swap_dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
                    cpus_dict[cpu_i]['Double'] = \
                        swap_dev.get_info(cl.device_info.DOUBLE_FP_CONFIG)
                    cpus_dict[cpu_i]['DrvVersion'] = \
                        swap_dev.get_info(cl.device_info.DRIVER_VERSION)

        return cpus_dict
