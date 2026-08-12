"""
Provides a minimal interface for the use of pymetallic.
Method names, if shared, match those of idpy.core.backends.cuda.backend class.

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

import numpy as np
import pymetallic

from . import METAL_T

'''
The main idea is to create the child classes
needed to make the Tenet class work homogenously
throughout different classes
'''


class MetalMemoryPool:
    """
    Free-list of pymetallic.Buffer objects keyed by (nbytes, dtype.str).

    Recycled buffers keep their Metal allocation; callers must zero/fill as needed.
    """

    def __init__(self, device):
        self.device = device
        self._free = {}
        self._active_bytes = 0
        self._held_bytes = 0
        self._closed = False

    @staticmethod
    def _key(shape, dtype):
        dtype = np.dtype(dtype)
        shape = shape if isinstance(shape, tuple) else (shape,)
        nbytes = int(np.prod(shape)) * int(dtype.itemsize)
        return (nbytes, dtype.str), shape, dtype, nbytes

    def allocate_buffer(self, shape, dtype):
        if self._closed:
            raise RuntimeError("MetalMemoryPool is closed")
        key, shape, dtype, nbytes = self._key(shape, dtype)
        free = self._free.setdefault(key, [])
        if free:
            buf = free.pop()
            self._held_bytes -= nbytes
        else:
            zeros = np.zeros(shape, dtype=dtype)
            buf = pymetallic.Buffer.from_numpy(self.device, zeros)
        self._active_bytes += nbytes
        return buf, shape, dtype, nbytes

    def free_buffer(self, buf, nbytes, dtype_str):
        if self._closed or buf is None:
            return
        key = (int(nbytes), str(dtype_str))
        self._free.setdefault(key, []).append(buf)
        self._active_bytes = max(0, self._active_bytes - int(nbytes))
        self._held_bytes += int(nbytes)

    def active_bytes(self):
        return int(self._active_bytes)

    def held_bytes(self):
        return int(self._held_bytes)

    def clear(self):
        """Drop free-list references so Metal buffers can be released."""
        self._free.clear()
        self._held_bytes = 0
        self._closed = True


def _TouchOverlaps(touched, buffer_key, start_byte, stop_byte):
    '''
    Does a command buffer's recorded touch-set intersect [start_byte, stop_byte)
    of 'buffer_key'?

    touched is None  -> unknown, so assume yes. Every path that cannot describe
                        what it touched must land here, keeping the fallback at
                        "wait" rather than "race".
    key absent       -> that command buffer never bound this buffer at all.
    '''
    if touched is None:
        return True
    _span = touched.get(buffer_key)
    if _span is None:
        return False
    return start_byte < _span[1] and _span[0] < stop_byte


class Tenet:
    GPU_T = "gpu"

    def __init__(self, device, queue, device_name):
        self.device = device
        self.queue = queue
        self.device_name = device_name
        self.mem_pool = None
        self.allocator = None
        # Last committed command buffer (same-queue order ⇒ waiting this drains priors)
        self.last_command_buffer = None
        '''
        Ordered record of submitted, possibly-incomplete command buffers, for
        range-scoped host waiting (finding F2). Each entry is

            {'cb': CommandBuffer, 'touched': {buffer_key: (start_byte, stop_byte)}}

        with 'touched' None meaning "unknown -- assume it touched everything".
        Unknown is the safe default: it degrades to the old drain-everything
        behaviour rather than to a race.

        The whole scheme rests on one property: a single queue completes command
        buffers IN ORDER. So waiting on entry i also completes 0..i-1, which is
        why WaitForRange can drop a whole prefix at once and why finding the
        LATEST overlapping entry is sufficient.
        '''
        self._in_flight = []

    def FreeMemoryDict(self, memory_dict=None):
        pass

    def Submit(self, command_buffer, touched=None):
        '''Record a committed command buffer and what it may have touched.'''
        self._in_flight.append({'cb': command_buffer, 'touched': touched})
        self.last_command_buffer = command_buffer
        return command_buffer

    def _DropThrough(self, index):
        '''In-order completion: everything up to and including index is done.'''
        del self._in_flight[:index + 1]
        if not self._in_flight:
            self.last_command_buffer = None

    def PruneCompleted(self):
        '''
        Drop entries the GPU has already finished, without blocking. Scans
        newest-first and drops the whole prefix at the first completed entry.
        '''
        for i in range(len(self._in_flight) - 1, -1, -1):
            _get_status = getattr(self._in_flight[i]['cb'], 'get_status', None)
            if _get_status is None:
                continue
            try:
                if _get_status() == 1:      # 1 == completed
                    self._DropThrough(i)
                    return
            except Exception:
                # status unavailable ⇒ leave the entry in flight (conservative)
                return

    def WaitForRange(self, buffer_key, start_byte, stop_byte):
        '''
        Block only for in-flight work that may touch [start_byte, stop_byte) of
        the buffer identified by 'buffer_key'. Returns True if it waited.

        This is the replacement for draining the queue on every host touch: a
        host write to a slot no in-flight kernel is reading costs nothing.
        '''
        self.PruneCompleted()
        for i in range(len(self._in_flight) - 1, -1, -1):
            if _TouchOverlaps(self._in_flight[i]['touched'],
                              buffer_key, start_byte, stop_byte):
                self._in_flight[i]['cb'].wait_until_completed()
                self._DropThrough(i)
                return True
        return False

    def Finish(self):
        """Host-sync: wait until the last submitted command buffer completes."""
        cb = self.last_command_buffer
        if cb is not None:
            cb.wait_until_completed()
            self.last_command_buffer = None
        # In-order completion: waiting the newest CB completes every earlier one.
        self._in_flight.clear()

    def AllocatedBytes(self):
        if self.mem_pool is None:
            return 0
        return self.mem_pool.active_bytes()

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
        self.Finish()
        if self.mem_pool is not None:
            self.mem_pool.clear()
            self.mem_pool = None
        self.allocator = None
        self.queue = None
        self.device = None
        return None

    def SetMemoryPool(self):
        self.mem_pool = MetalMemoryPool(self.device)
        self.allocator = self.mem_pool
        return self.mem_pool


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
