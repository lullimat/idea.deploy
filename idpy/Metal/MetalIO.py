__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2026 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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
MTLIOCommandQueue binding: storage -> device on Metal (Phase 3).

This is the case the whole HostModule design was aimed at. Metal's storage API
has no Python binding and pymetallic does not wrap it, so reaching it needs
compiled host code -- and Swift is the only language that can see the API. That
does NOT make Swift a kernel language: nothing here is generated per kernel, it
is fixed host code written once. Swift is a *compiler choice*, which is why
there is no SWIFT_T in idpy_langs_dict and why this file is a binding rather
than a backend.

The mechanics, all of which had to be verified rather than assumed:

- pymetallic exposes its MTLDevice and MTLBuffer as raw pointers
  (`_device_ptr`, `_buffer_ptr`). Swift reconstitutes them with
  Unmanaged.fromOpaque(...).takeUnretainedValue(), which works: the probe
  created an MTLIOCommandQueue from pymetallic's own device.
- The load writes into a byte offset of an existing buffer, so it can target a
  SubView of the residency cache directly rather than a whole allocation. That
  is what makes it usable as a cache-slot filler; the offset bookkeeping added
  for Metal in Phase 2b is exactly what supplies the destination.
- The Swift API is `load(_:offset:size:sourceHandle:sourceHandleOffset:)`.
  `loadBuffer(...)` is the Objective-C spelling and was obsoleted in Swift 3;
  the compiler says so plainly if you get it wrong.

Compilation goes through HostModule, whose md5 cache means the shim is built
once per source+flags and loaded from /tmp thereafter.
'''

import ctypes

from idpy import idpy_os_found
from idpy.Utils.HostModule import HostModule, SwiftToolchain

_SWIFT_SOURCE = r'''
import Metal
import Foundation

// Queue plus file handle, kept together for the lifetime of a store. Both are
// expensive to build relative to a single block read, so they are created once
// per file and reused for every load.
final class IdpyMetalIO {
    let queue: MTLIOCommandQueue
    let handle: MTLIOFileHandle
    init(device: MTLDevice, path: String) throws {
        self.queue = try device.makeIOCommandQueue(
            descriptor: MTLIOCommandQueueDescriptor())
        self.handle = try device.makeIOHandle(url: URL(fileURLWithPath: path))
    }
}

@_cdecl("idpy_metal_io_open")
public func idpy_metal_io_open(_ devicePtr: UnsafeRawPointer?,
                               _ path: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
    guard let dp = devicePtr, let cpath = path else { return nil }
    guard let device = Unmanaged<AnyObject>.fromOpaque(dp).takeUnretainedValue() as? MTLDevice
    else { return nil }
    do {
        return Unmanaged.passRetained(
            try IdpyMetalIO(device: device, path: String(cString: cpath))).toOpaque()
    } catch {
        return nil
    }
}

// Returns 1 on success, a negative code for a bad argument, or 3000+status
// when the command buffer completed in a non-complete state -- distinguishable
// from "declined" so a caller can tell a refusal from a failure.
@_cdecl("idpy_metal_io_load")
public func idpy_metal_io_load(_ ioPtr: UnsafeMutableRawPointer?,
                               _ bufferPtr: UnsafeRawPointer?,
                               _ bufferOffset: Int, _ size: Int,
                               _ fileOffset: Int) -> Int32 {
    guard let ip = ioPtr, let bp = bufferPtr else { return -1 }
    let io = Unmanaged<IdpyMetalIO>.fromOpaque(ip).takeUnretainedValue()
    guard let buffer = Unmanaged<AnyObject>.fromOpaque(bp).takeUnretainedValue() as? MTLBuffer
    else { return -2 }
    let cb = io.queue.makeCommandBuffer()
    cb.load(buffer, offset: bufferOffset, size: size,
            sourceHandle: io.handle, sourceHandleOffset: fileOffset)
    cb.commit()
    cb.waitUntilCompleted()
    return cb.status == .complete ? 1 : Int32(3000 + cb.status.rawValue)
}

@_cdecl("idpy_metal_io_close")
public func idpy_metal_io_close(_ ioPtr: UnsafeMutableRawPointer?) {
    guard let ip = ioPtr else { return }
    Unmanaged<IdpyMetalIO>.fromOpaque(ip).release()
}
'''

_SHIM = None
_SHIM_FAILED = False


def Shim():
    '''
    Build (once) and return the loaded shim, or None where it cannot exist.

    None rather than an exception: a store asks for this and falls back to the
    staged path when it is unavailable, which is an ordinary configuration
    outcome on any non-Darwin machine or one without a Swift toolchain.
    '''
    global _SHIM, _SHIM_FAILED
    if _SHIM is not None or _SHIM_FAILED:
        return _SHIM
    if idpy_os_found != 'darwin':
        _SHIM_FAILED = True
        return None

    _toolchain = SwiftToolchain(extra=' -framework Metal -framework Foundation')
    if not _toolchain.Available():
        _SHIM_FAILED = True
        return None

    try:
        _module = HostModule({}, _SWIFT_SOURCE, '', toolchain=_toolchain)
        if not _module.compile_status:
            _SHIM_FAILED = True
            return None
        _SHIM = {
            'open': _module.GetFunction(
                'idpy_metal_io_open',
                argtypes=(ctypes.c_void_p, ctypes.c_char_p),
                restype=ctypes.c_void_p),
            'load': _module.GetFunction(
                'idpy_metal_io_load',
                argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t,
                          ctypes.c_ssize_t, ctypes.c_ssize_t),
                restype=ctypes.c_int32),
            'close': _module.GetFunction(
                'idpy_metal_io_close', argtypes=(ctypes.c_void_p,)),
        }
    except Exception:
        _SHIM_FAILED = True
        return None
    return _SHIM
