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
Phase 1 acceptance: HostModule compiles and loads more than C.

The design's criterion is "existing tests pass unchanged; a trivial non-C
(Swift) shim compiles and loads on macOS". The first half is the whole existing
suite, which exercises the C path through CTYPES_T on every run. This file
covers the second half plus the parts of the lift that the CTYPES_T path never
touches.

  H1  C toolchain              a function compiles, loads and computes
  H2  build cache              a second identical build reuses the artifact,
                               and changed flags produce a different one
  H3  opaque argtypes          a pointer declared as an opaque handle becomes
                               c_void_p rather than an ndpointer
  H4  raw entry point          GetFunction takes explicit ctypes argtypes with
                               no numpy involvement
  H5  Swift shim (macOS)       swiftc builds a @_cdecl library that ctypes
                               loads and calls -- the acceptance criterion

H3 and H5 are the two that matter for what comes next. Together they are the
claim that Swift is a *compiler choice* rather than a language target: the same
facility that builds CTYPES_T's C kernels builds a Swift shim exposing C entry
points, so binding MTLIOCommandQueue needs no new lang in idpy_langs_dict.

Run directly:
    python -m idpy.Utils.test_hostmodule
'''

import ctypes
from collections import OrderedDict

import numpy as np

from idpy import idpy_os_found
from idpy.Utils.TestExit import report_exit as _report_exit
from idpy.Utils.HostModule import (
    HostModule, Toolchain, CToolchain, SwiftToolchain,
)

_C_SOURCE = '''
#include <stddef.h>
#include <stdint.h>

void idpy_scale(double * out, const double * in, int n, double factor){
    for(int i = 0; i < n; i++){ out[i] = in[i] * factor; }
}

/* Takes a raw address rather than an array: the shape a capability shim has.
   Converts through uintptr_t, which is the defined way to turn a pointer into
   an integer -- subtracting (char *)0 is undefined behaviour and a compiler is
   entitled to fold it. */
size_t idpy_probe_handle(void * handle, size_t offset){
    return (size_t)(uintptr_t)handle + offset;
}
'''

_SWIFT_SOURCE = '''
@_cdecl("idpy_swift_fma")
public func idpy_swift_fma(_ a: Int32, _ b: Int32, _ c: Int32) -> Int32 {
    return a * b + c
}
'''


def h1_c_toolchain():
    '''A C function compiles, loads and computes.'''
    params = OrderedDict([
        ('double * out', None), ('double * in', None),
        ('int n', None), ('double factor', None),
    ])
    mod = HostModule(params, _C_SOURCE, '', toolchain=CToolchain())
    if not mod.compile_status:
        raise RuntimeError("C module failed to compile")

    fn = mod.GetKernelFunction('idpy_scale', {})
    n = 1024
    src = np.arange(n, dtype=np.float64)
    dst = np.zeros(n, dtype=np.float64)
    fn(dst, src, n, 2.5)
    return float(np.max(np.abs(dst - src * 2.5)))


def h2_build_cache():
    '''Identical source+flags reuse the artifact; different flags do not.'''
    params = OrderedDict([('double * out', None)])
    a = HostModule(params, _C_SOURCE, '', toolchain=CToolchain())
    b = HostModule(params, _C_SOURCE, '', toolchain=CToolchain())
    c = HostModule(params, _C_SOURCE, ' -O1', toolchain=CToolchain())
    return {
        'same_artifact': a.so_file == b.so_file,
        'reused': b.is_so_file,
        'flags_change_artifact': a.so_file != c.so_file,
        'same_source_file': a.code_file == c.code_file,
    }


def h3_opaque_argtypes():
    '''
    A pointer declared opaque must stay c_void_p.

    ndpointer would demand a C-contiguous numpy array at call time, which is
    exactly wrong for a device pointer or a queue handle -- there is no host
    array behind them. This is the argtype path the old CTypesKernelModule
    could not express.
    '''
    params = OrderedDict([('void * handle', None), ('size_t offset', None)])
    mod = HostModule(params, _C_SOURCE, '', toolchain=CToolchain())

    resolved = [mod.ResolveArgType(p, {}) for p in params]
    ok_types = (resolved[0] is ctypes.c_void_p
                and resolved[1] is ctypes.c_size_t)

    # and it must actually be callable with a raw address
    fn = mod.GetKernelFunction('idpy_probe_handle', {})
    fn.restype = ctypes.c_size_t
    buf = ctypes.create_string_buffer(16)
    addr = ctypes.cast(buf, ctypes.c_void_p).value
    got = fn(ctypes.c_void_p(addr), ctypes.c_size_t(8))
    return ok_types, (got == addr + 8)


def h4_raw_entry_point():
    '''GetFunction: explicit ctypes argtypes, no numpy in sight.'''
    mod = HostModule({}, _C_SOURCE, '', toolchain=CToolchain())
    fn = mod.GetFunction(
        'idpy_probe_handle',
        argtypes=(ctypes.c_void_p, ctypes.c_size_t),
        restype=ctypes.c_size_t,
    )
    buf = ctypes.create_string_buffer(8)
    addr = ctypes.cast(buf, ctypes.c_void_p).value
    return fn(ctypes.c_void_p(addr), ctypes.c_size_t(4)) == addr + 4


def h5_swift_shim():
    '''
    The acceptance criterion: a non-C source compiles and loads through the
    same facility, reached by swapping the toolchain and nothing else.
    '''
    tc = SwiftToolchain()
    if not tc.Available():
        return None
    mod = HostModule({}, _SWIFT_SOURCE, '', toolchain=tc)
    if not mod.compile_status:
        raise RuntimeError("swiftc failed to build the shim")
    fn = mod.GetFunction(
        'idpy_swift_fma',
        argtypes=(ctypes.c_int32,) * 3, restype=ctypes.c_int32,
    )
    return fn(6, 7, 5) == 47


def main():
    print("=== Phase 1: HostModule ===\n")

    err = h1_c_toolchain()
    print(f"  H1 C toolchain        max|out-ref| = {err:g}"
          f"   -> {'OK' if err == 0.0 else 'FAIL'}")

    c = h2_build_cache()
    ok2 = all(c.values())
    print(f"  H2 build cache        same artifact for identical builds: "
          f"{c['same_artifact']}, reused: {c['reused']},")
    print(f"                        flags change the library: "
          f"{c['flags_change_artifact']}, source shared: "
          f"{c['same_source_file']}   -> {'OK' if ok2 else 'FAIL'}")

    ok_types, ok_call = h3_opaque_argtypes()
    print(f"  H3 opaque argtypes    void*/size_t resolve to c_void_p/c_size_t: "
          f"{ok_types}, call correct: {ok_call}"
          f"   -> {'OK' if (ok_types and ok_call) else 'FAIL'}")

    ok4 = h4_raw_entry_point()
    print(f"  H4 raw entry point    GetFunction with explicit argtypes: {ok4}"
          f"   -> {'OK' if ok4 else 'FAIL'}")

    ok5 = h5_swift_shim()
    if ok5 is None:
        print(f"  H5 Swift shim         [skip] swiftc not available "
              f"(os: {idpy_os_found})")
    else:
        print(f"  H5 Swift shim         @_cdecl library built by swiftc, "
              f"loaded and called: {ok5}   -> {'OK' if ok5 else 'FAIL'}")

    _ok = (err == 0.0 and ok2 and ok_types and ok_call and ok4
           and (ok5 is None or ok5))
    print(
        "\nH3 and H5 together are the claim that Swift is a compiler choice and\n"
        "not a language target: one facility builds both CTYPES_T's C kernels\n"
        "and a Swift shim exposing C entry points, so binding MTLIOCommandQueue\n"
        "needs no new entry in idpy_langs_dict."
    )
    _report_exit(_ok, checks_run=True)


if __name__ == '__main__':
    main()
