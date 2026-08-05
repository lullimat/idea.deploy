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
__email__ = "matteo.lulli"
__status__ = "Development"

'''
Phase 4: insertion point (2) -- static code linked into the kernel compile unit.

'include_dirs', 'definitions_files' and 'objects_files' were accepted by
IdpyKernel.__init__, type-checked, and then discarded. A caller who passed them
got silence. This is the wiring, and it is as much a bug fix as a feature.

The three mechanisms do genuinely different things and reach different backends,
which is the substance of the phase rather than an implementation detail:

  L1  definitions_files, path      text injected into the compile unit
  L2  definitions_files, function  an IdpyFunction pulled in, per-language
                                   qualifier and all
  L3  definitions_files, kernel    an IdpyKernel injected WITHOUT its preamble,
                                   so its typedefs do not collide with the
                                   host kernel's -- the case the design called
                                   out specifically
  L4  include_dirs                 '-I' search paths; unavailable on Metal
  L5  objects_files                native static linking; CTypes only

L4 and L5 also check the refusals. A mechanism a backend cannot express now
raises NotImplementedError from Code(), before anything is handed to a compiler.
Refusing is the whole point: silently accepting an argument that cannot work is
the behaviour being removed.

Run directly:
    python -m idpy.IdpyCode.test_linkage
'''

import os
import subprocess
import tempfile
from collections import OrderedDict

import numpy as np

from idpy.IdpyCode import (
    IDPY_T, CUDA_T, OCL_T, CTYPES_T, METAL_T,
    idpy_langs_sys, idpy_langs_human_dict, GetTenet,
)
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyFunction
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyCode.IdpyUnroll import _codify_assignment, _array_value
from idpy.Utils.CustomTypes import CustomTypes
from idpy.Utils.TestExit import report_exit as _report_exit

_LANGS = (CUDA_T, OCL_T, METAL_T, CTYPES_T)
_N = 1024
_BLOCK = 128

# A macro rather than a function: valid verbatim on all four backends, so it
# exercises raw text injection without dragging in per-language qualifiers
# (a CUDA device function needs __device__, an OpenCL one does not).
_DEFS_MACRO = "#define IDPY_TRIPLE(X) (3 * (X))\n"

_TYPES = CustomTypes({'FType': 'float'}).Push()


class F_AddSeven(IdpyFunction):
    '''An ordinary IdpyFunction, injected through definitions_files.'''

    def __init__(self, custom_types=None, f_type='FType'):
        IdpyFunction.__init__(self, custom_types=custom_types, f_type=f_type)
        self.params = {'FType x': ['const']}
        self.functions[IDPY_T] = """
        return x + 7;
        """


class K_Donor(IdpyKernel):
    '''dst[i] = src[i] - 1. Injected wholesale into K_Host's compile unit.'''

    def __init__(self, custom_types=None, optimizer_flag=None):
        IdpyKernel.__init__(self, custom_types=custom_types or _TYPES,
                            optimizer_flag=optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {
            'FType * dst': ['global', 'restrict'],
            'FType * src': ['global', 'restrict', 'const'],
        }
        self.kernels[IDPY_T] = "\n" + _codify_assignment(
            _array_value('dst', 'g_tid'), _array_value('src', 'g_tid') + " - 1"
        )


class K_UseMacro(IdpyKernel):
    '''dst[i] = IDPY_TRIPLE(src[i]) -- the macro arrives from a file.'''

    def __init__(self, defs=None, custom_types=None, optimizer_flag=None):
        IdpyKernel.__init__(self, custom_types=custom_types or _TYPES,
                            optimizer_flag=optimizer_flag,
                            definitions_files=defs)
        self.SetCodeFlags('g_tid')
        self.params = {
            'FType * dst': ['global', 'restrict'],
            'FType * src': ['global', 'restrict', 'const'],
        }
        self.kernels[IDPY_T] = "\n" + _codify_assignment(
            _array_value('dst', 'g_tid'),
            "IDPY_TRIPLE(" + _array_value('src', 'g_tid') + ")",
        )


class K_UseFunction(IdpyKernel):
    '''dst[i] = AddSeven(src[i]) -- the function arrives via definitions_files.'''

    def __init__(self, defs=None, custom_types=None, optimizer_flag=None):
        IdpyKernel.__init__(self, custom_types=custom_types or _TYPES,
                            optimizer_flag=optimizer_flag,
                            definitions_files=defs)
        self.SetCodeFlags('g_tid')
        self.params = {
            'FType * dst': ['global', 'restrict'],
            'FType * src': ['global', 'restrict', 'const'],
        }
        self.kernels[IDPY_T] = "\n" + _codify_assignment(
            _array_value('dst', 'g_tid'),
            "F_AddSeven(" + _array_value('src', 'g_tid') + ")",
        )


class K_Host(IdpyKernel):
    '''dst[i] = src[i] * 2, with K_Donor injected alongside it.'''

    def __init__(self, defs=None, custom_types=None, optimizer_flag=None):
        IdpyKernel.__init__(self, custom_types=custom_types or _TYPES,
                            optimizer_flag=optimizer_flag,
                            definitions_files=defs)
        self.SetCodeFlags('g_tid')
        self.params = {
            'FType * dst': ['global', 'restrict'],
            'FType * src': ['global', 'restrict', 'const'],
        }
        self.kernels[IDPY_T] = "\n" + _codify_assignment(
            _array_value('dst', 'g_tid'), _array_value('src', 'g_tid') + " * 2"
        )


def _deploy(tenet, kern, host_in):
    '''Run a two-buffer kernel and return the result.'''
    idea = kern(tenet=tenet, grid=(_N // _BLOCK, 1, 1), block=(_BLOCK, 1, 1))
    src = IdpyMemory.OnDevice(host_in, tenet=tenet)
    dst = IdpyMemory.Zeros(shape=(_N,), dtype=np.float32, tenet=tenet)
    idea.Deploy([dst, src])
    if hasattr(tenet, 'FlushAndWait'):
        tenet.FlushAndWait()
    return np.array(dst.D2H())


def _tenet_params(lang):
    params = {'lang': lang}
    if lang == OCL_T:
        params['cl_kind'] = 'gpu'
    return params


def run_on(lang, tmpdir):
    host_in = np.arange(_N, dtype=np.float32)
    out = OrderedDict()

    # -- refusals are pure codegen: check them before touching a device
    try:
        _k = K_UseMacro(custom_types=_TYPES)
        _k.objects_files = ['/nonexistent.o']
        _k.Code(lang)
        out['L5 refusal'] = 'FAIL: accepted objects_files'
    except NotImplementedError:
        out['L5 refusal'] = 'raised' if lang != CTYPES_T else 'n/a (supported)'
    except Exception as exc:
        out['L5 refusal'] = 'FAIL: %s' % type(exc).__name__
    if lang == CTYPES_T:
        out['L5 refusal'] = 'n/a (supported)'

    try:
        _k = K_UseMacro(custom_types=_TYPES)
        _k.include_dirs = [tmpdir]
        _k.Code(lang)
        out['L4 refusal'] = 'n/a (supported)' if lang != METAL_T \
            else 'FAIL: accepted include_dirs'
    except NotImplementedError:
        out['L4 refusal'] = 'raised'

    tenet = GetTenet(_tenet_params(lang))
    try:
        # -- L1: macro injected from a file path
        _defs_path = os.path.join(tmpdir, 'idpy_defs.h')
        with open(_defs_path, 'w') as _fh:
            _fh.write(_DEFS_MACRO)
        got = _deploy(tenet, K_UseMacro(defs=[_defs_path]), host_in)
        out['L1 err'] = float(np.max(np.abs(got - host_in * 3)))

        # -- L2: an IdpyFunction object injected
        got = _deploy(
            tenet, K_UseFunction(defs=[F_AddSeven(custom_types=_TYPES)]),
            host_in,
        )
        out['L2 err'] = float(np.max(np.abs(got - (host_in + 7))))

        # -- L3: a whole IdpyKernel injected, preamble suppressed
        _donor = K_Donor(custom_types=_TYPES)
        _host = K_Host(defs=[_donor])
        _src = _host.Code(lang)
        out['L3 donor_present'] = 'K_Donor' in _src
        out['L3 single_typedef'] = _src.count('typedef float FType;') == 1
        got = _deploy(tenet, K_Host(defs=[K_Donor(custom_types=_TYPES)]),
                      host_in)
        out['L3 err'] = float(np.max(np.abs(got - host_in * 2)))

        # -- L4: include_dirs, where the backend supports it
        if lang != METAL_T:
            _inc_dir = os.path.join(tmpdir, 'inc')
            os.makedirs(_inc_dir, exist_ok=True)
            with open(os.path.join(_inc_dir, 'idpy_hdr.h'), 'w') as _fh:
                _fh.write("#define IDPY_TRIPLE(X) (3 * (X))\n")

            class K_ViaHeader(K_UseMacro):
                def __init__(self):
                    IdpyKernel.__init__(
                        self, custom_types=_TYPES,
                        headers_files=['idpy_hdr.h'], include_dirs=[_inc_dir],
                    )
                    self.SetCodeFlags('g_tid')
                    self.params = {
                        'FType * dst': ['global', 'restrict'],
                        'FType * src': ['global', 'restrict', 'const'],
                    }
                    self.kernels[IDPY_T] = "\n" + _codify_assignment(
                        _array_value('dst', 'g_tid'),
                        "IDPY_TRIPLE(" + _array_value('src', 'g_tid') + ")",
                    )

            got = _deploy(tenet, K_ViaHeader(), host_in)
            out['L4 err'] = float(np.max(np.abs(got - host_in * 3)))
        else:
            out['L4 err'] = None

        # -- L5: static linking of a native object, CTypes only
        if lang == CTYPES_T:
            out['L5 err'] = l5_objects_files(tenet, tmpdir, host_in)
        else:
            out['L5 err'] = None
    finally:
        if hasattr(tenet, 'End'):
            tenet.End()
    return out


def l5_objects_files(tenet, tmpdir, host_in):
    '''
    Compile a native object outside idpy, then link it into a CTypes kernel.

    This is the mechanism's reason to exist: code that is not generated, not
    injectable as text, and only available as a compiled artifact -- which is
    the shape of every capability shim Phase 3 will need.
    '''
    from idpy.CTypes import idpy_ctypes_compiler_string_h

    _c_path = os.path.join(tmpdir, 'idpy_ext.c')
    _o_path = os.path.join(tmpdir, 'idpy_ext.o')
    with open(_c_path, 'w') as _fh:
        _fh.write("float idpy_ext_half(float x){ return x * 0.5f; }\n")

    _cc = idpy_ctypes_compiler_string_h.split(" ")[0]
    if subprocess.run([_cc, '-fPIC', '-c', '-o', _o_path, _c_path]).returncode:
        raise RuntimeError("could not build the test object file")

    _proto = os.path.join(tmpdir, 'idpy_ext.h')
    with open(_proto, 'w') as _fh:
        _fh.write("float idpy_ext_half(float x);\n")

    class K_Linked(IdpyKernel):
        def __init__(self):
            IdpyKernel.__init__(self, custom_types=_TYPES,
                                definitions_files=[_proto],
                                objects_files=[_o_path])
            self.SetCodeFlags('g_tid')
            self.params = {
                'FType * dst': ['global', 'restrict'],
                'FType * src': ['global', 'restrict', 'const'],
            }
            self.kernels[IDPY_T] = "\n" + _codify_assignment(
                _array_value('dst', 'g_tid'),
                "idpy_ext_half(" + _array_value('src', 'g_tid') + ")",
            )

    got = _deploy(tenet, K_Linked(), host_in)
    return float(np.max(np.abs(got - host_in * 0.5)))


def main():
    print("=== Phase 4: linking static code into the kernel compile unit ===\n")
    _ok, _ran = True, False
    with tempfile.TemporaryDirectory() as tmpdir:
        for lang in _LANGS:
            human = idpy_langs_human_dict[lang]
            if not idpy_langs_sys[lang]:
                print(f"  [skip] {human}: backend not available\n")
                continue
            try:
                r = run_on(lang, tmpdir)
            except Exception as exc:
                _ok = False
                print(f"  [err ] {human}: {type(exc).__name__}: {exc}\n")
                continue
            _ran = True

            _errs = [r[k] for k in ('L1 err', 'L2 err', 'L3 err', 'L4 err',
                                    'L5 err') if r[k] is not None]
            ok = (all(e == 0.0 for e in _errs)
                  and r['L3 donor_present'] and r['L3 single_typedef']
                  and not str(r['L4 refusal']).startswith('FAIL')
                  and not str(r['L5 refusal']).startswith('FAIL'))
            print(f"  {human}:")
            print(f"    L1 defs from path       max|out-ref| = {r['L1 err']:g}")
            print(f"    L2 defs from function   max|out-ref| = {r['L2 err']:g}")
            print(f"    L3 defs from kernel     max|out-ref| = {r['L3 err']:g}, "
                  f"donor present {r['L3 donor_present']}, "
                  f"one typedef {r['L3 single_typedef']}")
            print(f"    L4 include_dirs         "
                  + (f"max|out-ref| = {r['L4 err']:g}" if r['L4 err'] is not None
                     else "unsupported") + f", refusal: {r['L4 refusal']}")
            print(f"    L5 objects_files        "
                  + (f"max|out-ref| = {r['L5 err']:g}" if r['L5 err'] is not None
                     else "unsupported") + f", refusal: {r['L5 refusal']}")
            print(f"    -> {'OK' if ok else 'FAIL'}\n")
            _ok = _ok and ok

    print(
        "Where a mechanism has no meaning on a backend it now raises from\n"
        "Code(), before anything reaches a compiler. That refusal is the point:\n"
        "these three parameters used to be validated and then discarded."
    )

    _report_exit(_ok, checks_run=_ran, what='backends')


if __name__ == '__main__':
    main()
