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
HostModule: compile a source string to a shared library and load it.

Phase 1 of the residency work. This is 'CTypesKernelModule' lifted out of
idpy/CTypes/ with the compiler turned into a parameter, because nothing in that
class was ever CPU-compute-specific -- it hashes a source string, caches the
build, and hands back callables through ctypes. Those are the mechanics any
backend needs for its *host-side capability layer*, not just the one whose
kernels happen to be C.

Why that matters for the design: the same facility compiles the C shims (cuFile,
rocm-xio) and the '@_cdecl' Swift shim for MTLIOCommandQueue. Swift becomes a
compiler choice rather than a language target -- which is exactly why there is no
'SWIFT_T' in idpy_langs_dict. A lang answers "where does the kernel execute and
what drives it"; a shim is fixed host code written once, and belongs here.

Two axes were hard-coded before and are parameters now:

  Toolchain   command + flags, source extension, library extension. CTYPES_T
              passes the C toolchain and behaves exactly as it always did.
  argtypes    GetKernelFunction keeps the numpy-shaped ABI (every pointer
              becomes an ndpointer). GetFunction is the escape hatch: explicit
              ctypes argtypes, no numpy assumption, which is what an opaque
              device handle or a file descriptor needs.

Placement: idpy/Utils/ rather than idpy/IdpyCode/. IdpyCode's __init__ imports
every backend package, so a backend importing back from it risks a cycle; Utils
has no import-time dependency on IdpyCode. Under the STRATEGY.md restructure
both land in idpy.core anyway.
'''

import hashlib
import subprocess
from pathlib import Path

import ctypes
from numpy.ctypeslib import ndpointer

from idpy.Utils.CTypesTypes import CTypesTypes

CTT = CTypesTypes()

'''
Default build cache. Deliberately the same directory CTypesKernelModule always
used, so lifting the class does not silently invalidate anyone's warm cache.
Source and library names are content hashes, and the toolchain's own command
string is part of the library hash, so several toolchains share the directory
without colliding.
'''
idpy_host_cache_dir = Path('/tmp/idpy_ctypes_kernels')


class Toolchain:
    '''
    How to turn a source string into a loadable shared library.

    'command' is the compiler invocation up to but excluding the output and
    input paths, e.g. 'clang -fPIC -shared -std=c99'. It is split on spaces to
    build the argument vector, which is how this has always worked -- and means
    paths containing spaces are not supported. That limitation is inherited
    deliberately rather than fixed here, to keep this a behaviour-preserving
    lift; the cache directory is under /tmp and the names are hashes, so no
    path in play today contains one.
    '''

    def __init__(self, command, source_ext='.c', lib_ext='.so', name=None):
        if not command:
            raise ValueError("Toolchain needs a compiler command")
        self.command = command
        self.source_ext = source_ext
        self.lib_ext = lib_ext
        self.name = name if name is not None else command.split(" ")[0]

    def Head(self, options=''):
        '''
        The part of the command that identifies the build for hashing.

        'options' is concatenated without a separator, matching the original
        CTypesKernelModule, so callers' option strings must carry their own
        leading space. Preserved rather than tidied so existing cache entries
        keep hashing to the same names.
        '''
        return self.command + options

    def Available(self):
        try:
            return subprocess.run(
                [self.command.split(" ")[0], '--version'],
                capture_output=True,
            ).returncode == 0
        except (OSError, ValueError):
            return False

    def __repr__(self):
        return "Toolchain(%r, %r)" % (self.name, self.command)


def CToolchain(compiler_string=None, cache_ext='.c'):
    '''
    The C toolchain CTYPES_T has always used. Imported lazily so that this
    module stays loadable on systems where idpy.CTypes is not importable.
    '''
    if compiler_string is None:
        from idpy.CTypes import idpy_ctypes_compiler_string_h
        compiler_string = idpy_ctypes_compiler_string_h
    return Toolchain(compiler_string, source_ext=cache_ext, lib_ext='.so',
                     name='c')


def SwiftToolchain(extra=''):
    '''
    swiftc as a *compiler*, not as a language target.

    Produces a plain dynamic library from Swift sources exposing C entry points
    via '@_cdecl', which ctypes then loads like any other. This is the path for
    the MTLIOCommandQueue shim: fixed host code, written once, compiled by the
    same facility as the C shims.
    '''
    return Toolchain('swiftc -emit-library' + extra,
                     source_ext='.swift', lib_ext='.dylib', name='swift')


class HostModule:
    '''
    A compiled, cached, loadable module built from a source string.

    The source is hashed to name its file; the source hash plus the toolchain
    command hash name the library, so changing either flags or code produces a
    distinct artifact and a stale build is never picked up. Compilation is
    skipped entirely when the library already exists.
    '''

    def __init__(self, params, code, options, toolchain=None, cache_dir=None):
        if toolchain is None:
            toolchain = CToolchain()
        self.params, self.code, self.options = params, code, options
        self.toolchain = toolchain
        self.cache_dir = Path(cache_dir) if cache_dir is not None \
            else idpy_host_cache_dir

        self.compile_string_head = self.toolchain.Head(self.options)

        self.code_utf = self.code.encode(encoding='UTF-8', errors='strict')
        self.compile_str_utf = self.compile_string_head.encode(
            encoding='UTF-8', errors='strict'
        )

        self.code_hash = self.GetCodeHash()
        self.compile_str_hash = self.GetCompileStrHash()

        self.code_file = self.cache_dir / (
            str(self.code_hash) + self.toolchain.source_ext
        )
        self.is_code_file = self.CheckCodeFile()
        if not self.is_code_file:
            with open(self.code_file, 'w') as _code_file:
                _code_file.write(self.code)

        self.so_file = self.cache_dir / (
            str(self.code_hash) + '_' + str(self.compile_str_hash)
            + self.toolchain.lib_ext
        )
        self.is_so_file = self.CheckSOFile()
        self.compile_status = True if self.is_so_file else self.Compile()

        self.kernel_function = None
        self.argtypes = ()

    # -- build -------------------------------------------------------------
    def Compile(self):
        self.compile_string = (
            self.compile_string_head + " -o "
            + str(self.so_file) + " " + str(self.code_file)
        )
        _compile_tuple = tuple(self.compile_string.split(" "))
        return subprocess.run(_compile_tuple).returncode == 0

    def GetCodeHash(self):
        return hashlib.md5(self.code_utf).hexdigest()

    def GetCompileStrHash(self):
        return hashlib.md5(self.compile_str_utf).hexdigest()

    def CheckCacheDir(self):
        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            return False
        return True

    def CheckCodeFile(self):
        return self.code_file.is_file() if self.CheckCacheDir() else False

    def CheckSOFile(self):
        return self.so_file.is_file() if self.CheckCacheDir() else False

    def Library(self):
        '''Load (once) and return the ctypes handle to the built library.'''
        if self.kernel_function is None:
            if not self.compile_status:
                raise RuntimeError(
                    "compilation failed for %s module %s"
                    % (self.toolchain.name, self.so_file)
                )
            self.kernel_function = ctypes.CDLL(str(self.so_file))
        return self.kernel_function

    # -- entry points ------------------------------------------------------
    def ResolveArgType(self, param, custom_types):
        '''
        Map one declared parameter to a ctypes argtype.

        Pointers normally become ndpointer, which enforces a C-contiguous numpy
        array at call time -- right for kernels operating on host arrays. But a
        capability shim takes things numpy cannot describe: a device pointer as
        an integer handle, a file descriptor, a stream handle. Those declare an
        opaque C type, and an opaque type stays c_void_p instead of being
        wrapped, since ndpointer(c_void_p) is meaningless.
        '''
        _param_nws = param.split(" ")
        _is_pointer = '*' in _param_nws
        _type = _param_nws[0]

        if _type not in list(custom_types.keys()) \
                or _type in list(custom_types.values()):
            _ctype = CTT.C[_type]
        else:
            _ctype = CTT.C[custom_types[_type]]

        if not _is_pointer:
            return _ctype
        if _ctype is ctypes.c_void_p:
            return _ctype
        return ndpointer(_ctype, flags="C_CONTIGUOUS")

    def GetKernelFunction(self, name, custom_types):
        '''
        Numpy-shaped entry point: the ABI idpy kernels have always used.
        '''
        _lib = self.Library()
        self.argtypes = tuple(
            self.ResolveArgType(_param, custom_types) for _param in self.params
        )
        getattr(_lib, name).argtypes = self.argtypes
        return getattr(_lib, name)

    def GetFunction(self, name, argtypes=None, restype=None):
        '''
        Raw entry point: explicit ctypes argtypes, no numpy assumption.

        This is what a capability shim uses. Pass argtypes as ctypes objects
        directly; leave them None to call through with ctypes' defaults.
        '''
        _fn = getattr(self.Library(), name)
        if argtypes is not None:
            _fn.argtypes = tuple(argtypes)
        if restype is not None:
            _fn.restype = restype
        return _fn
