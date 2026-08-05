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
Type-aware constant emission.

A bare Python float reaches the generated source as a bare literal, which is a
*double* in C. Any surrounding fp32 arithmetic promotes -- measured at ~208x
slower on a GeForce part, and worse than slow: the same kernel then computes
fp64 intermediates on CUDA and fp32 on Apple, where there is no fp64 to promote
to. That is a silent cross-backend numerical divergence, and it is exactly what
STRATEGY.md's "verified-identical results across backends" criterion forbids.

Precision now comes from, in order:

  1. constants_types={'X': 'FType'}   resolved through custom_types at EMISSION
                                      time, so it follows the runtime
                                      double->float downcast CheckOCLFP and
                                      CheckMetalFP apply on devices without
                                      fp64. Right for physics constants, which
                                      should match the precision of the arrays
                                      they are used with.
  2. constants['X'] = np.float32(v)   pinned to fp32 on every device. Right for
                                      algorithmic constants that must not
                                      change with the hardware.
  3. neither                          C's default (double) applies, and an
                                      IdpyConstantPrecisionWarning fires.

The two declarations mean genuinely different things, which is why both exist:
one tracks the kernel's working precision, the other overrides it. Explicitness
by either route silences the warning -- np.float64 states 'double' as clearly as
np.float32 states 'float'.

This is pure codegen, so it needs no device and runs anywhere.

Run directly:
    python -m idpy.IdpyCode.test_constants
'''

import warnings
from collections import OrderedDict

import numpy as np

from idpy.IdpyCode import IDPY_T, CUDA_T, METAL_T
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyConstantPrecisionWarning
from idpy.Utils.CustomTypes import CustomTypes


def _kernel(constants, constants_types=None, ftype='float'):
    class K_Const(IdpyKernel):
        def __init__(self):
            IdpyKernel.__init__(
                self, custom_types=CustomTypes({'FType': ftype}).Push(),
                constants=constants, constants_types=constants_types or {},
            )
            self.SetCodeFlags('g_tid')
            self.params = {'FType * a': ['global']}
            self.kernels[IDPY_T] = "\na[g_tid] = A;\n"
    return K_Const()


def _define(kern, lang=CUDA_T):
    for _line in kern.Code(lang).splitlines():
        if _line.startswith('#define A '):
            return _line.split(' ', 2)[2]
    return None


def cases():
    _c = OrderedDict()

    # -- alias resolution tracks custom_types, including a runtime downcast
    _c['alias on an fp32 kernel'] = (
        _define(_kernel({'A': 0.99999}, {'A': 'FType'}, ftype='float')),
        '0.99999f',
    )
    _c['alias on an fp64 kernel'] = (
        _define(_kernel({'A': 0.99999}, {'A': 'FType'}, ftype='double')),
        '0.99999',
    )

    # -- the value's own dtype pins precision regardless of the kernel
    _c['np.float32 pins fp32'] = (
        _define(_kernel({'A': np.float32(0.99999)})), '0.99999f',
    )
    _c['np.float64 stays double'] = (
        _define(_kernel({'A': np.float64(0.99999)})), '0.99999',
    )

    # -- str passes through verbatim: the escape hatch for an exact literal
    _c['str escape hatch'] = (_define(_kernel({'A': '0.5f'})), '0.5f')

    # -- integers unchanged, and the unsigned suffix now reaches numpy ints
    _c['int unchanged'] = (_define(_kernel({'A': 7})), '7')
    _c['python int > int64max'] = (
        _define(_kernel({'A': 2 ** 64 - 1})), '18446744073709551615ULL',
    )
    _c['numpy uint64 > int64max'] = (
        _define(_kernel({'A': np.uint64(2 ** 64 - 1)})),
        '18446744073709551615ULL',
    )
    _c['metal uses UL not ULL'] = (
        _define(_kernel({'A': 2 ** 64 - 1}), METAL_T),
        '18446744073709551615UL',
    )

    # -- unchanged default: a bare float is still a double literal
    _c['bare float unchanged'] = (_define(_kernel({'A': 0.99999})), '0.99999')
    return _c


def warning_behaviour():
    '''Only a bare Python float is unsaid; every explicit form is silent.'''
    _out = OrderedDict()
    for _name, _kern in (
        ('bare float warns', lambda: _kernel({'A': 0.5})),
        ('np.float32 silent', lambda: _kernel({'A': np.float32(0.5)})),
        ('np.float64 silent', lambda: _kernel({'A': np.float64(0.5)})),
        ('declared silent', lambda: _kernel({'A': 0.5}, {'A': 'FType'})),
        ('int silent', lambda: _kernel({'A': 5})),
    ):
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            _kern().Code(CUDA_T)
            _out[_name] = any(
                issubclass(_x.category, IdpyConstantPrecisionWarning)
                for _x in _w
            )
    return _out


def main():
    print("=== Type-aware constant emission ===\n")
    _ok = True

    for _name, (_got, _want) in cases().items():
        _pass = (_got == _want)
        _ok = _ok and _pass
        print(f"  [{'OK  ' if _pass else 'FAIL'}] {_name:28} -> {_got!r}"
              + ('' if _pass else f"   (expected {_want!r})"))

    print()
    _expect_warn = {
        'bare float warns': True, 'np.float32 silent': False,
        'np.float64 silent': False, 'declared silent': False,
        'int silent': False,
    }
    for _name, _warned in warning_behaviour().items():
        _pass = (_warned == _expect_warn[_name])
        _ok = _ok and _pass
        print(f"  [{'OK  ' if _pass else 'FAIL'}] {_name:28} -> "
              f"warned={_warned}")

    print(f"\n  -> {'OK' if _ok else 'FAIL'}")
    print(
        "\nThe two declarations are not interchangeable: an alias tracks the\n"
        "kernel's working precision through the double->float downcast applied\n"
        "on devices without fp64, while a numpy scalar pins precision against\n"
        "it. Physics constants usually want the first, algorithmic ones the\n"
        "second."
    )


if __name__ == '__main__':
    main()
