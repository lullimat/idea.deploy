"""
Periodic stencil convolution via IdpyKernel metaprogramming.

Lives under IdpyStencils alongside GradientCode / StreamingCode.
Uses the same IdpyUnroll index machinery (_sp/_sm neighbor coords, lex rebuild)
but accumulates hardcoded tap coefficients:
    dst[g_tid] += c_xi * src[n_xi]
"""

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

from functools import reduce
from collections import OrderedDict

import numpy as np

from idpy.IdpyCode import IDPY_T
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyFunction
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyCode.IdpyUnroll import (
    _get_cartesian_coordinates_macro,
    _get_single_neighbor_pos_macro_fully_sym,
    _get_seq_macros,
    _sp_macro,
    _sm_macro,
    _codify_declaration_const_check,
    _codify_sympy_assignment,
    _codify_comment,
    _codify_newl,
    _array_value,
)
from idpy.Utils.CustomTypes import CustomTypes
from idpy.Utils.NpTypes import NpTypes

_NPT = NpTypes()

# Session Tenet for Deploy (same role as GetTenet usage in sims)
_active_tenet = None
_active_shape = None
_idea_cache = {}


def set_active_tenet(tenet):
    global _active_tenet
    _active_tenet = tenet
    return tenet


def get_active_tenet():
    return _active_tenet


def clear_active_tenet():
    global _active_tenet
    _active_tenet = None


def set_convolution_shape(shape):
    """Logical multi-d shape when device buffers are flat (length V)."""
    global _active_shape
    _active_shape = tuple(int(s) for s in shape) if shape is not None else None
    return _active_shape


def get_convolution_shape():
    return _active_shape


def clear_convolution_shape():
    global _active_shape
    _active_shape = None


class F_PosFromIndex(IdpyFunction):
    def __init__(self, custom_types=None, f_type='void'):
        IdpyFunction.__init__(self, custom_types=custom_types, f_type=f_type)
        self.params = {
            'SType * pos': [],
            'SType * dim_sizes': ['global', 'const'],
            'SType * dim_strides': ['global', 'const'],
            'unsigned int index': ['const'],
        }
        self.functions[IDPY_T] = """
        pos[0] = index % dim_strides[0];
        for(int d=1; d<DIM; d++){
            pos[d] = (index / dim_strides[d - 1]) % dim_sizes[d];
        }
        return;
        """


class F_IndexFromPos(IdpyFunction):
    def __init__(self, custom_types=None, f_type='SType'):
        IdpyFunction.__init__(self, custom_types=custom_types, f_type=f_type)
        self.params = {
            'SType * pos': ['const'],
            'SType * dim_strides': ['global', 'const'],
        }
        self.functions[IDPY_T] = """
        SType index = pos[0];
        for(int d=1; d<DIM; d++){
            index += pos[d] * dim_strides[d - 1];
        }
        return index;
        """


def _dim_sizes_strides(shape):
    """LBM-style sizes/strides: first axis varies fastest (numpy order='F')."""
    dim_sizes = [int(s) for s in shape]
    dim = len(dim_sizes)
    if dim < 1:
        raise ValueError("shape must be non-empty")
    if dim == 1:
        dim_strides = list(dim_sizes)
    else:
        dim_strides = [
            int(reduce(lambda x, y: x * y, dim_sizes[0:i + 1]))
            for i in range(dim - 1)
        ]
    V = int(reduce(lambda x, y: x * y, dim_sizes))
    return dim, dim_sizes, dim_strides, V


def _taps_fingerprint(offsets, coeffs):
    off = np.asarray(offsets, dtype=np.int64)
    cf = np.asarray(coeffs)
    return (off.tobytes(), cf.dtype.str, cf.tobytes())


def _normalize_taps(offsets, coeffs):
    offsets = np.asarray(offsets, dtype=np.int64)
    coeffs = np.asarray(coeffs)
    if offsets.size == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0,), dtype=coeffs.dtype)
    if offsets.ndim == 1:
        offsets = offsets.reshape(-1, 1)
    if len(coeffs) != len(offsets):
        raise ValueError("offsets and coeffs length mismatch")
    return offsets, coeffs


def _gather_vectors_from_roll_offsets(offsets):
    """
    np.roll(b, shift=off) contributes b[x - off] at x.
    Neighbor macros use position x + xi, so xi = -off.
    """
    return [tuple(int(-x) for x in off) for off in offsets]


def _define_neighbor_coords(xis, declared_variables, declared_constants,
                            pos_type='SType', root_dim_sizes='L',
                            root_coord='x', declare_const_flag=True):
    """Predeclare x_d_pδ / x_d_mδ with periodic _sp/_sm macros (GradientCode style)."""
    if not xis:
        return ""
    dim = len(xis[0])
    dim_sizes_macros = _get_seq_macros(dim, root_dim_sizes)
    largest = 0
    for xi in xis:
        for c in xi:
            largest = max(largest, abs(int(c)))

    code = ""
    for delta in range(1, largest + 1):
        for d in range(dim):
            sp = _sp_macro(root_coord + '_' + str(d), str(delta), dim_sizes_macros[d])
            sm = _sm_macro(root_coord + '_' + str(d), str(delta), dim_sizes_macros[d])
            code += _codify_declaration_const_check(
                root_coord + '_' + str(d) + '_p' + str(delta),
                sp, pos_type, declared_variables, declared_constants, declare_const_flag,
            )
            code += _codify_declaration_const_check(
                root_coord + '_' + str(d) + '_m' + str(delta),
                sm, pos_type, declared_variables, declared_constants, declare_const_flag,
            )
    code += _codify_newl
    return code


def ConvolveCode(offsets, coeffs, declared_variables=None, declared_constants=None,
                 src_var='src', dst_var='dst', src_type='FType',
                 pos_type='SType', use_ptrs=False,
                 root_dim_sizes='L', root_strides='STR',
                 root_coord='x', lex_index='g_tid',
                 declare_const_dict=None):
    """
    Emit meta-code: cartesian coords, neighbor wraps, unrolled
    dst[g_tid] = sum_i c_i * src[n(x + xi_i)] with xi = -roll_offset.
    """
    if declared_variables is None or declared_constants is None:
        raise Exception("declared_variables and declared_constants are required")
    if declare_const_dict is None:
        declare_const_dict = {'cartesian_coords': True, 'cartesian_coord_neigh': True}

    offsets, coeffs = _normalize_taps(offsets, coeffs)
    if offsets.shape[0] == 0:
        return (
            _codify_comment("empty stencil")
            + _array_value(dst_var, lex_index, use_ptrs) + " = 0;\n"
        )

    dim = offsets.shape[1]
    xis = _gather_vectors_from_roll_offsets(offsets)
    dim_sizes_macros = _get_seq_macros(dim, root_dim_sizes)
    dim_strides_macros = _get_seq_macros(max(dim - 1, 1), root_strides)
    # For dim==1 LBM uses a single stride macro equal to L_0
    if dim == 1:
        dim_strides_macros = _get_seq_macros(1, root_strides)

    code = ""
    code += _get_cartesian_coordinates_macro(
        declared_variables, declared_constants,
        root_coord, lex_index,
        dim_sizes_macros, dim_strides_macros,
        _type=pos_type,
        declare_const_flag=declare_const_dict.get('cartesian_coords', True),
    )
    code += _codify_newl
    code += _define_neighbor_coords(
        xis, declared_variables, declared_constants,
        pos_type=pos_type, root_dim_sizes=root_dim_sizes,
        root_coord=root_coord,
        declare_const_flag=declare_const_dict.get('cartesian_coord_neigh', True),
    )

    acc = 'conv_acc'
    code += _codify_declaration_const_check(
        acc, 0, src_type, declared_variables, declared_constants, False,
    )
    code += _codify_declaration_const_check(
        'n_' + root_coord, 0, pos_type, declared_variables, declared_constants, False,
    )
    code += _codify_newl

    for xi, c in zip(xis, coeffs):
        if abs(float(c)) == 0.0:
            continue
        code += _codify_comment("tap xi=" + str(xi) + " coeff=" + str(c))
        if all(int(v) == 0 for v in xi):
            code += _codify_sympy_assignment('n_' + root_coord, lex_index)
        else:
            n_expr = _get_single_neighbor_pos_macro_fully_sym(
                xi, dim_sizes_macros, dim_strides_macros, root_coord, lex_index,
            )
            code += _codify_sympy_assignment('n_' + root_coord, n_expr)

        c_lit = np.format_float_positional(np.asarray(c).item(), unique=True, trim='k')
        src_hnd = _array_value(src_var, 'n_' + root_coord, use_ptrs)
        code += (
            acc + " += ((" + src_type + ")(" + c_lit + ")) * " + src_hnd + ";\n"
        )
        code += _codify_newl

    dst_hnd = _array_value(dst_var, lex_index, use_ptrs)
    code += _codify_assignment_safe(dst_hnd, acc)
    return code


def _codify_assignment_safe(lhs, rhs):
    return str(lhs) + " = " + str(rhs) + ";\n"


class K_ConvolvePeriodic(IdpyKernel):
    """
    Meta-generated periodic convolution. Tap coefficients are literals in IDPY_T.
    """

    def __init__(self, offsets, coeffs, shape,
                 custom_types=None, optimizer_flag=None,
                 root_dim_sizes='L', root_strides='STR',
                 root_coord='x'):
        offsets, coeffs = _normalize_taps(offsets, coeffs)
        dim, dim_sizes, dim_strides, V = _dim_sizes_strides(shape)
        if offsets.shape[0] and offsets.shape[1] != dim:
            raise ValueError(
                f"tap dim {offsets.shape[1]} != shape dim {dim}"
            )

        if custom_types is None:
            custom_types = CustomTypes({
                'FType': 'double',
                'SType': 'int',
            }).Push()

        constants = OrderedDict()
        constants['V'] = V
        constants['DIM'] = dim
        size_macros = _get_seq_macros(dim, root_dim_sizes)
        for i, L in enumerate(dim_sizes):
            constants[size_macros[i]] = int(L)
        stride_macros = _get_seq_macros(len(dim_strides), root_strides)
        for i, S in enumerate(dim_strides):
            constants[stride_macros[i]] = int(S)

        IdpyKernel.__init__(
            self,
            custom_types=custom_types,
            constants=constants,
            f_classes=[],
            optimizer_flag=optimizer_flag,
        )
        self.SetCodeFlags('g_tid')
        self.params = {
            'FType * dst': ['global', 'restrict'],
            'FType * src': ['global', 'restrict', 'const'],
        }
        self.offsets = offsets
        self.coeffs = coeffs
        self.shape = tuple(shape)
        self.root_dim_sizes = root_dim_sizes
        self.root_strides = root_strides
        self.root_coord = root_coord

        declared_variables = [[]]
        declared_constants = [[]]
        # Size/stride macros must appear as declared constants for neighbor codegen checks
        for name in list(constants.keys()):
            if name.startswith(root_dim_sizes) or name.startswith(root_strides):
                declared_constants[0].append(name)

        body = ConvolveCode(
            offsets, coeffs,
            declared_variables=declared_variables,
            declared_constants=declared_constants,
            src_var='src', dst_var='dst',
            src_type='FType', pos_type='SType',
            root_dim_sizes=root_dim_sizes,
            root_strides=root_strides,
            root_coord=root_coord,
            lex_index='g_tid',
        )
        self.kernels[IDPY_T] = (
            "\n        if(g_tid < V){\n"
            + body
            + "\n        }\n"
        )

    def dump_code(self, lang=None):
        if lang is None:
            lang = IDPY_T
        return self.Code(lang)


def convolve_periodic(src, offsets, coeffs, shape=None, tenet=None,
                      custom_types=None, block_size=128):
    """
    Apply meta-generated periodic convolution on a device (or CTypes) array.

    Memory layout must be first-axis-fastest (Fortran order), matching Idpy lex macros.
    `src` may be multi-d (F-contiguous) or flat with explicit `shape`.
    """
    if tenet is None:
        tenet = get_active_tenet()
    if tenet is None:
        raise RuntimeError(
            "convolve_periodic requires an active Tenet "
            "(call set_active_tenet(tenet) or pass tenet=...)"
        )

    offsets, coeffs = _normalize_taps(offsets, coeffs)

    if shape is None:
        if getattr(src, 'ndim', 1) >= 2:
            shape = tuple(int(s) for s in src.shape)
        else:
            shape = get_convolution_shape()
            if shape is None:
                raise ValueError(
                    "shape is required for flat src "
                    "(pass shape=... or set_convolution_shape)"
                )
    else:
        shape = tuple(int(s) for s in shape)

    dim, dim_sizes, dim_strides, V = _dim_sizes_strides(shape)
    if int(np.prod(getattr(src, 'shape', ()))) != V:
        raise ValueError(
            f"src size {np.prod(src.shape)} != volume {V} for shape {shape}"
        )

    if custom_types is None:
        src_dtype = getattr(src, 'dtype', np.float64)
        f_name = 'float' if src_dtype == np.float32 else 'double'
        custom_types = CustomTypes({'FType': f_name, 'SType': 'int'}).Push()

    lang = tenet.GetLang()
    cache_key = (
        lang, shape, str(custom_types), _taps_fingerprint(offsets, coeffs),
    )
    cached = _idea_cache.get(cache_key)
    if cached is None:
        kern = K_ConvolvePeriodic(
            offsets, coeffs, shape, custom_types=custom_types,
        )
        grid = ((V + block_size - 1) // block_size, 1, 1)
        block = (block_size, 1, 1)
        idea = kern(tenet=tenet, grid=grid, block=block)
        _idea_cache[cache_key] = (idea, kern)
    else:
        idea, kern = cached

    dst = IdpyMemory.Zeros(shape=(V,), dtype=src.dtype, tenet=tenet)
    if getattr(src, 'ndim', 1) == 1:
        src_flat = src
    else:
        host = np.asfortranarray(np.asarray(src.D2H() if hasattr(src, 'D2H') else src))
        src_flat = IdpyMemory.OnDevice(host.ravel(order='F'), tenet=tenet)

    idea.Deploy([dst, src_flat])

    if getattr(src, 'ndim', 1) >= 2:
        host_out = np.asarray(dst.D2H() if hasattr(dst, 'D2H') else dst)
        host_out = np.asfortranarray(host_out.reshape(shape, order='F'))
        return IdpyMemory.OnDevice(host_out, tenet=tenet)
    return dst


def clear_idea_cache():
    _idea_cache.clear()
