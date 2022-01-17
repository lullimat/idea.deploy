__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2021 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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

from idpy.Utils.Statements import AllTrue

'''
This module provides the basic functions to generate unrolled code in the kernels
'''

_sp_macro = lambda _x, _m : "(" + str(_x) + "&(~(-(" + str(_x) + " >= " + str(_m) + "))))"
_sm_macro = lambda _x, _m : "(" + str(_x) + "+((-(" + str(_x) + " < 0))&" + str(_m) + "))"

'''
Simple wrapper functions for code generation
'''
def _codify(_expr):
    return str(_expr)

_codify_newl = '\n'

def _codify_comment(_expr):
    return '// ' + _codify(_expr) + '\n'

def _codify_assignment(_var_str, _expr):
    _macro = """"""
    _macro += _var_str + " = " + _codify(_expr) + ";\n"
    return _macro

def _codify_add_assignment(_var_str, _expr):
    _macro = """"""
    _macro += _var_str + " += " + _codify(_expr) + ";\n"
    return _macro

def _codify_sub_assignment(_var_str, _expr):
    _macro = """"""
    _macro += _var_str + " -= " + _codify(_expr) + ";\n"
    return _macro

def _codify_mul_assignment(_var_str, _expr):
    _macro = """"""
    _macro += _var_str + " *= " + _codify(_expr) + ";\n"
    return _macro

def _codify_div_assignment(_var_str, _expr):
    _macro = """"""
    _macro += _var_str + " /= " + _codify(_expr) + ";\n"
    return _macro

def _codify_declaration(_var_str, _expr, _type = None):
    if _type is not None:
        return _type + " " + _codify_assignment(_var_str, _expr)
    else:
        return _codify_assignment(_var_str, _expr)

def _codify_declaration_const(_var_str, _expr, _type = None):
    if _type is not None:
        return "const " + _type + " " + _codify_assignment(_var_str, _expr)
    else:
        return _codify_assignment(_var_str, _expr)

def _codify_declaration_const_check(_var_str, _expr, _type = None,
                                    declared_variables = None,
                                    declared_constants = None,
                                    declare_const_flag = False):
    _swap_code = """"""
    if _var_str not in declared_variables[0] and _var_str not in declared_constants[0]:
        _declare_f = _codify_declaration_const if declare_const_flag else _codify_declaration
        _swap_code += _declare_f(_var_str, _expr, _type)
        if declare_const_flag:
            declared_constants[0] += [_codify(_var_str)]
        else:
            declared_variables[0] += [_codify(_var_str)]
            
    elif _var_str in declared_constants[0] and declare_const_flag:
        pass
        ##raise Exception("Variable", _var_str, "already declared as a constant!")
    elif _var_str in declared_variables[0]:
        pass

    return _swap_code

'''
_get_cartesian_coordinates_macro
- here the idea is that the both sizes and strides are passed as macros to the compiler
- this implies that the values for the strides are precomputed
- need to add some checks
'''
'''
if _custom_type is passed to the function then the variable declaration is assumed
'''

def _get_cartesian_coordinates_macro(declared_variables, declared_constants,
                                     _root_coordinate, _lexicographic_index, 
                                     _dim_sizes, _dim_strides,
                                     _type = None, declare_const_flag = False,
                                     _avoid_module = False):
    '''
    Checking that the list of declared variables is available
    '''
    if declared_variables is None:
        raise Exception("Missing argument 'declared_variables'")
    if type(declared_variables) != list:
        raise Exception("Argument 'declared_variables' must be a list containing one list")
    if len(declared_variables) == 0 or type(declared_variables[0]) != list:
        raise Exception("List 'declared_variables' must contain another list!")

    '''
    Checking that the list of declared constants is available
    '''
    if declared_constants is None:
        raise Exception("Missing argument 'declared_constants'")
    if type(declared_constants) != list:
        raise Exception("Argument 'declared_constants' must be a list containing one list")
    if len(declared_constants) == 0 or type(declared_constants[0]) != list:
        raise Exception("List 'declared_constants' must contain another list!")

    '''
    Checking needed variables: can be abstracted out
    '''
    _needed_variables = _dim_strides + _dim_sizes
    _chk_needed_variables = []
    for _ in _needed_variables:
        _chk_needed_variables += [_ in declared_variables[0] or
                                  _ in declared_constants[0]]

    if not AllTrue(_chk_needed_variables):
        print()
        for _i, _ in enumerate(_chk_needed_variables):
            if not _:
                print("Variable/constant ", _needed_variables[_i], "not declared!")
        raise Exception("Some needed variables/constants have not been declared yet (!)")            

    
    _D = len(_dim_sizes)
    _macro = """"""
    _macro += \
        _codify_declaration_const_check(_root_coordinate + "_0", 
                                        _lexicographic_index + " % " + _dim_strides[0], 
                                        _type, declared_variables, declared_constants,
                                        declare_const_flag)
    for _d in range(1, _D - 1):
        _macro += \
            _codify_declaration_const_check(_root_coordinate + "_" + str(_d),
                                            ("(" + _lexicographic_index + " / " + _dim_strides[_d - 1] + ") % " + 
                                             _dim_sizes[_d]), 
                                            _type, declared_variables, declared_constants,
                                            declare_const_flag)
    _macro += \
        _codify_declaration_const_check(_root_coordinate + "_" + str(_D - 1),
                                        _lexicographic_index + " / " + _dim_strides[_D - 2],
                                        _type, declared_variables, declared_constants,
                                        declare_const_flag)
    return _macro

'''
_get_array_value
'''
def _array_value(array, index, use_ptrs = False):
    return '(*(' + array + ' + ' + index + '))' if use_ptrs else array + '[' + index + ']'

def _array_ptr(array, index):
    return '(' + array + ' + ' + index + ')'

'''
_get_seq_macros:
- returns a list of macros according to the number of dimensions
'''
def _get_seq_macros(N, _root_symbol):
    return [_root_symbol + '_' + str(_i) for _i in range(N)]

def _get_seq_vars(N, _root_symbol):
    return _get_seq_macros(N, _root_symbol)

def _get_sympy_seq_macros(N, _root_symbol):
    return [sp.Symbol(_) for _ in _get_seq_macros(N, _root_symbol)]

def _get_sympy_seq_vars(N, _root_symbol):
    return _get_sympy_seq_macros(N, _root_symbol)

'''
_get_single_neighbor_pos_macro_fully_sym_sliced
Need to write this function for the sliced memory pattern
'''

'''
_get_single_neighbor_pos_macro_fully_sym
- this function returns a sympy expression for the calculations of the lexicographic index
of a neighbor of the the site pointed by '_lexicographic_index' and connected by '_vector'
- this is a common situation in both Ising models and lattice Boltzmann
- the final result is obtained after sympy.simplify and sympy.collect in order to minimize
the number of multiplications 
'''
from functools import reduce
import sympy as sp

_vector_delta_str = lambda _c: '_p' + str(_c) if _c > 0 else '_m' + str(abs(_c))

import numpy as np

def _get_single_neighbor_pos_macro_fully_sym(_vector, _dim_sizes, _dim_strides,
                                             _root_coordinate, _lexicographic_index):
    _D = len(_dim_sizes)
    if _D != len(_vector):
        raise Exception("Mismatching dimensions for sizes and vectors!")
        
    _sp_dim_sizes = [sp.Symbol(_) for _ in _dim_sizes]
    _sp_dim_strides = [sp.Symbol(_) for _ in _dim_strides]
        
    _expr = sp.Symbol(_lexicographic_index)
    for _i in range(_D):
        _sp_root_coord = sp.Symbol(_root_coordinate + '_' + str(_i))

        _str_vector_delta = _vector_delta_str(_vector[_i])
        
        _sp_root_coord_neigh = sp.Symbol(_root_coordinate + '_' + str(_i) + _str_vector_delta)
        _coord_flag = int(abs(_vector[_i]) > 0)
        _expr -= (_sp_root_coord * (1 if _i < 1 else _sp_dim_strides[_i - 1])) * _coord_flag
        _expr += (_sp_root_coord_neigh * (1 if _i < 1 else _sp_dim_strides[_i - 1])) * _coord_flag
    
    _expr = sp.simplify(_expr)
    for _ in _dim_sizes:
        _expr = sp.collect(_expr, _)
        
    return _expr

'''
Small functions for codifying sympy expressions
'''

def _codify_sympy(_expr):
    return _codify(_expr).replace('\\', '_')

def _codify_sympy_assignment(_var_str, _expr):
    return _codify_assignment(_codify_sympy(_var_str), _codify_sympy(_expr))

def _codify_sympy_add_assignment(_var_str, _expr):
    return _codify_add_assignment(_codify_sympy(_var_str), _codify_sympy(_expr))

def _codify_sympy_sub_assignment(_var_str, _expr):
    return _codify_sub_assignment(_codify_sympy(_var_str), _codify_sympy(_expr))

def _codify_sympy_mul_assignment(_var_str, _expr):
    return _codify_mul_assignment(_codify_sympy(_var_str), _codify_sympy(_expr))

def _codify_sympy_div_assignment(_var_str, _expr):
    return _codify_div_assignment(_codify_sympy(_var_str), _codify_sympy(_expr))

def _codify_sympy_declaration(_var_str, _expr, _type = None):
    return _codify_declaration(_codify_sympy(_var_str), _codify_sympy(_expr), _type)

def _codify_sympy_declaration_const(_var_str, _expr, _type = None):
    return _codify_declaration_const(_codify_sympy(_var_str), _codify_sympy(_expr), _type)

'''
_neighbors_register_pressure_macro_sliced:
- need to write this function for the sliced memory access pattern
- we can try the sliced memory pattern for LBM
'''

'''
_neighbors_register_pressure_macro:
- It is assumed that in '_stencil_vectors' the groups are ordered by squared norm
'''
from idpy.Utils.Geometry import GetLen2Pos

def _neighbors_register_pressure_macro(_declared_variables, _declared_constants,
                                       _root_coordinate, _lexicographic_index,
                                       _stencil_vectors, _dim_sizes, _dim_strides,
                                       _custom_type, _exclude_zero_norm = False,
                                       _collect_mul = False, _declare_const_flag = False):
    '''
    Checking that the list of declared variables is available
    '''
    if _declared_variables is None:
        raise Exception("Missing argument '_declared_variables'")
    if type(_declared_variables) != list:
        raise Exception("Argument '_declared_variables' must be a list containing one list")
    if len(_declared_variables) == 0 or type(_declared_variables[0]) != list:
        raise Exception("List '_declared_variables' must contain another list!")

    '''
    Checking that the list of declared constants is available
    '''
    if _declared_constants is None:
        raise Exception("Missing argument '_declared_constants'")
    if type(_declared_constants) != list:
        raise Exception("Argument '_declared_constants' must be a list containing one list")
    if len(_declared_constants) == 0 or type(_declared_constants[0]) != list:
        raise Exception("List '_declared_constants' must contain another list!")
    
    '''
    Checking if the zero-th vector has zero norm and in case switch the flag
    '''
    if _exclude_zero_norm and GetLen2Pos(_stencil_vectors[0]) > 0:
        _exclude_zero_norm = False
    
    '''
    Getting the full list of sympy expressions
    '''
    _lexicographic_index_neigh_list = []
    
    for _i in range(len(_stencil_vectors)):
        _vector = _stencil_vectors[_i]
        _lexicographic_index_neigh = \
            _get_single_neighbor_pos_macro_fully_sym(_vector, _dim_sizes, _dim_strides,
                                                     _root_coordinate, _lexicographic_index)
        _lexicographic_index_neigh_list += [_lexicographic_index_neigh]
        
    '''
    Check if the set of calculations can be simplified avoiding repetitions and
    minimizing the number of multiplications (first) and operations (second).
    The number of operations is estimated by the number of characters in the expression
    '''
    
    _lexicographic_index_neigh_red_list = []
    
    '''
    The starting index is 1 in order to perform backwards comparisons
    '''
    for _i in range(1, len(_lexicographic_index_neigh_list)):
        _swap_elem_i = _lexicographic_index_neigh_list[_i]

        _swap_list_subs, _swap_list_char_count = [], []
        _swap_list_mult_count = []
        for _j in reversed(range(_i)):
            '''
            I need to expand first to match previous expressions:
            the output of _get_single_neighbor_pos_macro_fully_sym is
            already simplified and the patterns might not be recognized
            '''
            _swap_elem_j = sp.expand(_lexicographic_index_neigh_list[_j])
            _swap_elem_check = \
                sp.expand(_swap_elem_i).subs(_swap_elem_j,
                                             sp.Symbol('n_' + _root_coordinate + '_' + str(_j)))
            
            '''
            After checking for matching patterrns it is possible to simplify again
            '''
            for _ in _dim_sizes:
                _swap_elem_check = sp.collect(_swap_elem_check, _)

            _swap_list_subs += [_swap_elem_check]
            _swap_list_mult_count += [str(_swap_elem_check).count('*')]
            _swap_list_char_count += [len(str(_swap_elem_check))]

        _swap_list_char_count = np.array(_swap_list_char_count)
        _min_mult_index = np.argmin(_swap_list_mult_count)
        _min_char_index = np.argmin(_swap_list_char_count)
        _min_mult = np.amin(_swap_list_mult_count)
        _min_char = np.amin(_swap_list_char_count)    

        '''
        Comparison between number of multplications and length of the expression
        For the same number of products the shortest expression is selected
        '''
        if _swap_list_mult_count[_min_char_index] == _min_mult:
            _best_index = _min_char_index
        else:
            _best_index = _min_mult_index

        _swap_elem_i = _swap_list_subs[_best_index]
        _lexicographic_index_neigh_red_list += [_swap_elem_i]
        
    '''
    Extend the reduced expression list to take into account the first one
    which was initially discarded to perform the comparisons
    '''
    if not _exclude_zero_norm:
        _lexicographic_index_neigh_red_list = \
            [_lexicographic_index_neigh_list[0]] + _lexicographic_index_neigh_red_list
    
    '''
    Preparing the macro
    '''
    _macro_neighbors = """"""
    for _i in range(len(_lexicographic_index_neigh_red_list)):
        _i_offset = _i + 1 if _exclude_zero_norm else _i
        _macro_neighbors += \
            _codify_comment('(Lex)Index for neighbor at ' + str(_stencil_vectors[_i_offset]) + ', q:' + str(_i_offset))

        _swap_elem = _lexicographic_index_neigh_red_list[_i]
        '''
        collect multiplications
        '''
        if _collect_mul:
            for _stride in _dim_strides:
                _swap_elem = sp.collect(_swap_elem, sp.Symbol(_stride))
            
        if _i == 0 and _exclude_zero_norm:
            _swap_elem = _codify_sympy(_swap_elem).replace('n_' + _root_coordinate + '_0', _lexicographic_index)

        _sx_hnd = 'n_' + _root_coordinate + '_' + str(_i)
        _dx_hnd = str(_swap_elem)
        _macro_neighbors += _codify_declaration_const_check(_sx_hnd, _dx_hnd, _custom_type,
                                                            _declared_variables, 
                                                            _declared_constants, 
                                                            _declare_const_flag)
            
        if False:
            _macro_neighbors += \
                _custom_type + " n_" + _root_coordinate + "_" + str(_i) + " = " + str(_swap_elem) + ";\n"
        _macro_neighbors += _codify_newl

    '''
    if _exclude_zero_norm == True I need to substitute 'n_*_0' with _lexicographic_index
    '''
    if False:
        _macro_neighbors = _macro_neighbors.replace('n_' + _root_coordinate + '_0', _lexicographic_index)
        
    '''
    Substitution of the expressions for the 'optimized' summations '_sp_macro' and '_sm_macro'
    '''
    if False:
        _macro_neighbors = \
            _subs_sp_sm_macros(_macro_neighbors, _root_coordinate, _stencil_vectors, _dim_sizes)
        
    return _macro_neighbors

def _subs_sp_sm_macros(_code, _root_coordinate, _stencil_vectors, _dim_sizes):
    for _vector in _stencil_vectors:    
        for _d in range(len(_vector)):
            _str_vector_delta = '_p' + str(_vector[_d]) if _vector[_d] > 0 else '_m' + str(abs(_vector[_d]))
            _which_macro = _sp_macro if _vector[_d] > 0 else _sm_macro
            _sign_str = ' + ' if _vector[_d] > 0 else ' - '
            _to_be_chgd = _root_coordinate + '_' + str(_d) + _str_vector_delta
            _to_be_put = _which_macro('(' + _root_coordinate + '_' + str(_d) + _sign_str + str(abs(_vector[_d])) + ')', 
                                      _dim_sizes[_d])
            _code = _code.replace(_to_be_chgd, _to_be_put)
    return _code
