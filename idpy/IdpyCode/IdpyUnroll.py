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

'''
This module provides the basic functions to generate unrolled code in the kernels
'''

_spx_macro = lambda _x, _m : "(" + str(_x) + "&(~(-(" + str(_x) + " >= " + str(_m) + "))))"
_smx_macro = lambda _x, _m : "(" + str(_x) + "+((-(" + str(_x) + " < 0))&" + str(_m) + "))"


'''
_get_cartesian_coordinates_macro
- here the idea is that the both sizes and strides are passed as macros to the compiler
- this implies that the values for the strides are precomputed
- need to add some checks
'''
def _get_cartesian_coordinates_macro(_root_coordinate, _lexicographic_index, 
                                     _dim_sizes, _dim_strides):
    _D = len(_dim_sizes)
    _macro = """"""
    _macro += _root_coordinate + "_0 = " + _lexicographic_index + " % " + _dim_strides[0] + ";\n"
    for _d in range(1, _D - 1):
        _macro += (_root_coordinate + "_" + str(_d) + " = (" + 
                   _lexicographic_index + " / " + _dim_strides[_d - 1] + ") % " + _dim_sizes[_d] + ";\n")
        
    _macro += (_root_coordinate + "_" + str(_D - 1) + " = " + 
               _lexicographic_index + " / " + _dim_strides[_D - 2] + ";\n")
        
    return _macro


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

def _get_dim_strides(_dim_sizes):
    return [reduce(lambda x, y: x * y, _dim_sizes[0: i + 1]) 
                   for i in range(len(_dim_sizes) - 1)]

_vector_delta_str = lambda _c: '_p' + str(_c) if _c > 0 else '_m' + str(abs(_c))

def _get_single_neighbor_pos_macro_fully_sym(_root_coordinate, _lexicographic_index,
                                             _dim_sizes, _vector):
    _D = len(_dim_sizes)
    if _D != len(_vector):
        raise Exception("Mismatching dimensions for sizes and vectors!")
        
    _sp_dim_sizes = [sp.Symbol(_) for _ in _dim_sizes]
    _sp_dim_strides = _get_dim_strides(_sp_dim_sizes)
        
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

def _codify_sympy_expr(_expr):
    return str(_expr)

def _codify_sympy_assignment(_var_str, _expr):
    _macro = """"""
    _macro += _var_str + " = " + _codify_sympy_expr(_expr) + ";\n"
    return _macro

def _codify_sympy_declaration(_type, _var_str, _expr):
    return _type + " " + _codify_sympy_assignment(_var_str, _expr)


'''
_neighbors_register_pressure_macro:
- It is assumed that in '_stencil_vectors' the groups are ordered by squared norm
'''
def _neighbors_register_pressure_macro(_root_coordinate, _lexicographic_index, 
                                       _stencil_vectors, _dim_sizes, _custom_type):
    '''
    Getting the full list of sympy expressions
    '''
    _lexicographic_index_neigh_list = []
    
    for _i in range(len(_stencil_vectors)):
        _vector = _stencil_vectors[_i]
        _lexicographic_index_neigh = \
            _get_single_neighbor_pos_macro_fully_sym(_root_coordinate, _lexicographic_index, 
                                                     _dim_sizes, _vector)
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
            _swap_elem_check = sp.expand(_swap_elem_i).subs(_swap_elem_j, sp.Symbol('n_' + str(_j)))
            
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
        Comparison betwee number of multplications and length of the expression
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
    _lexicographic_index_neigh_red_list = \
        [_lexicographic_index_neigh_list[0]] + _lexicographic_index_neigh_red_list
    
    '''
    Preparing the macro
    '''
    _macro_neighbors = """"""
    for _i in range(len(_lexicographic_index_neigh_red_list)):
        _swap_elem = _lexicographic_index_neigh_red_list[_i]
        _macro_neighbors += \
            _custom_type + " n_" + _root_coordinate + "_" + str(_i) + " = " + str(_swap_elem) + ";\n"
        
    '''
    Substitution of the expressions for the 'optimized' summations '_sp_macro' and '_sm_macro'
    '''
    for _vector in _stencil_vectors:    
        for _d in range(len(_vector)):
            _str_vector_delta = '_p' + str(_vector[_d]) if _vector[_d] > 0 else '_m' + str(abs(_vector[_d]))
            _which_macro = _sp_macro if _vector[_d] > 0 else _sm_macro
            _to_be_chgd = _root_coordinate + '_' + str(_d) + _str_vector_delta
            _to_be_put = _which_macro('(' + _root_coordinate + '_' + str(_d) + ' + ' + str(_vector[_d]) + ')', 
                                      _dim_sizes[_d])
            _macro_neighbors = \
                _macro_neighbors.replace(_to_be_chgd, _to_be_put)    
    
        
    return _macro_neighbors
