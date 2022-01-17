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

from idpy.IdpyCode.IdpyUnroll import _get_sympy_seq_vars, _codify_sympy, _codify_newl
from idpy.IdpyCode.IdpyUnroll import _codify_sympy_assignment, _get_seq_macros, _array_value
from idpy.IdpyCode.IdpyUnroll import _get_seq_vars, _codify_declaration_const_check
from idpy.IdpyCode.IdpyUnroll import _codify_comment, _get_single_neighbor_pos_macro_fully_sym
from idpy.IdpyCode.IdpyUnroll import _codify_assignment
from idpy.IdpyCode.IdpyUnroll import _neighbors_register_pressure_macro

from idpy.Utils.Statements import AllTrue

import sympy as sp

class BGK:
    def __init__(self, root_omega = '\\omega', omega_val = 1):
        self.SetOmegaSym(root_omega, omega_val)

    def SetOmegaSym(self, root_omega, omega_val):
        self.omega_sym = sp.Symbol(root_omega)
        self.omega_val = omega_val
        
    def SRTCollisionPlusGuoSym(self, order = 2, eq_obj = None, guo_obj = None,
                               neq_pop = 'pop'):
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")
        if guo_obj is None:
            raise Exception("Missing argument 'guo_obj'")

        if eq_obj.D != guo_obj.D:
            raise Exception("The dimension of 'equilibria' and 'guo' objects differ!")

        if False:
            '''
            Need to implement the comparison operator for the stencils
            '''
            if eq_object.idpy_stencil != guo_obj.idpy_stencil:
                raise Exception("The dimension of 'equibria' and 'guo' objects differ!")

        _neq_pop_vector = sp.Matrix(_get_sympy_seq_vars(eq_obj.Q, neq_pop))
            
        _f_eq = eq_obj.GetSymEquilibrium(order = order)
        _f_guo = guo_obj.GetSymForcing(order = order)

        return (1 - self.omega_sym) * _neq_pop_vector + self.omega_sym * _f_eq + _f_guo

    def CodifySingleSRTCollisionPlusGuoSym(self, order = 2, i = None, eq_obj = None, guo_obj = None,
                                           neq_pop = 'pop', tuples_eq = [], tuples_guo = []):
        if i is None:
            raise Exception("Missing parameter 'i'")        
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")
        if guo_obj is None:
            raise Exception("Missing argument 'guo_obj'")

        _collision_pop = \
            self.SRTCollisionPlusGuoSym(
                order = order, eq_obj = eq_obj, guo_obj = guo_obj,
                neq_pop = neq_pop
            )

        _collision_pop = sp.expand(_collision_pop[i].subs(self.omega_sym, self.omega_val))
        for _expr_tuple in tuples_eq + tuples_guo:
            _collision_pop = _collision_pop.subs(_expr_tuple[0], _expr_tuple[1])

        return _codify_sympy(_collision_pop.evalf())

    '''
    By defining the 'pressure_mode' here and passing the tuples I keep the granularity fine
    i.e. I can tune the pressure mode for the streaming independently from the pressure mode
    of the equilibrium and collision computations
    '''
    def SRTCollisionPlusGuoPushStreamCode(self, declared_variables = None, declared_constants = None,
                                          ordering_lambda = None, order = 2,
                                          dst_arrays_var = 'pop_swap',
                                          stencil_obj = None, eq_obj = None, guo_obj = None,
                                          neq_pop = 'pop', pressure_mode = 'compute',
                                          tuples_eq = [], tuples_guo = [], pos_type = 'int', 
                                          use_ptrs = False, collect_mul = False,
                                          root_dim_sizes = 'L', root_strides = 'STR', 
                                          root_coord = 'x', lex_index = 'g_tid', 
                                          declare_const_dict = {'cartesian_coord_neigh': False}):

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
        Checking that the ordering_lambda is defined
        '''
        if ordering_lambda is None:
            raise Exception("Missing argument 'ordering_lambda'")
        
        '''
        Possibly I do not need this one
        '''
        if stencil_obj is None:
            raise Exception("Missing argument 'stencil_obj'")
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")
        if guo_obj is None:
            raise Exception("Missing argument 'guo_obj'")
        
        '''
        Defining and checking the variables that are needed: missing (!)
        '''
        _Q, _dim = stencil_obj.Q, stencil_obj.D
        _XIs = stencil_obj.XIs

        _src_pop_vars = _get_seq_vars(_Q, neq_pop)
        _dim_sizes_macros = _get_seq_macros(_dim, root_dim_sizes)
        _dim_strides_macros = _get_seq_macros(_dim - 1, root_strides)        
        
        _needed_variables = \
            [dst_arrays_var] + _src_pop_vars + _dim_sizes_macros + _dim_strides_macros
        
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
        
        
        _swap_code = """"""
        
        _swap_code += \
            stencil_obj._define_cartesian_neighbors_coords(
                declared_variables = declared_variables,
                declared_constants = declared_constants,
                pos_type = pos_type,
                root_dim_sizes = root_dim_sizes, 
                root_strides = root_strides, 
                root_coord = root_coord,
                lex_index = lex_index, 
                declare_const_flag = declare_const_dict['cartesian_coord_neigh']
            )

        if pressure_mode == 'compute':
            '''
            Declaring the variables for the position of src and dst
            '''
            _coord_dst = sp.Symbol(root_coord + '_dst')
            _swap_code += _codify_declaration_const_check(_codify_sympy(_coord_dst), 0, pos_type,
                                                          declared_variables, declared_constants)
            _swap_code += _codify_newl

            for _q, _xi in enumerate(_XIs):
                '''
                First get the neighbor position and the opposite
                '''
                _swap_code += _codify_comment('(Lex)Index for neighbor at ' + str(_xi) + ', q: ' + str(_q))
                _swap_expr = \
                    _get_single_neighbor_pos_macro_fully_sym(_xi, _dim_sizes_macros, 
                                                             _dim_strides_macros,
                                                             root_coord, lex_index)

                '''
                if collect_mul == True collect multiplications by the strides
                '''
                if collect_mul:
                    for _stride in _dim_strides_macros:
                        _swap_expr = sp.collect(_swap_expr, sp.Symbol(_stride))

                _swap_code += _codify_sympy_assignment(_coord_dst, _swap_expr)

                _sx_hnd = _array_value(dst_arrays_var, ordering_lambda(_coord_dst, _q),
                                       use_ptrs)
                _dx_hnd = \
                    self.CodifySingleSRTCollisionPlusGuoSym(
                        order = order, i = _q, eq_obj = eq_obj, guo_obj = guo_obj,
                        neq_pop = neq_pop, tuples_eq = tuples_eq, tuples_guo = tuples_guo
                    )
                
                _swap_code += _codify_assignment(_sx_hnd, _dx_hnd)
                _swap_code += _codify_newl

        '''
        Registers-pressure mode
        '''
        if pressure_mode == 'registers':
            '''
            Here I should define all the neighbors positions at once:
            need to add the management of the declared neighbors variables
            '''
            '''
            I need to include the option to shift the neighbor index
            in IdpyStencil in order to make the two pieces work together
            '''
            _swap_code += \
                _neighbors_register_pressure_macro(
                    _declared_variables = declared_variables,
                    _declared_constants = declared_constants,
                    _root_coordinate = root_coord,
                    _lexicographic_index = lex_index,
                    _stencil_vectors = _XIs,
                    _dim_sizes = _dim_sizes_macros,
                    _dim_strides = _dim_strides_macros, 
                    _custom_type = pos_type,
                    _exclude_zero_norm = False,
                    _collect_mul = collect_mul,
                    _declare_const_flag = declare_const_dict['cartesian_coord_neigh']
                )
            _swap_code += _codify_newl
                
            
        return _swap_code
