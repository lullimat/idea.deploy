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
import numpy as np
from IPython.display import display, Math, Latex

class RelaxationMatrix:
    def __init__(self, eq_obj = None, search_depth = 6, WOrth_flag = True):
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")

        if WOrth_flag:
            _M_dict = \
                eq_obj.idpy_stencil.GetWOrthInvertibleHermiteSet(
                    search_depth = search_depth
                )
            self.MHermitePolys = _M_dict['MWOrthHermitePolys']

        else:
            _M_dict = \
                eq_obj.idpy_stencil.GetInvertibleHermiteSet(search_depth = search_depth)
            self.MHermitePolys = _M_dict['MHermitePolys']
            
        self.D, self.Q = eq_obj.D, eq_obj.Q

    def GetMoments(self):
        return self.MHermitePolys

    def DisplayMoments(self):
        for _i, _m in enumerate(self.MHermitePolys):
            display(
                Latex(
                    str(_i) + " :$" + sp.latex(_m) + "$"
                )
            )

    def GetIndexFromMoment(self, moment):
        return np.where(self.MHermitePolys == moment)[0][0]

    def GetDiagonalMatrix(self, omega_list = None):
        self.omega_list = omega_list
        return sp.Matrix.diag(omega_list)

    def SetOmegasByMoment(self, omega_dict = None):
        _S = sp.zeros(self.Q)
        for _key, _val in omega_dict.items():
            _i = self.GetIndexFromMoment(_key[0])
            _j = self.GetIndexFromMoment(_key[1])
            _S[_i, _j] = _val
        return _S
            
        

class BGK:
    def __init__(self, root_omega = '\\omega', omega_val = 1):
        self.SetOmegaSym(root_omega, omega_val)

    def SetOmegaSym(self, root_omega, omega_val):
        self.omega_sym = sp.Symbol(root_omega)
        self.omega_val = omega_val
                
    def MRTCollisionPlusGuoSym(self, order = 2, eq_obj = None, guo_obj = None,
                               neq_pop_root = 'pop', neq_mom_root = 'mneq',
                               relaxation_matrix = None,
                               search_depth = 6, WOrth_flag = True):
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")
        if guo_obj is None:
            raise Exception("Missing argument 'guo_obj'")
        if relaxation_matrix is None:
            raise Exception("Missing argument 'relaxation_matrix'")        

        if eq_obj.D != guo_obj.D:
            raise Exception("The dimension of 'equilibria' and 'guo' objects differ!")

        _Q, _D = eq_obj.Q, eq_obj.D

        _S = relaxation_matrix
        _1 = sp.eye(_Q)
        
        _neq_pop_vector = sp.Matrix(_get_sympy_seq_vars(_Q, neq_pop_root))

        if WOrth_flag:
            _M_dict = \
                eq_obj.idpy_stencil.GetWOrthInvertibleHermiteSet(
                    search_depth = search_depth
                )
            _M, _Mm1 = _M_dict['MWOrth'], _M_dict['MWOrth'].inv()
        else:
            _M_dict = \
                eq_obj.idpy_stencil.GetInvertibleHermiteSet(search_depth = search_depth)
            _M, _Mm1 = _M_dict['M'], _M_dict['M'].inv()

        _neq_m = \
            sp.Matrix([sp.Symbol(neq_mom_root + '_' + str(_q)) for _q in range(_Q)])
        _neq_m_pop = _M * _neq_pop_vector

        _f_eq = eq_obj.GetSymEquilibrium(order = order)
        _eq_m = _M * _f_eq
        _f_guo = guo_obj.GetSymForcingMRT(order = order, eq_obj = eq_obj,
                                          relaxation_matrix = relaxation_matrix,
                                          search_depth = search_depth)

        
        _rel_eq_m = _S * _M * _f_eq
        _rel_neq_m = (_1 - _S) * _neq_m
        
        _rel_m = - _S * (_neq_m - _eq_m)

        '''
        After checking the operations counts it seems reasonable to use the populations
        in the final result.
        Hence, we transform back to population space and substitute the momenta
        definitions
        '''
        _pop_out = _Mm1 * (_rel_eq_m + _rel_neq_m) + _f_guo
        #_pop_out = _neq_pop_vector + _Mm1 * _rel_m + _f_guo        
        for _i, _pop in enumerate(_pop_out):
            _pop_out_swap = _pop
            for _j, _m_pop in enumerate(_neq_m_pop):
                _pop_out_swap = \
                    _pop_out_swap.subs(_neq_m[_j], _m_pop)
                
            _pop_out[_i] = _pop_out_swap
        
        return sp.expand(_pop_out)
        
        
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

    def SRTCollisionSym(self, order = 2, eq_obj = None, neq_pop = 'pop'):
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")

        _neq_pop_vector = sp.Matrix(_get_sympy_seq_vars(eq_obj.Q, neq_pop))
            
        _f_eq = eq_obj.GetSymEquilibrium(order = order)

        return (1 - self.omega_sym) * _neq_pop_vector + self.omega_sym * _f_eq

    def MRTCollisionSym(self, order = 2, eq_obj = None,
                        neq_pop_root = 'pop', neq_mom_root = 'mneq',
                        relaxation_matrix = None,
                        search_depth = 6, WOrth_flag = True):
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")
        if relaxation_matrix is None:
            raise Exception("Missing argument 'relaxation_matrix'")        

        _Q, _D = eq_obj.Q, eq_obj.D

        _S = relaxation_matrix
        _1 = sp.eye(_Q)
        
        _neq_pop_vector = sp.Matrix(_get_sympy_seq_vars(_Q, neq_pop_root))

        if WOrth_flag:
            _M_dict = \
                eq_obj.idpy_stencil.GetWOrthInvertibleHermiteSet(
                    search_depth = search_depth
                )
            _M, _Mm1 = _M_dict['MWOrth'], _M_dict['MWOrth'].inv()
        else:
            _M_dict = \
                eq_obj.idpy_stencil.GetInvertibleHermiteSet(search_depth = search_depth)
            _M, _Mm1 = _M_dict['M'], _M_dict['M'].inv()

        _neq_m = \
            sp.Matrix([sp.Symbol(neq_mom_root + '_' + str(_q))
                       for _q in range(_Q)])
        _neq_m_pop = _M * _neq_pop_vector

        _f_eq = eq_obj.GetSymEquilibrium(order = order)

        _rel_eq_m = _S * _M * _f_eq
        _rel_neq_m = (_1 - _S) * _neq_m

        '''
        After checking the operations counts it seems reasonable to use the populations
        in the final result.
        Hence, we transform back to population space and substitute the momenta
        definitions
        '''
        _pop_out = _Mm1 * (_rel_eq_m + _rel_neq_m)
        for _i, _pop in enumerate(_pop_out):
            _pop_out_swap = _pop
            for _j, _m_pop in enumerate(_neq_m_pop):
                _pop_out_swap = \
                    _pop_out_swap.subs(_neq_m[_j], _m_pop)
                
            _pop_out[_i] = _pop_out_swap
        
        return sp.expand(_pop_out)
    
    def CodifySingleSRTCollisionPlusGuoSym(
            self, order = 2, i = None,
            eq_obj = None, guo_obj = None,
            neq_pop = 'pop', tuples_eq = [], tuples_guo = []
    ):
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

    def CodifySingleMRTCollisionPlusGuoSym(
            self, order = 2, i = None,
            eq_obj = None, guo_obj = None,
            neq_pop = 'pop', tuples_eq = [], tuples_guo = [],
            relaxation_matrix = None, search_depth = 6, WOrth_flag = True,
            neq_mom_root = 'mneq', omega_syms_vals = None
    ):
        if i is None:
            raise Exception("Missing parameter 'i'")        
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")
        if guo_obj is None:
            raise Exception("Missing argument 'guo_obj'")
        if relaxation_matrix is None:
            raise Exception("Missing argument 'relaxation_matrix'")
        if omega_syms_vals is None:
            raise Exception("Missing argument 'omega_syms_vals'")

        _collision_pop = \
            self.MRTCollisionPlusGuoSym(
                order = order, eq_obj = eq_obj, guo_obj = guo_obj,
                neq_pop_root = neq_pop, neq_mom_root = neq_mom_root,
                relaxation_matrix = relaxation_matrix,
                search_depth = search_depth, WOrth_flag = WOrth_flag
            )

        '''
        Substituting omega values
        '''
        _collision_pop = _collision_pop[i]
        for _omega_tuple in omega_syms_vals:
            _collision_pop = \
                _collision_pop.subs(_omega_tuple[0], _omega_tuple[1])
                
        for _expr_tuple in tuples_eq + tuples_guo:
            _collision_pop = _collision_pop.subs(_expr_tuple[0], _expr_tuple[1])

        _swap_code = _codify_sympy(_collision_pop.evalf())
        if _swap_code.count('omega') > 0:
            raise Exception("Some of the rates values are missing!")
        
        return _swap_code
    
    def CodifySingleSRTCollisionSym(self, order = 2, i = None, eq_obj = None,
                                    neq_pop = 'pop', tuples_eq = []):
        if i is None:
            raise Exception("Missing parameter 'i'")        
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")

        _collision_pop = \
            self.SRTCollisionSym(
                order = order, eq_obj = eq_obj,
                neq_pop = neq_pop
            )

        _collision_pop = sp.expand(_collision_pop[i].subs(self.omega_sym, self.omega_val))
        for _expr_tuple in tuples_eq:
            _collision_pop = _collision_pop.subs(_expr_tuple[0], _expr_tuple[1])

        return _codify_sympy(_collision_pop.evalf())

    def CodifySingleMRTCollisionSym(
            self, order = 2, i = None,
            eq_obj = None, neq_pop = 'pop', tuples_eq = [],
            relaxation_matrix = None, search_depth = 6, WOrth_flag = True,
            neq_mom_root = 'mneq', omega_syms_vals = None
    ):
        if i is None:
            raise Exception("Missing parameter 'i'")        
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")
        if relaxation_matrix is None:
            raise Exception("Missing argument 'relaxation_matrix'")
        if omega_syms_vals is None:
            raise Exception("Missing argument 'omega_syms_vals'")

        _collision_pop = \
            self.MRTCollisionSym(
                order = order, eq_obj = eq_obj,
                neq_pop_root = neq_pop, neq_mom_root = neq_mom_root,
                relaxation_matrix = relaxation_matrix,
                search_depth = search_depth, WOrth_flag = WOrth_flag
            )

        '''
        Substituting omega values
        '''
        _collision_pop = _collision_pop[i]
        for _omega_tuple in omega_syms_vals:
            _collision_pop = \
                _collision_pop.subs(_omega_tuple[0], _omega_tuple[1])
                
        for _expr_tuple in tuples_eq:
            _collision_pop = _collision_pop.subs(_expr_tuple[0], _expr_tuple[1])

        _swap_code = _codify_sympy(_collision_pop.evalf())
        if _swap_code.count('omega') > 0:
            raise Exception("Some of the rates values are missing!")
        
        return _swap_code
    
    
    '''
    By defining the 'pressure_mode' here and passing the tuples I keep the granularity fine
    i.e. I can tune the pressure mode for the streaming independently from the pressure mode
    of the equilibrium and collision computations
    '''
    def SRTCollisionPlusGuoPushStreamCode(
            self, declared_variables = None, declared_constants = None,
            ordering_lambda = None, order = 2,
            dst_arrays_var = 'pop_swap',
            stencil_obj = None, eq_obj = None,
            guo_obj = None,
            neq_pop = 'pop', pressure_mode = 'compute',
            tuples_eq = [], tuples_guo = [],
            pos_type = 'int', 
            use_ptrs = False, collect_mul = False,
            root_dim_sizes = 'L', root_strides = 'STR', 
            root_coord = 'x', lex_index = 'g_tid', 
            declare_const_dict = {'cartesian_coord_neigh': False},
            rnd_pops = None
    ):

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

        if rnd_pops is not None and type(rnd_pops) != list:
            raise Exception("Argument 'rnd_pops' msut be a list!")
        if rnd_pops is not None and len(rnd_pops) != _Q:
            raise Exception("The length of 'rnd_pops'", len(rnd_pops), "does not match Q:", _Q)

        _src_pop_vars = _get_seq_vars(_Q, neq_pop)
        _dim_sizes_macros = _get_seq_macros(_dim, root_dim_sizes)
        _dim_strides_macros = _get_seq_macros(_dim - 1, root_strides)        
        
        _needed_variables = \
            [dst_arrays_var] + _src_pop_vars + _dim_sizes_macros + _dim_strides_macros

        '''
        This part does not seem useful, since I need to add the terms from
        outside I would also need to make sure they are declared
        '''
        if False:
            if rnd_pops is not None:
                _needed_variables += rnd_pops
        
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
            _swap_code += \
                _codify_declaration_const_check(_codify_sympy(_coord_dst), 0, pos_type,
                                                declared_variables, declared_constants)
            _swap_code += _codify_newl

            for _q, _xi in enumerate(_XIs):
                '''
                First get the neighbor position and the opposite
                '''
                _swap_code += \
                    _codify_comment('(Lex)Index for neighbor at ' + str(_xi) +
                                    ', q: ' + str(_q))
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

                if rnd_pops is not None:
                    _dx_hnd += ' + ' + rnd_pops[_q]
                
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

    def SRTCollisionPushStreamCode(self,
                                   declared_variables = None,
                                   declared_constants = None,
                                   ordering_lambda = None, order = 2,
                                   dst_arrays_var = 'pop_swap',
                                   stencil_obj = None, eq_obj = None,
                                   neq_pop = 'pop', pressure_mode = 'compute',
                                   tuples_eq = [], pos_type = 'int', 
                                   use_ptrs = False, collect_mul = False,
                                   root_dim_sizes = 'L', root_strides = 'STR', 
                                   root_coord = 'x', lex_index = 'g_tid', 
                                   declare_const_dict = {'cartesian_coord_neigh': False},
                                   rnd_pops = None):

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
        
        '''
        Defining and checking the variables that are needed: missing (!)
        '''
        _Q, _dim = stencil_obj.Q, stencil_obj.D
        _XIs = stencil_obj.XIs

        if rnd_pops is not None and type(rnd_pops) != list:
            raise Exception("Argument 'rnd_pops' msut be a list!")
        if rnd_pops is not None and len(rnd_pops) != _Q:
            raise Exception("The length of 'rnd_pops'", len(rnd_pops), "does not match Q:", _Q)        

        _src_pop_vars = _get_seq_vars(_Q, neq_pop)
        _dim_sizes_macros = _get_seq_macros(_dim, root_dim_sizes)
        _dim_strides_macros = _get_seq_macros(_dim - 1, root_strides)        
        
        _needed_variables = \
            [dst_arrays_var] + _src_pop_vars + _dim_sizes_macros + _dim_strides_macros

        if rnd_pops is not None:
            _needed_varaibles += rnd_pops        
        
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
        _swap_code += _codify_comment("Defining cartesian neighbors coordinates if needed")
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
                    self.CodifySingleSRTCollisionSym(
                        order = order, i = _q, eq_obj = eq_obj,
                        neq_pop = neq_pop, tuples_eq = tuples_eq
                    )

                if rnd_pops is not None:
                    _dx_hnd += ' + ' + rnd_pops[_q]                
                
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
    
    def MRTCollisionPlusGuoPushStreamCode(
            self, declared_variables = None, declared_constants = None,
            ordering_lambda = None, order = 2,
            dst_arrays_var = 'pop_swap',
            stencil_obj = None, eq_obj = None,
            guo_obj = None, relaxation_matrix = None, omega_syms_vals = None,
            search_depth = 6, WOrth_flag = True,
            neq_pop = 'pop', pressure_mode = 'compute',
            tuples_eq = [], tuples_guo = [], pos_type = 'int', 
            use_ptrs = False, collect_mul = False,
            root_dim_sizes = 'L', root_strides = 'STR', 
            root_coord = 'x', lex_index = 'g_tid', 
            declare_const_dict = {'cartesian_coord_neigh': False},
            rnd_pops = None
    ):

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
        
        if stencil_obj is None:
            raise Exception("Missing argument 'stencil_obj'")
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")
        if guo_obj is None:
            raise Exception("Missing argument 'guo_obj'")

        if relaxation_matrix is None:
            raise Exception("Missing argument 'relaxation_matrix'")
        if omega_syms_vals is None:
            raise Exception("Missing argument 'omega_syms_vals'")
        
        
        '''
        Defining and checking the variables that are needed: missing (!)
        '''
        _Q, _dim = stencil_obj.Q, stencil_obj.D
        _XIs = stencil_obj.XIs

        if rnd_pops is not None and type(rnd_pops) != list:
            raise Exception("Argument 'rnd_pops' msut be a list!")
        if rnd_pops is not None and len(rnd_pops) != _Q:
            raise Exception("The length of 'rnd_pops'", len(rnd_pops), "does not match Q:", _Q)        

        _src_pop_vars = _get_seq_vars(_Q, neq_pop)
        _dim_sizes_macros = _get_seq_macros(_dim, root_dim_sizes)
        _dim_strides_macros = _get_seq_macros(_dim - 1, root_strides)        
        
        _needed_variables = \
            [dst_arrays_var] + _src_pop_vars + _dim_sizes_macros + _dim_strides_macros

        if rnd_pops is not None:
            _needed_varaibles += rnd_pops        
        
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
            _swap_code += \
                _codify_declaration_const_check(_codify_sympy(_coord_dst), 0, pos_type,
                                                declared_variables, declared_constants)
            _swap_code += _codify_newl

            for _q, _xi in enumerate(_XIs):
                '''
                First get the neighbor position and the opposite
                '''
                _swap_code += \
                    _codify_comment('(Lex)Index for neighbor at ' + str(_xi) +
                                    ', q: ' + str(_q))
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
                    self.CodifySingleMRTCollisionPlusGuoSym(
                        order = order, i = _q, eq_obj = eq_obj, guo_obj = guo_obj,
                        neq_pop = neq_pop, tuples_eq = tuples_eq, tuples_guo = tuples_guo,
                        relaxation_matrix = relaxation_matrix,
                        omega_syms_vals = omega_syms_vals,
                        search_depth = search_depth, WOrth_flag = WOrth_flag
                    )

                if rnd_pops is not None:
                    _dx_hnd += ' + ' + rnd_pops[_q]                
                
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

    def MRTCollisionPushStreamCode(
            self, declared_variables = None, declared_constants = None,
            ordering_lambda = None, order = 2,
            dst_arrays_var = 'pop_swap',
            stencil_obj = None, eq_obj = None,
            relaxation_matrix = None, omega_syms_vals = None,
            search_depth = 6, WOrth_flag = True,            
            neq_pop = 'pop', pressure_mode = 'compute',
            tuples_eq = [], pos_type = 'int', 
            use_ptrs = False, collect_mul = False,
            root_dim_sizes = 'L', root_strides = 'STR', 
            root_coord = 'x', lex_index = 'g_tid', 
            declare_const_dict = {'cartesian_coord_neigh': False},
            rnd_pops = None
    ):

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
        
        if stencil_obj is None:
            raise Exception("Missing argument 'stencil_obj'")
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")

        if relaxation_matrix is None:
            raise Exception("Missing argument 'relaxation_matrix'")
        if omega_syms_vals is None:
            raise Exception("Missing argument 'omega_syms_vals'")        
        
        '''
        Defining and checking the variables that are needed: missing (!)
        '''
        _Q, _dim = stencil_obj.Q, stencil_obj.D
        _XIs = stencil_obj.XIs

        if rnd_pops is not None and type(rnd_pops) != list:
            raise Exception("Argument 'rnd_pops' msut be a list!")
        if rnd_pops is not None and len(rnd_pops) != _Q:
            raise Exception("The length of 'rnd_pops'", len(rnd_pops), "does not match Q:", _Q)        

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
        _swap_code += _codify_comment("Defining cartesian neighbors coordinates if needed")
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
                    self.CodifySingleMRTCollisionSym(
                        order = order, i = _q, eq_obj = eq_obj,
                        relaxation_matrix = relaxation_matrix,
                        omega_syms_vals = omega_syms_vals,
                        search_depth = search_depth, WOrth_flag = WOrth_flag,
                        neq_pop = neq_pop, tuples_eq = tuples_eq
                    )

                if rnd_pops is not None:
                    _dx_hnd += ' + ' + rnd_pops[_q]                
                
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
    
