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

from idpy.IdpyStencils.IdpyStencils import IdpyStencil
from idpy.Utils.IdpySymbolic import SymmetricTensor, TaylorTuples
from idpy.IdpyCode.IdpyUnroll import _get_seq_vars, _get_seq_macros, _codify_add_assignment
from idpy.IdpyCode.IdpyUnroll import _get_sympy_seq_vars, _codify_sympy, _codify_newl
from idpy.IdpyCode.IdpyUnroll import _codify_declaration_const_check, _get_sympy_seq_vars
from idpy.Utils.Statements import AllTrue
from idpy.Utils.Geometry import FlipVector, IsSameVector

from functools import reduce

import sympy as sp

class GuoForcing:
    def __init__(self, xi_stencil = None,
                 root_pop = 'fG', root_u = 'u', omega_val = 1,
                 root_f = 'F', root_w = 'w', root_omega = '\\omega'):
        
        if xi_stencil is None:
            raise Exception("Missing argument 'xi_stencil'")

        self.root_pop, self.root_u, self.root_w, self.root_omega = \
            root_pop, root_u, root_w, root_omega
        
        self.idpy_stencil = IdpyStencil(xi_stencil)
        self.D, self.Q = len(self.idpy_stencil.XIs[0]), len(self.idpy_stencil.XIs)

        self.SetZerothAndFirstMomentsSyms(root_u, root_f)
        self.SetPopulationsSyms(root_pop)
        self.SetWSyms(root_w)
        self.SetOmegaSym(root_omega, omega_val)

    def SetZerothAndFirstMomentsSyms(self, root_u, root_F):
        self.u = sp.Matrix([sp.Symbol(root_u + '_' + str(_))
                            for _ in range(self.D)])
        
        self.u_symt = SymmetricTensor(
            list_ttuples = TaylorTuples(list(range(self.D)), 1),
            list_values = self.u, d = self.D, rank = 1
        )

        self.F = sp.Matrix([sp.Symbol(root_F + '_' + str(_))
                            for _ in range(self.D)])
        self.F_symt = SymmetricTensor(
            list_ttuples = TaylorTuples(list(range(self.D)), 1),
            list_values = self.F, d = self.D, rank = 1
        )

    def SetPopulationsSyms(self, root_pop):
        self.f_i = sp.Matrix([sp.Symbol(root_pop + '_' + str(_))
                              for _ in range(self.Q)])

    def SetOmegaSym(self, root_omega, omega_val):
        self.omega_sym = sp.Symbol(root_omega)
        self.omega_n = omega_val
        
    def SetWSyms(self, root_w):
        self.w_i = sp.Matrix([sp.Symbol(root_w + '_' + str(_))
                              for _ in range(self.Q)])
        self.w_i_n = sp.Matrix(self.idpy_stencil.Ws)
        self.w_symt = SymmetricTensor(c_dict = {0: self.w_i}, d = self.D, rank = 0)

    def GetHigherOrderSymmetricUFTensor(self, order = 2):
        '''
        Need to generalize to higher orders
        '''
        _taylor_indices = TaylorTuples(list(range(self.D)), 2)
        _swap_dict = {}
        for _i, _index_tuple in enumerate(_taylor_indices):
            _swap_dict[_index_tuple] = \
                self.u[_index_tuple[0]] * self.F[_index_tuple[1]] + \
                self.u[_index_tuple[1]] * self.F[_index_tuple[0]]
            
        return SymmetricTensor(c_dict = _swap_dict, d = self.D, rank = order)

    def GetSymForcing(self, order = 2):
        '''
        Using the class SymmetricTensor to simplify the writing
        '''
        if order == 2:
            _eq_set = \
                self.idpy_stencil.GetEquilibriumHermiteSet(order = order, symt_flag = True)
            _c2 = self.idpy_stencil.c2
            _uF = self.GetHigherOrderSymmetricUFTensor()

            
            _F_guo = \
                _eq_set[1] * self.F_symt / _c2 + \
                _eq_set[2] * _uF / (2 * _c2 * _c2)

            for _i in range(self.Q):
                _F_guo[_i] *= (1 - self.omega_sym / 2) * self.w_i_n[_i]

            return _F_guo
            
    '''
    Getting thr forcing separately for each population
    '''
    def CodifyForcingSinglePopulation(self, order = 2, i = None, collect_exprs = []):
        if i is None:
            raise Exception("Missing parameter 'i'")

        _F_guo = sp.expand(self.GetSymForcing(order = order)[i])
        for _expr_tuple in collect_exprs:
            _F_guo = _F_guo.subs(_expr_tuple[0], _expr_tuple[1])
        return _codify_sympy(_F_guo.evalf())

    '''
    SetMomentsProducts
    '''
    def SetMomentsProductsCode(self, declared_variables = None, declared_constants = None,
                               pressure_mode = 'compute',
                               order = 2, root_f = 'F', root_u = 'u', mom_type = 'MType'):
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

        if pressure_mode not in ['compute', 'registers']:
            raise Exception("Parameter 'pressure_mode' should be eithe 'compute' or 'registers'")
        
        '''
        Check for needed variables
        '''
        _f_vars, _u_vars = _get_seq_vars(self.D, root_f), _get_seq_vars(self.D, root_u)
        _f_sym_vars, _u_sym_vars = \
            _get_sympy_seq_vars(self.D, root_f), _get_sympy_seq_vars(self.D, root_u)
        
        _needed_variables = _f_vars + _u_vars
        
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

        if pressure_mode not in ['compute', 'registers']:
            raise Exception("Parameter 'pressure_mode' should be eithe 'compute' or 'registers'")

        '''
        Need to change this part to correctly handle powers of the same variables
        as it is presently done in the same function in the Equilibria module
        '''
        _n_prod_sympy, _n_prod_code_sympy, _n_prod_vars = [], [], []
        for _order in range(2, order + 1):
            _swap_sympy, _swap_sympy_code = [], []
            for _tuple in TaylorTuples(list(range(self.D)), _order):
                _flip_tuple = FlipVector(_tuple)
                _is_symmetric_tuple = IsSameVector(_tuple, _flip_tuple)

                _swap_prod = _u_sym_vars[_tuple[0]] * _f_sym_vars[_tuple[1]]
                _swap_code_prod = _codify_sympy(_swap_prod)
                
                _swap_sympy += [_swap_prod]
                _swap_sympy_code += [_swap_code_prod]
                
                if not _is_symmetric_tuple:
                    _swap_prod = _u_sym_vars[_tuple[1]] * _f_sym_vars[_tuple[0]]
                    _swap_code_prod = _codify_sympy(_swap_prod)
                    
                    _swap_sympy += [_swap_prod]
                    _swap_sympy_code += [_swap_code_prod]
                
            _n_prod_sympy += _swap_sympy
            _n_prod_code_sympy += _swap_sympy_code
            
            _n_prod_vars += [sp.Symbol(str(_).replace('*', '_')) for _ in _swap_sympy]
            
        _variables_tuples = []
        if pressure_mode == 'registers':
            for _i, _prod in enumerate(_n_prod_vars):
                _variables_tuples += [(_n_prod_sympy[_i], _prod)]
        else:
            for _i, _prod in enumerate(_n_prod_sympy):
                _variables_tuples += [(_prod, sp.Symbol(_n_prod_code_sympy[_i]))]
            
        '''
        declaring the variables
        '''
        _swap_code = """"""

        if pressure_mode == 'registers':
            for _i, _var in enumerate(_n_prod_vars):
                _swap_code += \
                    _codify_declaration_const_check(
                        _var_str = _codify_sympy(_var),
                        _expr = _codify_sympy(_n_prod_sympy[_i]),
                        _type = mom_type,
                        declared_variables = declared_variables,
                        declared_constants = declared_constants,
                        declare_const_flag = True
                    )

        '''
        Passing the tuples in the reversed order so that higher order products
        get substituted first
        '''
        return _swap_code, _variables_tuples[::-1]
        
    def ShiftVelocityCode(self, n_sets = ['n'], u_sets = ['u'],
                          f_sets = ['F']):

        _swap_code = """"""

        _n_syms = [sp.Symbol(_) for _ in n_sets]
        _u_syms, _f_syms = [], []
        for _i in range(len(_n_syms)):
            _u_syms += [_get_sympy_seq_vars(self.D, u_sets[_i])]
            _f_syms += [_get_sympy_seq_vars(self.D, f_sets[_i])]            
        
        for _i, _n in enumerate(_n_syms):
            for _d in range(self.D):
                _swap_code += \
                    _codify_add_assignment(
                        _codify_sympy(_u_syms[_i][_d]),
                        _codify_sympy(0.5 * _f_syms[_i][_d] / _n)
                    )
            _swap_code += _codify_newl

        return _swap_code
        
