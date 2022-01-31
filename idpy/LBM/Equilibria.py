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

import sympy as sp

from idpy.IdpyStencils.IdpyStencils import IdpyStencil
from idpy.Utils.IdpySymbolic import SymmetricTensor, TaylorTuples
from idpy.Utils.Geometry import IsSameVector, FlipVector
from idpy.IdpyCode.IdpyUnroll import _codify_sympy, _codify_declaration_const_check
from idpy.IdpyCode.IdpyUnroll import _get_seq_macros, _get_seq_vars
from idpy.IdpyCode.IdpyUnroll import _get_sympy_seq_vars

from idpy.Utils.Statements import AllTrue

from functools import reduce

class HermiteEquilibria:
    def __init__(self, xi_stencil = None,
                 root_pop = 'f', root_n = 'n', root_u = 'u', root_w = 'w'):
        if xi_stencil is None:
            raise Exception("Missing argument 'xi_stencil'")

        self.root_pop, self.root_n, self.root_u, self.root_w = \
            root_pop, root_n, root_u, root_w
        
        self.idpy_stencil = IdpyStencil(xi_stencil)
        self.D, self.Q = len(self.idpy_stencil.XIs[0]), len(self.idpy_stencil.XIs)
        self.SetZerothAndFirstMomentsSyms(root_n, root_u)
        self.SetPopulationsSyms(root_pop)
        self.SetWSyms(root_w)

    def SetZerothAndFirstMomentsSyms(self, root_n, root_u):
        self.n = sp.Symbol(root_n)
        self.u = sp.Matrix([sp.Symbol(root_u + '_' + str(_))
                            for _ in range(self.D)])
        self.u_symt = SymmetricTensor(
            list_ttuples = TaylorTuples(list(range(self.D)), 1),
            list_values = self.u, d = self.D, rank = 1
        )

    def GetHigherOrderSymmetricUTensor(self, order):
        _taylor_tuples = TaylorTuples(list(self.u), order)
        _taylor_indices = TaylorTuples(list(range(self.D)), order)
        _swap_dict = {}
        for _i, _index_tuple in enumerate(_taylor_indices):
            _swap_dict[_index_tuple] = reduce(lambda x, y: x * y, _taylor_tuples[_i])
        return SymmetricTensor(c_dict = _swap_dict, d = self.D, rank = order)

    '''
    This function looks general enough to be put in idpy.Utils.IdpySymbolic
    '''
    def GetASymmetricTensor(self, order, root_sym = 'A'):
        _taylor_indices = TaylorTuples(list(range(self.D)), order)
        _swap_dict = {}
        for _i, _index_tuple in enumerate(_taylor_indices):
            _lower_indices = reduce(lambda x, y: str(x) + str(y), _index_tuple)
            _swap_dict[_index_tuple] = sp.Symbol(root_sym + "_" + _lower_indices)
        return SymmetricTensor(c_dict = _swap_dict, d = self.D, rank = order)
    
    def SetPopulationsSyms(self, root_pop):
        self.f_i = sp.Matrix([sp.Symbol(root_pop + '_eq_' + str(_))
                              for _ in range(self.Q)])

    def SetWSyms(self, root_w):
        self.w_i = sp.Matrix([sp.Symbol(root_w + '_' + str(_))
                              for _ in range(self.Q)])
        self.w_i_n = sp.Matrix(self.idpy_stencil.Ws)
        self.w_symt = SymmetricTensor(c_dict = {0: self.w_i}, d = self.D, rank = 0)

        
    def GetSymEquilibrium(self, order = 2, tensor_dict = None):
        '''
        The idea is to use the class SymmetricTensor to simplify the writing
        '''        
        if order == 2:
            self.u2_symt = self.GetHigherOrderSymmetricUTensor(order = 2)
            ##return self.n, self.u_symt, self.u2_symt, self.w_symt

            '''
            Maybe this one should return a dict of Symmetric tensors 
            starting from order = 2
            '''            
            _eq_set = \
                self.idpy_stencil.GetEquilibriumHermiteSet(order = order,
                                                           symt_flag = True)
            
            _c2 = self.idpy_stencil.c2
            _f_eq = \
                self.n * (_eq_set[0] +
                          _eq_set[1] * self.u_symt / _c2 +
                          _eq_set[2] * self.u2_symt / (2 * _c2 * _c2))

            if tensor_dict is not None:
                if '1' in tensor_dict:
                    _f_eq += self.n * (_eq_set[1] * tensor_dict['1'] / _c2)
                if '2' in tensor_dict:
                    _f_eq += self.n * (_eq_set[2] * tensor_dict['2'] / (2 * _c2 * _c2))
                    
            for _i in range(self.Q):
                _f_eq[_i] *= self.w_i_n[_i]

            return _f_eq

        if order == 3:
            self.u2_symt = self.GetHigherOrderSymmetricUTensor(order = 2)
            self.u3_symt = self.GetHigherOrderSymmetricUTensor(order = 3)            
            ##return self.n, self.u_symt, self.u2_symt, self.w_symt

            '''
            Maybe this one should return a dict of Symmetric tensors 
            starting from order = 2
            '''            
            _eq_set = \
                self.idpy_stencil.GetEquilibriumHermiteSet(order = order,
                                                           symt_flag = True)
            
            _c2 = self.idpy_stencil.c2
            _f_eq = \
                self.n * (
                    _eq_set[0] +
                    _eq_set[1] * self.u_symt / _c2 +
                    _eq_set[2] * self.u2_symt / (2 * _c2 * _c2) + 
                    _eq_set[3] * self.u3_symt / (6 * _c2 * _c2 * _c2)
                )

            for _i in range(self.Q):
                _f_eq[_i] *= self.w_i_n[_i]

            return _f_eq
        
        if order == 4:
            self.u2_symt = self.GetHigherOrderSymmetricUTensor(order = 2)
            self.u3_symt = self.GetHigherOrderSymmetricUTensor(order = 3)            
            self.u4_symt = self.GetHigherOrderSymmetricUTensor(order = 4)
            ##return self.n, self.u_symt, self.u2_symt, self.w_symt

            '''
            Maybe this one should return a dict of Symmetric tensors 
            starting from order = 2
            '''            
            _eq_set = \
                self.idpy_stencil.GetEquilibriumHermiteSet(order = order,
                                                           symt_flag = True)
            
            _c2 = self.idpy_stencil.c2
            _f_eq = \
                self.n * (
                    _eq_set[0] +
                    _eq_set[1] * self.u_symt / _c2 +
                    _eq_set[2] * self.u2_symt / (2 * _c2 * _c2) + 
                    _eq_set[3] * self.u3_symt / (6 * _c2 * _c2 * _c2) +
                    _eq_set[4] * self.u4_symt / (24 * _c2 * _c2 * _c2 * _c2)
                )

            for _i in range(self.Q):
                _f_eq[_i] *= self.w_i_n[_i]

            return _f_eq
        
    '''
    Need to do the same for the forcing so that each line can be reassembled in 
    the collision case: at the end of the day though, I should be able to write
    everything only in terms of the populations, no matter BGK or MRT
    collect_exprs should be a list of tuples: [0] -> expr, [1] -> new variable
    '''
    def CodifyEquilibriumSinglePopulation(self, order = 2, i = None,
                                          collect_exprs = []):
        if i is None:
            raise Exception("Missing paramter 'i'")
        
        _f_eq = sp.expand(self.GetSymEquilibrium(order = order)[i])
        for _expr_tuple in collect_exprs:
            _f_eq = _f_eq.subs(_expr_tuple[0], _expr_tuple[1])
        return _codify_sympy(_f_eq.evalf())

    '''
    SetMomentsProducts
    '''
    def SetMomentsProductsCode(self, declared_variables = None, declared_constants = None,
                               pressure_mode = 'compute',
                               order = 2, root_n = 'n', root_u = 'u', mom_type = 'MType'):
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
        _n_var = root_n
        _n_sympy_var = sp.Symbol(root_n)

        _u_vars = _get_seq_vars(self.D, root_u)
        _u_sympy_vars = _get_sympy_seq_vars(self.D, root_u)
        
        _needed_variables = [_n_var] + _u_vars
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

        
        _n_prod_sympy, _n_prod_code_sympy, _n_prod_vars = [], [], []
        for _order in range(1, order + 1):
            _swap_sympy, _swap_sympy_code = [], []
            for _tuple in TaylorTuples(_u_sympy_vars, _order):
                _swap_prod = \
                    reduce(lambda x, y: _n_sympy_var * x * y, _tuple) \
                    if _order > 1 else _n_sympy_var * _tuple

                _swap_code_prod = \
                    reduce(lambda x, y:
                           _codify_sympy(x) + '*' + _codify_sympy(y), _tuple) \
                           if _order > 1 else _codify_sympy(_tuple)
                
                _swap_sympy += [_swap_prod]
                _swap_sympy_code += [_n_var + '*' + _swap_code_prod]
                
            _n_prod_sympy += _swap_sympy
            _n_prod_code_sympy += _swap_sympy_code
            _n_prod_vars += [sp.Symbol(str(_).replace('*', '_'))
                             for _ in _swap_sympy_code]
            
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
                        _expr = _codify_sympy(_n_prod_code_sympy[_i]),
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
    
    '''
    SetConstsEquilibrium
    '''
    def SetConstsEquilibrium(self, declared_variables = None, declared_constants = None,
                             root_n = 'n', root_u = 'u'):
        pass
    
    '''
    ComputeEquilibriumCode
    '''       
    def ComputeEquilibriumCode(self, declared_variables = None, declared_constants = None,
                               pressure_mode = 'compute'):
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
        Check for needed variables
        '''
        _needed_variables = [root_n] + _get_seq_macros(self.D, root_u)
        _chk_needed_variables = []
        for _ in _needed_variables:
            _chk_needed_variables += [_ in declared_variables[0] or
                                      _ in declared_constants[0]]
        
