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

from collections import defaultdict
from idpy.LBM.LBM import LBMTypes
from idpy.IdpyStencils.IdpyStencils import IdpyStencil
from idpy.IdpyCode.IdpyUnroll import _get_seq_vars, _codify_sympy_declaration
from idpy.IdpyCode.IdpyUnroll import _array_value, _codify_mul_assignment, _codify_newl
from idpy.IdpyCode.IdpyUnroll import _codify_declaration_const_check, _codify_sympy

import sympy as sp

class IDShanChenForce(IdpyStencil):
    def __init__(self, FStencil = None):
        if FStencil is None:
            raise Exception("Missing arhument 'FStencil'")
        
        IdpyStencil.__init__(self, FStencil)
        
    def ForceCodeMultiPhase(self, declared_variables = None, declared_constants = None,
                            pressure_mode = 'compute', use_ptrs = False,
                            psi_array = 'psi', psi_type = 'PsiType',
                            root_dim_sizes = 'L', root_strides = 'STR',
                            root_coord = 'x', lex_index = 'g_tid', g_coupling = 'SC_G'):

        '''
        Checking that the list of declared variables is available
        '''
        if declared_variables is None:
            raise Exception("Missing argument 'declared_variables'")
        if type(declared_variables) != list:
            raise Exception("Argument 'declared_variables' must be a list containing one list")
        if len(declared_variables) == 0:
            raise Exception("List 'declared_values' must contain another list!")

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
        Defining and checking the variables that are needed
        '''
        _dim = len(self.XIs[0])
        _needed_variables = _get_seq_vars(_dim, root_coord) + [g_coupling]
        
        _chk_needed_variables = []
        for _ in _needed_variables:
            _chk_needed_variables += [_ in declared_variables[0] or
                                      _ in declared_constants[0]]
                
        _swap_code = """"""
        '''
        Set the local value of the pseudo-potential
        '''
        _l_psi_sym = sp.Symbol('l_' + psi_array)

        _swap_code += \
            _codify_declaration_const_check(
                _codify_sympy(_l_psi_sym),
                _array_value(psi_array, lex_index, use_ptrs), psi_type,
                declared_variables, declared_constants,
                declare_const_flag = True
            )
        _swap_code += _codify_newl

        '''
        Compute the gradient: output d_psi_0, d_psi_1
        '''
        _swap_code += IdpyStencil.GradientCode(self, declared_variables, declared_constants,
                                               arrays_vars = ['psi'],
                                               arrays_types = ['PsiType'],
                                               pressure_mode = pressure_mode,
                                               pos_type = 'SType', 
                                               use_ptrs = use_ptrs, 
                                               root_dim_sizes = root_dim_sizes, 
                                               root_strides = root_strides,
                                               root_coord = root_coord, 
                                               lex_index = lex_index)
        '''
        Compute the Shan-Chen force
        '''
        for _d in range(_dim):
            _swap_code += _codify_mul_assignment('d_psi_' + str(_d), 
                                                 '-SC_G * ' + str(_l_psi_sym))
        
        return _swap_code
