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
import numpy as np
from collections import defaultdict
from idpy.Utils.IdpySymbolic import SymmetricTensor, TaylorTuples

IDStencils = defaultdict( # Pertinent idpy module
    lambda: defaultdict( # IDStencil name
        lambda: defaultdict(dict) # IDStencil properties
    )
)

'''
using sp.Rational for the weights should be fine when creating the W_list
in the naive version of the LBM module
'''

IDStencils['Ising']['NN_D2'] = {'XIs': ((1, 0), (0, 1), (-1, 0), (0, -1)),
                                'Ws': (sp.Rational(1, 4), sp.Rational(1, 4),
                                       sp.Rational(1, 4), sp.Rational(1, 4)),
                                'e2': 1}

IDStencils['Ising']['NN_D3'] = {'XIs': ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
                                        (0, 0, 1), (0, 0, -1)),
                                'Ws': (sp.Rational(1, 6), sp.Rational(1, 6), sp.Rational(1, 6),
                                       sp.Rational(1, 6), sp.Rational(1, 6), sp.Rational(1, 6)),
                                'e2': 1}

IDStencils['Ising']['NN_D4'] = {'XIs': ((1, 0, 0, 0), (0, 1, 0, 0), (-1, 0, 0, 0), (0, -1, 0, 0),
                                        (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1)),
                                'Ws': (sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8),
                                       sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8)),
                                'e2': 1}

IDStencils['LBM']['XI_D2Q9'] = {'XIs': ((0, 0),
                                        (1, 0), (0, 1), (-1, 0), (0, -1),
                                        (1, 1), (-1, 1), (-1, -1), (1, -1)),
                                'Ws': (sp.Rational(4, 9),
                                       sp.Rational(1, 9), sp.Rational(1, 9), sp.Rational(1, 9), sp.Rational(1, 9),
                                       sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36)),
                                'c2': sp.Rational(1, 3)}

IDStencils['LBM']['XI_D3Q15'] = {'XIs': ((0, 0, 0),
                                         (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                                         (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1),
                                         (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)),
                                 'Ws': (sp.Rational(2, 9),
                                        sp.Rational(1, 9), sp.Rational(1, 9), sp.Rational(1, 9),
                                        sp.Rational(1, 9), sp.Rational(1, 9), sp.Rational(1, 9),
                                        sp.Rational(1, 72), sp.Rational(1, 72), sp.Rational(1, 72), sp.Rational(1, 72),
                                        sp.Rational(1, 72), sp.Rational(1, 72), sp.Rational(1, 72), sp.Rational(1, 72)),
                                 'c2': sp.Rational(1, 3)}


IDStencils['LBM']['XI_D3Q19'] = {'XIs': ((0, 0, 0),
                                         (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                                         (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
                                         (1, 0, 1), (-1, 0, 1), (-1, 0, -1), (1, 0, -1),
                                         (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1)),
                                 'Ws': (sp.Rational(1, 3),
                                        sp.Rational(1, 18), sp.Rational(1, 18), sp.Rational(1, 18),
                                        sp.Rational(1, 18), sp.Rational(1, 18), sp.Rational(1, 18),
                                        sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36),
                                        sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36),
                                        sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36)),
                                 'c2': sp.Rational(1, 3)}

IDStencils['LBM']['XI_D3Q27'] = {'XIs': ((0, 0, 0),
                                         (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                                         (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
                                         (1, 0, 1), (-1, 0, 1), (-1, 0, -1), (1, 0, -1),
                                         (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1),
                                         (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1),
                                         (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)),
                                 'Ws': (sp.Rational(8, 27),
                                        sp.Rational(2, 27), sp.Rational(2, 27), sp.Rational(2, 27),
                                        sp.Rational(2, 27), sp.Rational(2, 27), sp.Rational(2, 27),
                                        sp.Rational(1, 54), sp.Rational(1, 54), sp.Rational(1, 54), sp.Rational(1, 54),
                                        sp.Rational(1, 54), sp.Rational(1, 54), sp.Rational(1, 54), sp.Rational(1, 54),
                                        sp.Rational(1, 54), sp.Rational(1, 54), sp.Rational(1, 54), sp.Rational(1, 54),
                                        sp.Rational(1, 216), sp.Rational(1, 216), sp.Rational(1, 216), sp.Rational(1, 216),
                                        sp.Rational(1, 216), sp.Rational(1, 216), sp.Rational(1, 216), sp.Rational(1, 216)),
                                 'c2': sp.Rational(1, 3)}

IDStencils['LBM']['SC_D2E4'] = {'XIs': ((1, 0), (0, 1), (-1, 0), (0, -1),
                                        (1, 1), (-1, 1), (-1, -1), (1, -1)),
                                'Ws': (sp.Rational(1, 3), sp.Rational(1, 3), sp.Rational(1, 3), sp.Rational(1, 3),
                                       sp.Rational(1, 12), sp.Rational(1, 12), sp.Rational(1, 12), sp.Rational(1, 12)), 
                                'e2': 1}

IDStencils['LBM']['SC_D2E6'] = {'XIs': ((1, 0), (0, 1), (-1, 0), (0, -1),
                                        (1, 1), (-1, 1), (-1, -1), (1, -1),
                                        (2, 0), (0, 2), (-2, 0), (0, -2)),
                                'Ws': (sp.Rational(4, 15), sp.Rational(4, 15), sp.Rational(4, 15), sp.Rational(4, 15),
                                       sp.Rational(1, 10), sp.Rational(1, 10), sp.Rational(1, 10), sp.Rational(1, 10),
                                       sp.Rational(1, 120), sp.Rational(1, 120), sp.Rational(1, 120), sp.Rational(1, 120)), 
                                'e2': 1}

IDStencils['LBM']['SC_D2E8'] = {'XIs': ((1, 0), (0, 1), (-1, 0), (0, -1),
                                        (1, 1), (-1, 1), (-1, -1), (1, -1),
                                        (2, 0), (0, 2), (-2, 0), (0, -2),
                                        (2, 1), (1, 2), (-1, 2), (-2, 1),
                                        (-2, -1), (-1, -2), (1, -2), (2, -1),
                                        (2, 2), (-2, 2), (-2, -2), (2, -2)),
                                'Ws': (sp.Rational(4, 21), sp.Rational(4, 21), sp.Rational(4, 21), sp.Rational(4, 21),
                                       sp.Rational(4, 45), sp.Rational(4, 45), sp.Rational(4, 45), sp.Rational(4, 45),
                                       sp.Rational(1, 60), sp.Rational(1, 60), sp.Rational(1, 60), sp.Rational(1, 60),
                                       sp.Rational(2, 315), sp.Rational(2, 315), sp.Rational(2, 315), sp.Rational(2, 315),
                                       sp.Rational(2, 315), sp.Rational(2, 315), sp.Rational(2, 315), sp.Rational(2, 315),
                                       sp.Rational(1, 5040), sp.Rational(1, 5040), sp.Rational(1, 5040), sp.Rational(1, 5040)), 
                                'e2': 1}

IDStencils['LBM']['SC_D3E4'] = {'XIs': ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                                        (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
                                        (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1),
                                        (1, 0, 1), (1, 0, -1), (-1, 0, -1), (-1, 0, 1)),
                                'Ws': (sp.Rational(1, 6), sp.Rational(1, 6), sp.Rational(1, 6), sp.Rational(1, 6),
                                       sp.Rational(1, 6), sp.Rational(1, 6),
                                       sp.Rational(1, 12), sp.Rational(1, 12), sp.Rational(1, 12), sp.Rational(1, 12),
                                       sp.Rational(1, 12), sp.Rational(1, 12),
                                       sp.Rational(1, 12), sp.Rational(1, 12), sp.Rational(1, 12), sp.Rational(1, 12),
                                       sp.Rational(1, 12), sp.Rational(1, 12)), 
                                'e2': 1}


IDStencils['LBM']['SC_D3E6'] = \
    {'XIs': 
        ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), 
        (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
        (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1),
        (1, 0, 1), (1, 0, -1), (-1, 0, -1), (-1, 0, 1),
        (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1),
        (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1),
        (2, 0, 0), (0, 2, 0), (-2, 0, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)),
    'Ws': (sp.Rational(2, 15), sp.Rational(2, 15), sp.Rational(2, 15), sp.Rational(2, 15), sp.Rational(2, 15), sp.Rational(2, 15),
            sp.Rational(1, 15), sp.Rational(1, 15), sp.Rational(1, 15), sp.Rational(1, 15),
            sp.Rational(1, 15), sp.Rational(1, 15), sp.Rational(1, 15), sp.Rational(1, 15),
            sp.Rational(1, 15), sp.Rational(1, 15), sp.Rational(1, 15), sp.Rational(1, 15),
            sp.Rational(1, 60), sp.Rational(1, 60), sp.Rational(1, 60), sp.Rational(1, 60),
            sp.Rational(1, 60), sp.Rational(1, 60), sp.Rational(1, 60), sp.Rational(1, 60),
            sp.Rational(1, 120), sp.Rational(1, 120), sp.Rational(1, 120), sp.Rational(1, 120), sp.Rational(1, 120), sp.Rational(1, 120))
    }

IDStencils['LBM']['SC_D3E8'] = \
    {'XIs': 
        ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
        (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1),
        (1, 0, 1), (1, 0, -1), (-1, 0, -1), (-1, 0, 1),
        (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1),
        (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1),
        (2, 0, 0), (0, 2, 0), (-2, 0, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2),
        (2, 1, 0), (2, -1, 0), (2, 0, 1), (2, 0, -1),
        (1, 2, 0), (-1, 2, 0), (0, 2, 1), (0, 2, -1),
        (-2, 1, 0), (-2, -1, 0), (-2, 0, 1), (-2, 0, -1),
        (1, -2, 0), (-1, -2, 0), (0, -2, 1), (0, -2, -1),
        (1, 0, 2), (-1, 0, 2), (0, 1, 2), (0, -1, 2),
        (1, 0, -2), (-1, 0, -2), (0, 1, -2), (0, -1, -2),
        (2, 1, 1), (2, -1, 1), (2, -1, -1), (2, 1, -1),
        (1, 2, 1), (-1, 2, 1), (-1, 2, -1), (1, 2, -1),
        (-2, 1, 1), (-2, -1, 1), (-2, -1, -1), (-2, 1, -1),
        (1, -2, 1), (-1, -2, 1), (-1, -2, -1), (1, -2, -1),
        (1, 1, 2), (-1, 1, 2), (-1, -1, 2), (1, -1, 2),
        (1, 1, -2), (-1, 1, -2), (-1, -1, -2), (1, -1, -2),
        (2, 2, 0), (-2, 2, 0), (-2, -2, 0), (2, -2, 0),
        (2, 0, 2), (-2, 0, 2), (-2, 0, -2), (2, 0, -2),
        (0, 2, 2), (0, -2, 2), (0, -2, -2), (0, 2, -2)),
    'Ws': (sp.Rational(4, 45), sp.Rational(4, 45), sp.Rational(4, 45), sp.Rational(4, 45), sp.Rational(4, 45), sp.Rational(4, 45),
            sp.Rational(1, 21), sp.Rational(1, 21), sp.Rational(1, 21), sp.Rational(1, 21),
            sp.Rational(1, 21), sp.Rational(1, 21), sp.Rational(1, 21), sp.Rational(1, 21),
            sp.Rational(1, 21), sp.Rational(1, 21), sp.Rational(1, 21), sp.Rational(1, 21),
            sp.Rational(2, 105), sp.Rational(2, 105), sp.Rational(2, 105), sp.Rational(2, 105),
            sp.Rational(2, 105), sp.Rational(2, 105), sp.Rational(2, 105), sp.Rational(2, 105),
            sp.Rational(5, 504), sp.Rational(5, 504), sp.Rational(5, 504), sp.Rational(5, 504), sp.Rational(5, 504), sp.Rational(5, 504),
            sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315),
            sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315),
            sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315),
            sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315),
            sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315),
            sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315), sp.Rational(1, 315),
            sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630),
            sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630),
            sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630),
            sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630),
            sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630),
            sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630), sp.Rational(1, 630),
            sp.Rational(1, 5040), sp.Rational(1, 5040), sp.Rational(1, 5040), sp.Rational(1, 5040),
            sp.Rational(1, 5040), sp.Rational(1, 5040), sp.Rational(1, 5040), sp.Rational(1, 5040),
            sp.Rational(1, 5040), sp.Rational(1, 5040), sp.Rational(1, 5040), sp.Rational(1, 5040))
    }



from idpy.IdpyCode.IdpyUnroll import _get_single_neighbor_pos_macro_fully_sym
from idpy.IdpyCode.IdpyUnroll import _codify, _codify_comment, _codify_newl
from idpy.IdpyCode.IdpyUnroll import _codify_assignment, _codify_declaration
from idpy.IdpyCode.IdpyUnroll import _codify_sympy, _codify_sympy_assignment, _codify_sympy_declaration

from idpy.IdpyCode.IdpyUnroll import _codify_add_assignment, _codify_mul_assignment
from idpy.IdpyCode.IdpyUnroll import _codify_sub_assignment, _codify_div_assignment
from idpy.IdpyCode.IdpyUnroll import _codify_sympy_add_assignment, _codify_sympy_mul_assignment
from idpy.IdpyCode.IdpyUnroll import _codify_sympy_sub_assignment, _codify_sympy_div_assignment
from idpy.IdpyCode.IdpyUnroll import _sp_macro, _sm_macro, _subs_sp_sm_macros
from idpy.IdpyCode.IdpyUnroll import _neighbors_register_pressure_macro

from idpy.IdpyCode.IdpyUnroll import _get_seq_macros, _get_seq_vars, _array_value
from idpy.IdpyCode.IdpyUnroll import _get_sympy_seq_macros, _get_sympy_seq_vars
from idpy.IdpyCode.IdpyUnroll import _codify_declaration_const, _codify_sympy_declaration_const
from idpy.IdpyCode.IdpyUnroll import _codify_declaration_const_check

from idpy.Utils.Geometry import GetLen2Pos, IsOppositeVector, ProjectionVAlongU, ProjectVAlongU
from idpy.Utils.Statements import AllTrue

from idpy.Utils.ManageData import ManageData
from idpy.Utils.Hermite import Hermite, HermiteWProd

from idpy.Utils.IdpySymbolic import TaylorTuples
from idpy.Utils.Statements import AllTrue, OneTrue

from itertools import combinations

class IdpyStencil:
    def __init__(self, stencil_dict = None, root_xi_sym = '\\xi'):
        if stencil_dict is None:
            raise Exception("Missing 'stencil_dict' argument")

        if 'XIs' in stencil_dict:
            self.XIs = stencil_dict['XIs']
        if 'Es' in stencil_dict:
            self.XIs = stencil_dict['Es']
            
        self.Ws = stencil_dict['Ws']

        if 'c2' in stencil_dict:
            self.c2 = self.e2 = stencil_dict['c2']
        if 'e2' in stencil_dict:
            self.c2 = self.e2 = stencil_dict['e2']
            
        self.SetOppositeDirections()
        self.manage_data = ManageData()
        self.SetXISyms(root_sym = root_xi_sym)

    def SetWSTensor():
        pass

    def SetXISyms(self, root_sym = '\\xi'):
        self.D, self.Q = len(self.XIs[0]), len(self.XIs)
        self.XISyms = \
            sp.Matrix(
                [
                    [sp.Symbol(root_sym + '_' + str(_i) + '_' + str(_d))
                     for _d in range(self.D)]
                    for _i in range(self.Q)
                ]
            )

    def ZeroHermiteCode(self, declared_variables = None, declared_constants = None,
                        arrays = ['array'], arrays_types = ['AType'],
                        root_n = 'n', n_type = 'n_type',
                        lex_index = 'g_tid', keep_read = True,
                        ordering_lambdas = None,
                        use_ptrs = False,
                        declare_const_dict = {'arrays_xi': True, 'moments': False}):

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
        if ordering_lambdas is None:
            raise Exception("Missing argument 'ordering_lambdas'")
        if type(ordering_lambdas) != list:
            raise Exception("Argument 'ordering_lambdas' must be a list of lambdas")
        if len(ordering_lambdas) != len(arrays):
            raise Exception("The length of 'ordering_lambdas' must be the same as 'arrays'")
        
        '''
        Defining and checking the variables that are needed
        '''
        _dim, _Q = len(self.XIs[0]), len(self.XIs)
 
        _needed_variables = arrays        
        _chk_needed_variables = []
        for _ in _needed_variables:
            _chk_needed_variables += [_ in declared_variables[0] or _ in declared_constants[0]]
        if not AllTrue(_chk_needed_variables):
            print()
            for _i, _ in enumerate(_chk_needed_variables):
                if not _:
                    print("Variable/constant ", _needed_variables[_i], "not declared!")
            raise Exception("Some needed variables/constants have not been declared yet (!)")
        
        _swap_code = """"""
        
        '''
        defining the symbols and the vector for 'array'
        '''
        _array_vars, _vector_array_vars = [], []
        for _i_a, _array in enumerate(arrays):
            _swap_a = _get_sympy_seq_vars(_Q, _array)
            _array_vars += [_swap_a]
            _vector_array_vars += [sp.Matrix(_swap_a)]

            for _i_var, _var in enumerate(_swap_a):
                _sx_hnd = _codify_sympy(_var)
                _dx_hnd = \
                    _array_value(_array,
                                 ordering_lambdas[_i_a](lex_index, _i_var),
                                 use_ptrs)
                _swap_code += \
                    _codify_declaration_const_check(
                        _sx_hnd, _dx_hnd,
                        arrays_types[_i_a],
                        declared_variables,
                        declared_constants,
                        declare_const_flag = declare_const_dict['arrays_xi']
                    )

        _swap_code += _codify_newl

        '''
        Defining the symbols for the array entries
        '''
        _n, _u_vars = [], []

        for _a_i in range(len(arrays)):
            _n += [sp.Symbol(root_n + '_' + str(_a_i))]
        
        '''
        Computing and declaring the momenta
        '''
        _hydro_set = self.GetHydroHermiteSet()
        for _a_i in range(len(arrays)):
            _moment_swap = _hydro_set * _vector_array_vars[_a_i]
            
            _swap_code += \
                _codify_declaration_const_check(
                    _codify_sympy(_n[_a_i]), _codify_sympy(_moment_swap[0]),
                    n_type,
                    declared_variables,
                    declared_constants,
                    declare_const_flag = declare_const_dict['moments']
                )
            _swap_code += _codify_newl

        return _swap_code

    def VelocityCode(self, declared_variables = None, declared_constants = None,
                     arrays = ['array'], arrays_types = ['AType'],
                     root_n = 'n', root_u = 'u', arrays_l_root = 'l',
                     n_type = 'n_type', u_type = 'u_type',
                     lex_index = 'g_tid', keep_read = True,
                     ordering_lambdas = None,
                     use_ptrs = False,
                     declare_const_dict = {'arrays_xi': True, 'moments': False}):

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
        if ordering_lambdas is None:
            raise Exception("Missing argument 'ordering_lambdas'")
        if type(ordering_lambdas) != list:
            raise Exception("Argument 'ordering_lambdas' must be a list of lambdas")
        if len(ordering_lambdas) != len(arrays):
            raise Exception("The length of 'ordering_lambdas' must be the same as 'arrays'")
        
        '''
        Defining and checking the variables that are needed
        '''
        _dim, _Q = len(self.XIs[0]), len(self.XIs)

        '''
        Local densities values
        '''
        _n = _get_seq_vars(len(arrays), root_n)
        
        _needed_variables = arrays + _n
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
        
        '''
        defining the symbols and the vector for 'array'
        '''
        _array_vars, _vector_array_vars = [], []
        for _i_a, _array in enumerate(arrays):
            _swap_a = _get_sympy_seq_vars(_Q, arrays_l_root + _array)
            _array_vars += [_swap_a]
            _vector_array_vars += [sp.Matrix(_swap_a)]

            for _i_var, _var in enumerate(_swap_a):
                _sx_hnd = _codify_sympy(_var)
                _dx_hnd = \
                    _array_value(_array,
                                 ordering_lambdas[_i_a](lex_index, _i_var),
                                 use_ptrs)
                _swap_code += \
                    _codify_declaration_const_check(
                        _sx_hnd, _dx_hnd,
                        arrays_types[_i_a],
                        declared_variables,
                        declared_constants,
                        declare_const_flag = declare_const_dict['arrays_xi']
                    )

        _swap_code += _codify_newl

        '''
        Defining the symbols for the array entries
        ''' 
        _n_vars = _get_sympy_seq_vars(len(arrays), root_n)
        _u_vars = []
        for _i in range(len(arrays)):
            _u_vars += [_get_sympy_seq_vars(_dim, root_u + '_' + str(_i))]

        
        '''
        Computing and declaring the momenta
        '''
        _hydro_set = self.GetHydroHermiteSet()
        for _a_i in range(len(arrays)):
            _moment_swap = _hydro_set * _vector_array_vars[_a_i]

            for _u_i, _u in enumerate(_u_vars[_a_i]):
                _swap_code += \
                    _codify_declaration_const_check(
                        _codify_sympy(_u),
                        _codify_sympy(_moment_swap[_u_i + 1] / _n_vars[_a_i]),
                        u_type,
                        declared_variables,
                        declared_constants,
                        declare_const_flag = declare_const_dict['moments']
                    )
            _swap_code += _codify_newl
            
        return _swap_code
    
    def HydroHermiteCode(self, declared_variables = None, declared_constants = None,
                         arrays = ['array'], arrays_types = ['AType'],
                         root_n = 'n', root_u = 'u',
                         n_type = 'n_type', u_type = 'u_type',
                         lex_index = 'g_tid', keep_read = True,
                         ordering_lambdas = None,
                         use_ptrs = False,
                         declare_const_dict = {'arrays_xi': True, 'moments': False}):
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
        if ordering_lambdas is None:
            raise Exception("Missing argument 'ordering_lambdas'")
        if type(ordering_lambdas) != list:
            raise Exception("Argument 'ordering_lambdas' must be a list of lambdas")
        if len(ordering_lambdas) != len(arrays):
            raise Exception("The length of 'ordering_lambdas' must be the same as 'arrays'")


        '''
        Defining and checking the variables that are needed
        '''
        _dim, _Q = len(self.XIs[0]), len(self.XIs)
 
        _needed_variables = arrays        
        _chk_needed_variables = []
        for _ in _needed_variables:
            _chk_needed_variables += [_ in declared_variables[0] or _ in declared_constants[0]]
        if not AllTrue(_chk_needed_variables):
            print()
            for _i, _ in enumerate(_chk_needed_variables):
                if not _:
                    print("Variable/constant ", _needed_variables[_i], "not declared!")
            raise Exception("Some needed variables/constants have not been declared yet (!)")
        
        _swap_code = """"""

        '''
        defining the symbols and the vector for 'array'
        '''
        _array_vars, _vector_array_vars = [], []
        for _i_a, _array in enumerate(arrays):
            _swap_a = _get_sympy_seq_vars(_Q, _array)
            _array_vars += [_swap_a]
            _vector_array_vars += [sp.Matrix(_swap_a)]

            for _i_var, _var in enumerate(_swap_a):
                _sx_hnd = _codify_sympy(_var)
                _dx_hnd = \
                    _array_value(_array,
                                 ordering_lambdas[_i_a](lex_index, _i_var),
                                 use_ptrs)
                _swap_code += \
                    _codify_declaration_const_check(
                        _sx_hnd, _dx_hnd,
                        arrays_types[_i_a],
                        declared_variables,
                        declared_constants,
                        declare_const_flag = declare_const_dict['arrays_xi']
                    )

        _swap_code += _codify_newl

        '''
        Defining the symbols for the array entries
        '''
        _n, _u_vars = [], []

        for _a_i in range(len(arrays)):
            _n += [sp.Symbol(root_n + '_' + str(_a_i))]
            _u_vars += [_get_sympy_seq_vars(_dim, root_u + '_' + str(_a_i))]

        '''
        Computing and declaring the momenta
        '''
        _hydro_set = self.GetHydroHermiteSet()
        for _a_i in range(len(arrays)):
            _moment_swap = _hydro_set * _vector_array_vars[_a_i]
            
            _swap_code += \
                _codify_declaration_const_check(
                    _codify_sympy(_n[_a_i]), _codify_sympy(_moment_swap[0]),
                    n_type,
                    declared_variables,
                    declared_constants,
                    declare_const_flag = declare_const_dict['moments']
                )
            _swap_code += _codify_newl

            for _u_i, _u in enumerate(_u_vars[_a_i]):
                _swap_code += \
                    _codify_declaration_const_check(
                        _codify_sympy(_u),
                        _codify_sympy(_moment_swap[_u_i + 1] / _n[_a_i]),
                        u_type,
                        declared_variables,
                        declared_constants,
                        declare_const_flag = declare_const_dict['moments']
                    )
                _swap_code += _codify_newl

        return _swap_code

    '''
    def MomentsHermiteCode
    '''
    def MomentsHermiteCode(self, declared_variables = None, declared_constants = None,
                           arrays = ['array'], arrays_types = ['AType'],
                           root_neq_mom = 'neq_m',
                           mom_type = 'mom_type',
                           lex_index = 'g_tid', keep_read = True,
                           ordering_lambdas = None,
                           use_ptrs = False, search_depth = 6,
                           declare_const_dict = {'arrays_xi': True, 'moments': False}):
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
        if ordering_lambdas is None:
            raise Exception("Missing argument 'ordering_lambdas'")
        if type(ordering_lambdas) != list:
            raise Exception("Argument 'ordering_lambdas' must be a list of lambdas")
        if len(ordering_lambdas) != len(arrays):
            raise Exception("The length of 'ordering_lambdas' must be the same as 'arrays'")


        '''
        Defining and checking the variables that are needed
        '''
        _dim, _Q = len(self.XIs[0]), len(self.XIs)
 
        _needed_variables = arrays        
        _chk_needed_variables = []
        for _ in _needed_variables:
            _chk_needed_variables += [_ in declared_variables[0] or _ in declared_constants[0]]
        if not AllTrue(_chk_needed_variables):
            print()
            for _i, _ in enumerate(_chk_needed_variables):
                if not _:
                    print("Variable/constant ", _needed_variables[_i], "not declared!")
            raise Exception("Some needed variables/constants have not been declared yet (!)")
        
        _swap_code = """"""

        '''
        defining the symbols and the vector for 'array'
        '''
        _array_vars, _vector_array_vars = [], []
        for _i_a, _array in enumerate(arrays):
            _swap_a = _get_sympy_seq_vars(_Q, _array)
            _array_vars += [_swap_a]
            _vector_array_vars += [sp.Matrix(_swap_a)]

            for _i_var, _var in enumerate(_swap_a):
                _sx_hnd = _codify_sympy(_var)
                _dx_hnd = \
                    _array_value(_array,
                                 ordering_lambdas[_i_a](lex_index, _i_var),
                                 use_ptrs)
                _swap_code += \
                    _codify_declaration_const_check(
                        _sx_hnd, _dx_hnd,
                        arrays_types[_i_a],
                        declared_variables,
                        declared_constants,
                        declare_const_flag = declare_const_dict['arrays_xi']
                    )

        _swap_code += _codify_newl

        '''
        Defining the symbols for the array entries
        '''
        _neq_mom_vars = []
        for _a_i in range(len(arrays)):
            _neq_mom_vars += [_get_sympy_seq_vars(_Q, root_neq_mom + '_' + str(_a_i))]

        '''
        Computing and declaring the momenta
        '''
        _full_hermite_set = self.GetInvertibleHermiteSet(search_depth = search_depth)['M']
        for _a_i in range(len(arrays)):
            _moment_swap = _full_hermite_set * _vector_array_vars[_a_i]
            
            for _m_i, _m in enumerate(_neq_mom_vars[_a_i]):
                _swap_code += \
                    _codify_declaration_const_check(
                        _codify_sympy(_m),
                        _codify_sympy(_moment_swap[_m_i].evalf()),
                        mom_type,
                        declared_variables,
                        declared_constants,
                        declare_const_flag = declare_const_dict['moments']
                    )
            _swap_code += _codify_newl

        return _swap_code
    

    '''
    symt_flag: flag to toggle the output as a idpy.Utils.IdpySymbolic.SymmetricTensor
    '''
    def GetEquilibriumHermiteSet(self, order = 2, symt_flag = False, root_xi_sym = '\\xi'):
        _set_dict = {}
        for _k in range(0, order + 1):
            _set_swap = self.GetHermiteOrder(_order = _k, root_xi_sym = root_xi_sym)
            if not symt_flag:
                _set_dict[_k] = _set_swap
            else:
                if _k == 0:
                    _set_dict[_k] = _set_swap.T
                else:
                    _set_dict[_k] = \
                        SymmetricTensor(
                            list_ttuples = TaylorTuples(list(range(self.D)), _k),
                            list_values = [_set_swap.T[:,_]
                                           for _ in range(_set_swap.T.shape[1])],
                            d = self.D, rank = _k
                        )
        return _set_dict

    def GetEquilibriumHermiteSetSYMT(self, order = 2, root_xi_sym = '\\xi'):
        _in_dict = GetEquilibriumHermiteSet(order = order, root_xi_sym = root_xi_sym)
        _out_dict = {0: _in_dict[0].T, 1: _in_dict[1].T}
        for _k in range(2, order + 1):
            _swap_dict = {}
            for _i, _index_tuple in enumerate(TaylorTuples(list(range(self.D)), _k)):
                _swap_dict[_index_tuple] = _in_dict[_k].T[:,_i]                
            _out_dict[_k] = SymmetricTensor(_swap_dict, self.D, _k)
            
        return _out_dict
    
    def GetHydroHermiteSet(self, root_xi_sym = '\\xi'):
        return self.GetHermiteSet(_max_order = 1, root_xi_sym = root_xi_sym)

    def GetHermiteOrder(self, _order, root_xi_sym = '\\xi'):
        _hc = Hermite(d = self.D, root_sym = 'c')
        _c_sym_list = _hc.sym_list
        _xi_sym_list = [sp.Symbol(root_xi_sym + '_' + str(_)) for _ in range(self.D)]
        _c_s_sym = sp.Symbol('c_s')
        
        _M_coeffs = []
        if _order == 0:
            _M_coeffs = [[1] * self.Q]
            return sp.Matrix([_row for _row in _M_coeffs])
        else:
            for _tuple in TaylorTuples(list(range(self.D)), _order):
                _hermite_pol = _hc.GetH(_order)[_tuple]

                for _i in range(len(_c_sym_list)):
                    _hermite_pol = _hermite_pol.subs(_c_sym_list[_i],
                                                     _xi_sym_list[_i] / _c_s_sym)

                _hermite_pol = sp.simplify(((_c_s_sym ** _order) * _hermite_pol))
                _hermite_pol = _hermite_pol.subs(_c_s_sym, sp.sqrt(self.c2))

                _M_coeffs_swap = []
                for _xi in self.XIs:
                    _coeff_swap = _hermite_pol
                    for _i in range(len(_xi)):
                        _coeff_swap = _coeff_swap.subs(_xi_sym_list[_i], _xi[_i])
                    _M_coeffs_swap += [_coeff_swap]

                _M_coeffs += [_M_coeffs_swap]

            return sp.Matrix([_row for _row in _M_coeffs])
    
    def GetHermiteSet(self, _max_order = 1, root_xi_sym = '\\xi'):
        _hc = Hermite(d = self.D, root_sym = 'c')
        _c_sym_list = _hc.sym_list
        _xi_sym_list = [sp.Symbol(root_xi_sym + '_' + str(_)) for _ in range(self.D)]
        _c_s_sym = sp.Symbol('c_s')
        
        _M_coeffs = [[1] * self.Q]
        
        _count_momenta = 1

        for _order in range(1, _max_order + 1):
            for _tuple in TaylorTuples(list(range(self.D)), _order):
                _hermite_pol = _hc.GetH(_order)[_tuple]

                for _i in range(len(_c_sym_list)):
                    _hermite_pol = _hermite_pol.subs(_c_sym_list[_i],
                                                     _xi_sym_list[_i] / _c_s_sym)

                _hermite_pol = sp.simplify(((_c_s_sym ** _order) * _hermite_pol))
                _hermite_pol = _hermite_pol.subs(_c_s_sym, sp.sqrt(self.c2))

                _M_coeffs_swap = []
                for _xi in self.XIs:
                    _coeff_swap = _hermite_pol
                    for _i in range(len(_xi)):
                        _coeff_swap = _coeff_swap.subs(_xi_sym_list[_i], _xi[_i])
                    _M_coeffs_swap += [_coeff_swap]

                _M_coeffs += [_M_coeffs_swap]

        return sp.Matrix([_row for _row in _M_coeffs])

    '''
    Need to watch out for situations in which there is more than a possibility
    '''
    def GetInvertibleHermiteSet(self, search_depth, root_xi_sym = '\\xi'):
        if not hasattr(self, 'M'):
            _hc = Hermite(d = self.D, root_sym = 'c')
            _c_sym_list = _hc.sym_list
            _xi_sym_list = [sp.Symbol(root_xi_sym + '_' + str(_)) for _ in range(self.D)]
            _c_s_sym = sp.Symbol('c_s')

            _M_coeffs = [[1] * self.Q]

            _count_momenta, _order = 1, 1
            _hermite_polys_list, _taylor_tuples_list = [1], [None]

            while _count_momenta < self.Q + search_depth:
                for _tuple in TaylorTuples(list(range(self.D)), _order):
                    _hermite_pol = _hc.GetH(_order)[_tuple]

                    for _i in range(len(_c_sym_list)):
                        _hermite_pol = _hermite_pol.subs(_c_sym_list[_i],
                                                         _xi_sym_list[_i] / _c_s_sym)

                    _hermite_pol = sp.simplify(((_c_s_sym ** _order) * _hermite_pol))
                    _hermite_pol = _hermite_pol.subs(_c_s_sym, sp.sqrt(self.c2))

                    _M_coeffs_swap = []
                    for _xi in self.XIs:
                        _coeff_swap = _hermite_pol
                        for _i in range(len(_xi)):
                            _coeff_swap = _coeff_swap.subs(_xi_sym_list[_i], _xi[_i])
                        _M_coeffs_swap += [_coeff_swap]

                    '''
                    Excluding rows of zero-coefficients
                    '''
                    _all_zero_chk = []
                    for _coeff in _M_coeffs_swap:
                        _all_zero_chk += [_coeff == 0]

                    if not AllTrue(_all_zero_chk):
                        _hermite_polys_list += [_hermite_pol]
                        _taylor_tuples_list += [_tuple]
                        _M_coeffs += [_M_coeffs_swap]
                        _count_momenta += 1

                _order += 1

            '''
            Need to modify to handle the cases where there are more possibilities
            '''
            ##_M_coeffs = _M_coeffs[:6] + _M_coeffs[7:9] + _M_coeffs[12:13]
            _M_full = sp.Matrix([_row for _row in _M_coeffs])
            _, _indices = _M_full.T.rref()

            self.M = _M_full[_indices,:]
            self.MHermitePolys = \
                np.array(_hermite_polys_list, dtype = object)[list(_indices)]
            self.MTaylorTuples = \
                np.array(_taylor_tuples_list, dtype = object)[list(_indices)]
            self.MHermiteNorms = \
                np.array([self.GetNormHermiteMoment(self.M, _)
                          for _ in range(self.Q)])
            self.MWProdsHermite = \
                self.GetWeightedProdsHermiteMomentsBasis(self.M)

        return {'M': self.M, 'MHermitePolys': self.MHermitePolys,
                'MTaylorTuples': self.MTaylorTuples,
                'MHermiteNorms': self.MHermiteNorms,
                'MWProdsHermite': self.MWProdsHermite}

    '''
    Need to write a function for the calculation of the weighted 'norm' of the moments
    two steps: i) single moment, ii) create the list of moments
    '''    
    def GetNormHermiteMoment(self, M, i):
        _swap_norm = \
            M[i,:] * sp.matrix_multiply_elementwise(M[i,:].T, sp.Matrix(self.Ws))
        return _swap_norm[0]


    def GetWeightedProdsHermiteMomentsBasis(self, M):
        _norms, _Ws = [], sp.Matrix(self.Ws)
        for _i in range(self.Q):
            _row_swap = []
            for _j in range(self.Q):
                _row_swap += \
                    [M[_i,:] * (sp.matrix_multiply_elementwise(M[_j,:].T, _Ws))]
            _norms += [_row_swap]

        _norms = sp.Matrix(_norms)
        return _norms

    '''
    GetWOrthInvertibleHermiteSet:
    this function executes the Gram-Schmidt procedure on the independent set of Hermite poly
    '''
    def GetWOrthInvertibleHermiteSet(self, search_depth, root_xi_sym = '\\xi'):
        _M_dict = \
            self.GetInvertibleHermiteSet(
                search_depth = search_depth,
                root_xi_sym = root_xi_sym
            )

        _M, _H_polys = _M_dict['M'], _M_dict['MHermitePolys']

        _lambda_hermite_prod = lambda x, y: HermiteWProd(x, y, self.Ws)

        _HermiteW_Orth_M, _HermiteW_Orth_Polys, _coeffs_list = [_M[0,:]], [_H_polys[0]], []
        '''
        Here we compute the orthogonal vectors and keep track of the Hermite polynomials
        '''
        for _i in range(1, _M.shape[0]):
            _new_vector = _M[_i,:]
            _new_poly = _H_polys[_i]
            _coeffs_swap = []
            for _j in range(len(_HermiteW_Orth_M)):
                _coeff = \
                    -ProjectionVAlongU(_M[_i,:], _HermiteW_Orth_M[_j], _lambda_hermite_prod)
                _new_poly += _coeff * _HermiteW_Orth_Polys[_j]
                _new_vector += \
                    -ProjectVAlongU(_M[_i,:], _HermiteW_Orth_M[_j], _lambda_hermite_prod)
                _coeffs_swap += [_coeff]

            _HermiteW_Orth_M += [_new_vector]
            _HermiteW_Orth_Polys += [sp.simplify(_new_poly)]
            _coeffs_list += [_coeffs_swap]
            
        self.MWOrth = sp.Matrix(_HermiteW_Orth_M)
        self.MWOrthHermitePolys = \
            np.array(_HermiteW_Orth_Polys, dtype = object)
        self.MWOrthCs = np.array(_coeffs_list, dtype = object)
        self.MWOrthHermiteNorms = \
            np.array([self.GetNormHermiteMoment(self.MWOrth, _)
                      for _ in range(self.Q)])
        self.MWOrthWProdsHermite = \
            self.GetWeightedProdsHermiteMomentsBasis(self.MWOrth)

        return {'MWOrth': self.MWOrth,
                'MWOrthHermitePolys': self.MWOrthHermitePolys,
                'MWOrthHermiteNorms': self.MWOrthHermiteNorms,
                'MWOrthWProdsHermite': self.MWOrthWProdsHermite,
                'MWOrthCs': self.MWOrthCs}
        
    '''
    def MakeMemoryFriendly:
    need to define a function to make the order of the directions more memory friendly
    this should be relevant on cpus, but on gpus it should not make much difference
    because the served memory chunck is per-warp (cuda) so it should not really matter
    at least not much
    '''
        
            
    '''
    def StreamingCode
    for this part of the code compute or registers pressure should not really matter
    '''
    def _define_cartesian_neighbors_coords(self, declared_variables, declared_constants, 
                                           pos_type = 'int', root_dim_sizes = 'L',
                                           root_strides = 'STR', 
                                           root_coord = 'x', lex_index = 'g_tid', 
                                           declare_const_flag = False):
        '''
        Find the largest increment for the vectors and precompute the
        coordinates increments:
        - check if variables not already declared in 'declared_variables'
        '''
        _swap_code = """"""
        _dim = len(self.XIs[0])
        _dim_sizes_macros = _get_seq_macros(_dim, root_dim_sizes)
        _dim_strides_macros = _get_seq_macros(_dim - 1, root_strides)
        
        _largest_c = 0
        for _xi in self.XIs:
            for _xi_c in _xi:
                _largest_c = max(_largest_c, _xi_c)

        for _delta_c in range(1, _largest_c + 1):
            for _d in range(_dim):
                if False:
                    _sp = _sp_macro('(' + root_coord + '_' + str(_d) + ' + ' + str(_delta_c) + ')', 
                                    _dim_sizes_macros[_d])
                    _sm = _sm_macro('(' + root_coord + '_' + str(_d) + ' - ' + str(_delta_c) + ')', 
                                    _dim_sizes_macros[_d])

                _sp = _sp_macro(root_coord + '_' + str(_d), str(_delta_c), 
                                _dim_sizes_macros[_d])
                _sm = _sm_macro(root_coord + '_' + str(_d), str(_delta_c), 
                                _dim_sizes_macros[_d])
                    
                _neigh_c_var = root_coord + '_' + str(_d) + '_p' + str(_delta_c)
                _swap_code += _codify_declaration_const_check(_neigh_c_var, _sp, pos_type, 
                                                              declared_variables, 
                                                              declared_constants, 
                                                              declare_const_flag)
                _neigh_c_var = root_coord + '_' + str(_d) + '_m' + str(_delta_c)
                _swap_code += _codify_declaration_const_check(_neigh_c_var, _sm, pos_type,
                                                              declared_variables, 
                                                              declared_constants, 
                                                              declare_const_flag)

        _swap_code += '\n'
        return _swap_code


    '''
    Possibly I do not need this function...
    '''
    def DefineAndStreamSingle(self, declared_variables = None, declared_constants = None,
                              ordering_lambdas = None, selected_dir = None,
                              src_vars = ['src_array'], dst_arrays_vars = ['dst_array'], 
                              src_types = ['SRCAType'], dst_arrays_types = ['DSTAType'],
                              stream_mode = 'push', pressure_mode = 'compute', pos_type = 'int', 
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
        if ordering_lambdas is None:
            raise Exception("Missing argument 'ordering_lambdas'")
        if type(ordering_lambdas) != list:
            raise Exception("Argument 'ordering_lambdas' must be a list of lambdas")
        if len(ordering_lambdas) != len(src_arrays_vars):
            raise Exception("The length of 'ordering_lambdas' must be the same as '*_arrays_vars'")

        '''
        Checking selected_pop
        '''
        if selected_dir is None:
            raise Exception("Missing argument 'selected_dir'")

        _swap_code = """"""


        return _swap_code
    
    def StreamingCode(self, declared_variables = None, declared_constants = None,
                      ordering_lambdas = None,
                      src_arrays_vars = ['src_array'], dst_arrays_vars = ['dst_array'], 
                      src_arrays_types = ['SRCAType'], dst_arrays_types = ['DSTAType'],
                      stream_mode = 'pull', pressure_mode = 'compute', pos_type = 'int', 
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
        if ordering_lambdas is None:
            raise Exception("Missing argument 'ordering_lambdas'")
        if type(ordering_lambdas) != list:
            raise Exception("Argument 'ordering_lambdas' must be a list of lambdas")
        if len(ordering_lambdas) != len(src_arrays_vars):
            raise Exception("The length of 'ordering_lambdas' must be the same as '*_arrays_vars'")
            

        if len(src_arrays_vars) != len(dst_arrays_vars):
            raise Exception("The number of source arrays must be the same as the destination ones")
        
        if stream_mode not in ['pull', 'push']:
            raise Exception("Parameter 'stream_mode' must be either 'pull' or 'push'")

        if pressure_mode not in ['compute', 'registers']:
            raise Exception("Parameter 'pressure_mode' must be either 'compute' or 'registers'")
            
        '''
        Defining and checking the variables that are needed
        '''
        _dim, _Q = len(self.XIs[0]), len(self.XIs)
        _dim_sizes_macros = _get_seq_macros(_dim, root_dim_sizes)
        _dim_strides_macros = _get_seq_macros(_dim - 1, root_strides)
        _pos_vars = _get_seq_vars(_dim, root_coord)

        _needed_variables = \
            _dim_sizes_macros + _dim_strides_macros + _pos_vars + \
            src_arrays_vars + dst_arrays_vars
        
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
            self._define_cartesian_neighbors_coords(
                declared_variables = declared_variables,
                declared_constants = declared_constants,
                pos_type = pos_type,
                root_dim_sizes = root_dim_sizes, 
                root_strides = root_strides, 
                root_coord = root_coord,
                lex_index = lex_index, 
                declare_const_flag = declare_const_dict['cartesian_coord_neigh']
            )
        
        '''
        Compute-pressure mode
        '''
        if pressure_mode == 'compute':
            '''
            Declaring the variables for the position of src and dst
            '''
            _coord_src, _coord_dst = sp.Symbol(root_coord + '_src'), sp.Symbol(root_coord + '_dst')
            
            _swap_code += _codify_declaration_const_check(_codify_sympy(_coord_src),
                                                          0 if stream_mode == 'pull' else lex_index,
                                                          pos_type, 
                                                          declared_variables, declared_constants)
            _swap_code += _codify_declaration_const_check(_codify_sympy(_coord_dst),
                                                          0 if stream_mode == 'push' else lex_index,
                                                          pos_type,
                                                          declared_variables, declared_constants)           
            _swap_code += _codify_newl
            for _q, _xi in enumerate(self.XIs):
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

                _swap_code += \
                    _codify_sympy_assignment(_coord_src if stream_mode == 'pull' else
                                             _coord_dst,
                                             _swap_expr)

                for _i in range(len(src_arrays_vars)):
                    _array_src = src_arrays_vars[_i]
                    _array_dst = dst_arrays_vars[_i]

                    _q_exchg = self.opposite[_q] if stream_mode == 'pull' else _q
                    _sx_hnd = _array_value(_array_dst,
                                           ordering_lambdas[_i](_coord_dst, _q_exchg),
                                           use_ptrs)
                    _dx_hnd = _array_value(_array_src,
                                           ordering_lambdas[_i](_coord_src, _q_exchg),
                                           use_ptrs)
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
            _swap_code += \
                _neighbors_register_pressure_macro(
                    _declared_variables = declared_variables,
                    _declared_constants = declared_constants,
                    _root_coordinate = root_coord,
                    _lexicographic_index = lex_index,
                    _stencil_vectors = self.XIs,
                    _dim_sizes = _dim_sizes_macros,
                    _dim_strides = _dim_strides_macros, 
                    _custom_type = pos_type,
                    _exclude_zero_norm = False,
                    _collect_mul = collect_mul,
                    _declare_const_flag = declare_const_dict['cartesian_coord_neigh']
                )
            _swap_code += _codify_newl
            '''
            The values associated to the zero-norm vector do not stream
            '''
            if False:
                for _i in range(len(src_arrays_vars)):
                    _array_src = src_arrays_vars[_i]
                    _array_dst = dst_arrays_vars[_i]

                    _sx_hnd = _array_value(_array_dst, ordering_lambdas[_i](lex_index, 0), use_ptrs)
                    _dx_hnd = _array_value(_array_src, ordering_lambdas[_i](lex_index, 0), use_ptrs)
                    _swap_code += _codify_assignment(_sx_hnd, _dx_hnd) 

            if True:                    
                for _q, _xi in enumerate(self.XIs):
                    '''
                    First get the neighbor position and the opposite
                    '''
                    for _i in range(len(src_arrays_vars)):
                        _array_src = src_arrays_vars[_i]
                        _array_dst = dst_arrays_vars[_i]

                        _q_exchg = self.opposite[_q] if stream_mode == 'pull' else _q 
                        _coord_src = 'n_' + root_coord + '_' + str(_q) if stream_mode == 'pull' else lex_index
                        _coord_dst = 'n_' + root_coord + '_' + str(_q) if stream_mode == 'push' else lex_index

                        _sx_hnd = _array_value(_array_dst, ordering_lambdas[_i](_coord_dst, _q_exchg), use_ptrs)
                        _dx_hnd = _array_value(_array_src, ordering_lambdas[_i](_coord_src, _q_exchg), use_ptrs)
                        _swap_code += _codify_assignment(_sx_hnd, _dx_hnd)
                        
        return _swap_code
            
    '''
    def SetOppositeDirections
    '''
    def SetOppositeDirections(self):
        self.opposite = defaultdict(dict)
        _Q = len(self.XIs)
        for _i in range(_Q):
            for _j in range(_i, _Q):
                if IsOppositeVector(self.XIs[_i], self.XIs[_j]):
                    self.opposite[_i] = _j
                    self.opposite[_j] = _i
            
        
    '''
    After the execution of this function the coordinates of the gradient can be found in
    d_array_0, d_array_1,...,d_array_(dim-1)
    for all the arrays listed in 'arrays_vars'
    '''
    def GradientCode(self, declared_variables = None, declared_constants = None,
                     arrays_vars = ['array'], arrays_types = ['AType'],
                     pressure_mode = 'compute', pos_type = 'int', use_ptrs = False,
                     root_dim_sizes = 'L', root_strides = 'STR',
                     root_coord = 'x', lex_index = 'g_tid',
                     declare_const_dict = {'cartesian_coord_neigh': True}):

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
            raise Exception("Parameter 'pressure_mode' must be either 'compute' or 'registers'")
        
        _dim, _Q = len(self.XIs[0]), len(self.XIs)
        _dim_sizes_macros = _get_seq_macros(_dim, root_dim_sizes)
        _dim_strides_macros = _get_seq_macros(_dim - 1, root_strides)
        _weights_values = self.Ws        
        
        '''
        Creating symbols list for arrays variables names
        '''
        _arrays_vars_syms = [sp.Symbol(_) for _ in arrays_vars]
        
        _swap_code = """"""
        
        '''
        declare swap_products
        '''

        for _i_a, _array in enumerate(_arrays_vars_syms):
            for _d in range(_dim):
                _swap_code += \
                    _codify_declaration_const_check(
                        'd_' + _codify_sympy(_array) + '_' + str(_d), 0, arrays_types[_i_a],
                        declared_variables, 
                        declared_constants
                    )
        _swap_code += _codify_newl

        '''
        Find the largest increment for the vectors and precompute the
        coordinates increments: this part is general enough to be abstracted
        '''        
        
        _swap_code += \
            self._define_cartesian_neighbors_coords(
                declared_variables = declared_variables,
                declared_constants = declared_constants,
                pos_type = pos_type,
                root_dim_sizes = root_dim_sizes, 
                root_strides = root_strides, 
                root_coord = root_coord,
                lex_index = lex_index, 
                declare_const_flag = declare_const_dict['cartesian_coord_neigh']
            )
        _swap_code += _codify_newl        
        
        '''
        Compute-pressure mode
        '''
        if pressure_mode == 'compute':
            '''
            Computing the gradient
            '''
            _swap_code += _codify_sympy_declaration('n_' + root_coord, 0, pos_type)
            _swap_code += _codify_newl
            for _q, _xi in enumerate(self.XIs):
                '''
                Need to check the length of the vector to exclude the zero norm one
                '''
                if GetLen2Pos(_xi) > 0:
                    '''
                    First get the neighbor positions
                    '''
                    _swap_code += _codify_comment("Neighbor at: " + str(_xi))
                    _swap_expr = \
                        _get_single_neighbor_pos_macro_fully_sym(_xi, _dim_sizes_macros, 
                                                                 _dim_strides_macros,
                                                                 root_coord, lex_index)
                    _swap_code += _codify_sympy_assignment('n_' + root_coord, _swap_expr)
                    
                    for _i, _array in enumerate(_arrays_vars_syms):
                        for _d in range(_dim):
                            if abs(_xi[_d]) > 0:
                                
                                _assignement_function = (
                                    _codify_sympy_add_assignment if (_xi[_d] * _weights_values[_q]) > 0 else 
                                    _codify_sympy_sub_assignment
                                )
                                
                                _dx_hnd = \
                                    (str(abs((_xi[_d]) * _weights_values[_q]).evalf()) + '*' + 
                                     _array_value(str(_array), 'n_' + root_coord, use_ptrs))
                                                                
                                _swap_code += \
                                    _assignement_function('d_' + str(_array) + '_' + str(_d), _dx_hnd)
                    _swap_code += _codify_newl
                                
        '''
        Compute-pressure mode
        '''
        if pressure_mode == 'registers':
            '''
            Here I should define all the neighbors positions at once
            '''
            _swap_code += \
                _neighbors_register_pressure_macro(declared_variables,
                                                   declared_constants,
                                                   root_coord, lex_index, self.XIs,
                                                   _dim_sizes_macros, _dim_strides_macros, 
                                                   pos_type, _exclude_zero_norm = True)
            _swap_code += _codify_newl
            '''
            Now we need to sum all the products
            '''
            _i = 0
            for _q, _xi in enumerate(self.XIs):
                '''
                No need to check the length of the vector here
                '''
                for _array in _arrays_vars_syms:
                    for _d in range(_dim):
                        if abs(_xi[_d]) > 0:

                            _assignement_function = (
                                _codify_sympy_add_assignment if (_xi[_d] * _weights_values[_q]) > 0 else 
                                _codify_sympy_sub_assignment
                            )
                            
                            _dx_hnd = \
                                (str((abs(_xi[_d]) * _weights_values[_q]).evalf()) + '*' + 
                                 _array_value(str(_array), 'n_' + root_coord + '_' + str(_q), 
                                              use_ptrs))

                            _swap_code += \
                                _assignement_function('d_' + str(_array) + '_' + str(_d), _dx_hnd)
                            
                            _i += 1
                        
                
        return _swap_code
