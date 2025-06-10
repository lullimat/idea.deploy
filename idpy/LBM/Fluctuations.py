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

from idpy.PRNGS.CRNGS import _codify_CRNGS_list, _codify_read_seeds, _codify_write_seeds
from idpy.PRNGS.CRNGS import _codify_n_random_vars

from idpy.IdpyCode.IdpyUnroll import _check_declared_variables_constants
from idpy.IdpyCode.IdpyUnroll import _check_needed_variables_constants
from idpy.IdpyCode.IdpyUnroll import _get_seq_vars, _codify, _codify_sympy

from idpy.IdpyStencils.IdpyStencils import IdpyStencil

import sympy as sp
import numpy as np

class GrossShanChenMultiPhase:
    def __init__(self, eq_obj = None,
                 root_pop = 'fR', root_rmom = 'Rm', search_depth = 6,
                 WOrth_flag = True):
        
        if eq_obj is None:
            raise Exception("Missing argument 'eq_obj'")

        self.root_pop, self.root_rmom = root_pop, root_rmom
        
        if WOrth_flag:
            self.M_dict = eq_obj.idpy_stencil.GetWOrthInvertibleHermiteSet(search_depth)
            self.MHermiteNorms = self.M_dict['MWOrthHermiteNorms']
            self.Mm1 = self.M_dict['MWOrth'].inv()
        else:
            self.M_dict = eq_obj.idpy_stencil.GetInvertibleHermiteSet(search_depth)
            self.MHermiteNorms = self.M_dict['MHermiteNorms']
            self.Mm1 = self.M_dict['M'].inv()
        
        self.D, self.Q = len(eq_obj.idpy_stencil.XIs[0]), len(eq_obj.idpy_stencil.XIs)
        self.NRandMoments = self.Q - (self.D + 1)
        self.SetPopulationsSyms(root_pop)
        self.SetRandMomentsSyms(root_rmom)

        self.c2 = eq_obj.idpy_stencil.c2
                    
    def SetPopulationsSyms(self, root_pop):
        self.f_i = sp.Matrix([sp.Symbol(root_pop + '_' + str(_))
                              for _ in range(self.Q)])
        
    def SetRandMomentsSyms(self, root_rmom):
        self.rmom_i = sp.Matrix([sp.Symbol(root_rmom + '_' + str(_))
                                 for _ in range(self.NRandMoments)])

    '''
    We need to modify this for selecting a different value of the bulk viscosity
    This is specific of the single relaxation time approximation
    '''
    def GetVariances(self, n0 = None, kBT = None, tau = None, c_s2 = None):
        if n0 is None:
            raise Exception("Missing argument 'n0'")
        
        if kBT is None:
            raise Exception("Missing argument 'kBT'")

        if tau is None:
            raise Exception("Missing argument 'tau'")

        if c_s2 is None:
            raise Exception("Missing argument 'c_s2'")
        
        _variances_list = []
        
        for _i in range(self.D + 1, self.Q):
            _swap_norm = self.MHermiteNorms[_i]
            _variances_list += \
                [_swap_norm * kBT * (2 - 1 / tau) / tau / c_s2]

        return [_codify(_ * n0) for _ in _variances_list]
    
    def CodifySingleRandomPopulation(self, i = None):
        if i is None:
            raise Exception("Parameter 'i' must not be None")
        pass
        

    '''
    CodifyRandomPopulations:
    - according to whether or not n0 is a float or a sympy.Symbol
    the variances should be computed accordingly
    '''
    def CodifySRTRandomPopulations(
            self, declared_variables = None, declared_constants = None,
            seeds_array = 'seeds_array', rand_vars_type = 'RANDType',
            n0 = None, kBT = None, tau = None,
            output_const = True, assignment_type = None,
            generator = 'MINSTD32', distribution = 'flat', parallel_streams = 1,
            which_box_m = None, lambda_ordering = None, use_ptrs = False
    ):
        _check_declared_variables_constants(declared_variables, declared_constants)

        '''
        Check that variables to be modified exist already
        '''
        if type(n0) == sp.core.symbol.Symbol:
            _check_needed_variables_constants([_codify_sympy(n0)],
                                              declared_variables,
                                              declared_constants)
        
        '''
        - Here I can begin by generating the random moments
        Still keep a non-optimized stance by decalring the random moments variable
        The idea is to add the right combinations of random moments in the
        final collision/streaming step
        - Hence, this function should output both the code and the list of 
        random moments combinations, a codified list, to be added at the end
        '''
        _swap_code = """"""

        _variances_list = \
            self.GetVariances(n0 = n0, kBT = kBT, tau = tau, c_s2 = self.c2)

        _swap_code += \
            _codify_n_random_vars(
                declared_variables = declared_variables,
                declared_constants = declared_constants,
                seeds_array = seeds_array,
                root_seed = 'lseed',
                rand_vars = _get_seq_vars(self.NRandMoments, self.root_rmom),
                rand_vars_type = rand_vars_type,
                assignment_type = assignment_type,
                lambda_ordering = lambda_ordering,
                use_ptrs = use_ptrs,
                variances = _variances_list,
                means = [0] * self.NRandMoments,
                distribution = distribution,
                generator = generator,
                parallel_streams = parallel_streams,
                output_const = output_const,
                which_box_m = which_box_m
            )

        _rmom_vector = sp.Matrix([0] * (self.D + 1) + list(self.rmom_i))
        _rpop_vector = (self.Mm1 * _rmom_vector).evalf()
        _codify_rpop = [_codify_sympy(_) for _ in _rpop_vector]
        
        return _swap_code, _codify_rpop