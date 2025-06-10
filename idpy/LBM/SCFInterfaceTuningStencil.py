__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2025 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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

import sys
sys.path.append("../../")

import numpy as np
import sympy as sp

from idpy.Utils.IdpySymbolic import TaylorTuples, GetTaylorDerivatives
from idpy.Utils.IdpySymbolic import GetTaylorExpansion, GetDerivativeCut
# from idpy.Utils.IdpySymbolic import GetDerivativeMonomials, PruneWExpression
from idpy.LBM.SCFStencils import SCFStencils, BasisVectors

"""
Get the list of all the derivative 'monomials': i.e., all possible combinations of derivatives
summing to the same order up to a given 'max_order'
"""
def GetDerivativeMonomials(max_order, fx, x):
    _terms_list = []

    for _order in range(0, max_order + 1, 2):
        for _k in range(_order // 2 + 1):
            ##print(_k, _order - _k)
            _terms_list += [sp.diff(fx(x), (x, _k)) * sp.diff(fx(x), (x, _order - _k))]
            
    return _terms_list

def PruneWExpression(expr, prune_list, w_sym):
    for l2_w in prune_list:
        expr = expr.subs(w_sym[l2_w], 0)
    return expr

# from idpy.Utils.Combinatorics import GetUniquePermutations, GetSinglePermutations
from idpy.Utils.Geometry import GetDihedralVectorsG

class InterfaceTuning2D:
    I_sym = {(4, 0): sp.Symbol("I_{4,0}"), 
             (6, 0): sp.Symbol("I_{6,0}"), (6, 1): sp.Symbol("I_{6,1}"),
             (8, 0): sp.Symbol("I_{8,0}"), (8, 1): sp.Symbol("I_{8,1}"), (8, 2): sp.Symbol("I_{8,2}"),
             (10, 0): sp.Symbol("I_{10,0}"), (10, 1): sp.Symbol("I_{10,1}"), 
             (10, 2): sp.Symbol("I_{10,2}"), (10, 3): sp.Symbol("I_{10,3}"),
             (12, 0): sp.Symbol("I_{12,0}"), (12, 1): sp.Symbol("I_{12,1}"), 
             (12, 2): sp.Symbol("I_{12,2}"), (12, 3): sp.Symbol("I_{12,3}"), 
             'lambda_i': sp.Symbol('\\Lambda_I'), 'chi_i': sp.Symbol('\chi_I')}
    
    hat_sigma_0_sym = sp.Symbol('\hat{\sigma}_0')
    hat_sigma_1_sym = sp.Symbol('\hat{\sigma}_1')
    hat_sigma_2_sym = sp.Symbol('\hat{\sigma}_2')
    hat_sigma_3_sym = sp.Symbol('\hat{\sigma}_3')
    hat_sigma_4_sym = sp.Symbol('\hat{\sigma}_4')
    hat_delta_0_sym = sp.Symbol('\hat{\delta}_0')

    len_2s = [1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25]
    w_sym = {l2: sp.Symbol('w(' + str(l2) + ')') for l2 in len_2s}

    """
    Definitions of the pressure tensor coefficients symbols - 
    PHYSICAL REVIEW E 103, 063309 (2021)
    """

    '''
    a_xx coefficients
    '''
    _axx_m50p5 = sp.Symbol("a^{(xx)}_{[-5, 0, 5]}")
    _axx_m40p4 = sp.Symbol("a^{(xx)}_{[-4, 0, 4]}")
    _axx_m30p3 = sp.Symbol("a^{(xx)}_{[-3, 0, 3]}")
    _axx_m20p2 = sp.Symbol("a^{(xx)}_{[-2, 0, 2]}")
    _axx_m10p1 = sp.Symbol("a^{(xx)}_{[-1, 0, 1]}")

    '''
    a_yy coefficients
    '''
    _ayy_m50p5 = sp.Symbol("a^{(yy)}_{[-5, 0, 5]}")
    _ayy_m40p4 = sp.Symbol("a^{(yy)}_{[-4, 0, 4]}")
    _ayy_m30p3 = sp.Symbol("a^{(yy)}_{[-3, 0, 3]}")
    _ayy_m20p2 = sp.Symbol("a^{(yy)}_{[-2, 0, 2]}")
    _ayy_m10p1 = sp.Symbol("a^{(yy)}_{[-1, 0, 1]}")
    _ayy_0 = sp.Symbol("a^{(yy)}_{[0]}")

    '''
    b_xx coefficients
    '''
    _bxx_22 = sp.Symbol("b^{(xx)}_{[2,2]}")
    _bxx_11 = sp.Symbol("b^{(xx)}_{[1,1]}")
    _bxx_13 = sp.Symbol("b^{(xx)}_{[1,3]}")
    _bxx_12 = sp.Symbol("b^{(xx)}_{[1,2]}")
    _bxx_14 = sp.Symbol("b^{(xx)}_{[1,4]}")
    _bxx_23 = sp.Symbol("b^{(xx)}_{[2,3]}")

    '''
    b_yy coefficients
    '''
    _byy_22 = sp.Symbol("b^{(yy)}_{[2,2]}")
    _byy_11 = sp.Symbol("b^{(yy)}_{[1,1]}")
    _byy_13 = sp.Symbol("b^{(yy)}_{[1,3]}")
    _byy_12 = sp.Symbol("b^{(yy)}_{[1,2]}")

    '''
    a_xx_w coefficients
    '''
    _axx_m50p5_w = 5 * w_sym[25] / 2
    _axx_m40p4_w = 2 * w_sym[16] + 4 * w_sym[17] + 8 * w_sym[20]
    _axx_m30p3_w = 3 * w_sym[9] / 2 + 3 * w_sym[10] + 3 * w_sym[13] + 3 * w_sym[18]
    _axx_m20p2_w = w_sym[4] + 2 * w_sym[5] + 2 * w_sym[8] + 4 * w_sym[13] / 3 + 2 * w_sym[20]
    _axx_m10p1_w = w_sym[1] / 2 + w_sym[2] + w_sym[5] + w_sym[10] + w_sym[17]

    _a_coeff_xx = [_axx_m50p5, _axx_m40p4, _axx_m30p3, _axx_m20p2, _axx_m10p1]
    _a_coeff_xx_w = [_axx_m50p5_w, _axx_m40p4_w, _axx_m30p3_w, _axx_m20p2_w, _axx_m10p1_w]

    '''
    a_yy_w coefficients
    '''
    _ayy_m40p4_w = w_sym[17] / 4 + 2 * w_sym[20]
    _ayy_m30p3_w = w_sym[10] / 3 + 4 * w_sym[13] / 3 + 3 * w_sym[18]
    _ayy_m20p2_w = w_sym[5] / 2 + 2 * w_sym[8] + 3 * w_sym[13] + 8 * w_sym[20]
    _ayy_m10p1_w = w_sym[2] + 4 * w_sym[5] + 9 * w_sym[10] + 16 * w_sym[17]
    _ayy_0_w = w_sym[1] + 4 * w_sym[4] + 9 * w_sym[9] + 16 * w_sym[16]

    _a_coeff_yy = [_ayy_m40p4, _ayy_m30p3, _ayy_m20p2, _ayy_m10p1, _ayy_0]
    _a_coeff_yy_w = [_ayy_m40p4_w, _ayy_m30p3_w, _ayy_m20p2_w, _ayy_m10p1_w, _ayy_0_w]

    '''
    b_xx_w coefficients
    '''
    _bxx_22_w = 4 * w_sym[16] + 8 * w_sym[17]
    _bxx_11_w = 2 * w_sym[4] + 4 * w_sym[5] + 4 * w_sym[8] + 16 * w_sym[13] / 3 + 4 * w_sym[20]
    _bxx_13_w = 4 * w_sym[16] + 8 * w_sym[17] + 8 * w_sym[20]
    _bxx_12_w = 3 * w_sym[9] + 6 * w_sym[10] + 6 * w_sym[13] + 6 * w_sym[18]
    _bxx_14_w = 5 * w_sym[25]
    _bxx_23_w = 5 * w_sym[25]

    _b_coeff_xx = [_bxx_22, _bxx_11, _bxx_13, _bxx_12, _bxx_14, _bxx_23]
    _b_coeff_xx_w = [_bxx_22_w, _bxx_11_w, _bxx_13_w, _bxx_12_w, _bxx_14_w, _bxx_23_w]

    '''
    b_yy_w coefficients
    '''
    _byy_22_w = w_sym[17] / 2
    _byy_11_w = w_sym[5] + 4 * w_sym[8] + 12 * w_sym[13] + 16 * w_sym[20] + 25 * w_sym[25]
    _byy_13_w = w_sym[17] / 2 + 2 * w_sym[20]
    _byy_12_w = 2 * w_sym[10] / 3 + 8 * w_sym[13] / 3 + 6 * w_sym[18]

    _b_coeff_yy = [_byy_22, _byy_11, _byy_13, _byy_12]
    _b_coeff_yy_w = [_byy_22_w, _byy_11_w, _byy_13_w, _byy_12_w]

    """
    Functions to manage the Taylor expansion and extracting the coefficients
    """
    def __init__(self, l2_list = []):
        """
        Defining the Taylor expansions for the pseudo-potential
        """
        _a, _b, _d = sp.Symbol('a'), sp.Symbol('b'), sp.Symbol('d')
        _psi, _x = sp.Function('\psi'), sp.Symbol('x')

        _order = 10

        _psi_pa_tx = GetTaylorExpansion(_psi(_x), [_x], _a * _d, _order)
        _psi_ma_tx = GetTaylorExpansion(_psi(_x), [_x], -_a * _d, _order)
        _psi_pb_tx = GetTaylorExpansion(_psi(_x), [_x], _b * _d, _order)
        _psi_mb_tx = GetTaylorExpansion(_psi(_x), [_x], -_b * _d, _order)

        _psi_a_ma0pa_taylor = \
            (_psi(_x) * (_psi_pa_tx + _psi_ma_tx)).subs(_d, 1)

        _psi_b_ab_taylor = \
            (_psi_ma_tx * _psi_pb_tx + _psi_pa_tx * _psi_mb_tx)
        '''
        Need to cut the derivatives at _order
        '''
        _psi_b_ab_taylor = sp.expand(_psi_b_ab_taylor)
        _psi_b_ab_taylor = sp.simplify(GetDerivativeCut(_psi_b_ab_taylor, _d, _order).subs(_d, 1))

        '''
        Simplest case at the end \psi(x + a) * \psi(x - a)
        '''
        _psi_b_aa_taylor = _psi_b_ab_taylor.subs(_b, _a) / 2

        _P_N_tay = \
            self._axx_m50p5 * (_psi_a_ma0pa_taylor.subs(_a, 5)) + \
            self._axx_m40p4 * (_psi_a_ma0pa_taylor.subs(_a, 4)) + \
            self._axx_m30p3 * (_psi_a_ma0pa_taylor.subs(_a, 3)) + \
            self._axx_m20p2 * (_psi_a_ma0pa_taylor.subs(_a, 2)) + \
            self._axx_m10p1 * (_psi_a_ma0pa_taylor.subs(_a, 1)) + \
            self._bxx_22 * (_psi_b_aa_taylor.subs(_a, 2)) + \
            self._bxx_11 * (_psi_b_aa_taylor.subs(_a, 1)) + \
            self._bxx_13 * (_psi_b_ab_taylor.subs(_a, 1).subs(_b, 3)) + \
            self._bxx_14 * (_psi_b_ab_taylor.subs(_a, 1).subs(_b, 4)) + \
            self._bxx_12 * (_psi_b_ab_taylor.subs(_a, 1).subs(_b, 2)) + \
            self._bxx_23 * (_psi_b_ab_taylor.subs(_a, 2).subs(_b, 3))

        _P_N_tay = sp.expand(_P_N_tay)

        _P_T_tay = \
            self._ayy_m40p4 * (_psi_a_ma0pa_taylor.subs(_a, 4)) + \
            self._ayy_m30p3 * (_psi_a_ma0pa_taylor.subs(_a, 3)) + \
            self._ayy_m20p2 * (_psi_a_ma0pa_taylor.subs(_a, 2)) + \
            self._ayy_m10p1 * (_psi_a_ma0pa_taylor.subs(_a, 1)) + \
            self._ayy_0 * (_psi(_x) ** 2) + \
            self._byy_22 * (_psi_b_aa_taylor.subs(_a, 2)) + \
            self._byy_11 * (_psi_b_aa_taylor.subs(_a, 1)) + \
            self._byy_13 * (_psi_b_ab_taylor.subs(_a, 1).subs(_b, 3)) + \
            self._byy_12 * (_psi_b_ab_taylor.subs(_a, 1).subs(_b, 2))

        _P_T_tay = sp.expand(_P_T_tay)        

        """
        Then, we select the coefficients of each derivative 'monomial'
        """

        self._ds_monos = GetDerivativeMonomials(_order, _psi, _x)

        for _d_mono in self._ds_monos:
            _P_N_tay = sp.collect(_P_N_tay, _d_mono)
            _P_T_tay = sp.collect(_P_T_tay, _d_mono)
            
        self._ds_coeffs_N, self._ds_coeffs_T = {}, {}
        for _d_mono in self._ds_monos:
            self._ds_coeffs_N[_d_mono] = _P_N_tay.coeff(_d_mono)
            self._ds_coeffs_T[_d_mono] = _P_T_tay.coeff(_d_mono)

        '''
        Let's replace the expressions in terms of Ws
        '''
        self._ds_coeffs_N_w, self._ds_coeffs_T_w = {}, {}

        for _d_mono in self._ds_monos:
            '''
            N
            '''
            _expr_ab = self._ds_coeffs_N[_d_mono]
            for _i_c, _coeff in enumerate(self._a_coeff_xx_w):
                _expr_ab = _expr_ab.subs(self._a_coeff_xx[_i_c], _coeff)
                
            for _i_c, _coeff in enumerate(self._b_coeff_xx_w):
                _expr_ab = _expr_ab.subs(self._b_coeff_xx[_i_c], _coeff)
                
            self._ds_coeffs_N_w[_d_mono] = _expr_ab
            
            '''
            T
            '''
            _expr_ab = self._ds_coeffs_T[_d_mono]
            for _i_c, _coeff in enumerate(self._a_coeff_yy_w):
                _expr_ab = _expr_ab.subs(self._a_coeff_yy[_i_c], _coeff)
                
            for _i_c, _coeff in enumerate(self._b_coeff_yy_w):
                _expr_ab = _expr_ab.subs(self._b_coeff_yy[_i_c], _coeff)
                
            self._ds_coeffs_T_w[_d_mono] = _expr_ab

        """
        Now, we prune all the weights we do not want to consider: we set the symbol to  zero
        """
        self.l2_list = l2_list
        prune_w_list = list(set(self.len_2s) - set(self.l2_list))

        for _d_mono in self._ds_monos:
            for l2_w in prune_w_list:
                self._ds_coeffs_N_w[_d_mono] = PruneWExpression(self._ds_coeffs_N_w[_d_mono], prune_w_list, self.w_sym)
                self._ds_coeffs_T_w[_d_mono] = PruneWExpression(self._ds_coeffs_T_w[_d_mono], prune_w_list, self.w_sym)

        """
        Now, we build the tuning stencil
        """
        self.tuning_stencil = \
            SCFStencils(E = BasisVectors(x_max = int(np.sqrt(np.amax(self.l2_list)))), len_2s = self.l2_list)
        self.tuning_stencil.GetTypEqs()
        self.tuning_stencil.GetWolfEqs()
        self.e_sym = {**self.tuning_stencil.e_sym}

        """
        Putting typical force isotropy equations in a class variable
        """
        self.typ_eq_s = self.tuning_stencil.typ_eq_s
        self.e_expr = self.tuning_stencil.e_expr
        """
        Saving the pointer
        """
        self.w_sol = self.tuning_stencil.w_sol


    def DefineWeights(self, eq_s: list):
        self.tuning_stencil.FindWeights(eq_s)
        
        tuning_w_sol_e = {l2: self.tuning_stencil.w_sol[0][i] for i, l2 in enumerate(self.l2_list)}
        self.tuning_w_sol_dict = {self.tuning_stencil.w_sym[l2]: self.tuning_stencil.w_sol[0][i] for i, l2 in enumerate(self.l2_list)}

        """
        Now, we need to write in terms of the isotropy coefficients and of the isotropy conditions the
        coefficients of the monomials of the pressure tensor
        """
        self._ds_coeffs_N_eI, self._ds_coeffs_T_eI = {}, {}
        for _d_mono in self._ds_monos:
            self._ds_coeffs_N_eI[_d_mono] = self._ds_coeffs_N_w[_d_mono]
            self._ds_coeffs_T_eI[_d_mono] = self._ds_coeffs_T_w[_d_mono]
            
            for l2_w in self.l2_list:
                self._ds_coeffs_N_eI[_d_mono] = self._ds_coeffs_N_eI[_d_mono].subs(self.w_sym[l2_w], tuning_w_sol_e[l2_w])
                self._ds_coeffs_T_eI[_d_mono] = self._ds_coeffs_T_eI[_d_mono].subs(self.w_sym[l2_w], tuning_w_sol_e[l2_w])

            self._ds_coeffs_N_eI[_d_mono] = sp.simplify(self._ds_coeffs_N_eI[_d_mono])
            self._ds_coeffs_T_eI[_d_mono] = sp.simplify(self._ds_coeffs_T_eI[_d_mono])


        """
        Defining the Taylor expansion coefficients for P_N - P_T, for the weight and isotropy-coefficient/conditions
        The choice here is related to the order of expansion of the pressure tensor and the number of weights in the stencil
        """

        '''
        Need to find something to simplify this mapping
        '''
        self.cs_w = {
            (0, 2): self._ds_coeffs_N_w[self._ds_monos[1]] - self._ds_coeffs_T_w[self._ds_monos[1]], 
            (1, 1): self._ds_coeffs_N_w[self._ds_monos[2]] - self._ds_coeffs_T_w[self._ds_monos[2]],

            (0, 4): self._ds_coeffs_N_w[self._ds_monos[3]] - self._ds_coeffs_T_w[self._ds_monos[3]], 
            (1, 3): self._ds_coeffs_N_w[self._ds_monos[4]] - self._ds_coeffs_T_w[self._ds_monos[4]], 
            (2, 2): self._ds_coeffs_N_w[self._ds_monos[5]] - self._ds_coeffs_T_w[self._ds_monos[5]], 

            (0, 6): self._ds_coeffs_N_w[self._ds_monos[6]] - self._ds_coeffs_T_w[self._ds_monos[6]], 
            (1, 5): self._ds_coeffs_N_w[self._ds_monos[7]] - self._ds_coeffs_T_w[self._ds_monos[7]], 
            (2, 4): self._ds_coeffs_N_w[self._ds_monos[8]] - self._ds_coeffs_T_w[self._ds_monos[8]], 
            (3, 3): self._ds_coeffs_N_w[self._ds_monos[9]] - self._ds_coeffs_T_w[self._ds_monos[9]],

            (0, 8): self._ds_coeffs_N_w[self._ds_monos[10]] - self._ds_coeffs_T_w[self._ds_monos[10]],
            (1, 7): self._ds_coeffs_N_w[self._ds_monos[11]] - self._ds_coeffs_T_w[self._ds_monos[11]],
            (2, 6): self._ds_coeffs_N_w[self._ds_monos[12]] - self._ds_coeffs_T_w[self._ds_monos[12]],
            (3, 5): self._ds_coeffs_N_w[self._ds_monos[13]] - self._ds_coeffs_T_w[self._ds_monos[13]],
            (4, 4): self._ds_coeffs_N_w[self._ds_monos[14]] - self._ds_coeffs_T_w[self._ds_monos[14]],

            (0, 10): self._ds_coeffs_N_w[self._ds_monos[15]] - self._ds_coeffs_T_w[self._ds_monos[15]],
            (1, 9): self._ds_coeffs_N_w[self._ds_monos[16]] - self._ds_coeffs_T_w[self._ds_monos[16]],
            (2, 8): self._ds_coeffs_N_w[self._ds_monos[17]] - self._ds_coeffs_T_w[self._ds_monos[17]],
            (3, 7): self._ds_coeffs_N_w[self._ds_monos[18]] - self._ds_coeffs_T_w[self._ds_monos[18]],
            (4, 6): self._ds_coeffs_N_w[self._ds_monos[19]] - self._ds_coeffs_T_w[self._ds_monos[19]],
            (5, 5): self._ds_coeffs_N_w[self._ds_monos[20]] - self._ds_coeffs_T_w[self._ds_monos[20]]
            }

        self.cs_eI = {
            (0, 2): self._ds_coeffs_N_eI[self._ds_monos[1]] - self._ds_coeffs_T_eI[self._ds_monos[1]], 
            (1, 1): self._ds_coeffs_N_eI[self._ds_monos[2]] - self._ds_coeffs_T_eI[self._ds_monos[2]],

            (0, 4): self._ds_coeffs_N_eI[self._ds_monos[3]] - self._ds_coeffs_T_eI[self._ds_monos[3]],
            (1, 3): self._ds_coeffs_N_eI[self._ds_monos[4]] - self._ds_coeffs_T_eI[self._ds_monos[4]], 
            (2, 2): self._ds_coeffs_N_eI[self._ds_monos[5]] - self._ds_coeffs_T_eI[self._ds_monos[5]], 

            (0, 6): self._ds_coeffs_N_eI[self._ds_monos[6]] - self._ds_coeffs_T_eI[self._ds_monos[6]], 
            (1, 5): self._ds_coeffs_N_eI[self._ds_monos[7]] - self._ds_coeffs_T_eI[self._ds_monos[7]], 
            (2, 4): self._ds_coeffs_N_eI[self._ds_monos[8]] - self._ds_coeffs_T_eI[self._ds_monos[8]],
            (3, 3): self._ds_coeffs_N_eI[self._ds_monos[9]] - self._ds_coeffs_T_eI[self._ds_monos[9]],

            (0, 8): self._ds_coeffs_N_eI[self._ds_monos[10]] - self._ds_coeffs_T_eI[self._ds_monos[10]], 
            (1, 7): self._ds_coeffs_N_eI[self._ds_monos[11]] - self._ds_coeffs_T_eI[self._ds_monos[11]], 
            (2, 6): self._ds_coeffs_N_eI[self._ds_monos[12]] - self._ds_coeffs_T_eI[self._ds_monos[12]],
            (3, 5): self._ds_coeffs_N_eI[self._ds_monos[13]] - self._ds_coeffs_T_eI[self._ds_monos[13]],
            (4, 4): self._ds_coeffs_N_eI[self._ds_monos[14]] - self._ds_coeffs_T_eI[self._ds_monos[14]],

            (0, 10): self._ds_coeffs_N_eI[self._ds_monos[15]] - self._ds_coeffs_T_eI[self._ds_monos[15]], 
            (1, 9): self._ds_coeffs_N_eI[self._ds_monos[16]] - self._ds_coeffs_T_eI[self._ds_monos[16]], 
            (2, 8): self._ds_coeffs_N_eI[self._ds_monos[17]] - self._ds_coeffs_T_eI[self._ds_monos[17]],
            (3, 7): self._ds_coeffs_N_eI[self._ds_monos[18]] - self._ds_coeffs_T_eI[self._ds_monos[18]],
            (4, 6): self._ds_coeffs_N_eI[self._ds_monos[19]] - self._ds_coeffs_T_eI[self._ds_monos[19]],            
            (5, 5): self._ds_coeffs_N_eI[self._ds_monos[20]] - self._ds_coeffs_T_eI[self._ds_monos[20]]            
            }


        self.hat_sigma_0_ws = [
            self.cs_w[(1, 1)] - self.cs_w[(0, 2)], 
            self.cs_w[(2, 2)] - self.cs_w[(1, 3)] + self.cs_w[(0, 4)], 
            self.cs_w[(3, 3)] - self.cs_w[(2, 4)] + self.cs_w[(1, 5)] - self.cs_w[(0, 6)], 
            self.cs_w[(4, 4)] - self.cs_w[(3, 5)] + self.cs_w[(2, 6)] - self.cs_w[(1, 7)] + self.cs_w[(0, 8)],
            self.cs_w[(5, 5)] - self.cs_w[(4, 6)] + self.cs_w[(3, 7)] - self.cs_w[(2, 8)] + self.cs_w[(1, 9)] - self.cs_w[(0, 10)]]

        self.hat_sigma_0_eIs = [
            self.cs_eI[(1, 1)] - self.cs_eI[(0, 2)], 
            self.cs_eI[(2, 2)] - self.cs_eI[(1, 3)] + self.cs_eI[(0, 4)], 
            self.cs_eI[(3, 3)] - self.cs_eI[(2, 4)] + self.cs_eI[(1, 5)] - self.cs_eI[(0, 6)], 
            self.cs_eI[(4, 4)] - self.cs_eI[(3, 5)] + self.cs_eI[(2, 6)] - self.cs_eI[(1, 7)] + self.cs_eI[(0, 8)], 
            self.cs_eI[(5, 5)] - self.cs_eI[(4, 6)] + self.cs_eI[(3, 7)] - self.cs_eI[(2, 8)] + self.cs_eI[(1, 9)] - self.cs_eI[(0, 10)]]

        self.hat_delta_0_w = self.cs_w[(0, 2)]
        self.hat_delta_0_eI = self.cs_eI[(0, 2)]

    def GetStdIsoDict(self):
        self.std_iso_dict = {**{v: 0 for v in self.I_sym.values()}, self.e_sym[2]: 1}
        return self.std_iso_dict

    def GetFullySolvedWeights(self):
        """
        - Here we define the fully solved systems: i.e. e6 and e8 as a function of \hat{\sigma}_1 and \hat{\delta}_0
        - This step can only be done after checking that the expressions obtained from 'DefineWeights'
        self.hat_sigma_0_eIs and self.hat_delta_0_eI
        """
        
        eq_s = [sp.Eq(self.hat_sigma_0_eIs[1], self.hat_sigma_1_sym)]
        eq_s += [sp.Eq(self.hat_delta_0_eI, self.hat_delta_0_sym)]

        self.e6_e8_sol_dict = sp.solve(eq_s, [self.e_sym[6], self.e_sym[8]])

        self.e6_e8_w_sol_list = [w.subs(self.e6_e8_sol_dict) for w in self.w_sol[0]]

    def GetFullySolvedWeightsSigma1(self):
        """
        - Here we define the fully solved systems: i.e. e6 and e8 as a function of \hat{\sigma}_1 and \hat{\delta}_0
        - This step can only be done after checking that the expressions obtained from 'DefineWeights'
        self.hat_sigma_0_eIs and self.hat_delta_0_eI
        """
        
        eq_s = [sp.Eq(self.hat_sigma_0_eIs[1], self.hat_sigma_1_sym)]

        self.e6_e8_sol_dict = sp.solve(eq_s, [self.e_sym[6]])
        self.e6_e8_w_sol_list = [w.subs(self.e6_e8_sol_dict) for w in self.w_sol[0]]

    def GetFullySolvedWeightsSigma2(self):
        """
        - Here we define the fully solved systems: i.e. e6 and e8 as a function of \hat{\sigma}_1 and \hat{\delta}_0
        - This step can only be done after checking that the expressions obtained from 'DefineWeights'
        self.hat_sigma_0_eIs and self.hat_delta_0_eI
        """
        
        eq_s = [sp.Eq(self.hat_sigma_0_eIs[1], self.hat_sigma_1_sym), 
                sp.Eq(self.hat_sigma_0_eIs[2], self.hat_sigma_2_sym)]

        self.e6_e8_sol_dict = sp.solve(eq_s, [self.e_sym[6], self.e_sym[8]])
        self.e6_e8_w_sol_list = [w.subs(self.e6_e8_sol_dict) for w in self.w_sol[0]]

    def GetFullySolvedWeightsSigma3(self):
        """
        - Here we define the fully solved systems: i.e. e6 and e8 as a function of \hat{\sigma}_1 and \hat{\delta}_0
        - This step can only be done after checking that the expressions obtained from 'DefineWeights'
        self.hat_sigma_0_eIs and self.hat_delta_0_eI
        """
        
        eq_s = [sp.Eq(self.hat_sigma_0_eIs[1], self.hat_sigma_1_sym), 
                sp.Eq(self.hat_sigma_0_eIs[2], self.hat_sigma_2_sym), 
                sp.Eq(self.hat_sigma_0_eIs[3], self.hat_sigma_3_sym)]

        self.e6_e8_e10_sol_dict = sp.solve(eq_s, [self.e_sym[6], self.e_sym[8], self.e_sym[10]])
        print("e6_e8_e10_sol_dict:", self.e6_e8_e10_sol_dict)

        self.e6_e8_e10_w_sol_list = [w.subs(self.e6_e8_e10_sol_dict) for w in self.w_sol[0]]

    def GetFullySolvedWeightsSigma2Delta(self):
        """
        - Here we define the fully solved systems: i.e. e6 and e8 as a function of \hat{\sigma}_1 and \hat{\delta}_0
        - This step can only be done after checking that the expressions obtained from 'DefineWeights'
        self.hat_sigma_0_eIs and self.hat_delta_0_eI
        """
        self.g12 = sp.Symbol('\gamma_{12}')
        subs_e12_dict = {self.e_sym[12]: self.e_sym[10] * self.g12}
        eq_s = [sp.Eq(self.hat_sigma_0_eIs[1], self.hat_sigma_1_sym), 
                sp.Eq(self.hat_sigma_0_eIs[2], self.hat_sigma_2_sym), 
                sp.Eq(self.hat_delta_0_eI.subs(subs_e12_dict), self.hat_delta_0_sym)]

        self.e6_e8_e10_e12_sol_dict = sp.solve(eq_s, [self.e_sym[6], self.e_sym[8], self.e_sym[10]])
        swap_w_list = [w.subs(subs_e12_dict) for w in self.w_sol[0]]
        self.e6_e8_e10_e12_w_sol_list = [w.subs(self.e6_e8_e10_e12_sol_dict) for w in swap_w_list]

    def GetFullySolvedStencilISO2D(self, values_dict):
        self.GetStdIsoDict()
        self.GetFullySolvedWeights()

        subs_dict = {**self.e6_e8_sol_dict,
                     self.e_sym[4]: values_dict[self.e_sym[4]],
                     self.hat_sigma_1_sym: values_dict[self.hat_sigma_1_sym], 
                     self.hat_delta_0_sym: values_dict[self.hat_delta_0_sym]}
        
        self.e6_e8_w_sol_iso_list = [w.subs(subs_dict) for w in self.e6_e8_w_sol_list]
        self.e6_e8_w_sol_iso_list = [w.subs(self.std_iso_dict) for w in self.e6_e8_w_sol_iso_list]

        output_stencil = \
            SCFStencils(E = BasisVectors(x_max = int(np.sqrt(np.amax(self.l2_list)))), len_2s = self.l2_list)
        output_stencil.w_sol += [self.e6_e8_w_sol_iso_list]        
        return output_stencil

    def GetFullySolvedStencilISO2DSigma1(self, values_dict):
        self.GetStdIsoDict()
        self.GetFullySolvedWeightsSigma1()

        subs_dict = {**self.e6_e8_sol_dict,
                     self.e_sym[4]: values_dict[self.e_sym[4]],
                     self.e_sym[8]: values_dict[self.e_sym[8]],
                     self.hat_sigma_1_sym: values_dict[self.hat_sigma_1_sym]}
        
        self.e6_e8_w_sol_iso_list = [w.subs(subs_dict) for w in self.e6_e8_w_sol_list]
        self.e6_e8_w_sol_iso_list = [w.subs(self.std_iso_dict) for w in self.e6_e8_w_sol_iso_list]

        output_stencil = \
            SCFStencils(E = BasisVectors(x_max = int(np.sqrt(np.amax(self.l2_list)))), len_2s = self.l2_list)
        output_stencil.w_sol += [self.e6_e8_w_sol_iso_list]        
        return output_stencil
    
    def GetFullySolvedStencilISO2DSigma2(self, values_dict):
        self.GetStdIsoDict()
        self.GetFullySolvedWeightsSigma2()

        subs_dict = {**self.e6_e8_sol_dict,
                     self.e_sym[4]: values_dict[self.e_sym[4]],
                     self.hat_sigma_1_sym: values_dict[self.hat_sigma_1_sym], 
                     self.hat_sigma_2_sym: values_dict[self.hat_sigma_2_sym]}
        
        self.e6_e8_w_sol_iso_list = [w.subs(subs_dict) for w in self.e6_e8_w_sol_list]
        self.e6_e8_w_sol_iso_list = [w.subs(self.std_iso_dict) for w in self.e6_e8_w_sol_iso_list]

        output_stencil = \
            SCFStencils(E = BasisVectors(x_max = int(np.sqrt(np.amax(self.l2_list)))), len_2s = self.l2_list)
        output_stencil.w_sol += [self.e6_e8_w_sol_iso_list]        
        return output_stencil    

    def GetFullySolvedStencilISO2DSigma3(self, values_dict):
        self.GetStdIsoDict()
        self.GetFullySolvedWeightsSigma3()

        subs_dict = {**self.e6_e8_e10_sol_dict,
                     self.e_sym[4]: values_dict[self.e_sym[4]],
                     self.e_sym[12]: values_dict[self.e_sym[12]],
                     self.hat_sigma_1_sym: values_dict[self.hat_sigma_1_sym], 
                     self.hat_sigma_2_sym: values_dict[self.hat_sigma_2_sym], 
                     self.hat_sigma_3_sym: values_dict[self.hat_sigma_3_sym]}
        
        self.e6_e8_e10_w_sol_iso_list = [w.subs(subs_dict) for w in self.e6_e8_e10_w_sol_list]
        self.e6_e8_e10_w_sol_iso_list = [w.subs(self.std_iso_dict) for w in self.e6_e8_e10_w_sol_iso_list]

        output_stencil = \
            SCFStencils(E = BasisVectors(x_max = int(np.sqrt(np.amax(self.l2_list)))), len_2s = self.l2_list)
        output_stencil.w_sol += [self.e6_e8_e10_w_sol_iso_list]        
        return output_stencil    

    def GetFullySolvedStencilISO2DSigma2Delta(self, values_dict):
        self.GetStdIsoDict()
        self.GetFullySolvedWeightsSigma2Delta()

        subs_dict = {**{self.e_sym[12]: self.e_sym[10] * self.g12},
                     **self.e6_e8_e10_e12_sol_dict,
                     self.e_sym[4]: values_dict[self.e_sym[4]],
                     self.hat_sigma_1_sym: values_dict[self.hat_sigma_1_sym], 
                     self.hat_sigma_2_sym: values_dict[self.hat_sigma_2_sym], 
                     self.hat_delta_0_sym: values_dict[self.hat_delta_0_sym],
                     self.g12: values_dict[self.g12]}
        
        self.e6_e8_e10_e12_w_sol_iso_list = [w.subs(subs_dict) for w in self.e6_e8_e10_e12_w_sol_list]
        self.e6_e8_e10_e12_w_sol_iso_list = [w.subs(self.std_iso_dict) for w in self.e6_e8_e10_e12_w_sol_iso_list]

        output_stencil = \
            SCFStencils(E = BasisVectors(x_max = int(np.sqrt(np.amax(self.l2_list)))), len_2s = self.l2_list)
        output_stencil.w_sol += [self.e6_e8_e10_e12_w_sol_iso_list]        
        return output_stencil    

    def GetWeightsStencilSol(self, values_dict):
        self.GetStdIsoDict()
        
        self.w_sol_iso_list = [w.subs(values_dict) for w in self.tuning_stencil.w_sol[0]]
        self.w_sol_iso_list = [w.subs(self.std_iso_dict) for w in self.w_sol_iso_list]

        output_stencil = \
            SCFStencils(E = BasisVectors(x_max = int(np.sqrt(np.amax(self.l2_list)))), len_2s = self.l2_list)
        output_stencil.w_sol += [self.w_sol_iso_list]        
        return output_stencil

"""
In the paper I need to specify that I the symmetry groups that I am using are chosen so that only a limited amount of new
analytical calculations will be required
"""

from idpy.Utils.IdpySymbolic import GetFullyIsotropicTensor, GetGeneralizedKroneckerDelta
from idpy.Utils.IdpySymbolic import GetASymmetricTensor, SymmetricTensor
from idpy.Utils.IdpySymbolic import TaylorTuples
from functools import reduce

class InterfaceTuning3D(InterfaceTuning2D):
    ttuples = TaylorTuples(list(range(3)), 1)

    def __init__(self, root_vectors_3d = None, l2_list_2d = [1, 2, 4, 5, 8, 9, 18, 16]):
        if root_vectors_3d is None:
            self.root_vectors_3d = [(1, 0, 0), (1, 1, 0), (1, 1, 1), 
                                    (2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1), 
                                    (3, 0, 0), (3, 3, 0), 
                                    (4, 0, 0)]
        else:
            self.root_vectors_3d = root_vectors_3d

        InterfaceTuning2D.__init__(self, l2_list=l2_list_2d)
        
        self.G = {root_v: np.array(GetDihedralVectorsG(np.array(root_v))) for root_v in self.root_vectors_3d}

        get_dict_xi = lambda v: {self.ttuples[i]: v[i] for i in range(len(v))}

        get_symv_tprod_2n = lambda v, n2: reduce(lambda x, y: x ^ y, [v] * n2)

        self.XIS_sym = \
            {
                root_v:
                [SymmetricTensor(d = 3, rank = 1, c_dict=get_dict_xi(sp.Matrix(self.G[root_v][i,:]))) for i in range(len(self.G[root_v]))]
                for root_v in self.root_vectors_3d
            }
        
        self.tensor_products = \
            {
                root_v: [{n2: get_symv_tprod_2n(v, n2) for n2 in range(2, 11, 2)} for v in self.XIS_sym[root_v]] 
                for root_v in self.root_vectors_3d
            }
        
        self.GetIsotropyConditions3d()

    def GetIsotropyConditions3d(self):
        indices_lists = \
            {
                2: [(0, 0)],
                4: [(0, 0, 0, 0), (0, 0, 1, 1)],
                6: [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1), (0, 0, 1, 1, 2, 2)],
                8: [(0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 1, 1, 1, 1), (0, 0, 0, 0, 1, 1, 2, 2)],
                10: [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1, 1), 
                    (0, 0, 0, 0, 0, 0, 1, 1, 1, 1), (0, 0, 0, 0, 1, 1, 1, 1, 2, 2), 
                    (0, 0, 1, 1, 1, 1, 1, 1, 2, 2)]
            }

        get_w_sym = lambda root_v: sp.Symbol('W_{' + str(root_v) + '}')
        self.w_syms_list = [get_w_sym(root_v) for root_v in self.root_vectors_3d]

        indices_exprs = {}
        for rank in [2, 4, 6, 8, 10]:
            indices_exprs[rank] = {}
            for indices in indices_lists[rank]:
                indices_exprs[rank][indices] = 0
                for root_v in self.root_vectors_3d:
                    for tp in self.tensor_products[root_v]:
                        indices_exprs[rank][indices] += get_w_sym(root_v) * tp[rank][indices]


        # 2nd order
        self.e2_3d = indices_exprs[2][(0, 0)]

        # 4th order
        self.e4_3d = indices_exprs[4][(0, 0, 1, 1)]
        self.I40_3d = indices_exprs[4][(0, 0, 0, 0)] - sp.factorial2(3) * self.e4_3d

        # 6th order -- this is not the 'typical' formulation of the isotropy conditions for the 6th order
        # However, when used all together, they are equivalent -- need to change
        self.e6_3d = indices_exprs[6][(0, 0, 1, 1, 2, 2)]
        self.I61_3d = indices_exprs[6][(0, 0, 0, 0, 1, 1)] - sp.factorial2(3) * self.e6_3d
        self.I60_3d = indices_exprs[6][(0, 0, 0, 0, 0, 0)] - sp.factorial2(5) * self.e6_3d - sp.binomial(6, 2) * self.I61_3d

        # 8th order
        self.e8_3d = indices_exprs[8][(0, 0, 0, 0, 1, 1, 2, 2)] / sp.factorial2(3)
        self.I80_3d = indices_exprs[8][(0, 0, 0, 0, 0, 0, 0, 0)] - sp.factorial2(7) * indices_exprs[8][(0, 0, 0, 0, 0, 0, 1, 1)] / sp.factorial2(5)
        self.I81_3d = indices_exprs[8][(0, 0, 0, 0, 0, 0, 1, 1)] - sp.factorial2(5) * indices_exprs[8][(0, 0, 0, 0, 1, 1, 1, 1)] / (sp.factorial2(3) ** 2)
        self.I82_3d = indices_exprs[8][(0, 0, 0, 0, 1, 1, 1, 1)] - (sp.factorial2(3) ** 2) * indices_exprs[8][(0, 0, 0, 0, 1, 1, 2, 2)] / sp.factorial2(3)

        # 10th order
        self.e10_3d = indices_exprs[10][(0, 0, 0, 0, 1, 1, 1, 1, 2, 2)] / (sp.factorial2(3) ** 2)
        self.I100_3d = indices_exprs[10][(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)] - sp.factorial2(9) * indices_exprs[10][(0, 0, 0, 0, 0, 0, 0, 0, 1, 1)] / sp.factorial2(7)
        self.I101_3d = indices_exprs[10][(0, 0, 0, 0, 0, 0, 0, 0, 1, 1)] - sp.factorial2(7) * indices_exprs[10][(0, 0, 0, 0, 0, 0, 1, 1, 1, 1)] / (sp.factorial2(5) * sp.factorial2(3))
        self.I102_3d = indices_exprs[10][(0, 0, 0, 0, 0, 0, 1, 1, 1, 1)] - (sp.factorial2(5) * sp.factorial2(3)) * indices_exprs[10][(0, 0, 0, 0, 1, 1, 1, 1, 2, 2)] / (sp.factorial2(3) ** 2)
        self.I103_3d = indices_exprs[10][(0, 0, 0, 0, 1, 1, 1, 1, 2, 2)] - (sp.factorial2(3) ** 2) * indices_exprs[10][(0, 0, 1, 1, 1, 1, 1, 1, 2, 2)] / sp.factorial2(5)

    def DefineWeights3D(self, eq_s: list):
        return sp.solve(eq_s, self.w_syms_list)
    
    def PushWeights3D(self, weights_list):
        self.tuning_w_sol_list_3d = weights_list
    
    def DefineWeights3DTYP0(self, projection_f):
        self.DefineWeights(
            [self.e_sym[n] - self.e_expr[n] for n in range(2, 8 + 1, 2)] + 
            [self.I_sym[(4,0)] - self.typ_eq_s[4][0], 
             self.I_sym[(6,0)] - self.typ_eq_s[6][0], 
             self.I_sym[(8,0)] - self.typ_eq_s[8][0], 
             self.I_sym[(8,1)] - self.typ_eq_s[8][1]])
        
        eq_s = [
            self.e2_3d - self.e_sym[2], 
            self.e4_3d - self.e_sym[4], 
            self.e6_3d - self.e_sym[6], 
            self.e8_3d - self.e_sym[8], 
            self.I40_3d - self.I_sym[(4, 0)], 
            self.I60_3d - self.I_sym[(6, 0)], 
            self.I61_3d - self.I_sym[(6, 1)], 
            self.I80_3d - self.I_sym[(8, 0)], 
            self.I81_3d - self.I_sym[(8, 1)], 
            self.I82_3d - self.I_sym[(8, 2)], 
            self.I103_3d - self.I_sym[(10, 3)]]
        
        # print(self.hat_sigma_0_ws)
                
        self.hat_sigma_0_ws_3d = [projection_f(self, expr) for expr in self.hat_sigma_0_ws]
        self.hat_delta_0_w_3d = projection_f(self, self.hat_delta_0_w)

        self.tuning_w_sol_dict_3d = sp.solve(eq_s, self.w_syms_list)
        # print(self.tuning_w_sol_dict_3d)
        self.tuning_w_sol_list_3d = [self.tuning_w_sol_dict_3d[w] for w in self.w_syms_list]
        self.e6_e8_w_sol_list_3d = [self.tuning_w_sol_dict_3d[w] for w in self.w_syms_list]

        self.hat_sigma_0_eIs_3d = [expr.subs(self.tuning_w_sol_dict_3d) for expr in self.hat_sigma_0_ws_3d]
        self.hat_delta_0_eI_3d = self.hat_delta_0_w_3d.subs(self.tuning_w_sol_dict_3d)        

    def DefineWeights3DTYP0Sigma1(self, projection_f):
        self.DefineWeights(
            [self.e_sym[n] - self.e_expr[n] for n in range(2, 8 + 1, 2)] + 
            [self.I_sym[(4,0)] - self.typ_eq_s[4][0], self.I_sym[(6,0)] - self.typ_eq_s[6][0]])
        
        eq_s = [
            self.e2_3d - self.e_sym[2], 
            self.e4_3d - self.e_sym[4], 
            self.e6_3d - self.e_sym[6], 
            self.e8_3d - self.e_sym[8], 
            self.I40_3d - self.I_sym[(4, 0)], 
            self.I60_3d - self.I_sym[(6, 0)], 
            self.I61_3d - self.I_sym[(6, 1)], 
            self.I81_3d - self.I_sym[(8, 0)]]
                
        self.hat_sigma_0_ws_3d = [projection_f(self, expr) for expr in self.hat_sigma_0_ws]
        self.hat_delta_0_w_3d = projection_f(self, self.hat_delta_0_w)

        self.tuning_w_sol_dict_3d = sp.solve(eq_s, self.w_syms_list)
        self.e6_e8_w_sol_list_3d = [self.tuning_w_sol_dict_3d[w] for w in self.w_syms_list]

        self.hat_sigma_0_eIs_3d = [expr.subs(self.tuning_w_sol_dict_3d) for expr in self.hat_sigma_0_ws_3d]
        self.hat_delta_0_eI_3d = self.hat_delta_0_w_3d.subs(self.tuning_w_sol_dict_3d)                
        

    def DefineWeights3DTYP1(self, projection_f):
        self.DefineWeights(
            [self.e_sym[n] - self.e_expr[n] for n in range(2, 8 + 1, 2)] + 
            [self.I_sym[(4,0)] - self.typ_eq_s[4][0], 
             self.I_sym[(6,0)] - self.typ_eq_s[6][0], 
             self.I_sym[(8,0)] - self.typ_eq_s[8][0], 
             self.I_sym[(8,1)] - self.typ_eq_s[8][1]])
        
        eq_s = [
            self.e2_3d - self.e_sym[2], 
            self.e4_3d - self.e_sym[4], 
            self.e6_3d - self.e_sym[6], 
            self.e8_3d - self.e_sym[8], 
            self.I40_3d - self.I_sym[(4, 0)], 
            self.I60_3d - self.I_sym[(6, 0)], 
            self.I61_3d - self.I_sym[(6, 1)], 
            self.I80_3d - self.I_sym[(8, 0)], 
            self.I81_3d - self.I_sym[(8, 1)], 
            self.I82_3d - self.I_sym[(8, 2)], 
            self.I102_3d - self.I_sym[(10, 2)]]
                
        self.hat_sigma_0_ws_3d = [projection_f(self, expr) for expr in self.hat_sigma_0_ws]
        self.hat_delta_0_w_3d = projection_f(self, self.hat_delta_0_w)

        self.tuning_w_sol_dict_3d = sp.solve(eq_s, self.w_syms_list)
        self.e6_e8_w_sol_list_3d = [self.tuning_w_sol_dict_3d[w] for w in self.w_syms_list]

        self.hat_sigma_0_eIs_3d = [expr.subs(self.tuning_w_sol_dict_3d) for expr in self.hat_sigma_0_ws_3d]
        self.hat_delta_0_eI_3d = self.hat_delta_0_w_3d.subs(self.tuning_w_sol_dict_3d)        

    def DefineWeights3DTYP3(self, projection_f):
        self.DefineWeights(
            [self.e_sym[n] - self.e_expr[n] for n in range(2, 8 + 1, 2)] + 
            [self.I_sym[(4,0)] - self.typ_eq_s[4][0], 
             self.I_sym[(6,0)] - self.typ_eq_s[6][0], 
             self.I_sym[(8,0)] - self.typ_eq_s[8][0], 
             self.I_sym[(8,1)] - self.typ_eq_s[8][1]])
        
        eq_s = [
            self.e2_3d - self.e_sym[2], 
            self.e4_3d - self.e_sym[4], 
            self.e6_3d - self.e_sym[6], 
            self.e8_3d - self.e_sym[8], 
            self.I40_3d - self.I_sym[(4, 0)], 
            self.I60_3d - self.I_sym[(6, 0)], 
            self.I61_3d - self.I_sym[(6, 1)], 
            self.I80_3d - self.I_sym[(8, 0)], 
            self.I81_3d - self.I_sym[(8, 1)], 
            self.w_syms_list[7], 
            self.w_syms_list[9]]
                
        self.hat_sigma_0_ws_3d = [projection_f(self, expr) for expr in self.hat_sigma_0_ws]
        self.hat_delta_0_w_3d = projection_f(self, self.hat_delta_0_w)

        self.tuning_w_sol_dict_3d = sp.solve(eq_s, self.w_syms_list)
        self.e6_e8_w_sol_list_3d = [self.tuning_w_sol_dict_3d[w] for w in self.w_syms_list]

        self.hat_sigma_0_eIs_3d = [expr.subs(self.tuning_w_sol_dict_3d) for expr in self.hat_sigma_0_ws_3d]
        self.hat_delta_0_eI_3d = self.hat_delta_0_w_3d.subs(self.tuning_w_sol_dict_3d)        


    def DefineWeights3DTYP2(self, projection_f):
        self.DefineWeights(
            [self.e_sym[n] - self.e_expr[n] for n in range(2, 8 + 1, 2)] + 
            [self.I_sym[(4,0)] - self.typ_eq_s[4][0], 
             self.I_sym[(6,0)] - self.typ_eq_s[6][0], 
             self.I_sym[(8,0)] - self.typ_eq_s[8][0]])
        
        eq_s = [
            self.e2_3d - self.e_sym[2], 
            self.e4_3d - self.e_sym[4], 
            self.e4_3d - self.e_sym[6], 
            self.e4_3d - self.e_sym[8], 
            self.I40_3d - self.I_sym[(4, 0)], 
            self.I60_3d - self.I_sym[(6, 0)], 
            self.I61_3d - self.I_sym[(6, 1)], 
            self.I80_3d - self.I_sym[(8, 0)], 
            self.I81_3d - self.I_sym[(8, 1)], 
            self.I82_3d - self.I_sym[(8, 2)], 
            self.I102_3d - self.I_sym[(10, 0)]]
                
        self.hat_sigma_0_ws_3d = [projection_f(self, expr) for expr in self.hat_sigma_0_ws]
        self.hat_delta_0_w_3d = projection_f(self, self.hat_delta_0_w)

        self.tuning_w_sol_dict_3d = sp.solve(eq_s, self.w_syms_list)
        print("La:", self.tuning_w_sol_dict_3d)
        self.e6_e8_w_sol_list_3d = [self.tuning_w_sol_dict_3d[w] for w in self.w_syms_list]

        self.hat_sigma_0_eIs_3d = [expr.subs(self.tuning_w_sol_dict_3d) for expr in self.hat_sigma_0_ws_3d]
        self.hat_delta_0_eI_3d = self.hat_delta_0_w_3d.subs(self.tuning_w_sol_dict_3d)        

    """
    The LPT version is already taking the "optimal" sequence for the isotropy conditions
    """
    def DefineWeights3DLPT(self, projection_f, lambda_I_eq_2d, chi_I_eq_2d):
        self.DefineWeights(
            [self.e_sym[n] - self.e_expr[n] for n in range(2, 8 + 1, 2)] + 
            [lambda_I_eq_2d - self.I_sym['lambda_i'], 
             chi_I_eq_2d - self.I_sym['chi_i'], 
             self.I_sym[(6,0)] - self.typ_eq_s[6][0], 
             self.I_sym[(8,1)] - self.typ_eq_s[8][1]])
        
        self.lambda_I_eq_3d = projection_f(self, lambda_I_eq_2d) + self.w_syms_list[5]
        self.chi_I_eq_3d = projection_f(self, chi_I_eq_2d) - self.w_syms_list[5]
        
        eq_s = [
            self.e2_3d - self.e_sym[2], 
            self.e4_3d - self.e_sym[4], 
            self.e6_3d - self.e_sym[6], 
            self.e8_3d - self.e_sym[8],
            self.lambda_I_eq_3d - self.I_sym['lambda_i'], 
            self.chi_I_eq_3d - self.I_sym['chi_i'],
            self.I60_3d - self.I_sym[(6, 0)], 
            self.I61_3d - self.I_sym[(6, 1)], 
            self.I80_3d - self.I_sym[(8, 0)], 
            self.I81_3d - self.I_sym[(8, 1)], 
            self.I82_3d - self.I_sym[(8, 2)]]
                
        self.hat_sigma_0_ws_3d = [projection_f(self, expr) for expr in self.hat_sigma_0_ws]
        self.hat_delta_0_w_3d = projection_f(self, self.hat_delta_0_w)

        self.tuning_w_sol_dict_3d = sp.solve(eq_s, self.w_syms_list)
        self.e6_e8_w_sol_list_3d = [self.tuning_w_sol_dict_3d[w] for w in self.w_syms_list]
        self.tuning_w_sol_list_3d = [self.tuning_w_sol_dict_3d[w] for w in self.w_syms_list]

        self.hat_sigma_0_eIs_3d = [expr.subs(self.tuning_w_sol_dict_3d) for expr in self.hat_sigma_0_ws_3d]
        self.hat_delta_0_eI_3d = self.hat_delta_0_w_3d.subs(self.tuning_w_sol_dict_3d)
        
        return
    
    def GetFullySolvedWeights3D(self):
        """
        - Here we define the fully solved systems: i.e. e6 and e8 as a function of \hat{\sigma}_1 and \hat{\delta}_0
        - This step can only be done after checking that the expressions obtained from 'DefineWeights'
        self.hat_sigma_0_eIs and self.hat_delta_0_eI
        """
        
        eq_s = [sp.Eq(self.hat_sigma_0_eIs_3d[1], self.hat_sigma_1_sym)]
        eq_s += [sp.Eq(self.hat_delta_0_eI_3d, self.hat_delta_0_sym)]

        self.e6_e8_sol_dict_3d = sp.solve(eq_s, [self.e_sym[6], self.e_sym[8]])

        self.e6_e8_w_sol_list_3d = [w.subs(self.e6_e8_sol_dict_3d) for w in self.e6_e8_w_sol_list_3d]

    def GetFullySolvedWeights3DSigma2(self):
        """
        - Here we define the fully solved systems: i.e. e6 and e8 as a function of \hat{\sigma}_1 and \hat{\delta}_0
        - This step can only be done after checking that the expressions obtained from 'DefineWeights'
        self.hat_sigma_0_eIs and self.hat_delta_0_eI
        """
        
        eq_s = [sp.Eq(self.hat_sigma_0_eIs_3d[1], self.hat_sigma_1_sym), 
                sp.Eq(self.hat_sigma_0_eIs_3d[2], self.hat_sigma_2_sym)]

        self.e6_e8_sol_dict_3d = sp.solve(eq_s, [self.e_sym[6], self.e_sym[8]])
        self.e6_e8_w_sol_list_3d = [w.subs(self.e6_e8_sol_dict_3d) for w in self.e6_e8_w_sol_list_3d]        

    def GetStencilDict3D(self, w_list):
        swap_stencil = {}
        xis_list = reduce(lambda x, y: x + y, [self.G[root_v].tolist() for root_v in self.G])
        root_vs = [root_v for root_v in self.G]
        w_values = [w_list[i] for i in range(len(root_vs))]
        swap_stencil['Ws'] = reduce(lambda x, y: x + y, [[w_values[i]] * len(self.G[root_v]) for i, root_v in enumerate(root_vs)])
        swap_stencil['XIs'] = xis_list
        return swap_stencil


    def GetFullySolvedStencilISO3D(self, values_dict):
        self.GetStdIsoDict()
        self.GetFullySolvedWeights3D()

        subs_dict = {**self.e6_e8_sol_dict_3d,
                     self.e_sym[4]: values_dict[self.e_sym[4]],
                     self.hat_sigma_1_sym: values_dict[self.hat_sigma_1_sym], 
                     self.hat_delta_0_sym: values_dict[self.hat_delta_0_sym]}
        
        self.e6_e8_w_sol_iso_list_3d = [w.subs(subs_dict) for w in self.e6_e8_w_sol_list_3d]
        self.e6_e8_w_sol_iso_list_3d = [w.subs(self.std_iso_dict) for w in self.e6_e8_w_sol_iso_list_3d]
                
        return self.GetStencilDict3D(self.e6_e8_w_sol_iso_list_3d)

    def GetFullySolvedStencilISO3DSigma2(self, values_dict):
        self.GetStdIsoDict()
        self.GetFullySolvedWeights3DSigma2()

        subs_dict = {**self.e6_e8_sol_dict_3d,
                     self.e_sym[4]: values_dict[self.e_sym[4]],
                     self.hat_sigma_1_sym: values_dict[self.hat_sigma_1_sym], 
                     self.hat_sigma_2_sym: values_dict[self.hat_sigma_2_sym]}
        
        self.e6_e8_w_sol_iso_list_3d = [w.subs(subs_dict) for w in self.e6_e8_w_sol_list_3d]
        self.e6_e8_w_sol_iso_list_3d = [w.subs(self.std_iso_dict) for w in self.e6_e8_w_sol_iso_list_3d]
        # print(self.e6_e8_w_sol_iso_list_3d)
                
        return self.GetStencilDict3D(self.e6_e8_w_sol_iso_list_3d)

    def GetWeightsStencilSol3D(self, values_dict):
        self.GetStdIsoDict()
        
        self.w_sol_iso_list = [w.subs(values_dict) for w in self.tuning_w_sol_list_3d]
        self.w_sol_iso_list = [w.subs(self.std_iso_dict) for w in self.w_sol_iso_list]

        return self.GetStencilDict3D(self.w_sol_iso_list)

