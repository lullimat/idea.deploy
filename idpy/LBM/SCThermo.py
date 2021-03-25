__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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
Provides classes for the computation of the thermodynamic quantities related
to the Shan-Chen model
'''

import scipy.integrate as integrate
from scipy.optimize import fsolve, bisect
from numpy import linspace
from sympy import Rational, diff, simplify
from sympy import lambdify as sp_lambdify
from sympy import symbols as sp_symbols
from sympy import exp as sympy_exp
from sympy.solvers.solveset import nonlinsolve
from sympy.solvers import solve
from functools import reduce
import math

from idpy.LBM.SCFStencils import SCFStencils
from idpy.Utils.ManageData import ManageData

def FindSingleZeroRange(func, x_init, delta_val):
    old_val, new_val = func(x_init), func(x_init)
    while old_val * new_val > 0:
        old_val = new_val
        x_init += delta_val
        new_val = func(x_init)
    return (x_init - delta_val, x_init)
    

def FindZeroRanges(func, n_range, n_bins, n_delta, debug_flag = False):
    zero_ranges = []
    old_val, new_val = 0, 0
    # Here I can use linspace
    for n_i in range(n_bins):
        new_val = func(n_range[0] + n_delta * n_i)
        if debug_flag:
            print(n_bins, n_i, n_range[0] + n_delta * n_i, new_val, old_val)
            print(n_i > 0, old_val * new_val < 0, n_i > 0 and old_val * new_val < 0)
            print()
        
        if n_i > 0 and old_val * new_val < 0:
            zero_ranges.append((n_range[0] + n_delta * (n_i - 1), 
                                n_range[0] + n_delta * n_i))
        old_val = new_val
    return zero_ranges

def FindExtrema(func, f_arg, arg_range = (0.01,3.), arg_bins = 256):
    d_func = lambda f_arg_: diff(func,f_arg).subs(f_arg, f_arg_)
    arg_delta = (arg_range[1] - arg_range[0])/arg_bins
    zero_ranges = FindZeroRanges(d_func, arg_range, arg_bins, arg_delta)

    print("zero_ranges: ", zero_ranges)
    
    extrema = []
    for z_range in zero_ranges:
        # Initialization point from LEFT -> z_range[0] NOT z_range[1]
        arg_swap = bisect(d_func, z_range[0], z_range[1])
        f_swap = func.subs(f_arg, arg_swap)
        extrema.append((arg_swap,f_swap))
    return extrema

class ShanChen:
    # Symbols should be safe here
    n, G, theta, psi, d_psi, e2 = \
        sp_symbols("n G \\theta \\psi \\psi' e_{2}")
    P = theta*n + Rational('1/2')*G*e2*psi**2

    def __init__(self,
                 psi_f = None,
                 G_val = -3.6, theta_val = 1., e2_val = 1.,
                 n_eps = 0.01):

        # Variables Init
        self.psi_f = sympy_exp(-1/self.n) if psi_f is None else psi_f
        #print(self.psi_f)
        self.G_val, self.theta_val, self.e2_val =  G_val, theta_val, e2_val
        self.n_eps = n_eps
        self.d_psi_f = diff(self.psi_f, self.n)
        
        self.P_subs = self.P.subs(self.psi, self.psi_f).subs(self.G, self.G_val)
        self.P_subs = self.P_subs.subs(self.theta, self.theta_val).subs(self.e2, self.e2_val)
        self.P_subs_lamb = sp_lambdify(self.n, self.P_subs)

        # Find Critical Point
        ## This substitution leaves both n and G free
        P_subs_swap = self.P.subs(self.psi, self.psi_f)
        P_subs_swap = P_subs_swap.subs(self.theta, self.theta_val)
        P_subs_swap = P_subs_swap.subs(self.e2, self.e2_val)
        
        self.d_P = diff(P_subs_swap, self.n)
        self.dd_P = diff(self.d_P, self.n)
        #print([self.d_P, self.dd_P])
        self.critical_point = solve([self.d_P, self.dd_P], [self.G, self.n])
        
        self.G_c, self.n_c = float(self.critical_point[0][0]), float(self.critical_point[0][1])
        self.P_c = P_subs_swap.subs(self.n, self.n_c).subs(self.G, self.G_c)

        if self.G_val * self.e2_val > self.G_c * self.e2_val:
            print("The value of G: %f is above the critical point G_c: %f for the chosen %s" % (self.G_val, self.G_c, str(self.psi) + " = " + str(self.psi_f)))
            print("-> No phase separation")
        else:
            # Find Extrema
            lambda_tmp = sp_lambdify(self.n, self.P_subs - self.P_c)
            ## Here I want to find the value of the density that correxpond to the critical
            ## pressure because by construction this value of the density is larger than
            ## any coexistence extreme, and there is only one
            self.range_ext = FindSingleZeroRange(lambda_tmp, self.n_eps, self.n_eps)[1]
            ## Hence we can look for extrema starting from self.n_eps to self.range_ext
            ## Cannot begin from zero because for some choices of \psi the derivative
            ## might be singular
            print("Extrema:", self.range_ext)
            self.extrema = FindExtrema(self.P_subs, self.n,
                                       arg_range = (self.n_eps, self.range_ext))
            self.coexistence_range = self.FindCoexistenceRange()
            print("Coexistence range (n, P): ", self.coexistence_range)
            print()            

        ### Init Ends
        
    def PressureTensorInit(self, py_stencil):
        self.PTensor = self.PressureTensor(py_stencil)

    def FlatInterfaceProperties(self, which_sol = 0, eps_val = None):
        self.FInterface = self.FlatInterface(self, self.PTensor, which_sol, eps_val)
            
    def FindCoexistenceRange(self):
        coexistence_range = []
        '''
        With this check we can manage values of the coupling for which one has
        negative pressures
        '''
        if self.extrema[1][1] > 0:
            func_f = lambda f_arg_: (self.P_subs.subs(self.n, f_arg_) - self.extrema[1][1])
            # Looking for the LEFT limit starting from ZERO
            # and ending after the first stationary point
            arg_swap = bisect(func_f, self.n_eps, self.extrema[0][0])
            p_swap = self.P_subs.subs(self.n, arg_swap)
            coexistence_range.append((arg_swap, p_swap))
        else:
            coexistence_range.append((0, 0))
            
        # Looking for the RIGHT limit starting from the RIGHT extremum
        # that is certainly at the LEFT of the value we are looking for
        func_f = lambda f_arg_: (self.P_subs.subs(self.n, f_arg_) - self.extrema[0][1])
        arg_swap = bisect(func_f, self.extrema[1][0] + self.n_eps, self.range_ext + self.n_eps)
        p_swap = self.P_subs.subs(self.n, arg_swap)
        coexistence_range.append((arg_swap, p_swap))
        return coexistence_range

    ####################################################################################
    ### Subclass: FlatInterface
    ####################################################################################

    class FlatInterface:
        def __init__(self, SC, PTensor, which_sol, eps_val):
            self.SC, self.PTensor = SC, PTensor
            # defining epsilon
            if eps_val is None:
                self.eps_val = \
                    PTensor.p_consts_wf['\epsilon'](self.PTensor.py_stencil.w_sol[which_sol])
            else:
                self.eps_val = eps_val
            print("eps_val:", self.eps_val)
                
            self.beta_val = self.PTensor.p_consts_wf['\beta'](self.PTensor.py_stencil.w_sol[which_sol])
            self.sigma_c_val = self.PTensor.p_consts_wf['\sigma_c'](self.PTensor.py_stencil.w_sol[which_sol])
            self.tolman_c_val = self.PTensor.p_consts_wf['t_c'](self.PTensor.py_stencil.w_sol[which_sol])
            self.dndx = None
            # defining symbols
            self.p_0, self.n_g, self.n_l, self.n_p, self.d_n = \
                sp_symbols("p_0 n_g n_l n' \\frac{dn}{dr}")
            self.eps = self.PTensor.p_consts_sym['\epsilon']
            self.beta = self.PTensor.p_consts_sym['\beta']
            self.sigma_c = self.PTensor.p_consts_sym['\sigma_c']
            self.tolman_c = self.PTensor.p_consts_sym['t_c']
            # Defining the integrand
            self.integrand = (self.p_0 - self.SC.P)*self.SC.d_psi_f/(self.SC.psi_f**(1 + self.eps))
            # Substituting \theta and e_2 and psi and eps and G
            self.integrand = self.integrand.subs(self.SC.theta, self.SC.theta_val)
            self.integrand = self.integrand.subs(self.SC.e2, 1)
            self.integrand = self.integrand.subs(self.SC.psi, self.SC.psi_f)
            self.integrand = self.integrand.subs(self.eps, self.eps_val)
            self.integrand = self.integrand.subs(self.SC.G, self.SC.G_val)
            # Make a function of n and p_0
            self.integrand_np = \
                (lambda n_, p_ :
                 self.integrand.subs(self.SC.n, n_).subs(self.p_0, p_).evalf())
            
            # Numerical value of the Maxwell Construction's Integral
            self.maxwell_integral = \
                (lambda target_values:
                 integrate.quad((lambda n_ : self.integrand_np(n_, target_values[0][1])),
                                target_values[0][0], target_values[1][0])[0])
            # Numerical value as a function of the delta density
            self.maxwell_integral_delta = \
                (lambda delta_: self.maxwell_integral(self.GuessDensitiesFlat(delta_)))
            
        def GuessDensitiesFlat(self, delta):
            target_values = []
    
            arg_init = self.SC.coexistence_range[0][0] + delta
            
            func_init = self.SC.P_subs.subs(self.SC.n, arg_init)
            target_values.append((arg_init, func_init))
            arg_range, arg_bins = [arg_init, self.SC.coexistence_range[1][0]], 2 ** 10
            arg_delta = (arg_range[1] - arg_range[0])/arg_bins
            
            delta_func_f = (lambda arg_: 
                            (self.SC.P_subs.subs(self.SC.n, arg_) - 
                             self.SC.P_subs.subs(self.SC.n, arg_range[0])))
            
            zero_ranges = FindZeroRanges(delta_func_f, arg_range, arg_bins, arg_delta,
                                         debug_flag = False)

            # Always pick the last range for the stable solution: -1
            #print("zero_ranges:", zero_ranges)
            #print(bisect(delta_func_f, zero_ranges[0][0], zero_ranges[0][1]))
            #print(bisect(delta_func_f, zero_ranges[-1][0], zero_ranges[-1][1]))
            
            solution = bisect(delta_func_f, zero_ranges[-1][0], zero_ranges[-1][1])
            
            arg_swap = solution
            func_swap = self.SC.P_subs.subs(self.SC.n, arg_swap)
            target_values.append((arg_swap, func_swap))
            
            return target_values

        def MechanicEquilibrium(self, n_bins = 32):
            # Need to find the zero of self.maxwell_integral_delta
            # Delta can vary between (0, and the difference between the gas maximum
            # and the beginning of the coexistence region
            '''
            search_range = \
                [self.SC.n_eps,
                 self.SC.extrema[0][0] - self.SC.coexistence_range[0][0] - self.SC.n_eps]
            '''
            search_range = \
                [self.SC.n_eps,
                 self.SC.extrema[0][0] - self.SC.coexistence_range[0][0]]
            
            search_delta = (search_range[1] - search_range[0])/n_bins
            mech_eq_range = FindZeroRanges(self.maxwell_integral_delta,
                                           search_range, n_bins, search_delta,
                                           debug_flag = False)
            
            mech_eq_delta = bisect(self.maxwell_integral_delta,
                                   mech_eq_range[0][0], mech_eq_range[0][1])

            self.mech_eq_zero = self.maxwell_integral_delta(mech_eq_delta)
            self.mech_eq_target = self.GuessDensitiesFlat(mech_eq_delta)
            print(self.mech_eq_target)

        def DNDXLambda(self, rho_g):
            prefactor = 24 * ((self.SC.psi_f)**self.eps)/(self.beta * self.SC.G * (self.SC.d_psi_f)**2)
            prefactor = prefactor.subs(self.beta, self.beta_val)
            prefactor = prefactor.subs(self.SC.G, self.SC.G_val)
            prefactor = prefactor.subs(self.eps, self.eps_val)
            prefactor_n = lambda n_: prefactor.subs(self.SC.n, n_).evalf()
            self.dndx = lambda n_: math.sqrt(prefactor_n(n_) * self.maxwell_integral([rho_g, [n_, rho_g[1]]]))
                            
        def SurfaceTension(self, mech_eq_target):
            self.DNDXLambda(mech_eq_target[0])
            prefactor = self.SC.G_val * self.sigma_c_val
            integrand_n = lambda n_: self.dndx(n_) * (self.SC.d_psi_f**2).subs(self.SC.n, n_).evalf()
            integral = integrate.quad(integrand_n, mech_eq_target[0][0], mech_eq_target[1][0])
            self.sigma_f = prefactor * integral[0]
            return self.sigma_f

    ####################################################################################
    ### Subclass: PressureTensor
    ####################################################################################

    class PressureTensor:        
        def __init__(self, py_stencil):
            # One stencil at the time
            self.py_stencil = py_stencil
            # Associating weights symbols
            self.w_sym = self.py_stencil.w_sym
            self.w_sym_list = self.py_stencil.w_sym_list
            # Get e_expr
            if not hasattr(self.py_stencil, 'e_expr'):
                self.py_stencil.GetWolfEqs()
            if not hasattr(self.py_stencil, 'typ_eq_s'):
                self.py_stencil.GetTypEqs()
                
            self.e_expr = self.py_stencil.e_expr
            self.typ_eq_s = self.py_stencil.typ_eq_s
            self.B2q_expr = self.py_stencil.B2q_expr
            self.B2n_expr = self.py_stencil.B2n_expr
            
            # Initializing Pressure Tensor symbols
            self.PConstants()
            self.InitPCoeff()
            self.PExpressW()

        def GetExprValues(self, w_sol = None):
            ## Need to add the new constants: Chi/Lambda
            if w_sol is None:
                w_sol = self.py_stencil.w_sol[0]

            print(self.e_expr)
            print("Isotropy constants")
            for elem in self.e_expr:
                w_i = 0
                swap_expr = self.e_expr[elem]
                for w in self.w_sym_list:
                    swap_expr = swap_expr.subs(w, w_sol[w_i])
                    w_i += 1
                print(self.e_expr[elem], swap_expr)
            print("\n")

            print("Pressure Tensor Constants")
            for elem in self.p_consts_sym:
                w_i = 0
                swap_expr = self.p_consts_w[elem]
                for w in self.w_sym_list:
                    swap_expr = swap_expr.subs(w, w_sol[w_i])
                    w_i += 1
                print(self.p_consts_sym[elem], swap_expr)
            print("\n")
                
            print("Typical Equations")
            for elem in self.typ_eq_s:
                for eq in self.typ_eq_s[elem]:
                    swap_expr = eq
                    w_i = 0
                    for w_sym in self.w_sym_list:
                        swap_expr = swap_expr.subs(w_sym, w_sol[w_i])
                        w_i += 1

                    print(elem, self.typ_eq_s[elem], swap_expr)
            print("\n")

            print("Wolfram Equations: B2q")
            for elem in self.B2n_expr:
                for eq in self.B2n_expr[elem]:
                    swap_expr = eq
                    w_i = 0
                    for w_sym in self.w_sym_list:
                        swap_expr = swap_expr.subs(w_sym, w_sol[w_i])
                        w_i += 1

                    print(elem, self.B2n_expr[elem], swap_expr)
            print("\n")
            
            print("Wolfram Equations: B2q")
            for elem in self.B2q_expr:
                for eq in self.B2q_expr[elem]:
                    swap_expr = eq
                    w_i = 0
                    for w_sym in self.w_sym_list:
                        swap_expr = swap_expr.subs(w_sym, w_sol[w_i])
                        w_i += 1

                    print(elem, self.B2q_expr[elem], swap_expr)


                    
        def InitPCoeff(self):
            # List of coefficients for the pressure tensor constants
            # Need to do this because each stencil can have a different
            # number of groups: for now: no more than the first 5!
            self.alpha_c, self.beta_c, self.gamma_c, self.eta_c, self.kappa_c, self.lambda_c = \
                [0] * 25, [0] * 25, [0] * 25, [0] * 25, [0] * 25, [0] * 25
            self.sigma_c_c, self.tolman_c_c = [0] * 25, [0] * 25
            self.lambda_i_c, self.lambda_t_c, self.lambda_n_c = [0] * 25, [0] * 25, [0] * 25
            self.chi_i_c, self.chi_t_c, self.chi_n_c = [0] * 25, [0] * 25, [0] * 25
            
            # alpha
            self.alpha_c[4], self.alpha_c[5], self.alpha_c[8] = 2, 4, 4
            self.alpha_c[9], self.alpha_c[10] = 12, 24
            self.alpha_c[13], self.alpha_c[16], self.alpha_c[17] = Rational(88, 3), 40, 80
            # beta
            self.beta_c[1], self.beta_c[2], self.beta_c[4], self.beta_c[5], self.beta_c[8] = \
                Rational("1/2"), 1, 6, 13, 12
            self.beta_c[9], self.beta_c[10] = Rational(57, 2), 58
            self.beta_c[13], self.beta_c[16], self.beta_c[17] = Rational(203, 3), 88, 177
            # gamma
            self.gamma_c[5], self.gamma_c[8], self.gamma_c[10] = 1, 4, Rational(8, 3)
            self.gamma_c[13], self.gamma_c[17] = Rational(68, 3), 5
            # eta
            self.eta_c[2], self.eta_c[5], self.eta_c[8], self.eta_c[10] = 1, 7, 12, Rational(46,3)
            self.eta_c[13], self.eta_c[17] = Rational(148, 3), 27            
            # kappa
            self.kappa_c[5], self.kappa_c[8] = 4, 8
            # lambda
            self.lambda_c[2], self.lambda_c[5], self.lambda_c[8] = 2, 12, 24
            # sigma_c
            self.sigma_c_c[1], self.sigma_c_c[4], self.sigma_c_c[5] = -6, -96, -108
            self.sigma_c_c[9], self.sigma_c_c[10] = -486, -768
            self.sigma_c_c[13], self.sigma_c_c[16], self.sigma_c_c[17] = -300, -1536, 2700
            # tolman_c
            self.tolman_c_c[1], self.tolman_c_c[4], self.tolman_c_c[5] = \
                -Rational('1/2'), -6, -6
            # Lambda_s
            self.lambda_i_c[1], self.lambda_i_c[2], self.lambda_i_c[4] = Rational('1/2'), -2, 6
            self.lambda_i_c[5], self.lambda_i_c[8] = -6, -24

            self.lambda_t_c[2], self.lambda_t_c[5], self.lambda_t_c[8] = 2, 12, 24
            self.lambda_n_c[2], self.lambda_n_c[5], self.lambda_n_c[8] = 1, 7, 12
            # chi_s
            self.chi_i_c[4], self.chi_i_c[5], self.chi_i_c[8] = 2, -1, -8
            self.chi_t_c[5], self.chi_t_c[8] = 4, 8
            self.chi_n_c[5], self.chi_n_c[8] = 1, 4
            
        def PConstants(self):
            # Defining symbols
            self.p_consts_sym = {}
            self.p_consts_sym['\alpha'] = sp_symbols('\\alpha')
            self.p_consts_sym['\beta'] = sp_symbols('\\beta')
            self.p_consts_sym['\gamma'] = sp_symbols('\\gamma')
            self.p_consts_sym['\eta'] = sp_symbols('\\eta')
            self.p_consts_sym['\kappa'] = sp_symbols('\\kappa')
            self.p_consts_sym['\lambda'] = sp_symbols('\\lambda')
            self.p_consts_sym['\epsilon'] = sp_symbols('\\epsilon')
            self.p_consts_sym['\sigma_c'] = sp_symbols('\\sigma_c')
            self.p_consts_sym['t_c'] = sp_symbols('t_c')
            # These symbols are not good anymore for higher order expansions
            self.p_consts_sym['\Lambda_{N}'] = sp_symbols('\\Lambda_{N}')
            self.p_consts_sym['\Lambda_{T}'] = sp_symbols('\\Lambda_{T}')
            self.p_consts_sym['\Lambda_{I}'] = sp_symbols('\\Lambda_{I}')

            self.p_consts_sym['\chi_{N}'] = sp_symbols('\\chi_{N}')
            self.p_consts_sym['\chi_{T}'] = sp_symbols('\\chi_{T}')
            self.p_consts_sym['\chi_{I}'] = sp_symbols('\\chi_{I}')
            

        def PExpressW(self):
            # Defining expressions: e
            # Should use a dictionary for the coefficients
            self.p_consts_w = {}
            self.p_consts_w['\alpha'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\alpha'] += -12*self.alpha_c[len2] * self.w_sym[len2]

            self.p_consts_w['\beta'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\beta'] += 12*self.beta_c[len2] * self.w_sym[len2]
            
            self.p_consts_w['\gamma'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\gamma'] += -4*self.gamma_c[len2] * self.w_sym[len2]
            
            self.p_consts_w['\eta'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\eta'] += 4*self.eta_c[len2] * self.w_sym[len2]
            
            self.p_consts_w['\kappa'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\kappa'] += self.kappa_c[len2] * self.w_sym[len2]
            
            self.p_consts_w['\lambda'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\lambda'] += self.kappa_c[len2] * self.w_sym[len2]

            self.p_consts_w['\sigma_c'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\sigma_c'] += self.sigma_c_c[len2] * self.w_sym[len2]/12

            self.p_consts_w['t_c'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['t_c'] += self.tolman_c_c[len2] * self.w_sym[len2]

            # Lambdas, Chis
            self.p_consts_w['\Lambda_{I}'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\Lambda_{I}'] += self.lambda_i_c[len2] * self.w_sym[len2]

            self.p_consts_w['\Lambda_{T}'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\Lambda_{T}'] += self.lambda_t_c[len2] * self.w_sym[len2]
                
            self.p_consts_w['\Lambda_{N}'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\Lambda_{N}'] += self.lambda_n_c[len2] * self.w_sym[len2]


            self.p_consts_w['\chi_{I}'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\chi_{I}'] += self.chi_i_c[len2] * self.w_sym[len2]

            self.p_consts_w['\chi_{T}'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\chi_{T}'] += self.chi_t_c[len2] * self.w_sym[len2]
                
            self.p_consts_w['\chi_{N}'] = 0
            for len2 in self.py_stencil.len_2s:
                self.p_consts_w['\chi_{N}'] += self.chi_n_c[len2] * self.w_sym[len2]

            self.p_consts_w['\epsilon'] = -2*self.p_consts_w['\alpha']/self.p_consts_w['\beta']

            # Defining Lambdas
            self.p_consts_wf = {}
            for elem in self.p_consts_w:
                self.p_consts_wf[str(elem)] = \
                    sp_lambdify([self.w_sym_list],
                             self.p_consts_w[str(elem)])
            

class ShanChanEquilibriumCache(ManageData):
    def __init__(self,
                 stencil = None,
                 G = None, c2 = None, psi_f = None,
                 dump_file = 'SCEqCache'):
        ManageData.__init__(self, dump_file = dump_file)

        if stencil is None:
            raise Exception("Missing argument stencil")

        if G is None:
            raise Exception("Missing argument G")

        if c2 is None:
            raise Exception("Missing argument c2")

        if psi_f is None:
            raise Exception("Missing argument psi_f")


        '''
        Looking for the file and data
        '''
        self.is_file, self.is_key = ManageData.Read(self), False
        self.dict_string = (str(psi_f) + "_" + str(float(G)) + "_" +
                            str(c2) + "_" + str(stencil.w_sol[0]))
        if self.is_file:
            if self.dict_string in ManageData.WhichData(self):
                self.data = ManageData.PullData(self, self.dict_string)
                self.is_key = True

        if self.is_key is False:
            '''
            I need to do this until I write a new pressure tensor class
            that also computes the Taylor expansion for the flat interface
            and consequently the expression for \varepsilon
            '''
            w1, w2, w4, w5, w8 = sp_symbols("w(1) w(2) w(4) w(5) w(8)")
            w9, w10, w13, w16, w17 = sp_symbols("w(9) w(10) w(13) w(16) w(17)")
            w_sym_list = [w1, w2, w4, w5, w8, w9, w10, w13, w16, w17]

            _eps_expr = (+ 48*w4 + 96*w5 + 96*w8
                         + 288*w9 + 576*w10 + 704*w13 + 960*w16 + 1920*w17)
            
            _eps_expr /= (+ 6*w1 + 12*w2 + 72*w4 + 156*w5 + 144*w8
                          + 342*w9 + 696*w10 + 812*w13 + 1056*w16 + 2124*w17)
            
            self.eps_lambda = sp_lambdify([w_sym_list], _eps_expr)

            _e2_expr = stencil.e_expr[2]
            self.e2_lambda = sp_lambdify([w_sym_list], _e2_expr)

            _weights_list = None
            if len(stencil.w_sol[0]) != 10:
                len_diff = 10 - len(stencil.w_sol[0])
                if len_diff < 0:
                    raise Exception("The number of weights must be 5 at most!")
                _weights_list = stencil.w_sol[0] + [0 for i in range(len_diff)]
            else:
                _weights_list = stencil.w_sol[0]


            _shan_chen = \
                ShanChen(psi_f = psi_f, G_val = G,
                         theta_val = c2,
                         e2_val = self.e2_lambda(_weights_list))

            _shan_chen.PressureTensorInit(stencil)
            _shan_chen.FlatInterfaceProperties()
            _shan_chen.FInterface.MechanicEquilibrium()

            _mech_eq_target = _shan_chen.FInterface.mech_eq_target
            
            _sigma_f = \
                _shan_chen.FInterface.SurfaceTension(_mech_eq_target)
            
            _n_g = _shan_chen.FInterface.mech_eq_target[0][0]
            _n_l = _shan_chen.FInterface.mech_eq_target[1][0]
            _p_0 = _shan_chen.FInterface.mech_eq_target[1][1]
            _n_c = _shan_chen.n_c
            _G_c = _shan_chen.G_c

            _data_dict = {'G_c': _G_c, 'n_c': _n_c,
                          'n_l': _n_l, 'n_g': _n_g,
                          'p_0': _p_0, 'sigma_f': _sigma_f}

            
            self.PushData(data = _data_dict,
                          key = self.dict_string)

            self.Dump()

    def GetFromCache(self):
        return ManageData.PullData(self, key = self.dict_string)
