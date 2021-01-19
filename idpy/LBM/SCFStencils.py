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

import struct, binascii, sys, os, ctypes
from functools import reduce
from sympy import symbols, Rational
import scipy.special
from sympy.solvers.solveset import nonlinsolve


# A PYStencil contains the reference to the groups and the weights
# Need to compute the different Wolfram coefficients according to the desired
# order and then solve the system for the set of groups with and without external
# conditions: can we use From's work to generalize the procedure? Should we check it?
# Double factorial: any library providing it?
def dfact(n):
    if n > 1:
        return n*dfact(n - 2)
    else:
        return 1

class SCFStencils:
    def __init__(self, E, len_2s = [], groups_list = []):
        self.E, self.len_2s = E, len_2s
        self.len_2_indices = [] # TBD
        self.unique_groups_indices = []
        self.groups_list = len_2s
        # The program needs to know the group index in the binary file
        for len_2 in len_2s:
            for i in range(len(E.unique_len)):
                if len_2 == E.unique_len[i]:
                    self.len_2_indices.append(i + 1)
        # Moving to more precise grouping
        for group in groups_list:
            self.unique_groups_indices.append(
                E.unique_groups_labels.index(group))
                    
        self.wolfs_flag, self.typ_flag = False, False
        self.w_sol = []
        # Define weights symbols
        self.w_sym = {} # TBD
        for elem in self.len_2s:
            #self.w_sym['w(' + str(elem) + ')'] = symbols('w(' + str(elem) + ')')
            self.w_sym[elem] = symbols('w(' + str(elem) + ')')

        self.w_sym_gr = {}
        for group in groups_list:
            len2_tmp = reduce(lambda x, y : x + y, map(lambda x: x**2, group))
            self.w_sym_gr[group] = symbols('w_{' + str(group[0]) + str(group[1]) +
                                           '}(' + str(len2_tmp) + ')')

        # TBD
        self.w_sym_list = [self.w_sym[elem]
                           for elem in self.len_2s]
        self.w_sym_gr_list = [self.w_sym_gr[group]
                              for group in groups_list]

        # Get Maximum Isotropy
        # n_eq starts from one because we always impose the e_2 value
        self.e_max, found, n_eq = 2, False, 1
        if len_2s[-1] == 32:
            found = True
            self.e_max = 16
            
        while not found:
            n_eq += self.n_eqs_order(self.e_max)
            #if n_eq == len(self.unique_groups_indices):
            if n_eq == len(self.len_2s):
                found = True
            else:
                self.e_max += 2
                
        # Dummy m_iso: how are these quantities defined?
        self.m_iso = []
                
        # Define Isotropy constants symbols
        ## This does not work because equations above the highest
        ## isotropy order turn out to be linearly dependent...double
        ## check the determinant
        ## _max_order = max(2*(len(self.w_sym_list) - 1), self.e_max)
        _max_order = self.e_max
        self.e_sym = {}
        for i in range(2, _max_order + 2, 2):
            #self.e_sym['e_{' + str(i) + '}'] = symbols('e_{' + str(i) + '}')
            self.e_sym[i] = symbols('e_{' + str(i) + '}')

        self.e_sym_list = [self.e_sym[i]
                           for i in range(2, _max_order + 2, 2)]

    def PushStencil(self):
        _Es = tuple(elem
                    for len_2 in self.len_2s
                    for elem in self.E.e_len_2[len_2])

        _Ws = []
        len_k = 0
        for len_2 in self.len_2s:
            for elem in self.E.e_len_2[len_2]:
                _Ws.append(self.w_sol[0][len_k])
            len_k += 1
        _Ws = tuple(_Ws)
        
        _stencil_dict = {'Es': _Es, 'Ws': _Ws}
        return _stencil_dict
            
    def small_m(self, n):
        n_ = int(n)
        return n_ + n_ % 2
    
    def big_m(self, n):
        n_ = int(n)
        return n_ - (2 + n_ % 2)
    
    def n_eqs_order(self, order):
        n_ = int(order)
        return (n_//2 - (n_//2)%2)//2

    def ComputeA2n(self, order, len_2):
        n = order//2
        n_x, n_y = n + (n%2), n - (n%2)
        den = dfact(n_x - 1) * dfact(n_y - 1)
        a_tmp = 0.
        for elem in self.E.e_len_2[len_2]:
            a_tmp += pow(elem[0], n_x) * pow(elem[1], n_y)
        return Rational(str(a_tmp) + '/' + str(den))

    def ComputeB2n2q(self, order, q2, len_2):
        if q2 == order:
            return False
        b_tmp = 0.
        for elem in self.E.e_len_2[len_2]:
            b_tmp += pow(elem[0], q2) * pow(elem[1], order - q2)

        b_tmp = int(b_tmp)
        b_tmp -= self.ComputeA2n(order, len_2) * dfact(q2 - 1) * dfact(order - q2 - 1) 
        return b_tmp

    def ComputeB2n2n(self, order, len_2):
        b_tmp = 0.
        for elem in self.E.e_len_2[len_2]:
            b_tmp += pow(elem[0], order)

        b_tmp = int(b_tmp)
        b_tmp -= self.ComputeA2n(order, len_2) * dfact(order - 1)
        # Now subtract the other coefficients with their multiplicity
        b_m = self.big_m(order//2)
        for q2 in range(order - b_m, order, 2):
            Z_2n_2q = int(scipy.special.binom(order, q2))
            b_tmp -= self.ComputeB2n2q(order, q2, len_2) * Z_2n_2q
            
        return b_tmp

    def FindWeights(self, eqs_list = None, override = False):
        w_sol = []
            
        if eqs_list is None:
            eqs_list = []
            if not self.wolfs_flag:
                self.GetWolfEqs()
            if not self.typ_flag:
                self.GetTypEqs()

            eqs_list.append(self.e_expr[2] - 1)
            for _ in self.typ_eq_s:
                for __ in self.typ_eq_s[_]:
                    eqs_list.append(__)
                    
        if len(eqs_list) != len(self.w_sym_list) and not override:
            print("Number of equations %d != Number of weights: %d; %s" %
                  (len(eqs_list), len(self.w_sym_list), self.w_sym_list))
            return []

        w_sol_tmp = nonlinsolve(eqs_list, self.w_sym_list)
        for _ in w_sol_tmp:
            for __ in _:
                w_sol.append(__)

        self.w_sol.append(w_sol)
        self.m_iso.append([0.] * (self.e_max//2))
        return w_sol

    def RecoverTypEqs(self):
        self.rec_typ_eq_s = {}
        for order in range(4, self.e_max + 2, 2):
            self.rec_typ_eq_s[order] = []
            n = order//2
            s_m = self.small_m(n)
            b_m = self.big_m(n)

            # This is the first
            # When it exists
            expr = 0
            if b_m > 0:
                expr += self.B2q_expr[order][0]
                self.rec_typ_eq_s[order].append(expr)

            # These are the middle ones
            # When they exist
            if b_m > 2:
                for eq_i in range(len(self.B2q_expr[order]) - 1):
                    q2 = s_m + 2 + 2*eq_i
                    ratio = Rational(str(q2 + 1) + "/" + str(order - q2 - 1))
                    expr = - ratio*self.B2q_expr[order][eq_i]
                    expr += self.B2q_expr[order][eq_i + 1]
                    self.rec_typ_eq_s[order].append(expr)

            # This is the last
            # Even though it applies in all cases
            expr = 0
            expr += self.B2n_expr[order][0]
            if b_m > 0:
                expr += ((int(scipy.special.binom(order, order - 2)) - order + 1) *
                         self.B2q_expr[order][-1])
            if b_m > 2:
                q2_i = 0
                for q2 in range(s_m + 2, order - 2, 2):
                    expr += (int(scipy.special.binom(order, q2)) *
                             self.B2q_expr[order][q2_i])
                    q2_i += 1            
            self.rec_typ_eq_s[order].append(expr)
            
    def GetWolfEqs(self):        
        self.e_expr = {}
        self.B2q_expr, self.B2n_expr = {}, {}
        # A's or isotropy constants
        _max_order = self.e_max
        for order in range(2, _max_order + 2, 2):
            # Compute e_k
            e_coeffs = []
            for len_2 in self.len_2s:
                e_coeffs.append(self.ComputeA2n(order, len_2))

            expr_swap = reduce(lambda x, y : x + y,
                               map(lambda x, y : x * y,
                                   self.w_sym_list, e_coeffs))
            self.e_expr[order] = expr_swap

        # Min order for B coeffs is 4
        for order in range(4, self.e_max + 2, 2):
            self.B2q_expr[order], self.B2n_expr[order] = [], []
            # Compute B^(2n)_(2q)
            b_m = self.big_m(order//2)
            for q2 in range(order - b_m, order, 2):
                b2q_coeffs = []
                for len_2 in self.len_2s:
                    b2q_coeffs.append(self.ComputeB2n2q(order, q2, len_2))

                expr2q_swap = reduce(lambda x, y : x + y,
                                     map(lambda x, y : x * y,
                                         self.w_sym_list, b2q_coeffs))
                self.B2q_expr[order].append(expr2q_swap)
                
            # Compute B^(2n)_(2n)
            b2n_coeffs = []
            for len_2 in self.len_2s:
                b2n_coeffs.append(self.ComputeB2n2n(order, len_2))

            expr2n_swap = reduce(lambda x, y : x + y,
                                 map(lambda x, y : x * y,
                                     self.w_sym_list, b2n_coeffs))
            self.B2n_expr[order].append(expr2n_swap)
        # Set Flag
        self.wolfs_flag = True
        
            
    def GetTypEqs(self):
        self.typ_eq_s = {}
        for order in range(4, self.e_max + 2, 2):
            self.typ_eq_s[order] = []
            n = order//2
            s_m = self.small_m(n)
            eq_s_num = (order - s_m)//2
            eq_s_num_swap = eq_s_num + 1
            eq_s = [[] for i in range(eq_s_num_swap)]
            f_coeffs = [dfact(n_x - 1)*dfact(order - n_x - 1)
                        for n_x in range(s_m, order + 2, 2)]
        
            for len_2 in self.len_2s:
                for n_x in range(s_m, order + 2, 2):
                    coeff, n_y = 0, order - n_x
                    eq_i = (n_x - s_m)//2
                    
                    for elem in self.E.e_len_2[len_2]:
                        coeff += pow(elem[0], n_x) * pow(elem[1], n_y)
                    eq_s[eq_i].append(coeff)

            # Now I need to take the f_coeffs ratios and subtract
            # the equations in couples
            
            for i in range(eq_s_num):
                expr_den = reduce(lambda x, y : x + y,
                                  map(lambda x, y : x * y, self.w_sym_list, eq_s[i]))
                expr_num = reduce(lambda x, y : x + y,
                                  map(lambda x, y : x * y, self.w_sym_list, eq_s[i + 1]))
                coeff_ratio = Rational(str(f_coeffs[i + 1]) + '/' + str(f_coeffs[i]))
                self.typ_eq_s[order].append(expr_num - coeff_ratio * expr_den)
        # Set Flag
        self.typ_flag = True

# Need to modify to take into account vectors of the same length
# but belonging to different groups: all those that cannot be obtained by
# a permutation of the components
class BasisVectors:
    def __init__(self, x_max = 2, root_sym = 'e', dim = 2):
        self.x_max, self.root_sym, self.dim = x_max, root_sym, dim
        self.e = []
        # Find vectors that cannot be linked
        # by neither a pi/2 nor a pi rotation
        # Need to parametrize the dimension
        for x in range(1, x_max + 1):
            for y in range(x_max + 1):
                swap_l = []
                swap_l.append((x,y))
                # pi/2 rotation
                swap_l.append((-y, x))
                # pi rotation
                swap_l.append((-x, -y))        
                # -pi/2 rotation
                swap_l.append((y, -x))
                self.e.append(swap_l)
        
        # Need to group the vectors by length
        min_len_2, max_len_2 = 1, 2*x_max**2
        self.e_len_2 = [[] for x in range(max_len_2 + 1)]
        
        self.unique_len = []
        
        for elem in self.e:
            # Squared length
            #swap_len = self.len_2(elem[0])
            swap_len = reduce(lambda x, y : x + y,
                              map(lambda x : x**2, elem[0]))
            if swap_len not in self.unique_len:
                self.unique_len.append(swap_len)
            for sub_elem in elem:
                self.e_len_2[swap_len].append(sub_elem)
        self.unique_len.sort()

        # With this check we want to label the groups according
        # to the coordinates of the firs vector, taking into account
        # the symmetry (x,y) <-> (y,x)
        self.unique_groups = {}
        for elem in self.e:
            elem_p = (elem[0][1], elem[0][0])
            if elem_p in self.unique_groups:
                self.unique_groups[elem_p] += elem
            else:
                self.unique_groups[elem[0]] = elem
        self.unique_groups_labels = list(self.unique_groups.keys())

        # And the symbols for the basis vectors
        self.e_counter = 1
        self.e_sym = []
        for e_group in self.e_len_2:
            if len(e_group) > 0:
                for elem in e_group:
                    self.e_sym.append(symbols(self.root_sym +
                                              '_{' + str(self.e_counter) + '}'))
                    self.e_counter += 1

    def DumpJSON(self, sim_path):
        # Packing json data
        json_out = {}
        json_out["dimensions"] = 2
        json_out["q"] = self.e_counter - 1
        json_out["lengths_n"] = len(self.unique_len)
        json_out["lengths"] = self.unique_len
        #json_out["vectors"] = {}
        counter = 0

        swap_vectors = []
        for e_group in self.e_len_2:
            if len(e_group) > 0:
                for elem in e_group:
                    #json_out["vectors"][str(self.v_sym[counter])] = elem
                    swap_vectors.append(elem)
                    counter += 1
        json_out["vectors"] = swap_vectors                
        output = open(sim_path + 'e.json', 'w')
        py_json.dump(json_out, output)
        output.close()

