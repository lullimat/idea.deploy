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
Provides a class for the computation of the 2D discretized velocities
'''

from functools import reduce
from sympy import symbols, Rational, solve
from sympy.solvers.solveset import linsolve
from sympy import init_printing as sym_init_print
from numpy import array as np_array
import json as py_json

# I would like to write this class in terms of lambdas
# So that it can treat any dimension

# Double factorial: any library providing it?
def dfact(n):
    if n > 1:
        return n*dfact(n - 2)
    else:
        return 1

# x_max will need to be changed to a string
# typical of the name of the lattice...or both
class LatticeVectors:
    def __init__(self, x_max = 1, root_sym = '\\xi'):
        self.x_max, self.root_sym = x_max, root_sym
        self.lv = []
        # Add the origin: velocities vectors
        self.lv.append([(0, 0)])
        # Find vectors that cannot be linked
        # by neither a pi/2 nor a pi rotation
        # Need to parametrize the dimension
        for x in range(1, x_max + 1):
            for y in range(x_max + 1):
                swap_l = []
                swap_l.append((x, y))
                # pi/2 rotation
                swap_l.append((-y, x))
                # pi rotation
                swap_l.append((-x, -y))        
                # -pi/2 rotation
                swap_l.append((y, -x))
                self.lv.append(swap_l)
        
        # Need to group the vectors by length
        min_len_2, max_len_2 = 1, 2*x_max**2
        self.lv_len_2 = [[] for x in range(max_len_2 + 1)]
        
        self.unique_len = []
        for elem in self.lv:
            # Squared length
            swap_len = reduce(lambda x, y : x + y, map(lambda x : x**2, elem[0]))
            if swap_len not in self.unique_len:
                self.unique_len.append(swap_len)
            for sub_elem in elem:
                self.lv_len_2[swap_len].append(sub_elem)
        self.unique_len.sort()

        # Symbols for the vectors
        # sym_init_print()
        self.v_counter = 0
        self.v_sym = []
        for v_group in self.lv_len_2:
            if len(v_group) > 0:
                for elem in v_group:
                    self.v_sym.append(symbols(self.root_sym +
                                              '_{' + str(self.v_counter) + '}'))
                    self.v_counter += 1

    def DumpJSON(self, sim_path):
        # Packing json data
        json_out = {}
        json_out["dimensions"] = 2
        json_out["q"] = self.v_counter
        json_out["lengths_n"] = len(self.unique_len)
        json_out["lengths"] = self.unique_len
        #json_out["vectors"] = {}
        counter = 0

        swap_vectors = []
        for v_group in self.lv_len_2:
            if len(v_group) > 0:
                for elem in v_group:
                    #json_out["vectors"][str(self.v_sym[counter])] = elem
                    swap_vectors.append(elem)
                    counter += 1
        json_out["vectors"] = swap_vectors                
        output = open(sim_path + 'xi.json', 'w')
        py_json.dump(json_out, output)
        output.close()
        
# Class containing the weights: symbols and numerical values
class Weights:
    def __init__(self, LV):
        self.LV = LV
        self.w_sym = []
        self.c2_sym = symbols("c_0^2")
        # Create the symbols for the weights
        for len_i in range(self.LV.unique_len[-1] + 1):
            self.w_sym.append(symbols('w(' + str(len_i) + ')'))
        # Pruned weights list
        self.w_sym_pr = list(map(lambda x: self.w_sym[x], self.LV.unique_len))
        # W constants values in a dictionary (?)
        self.w = {}
        self.c = 0

    # Need to find a better name than TypicalSolution
    def TypicalSolution(self):
        # Need to define the system of equations by the generalized
        # Isotropy requirements
        self.eq_s, self.eq_s_c = [], []

        # Only 2D for now: but it can be generalized
        # Let's begin with the coefficient
        max_n, eqs_count = 0, 0
        self.exps_list = []
        # Exit condition: number of weights + speed of sound
        # In this way we compute the right number of indepenedent equations
        # ... need to be more specific
        exps_counter = 0
        while exps_counter < len(self.LV.unique_len) + 1:
            exps_swap, exps_counter_swap = [], 0
            # Heavily dependent on the dimension, for now
            for base_exp in range(0, max_n//2 - (max_n//2)%2 + 1, 2):
                #print(max_n, max_n - base_exp, base_exp)
                exps_swap.append([max_n - base_exp, base_exp])
                exps_counter_swap += 1
            # putting the same as sc_iso makes a problem...why ?
            exps_counter += exps_counter_swap
            self.exps_list.append(exps_swap)
            max_n += 2

        # Now we assemble the different equations
        for exps_seq in self.exps_list:
            partial_eqs = []
            c2_coeff_list = []
            for exps in exps_seq:
                coeff = 0
                swap_eqs = 0
                for len_2 in self.LV.unique_len:
                    coeff = 0
                    for elem in self.LV.lv_len_2[len_2]:
                        coeff += reduce(lambda x, y: x*y,
                                        map(lambda x, y : x**y, elem, exps))
                    swap_eqs += coeff * self.w_sym[len_2]
                    #print(exps, self.LV.lv_len_2[len_2], coeff)
                # Now compute the term proportional to some power of the sound speed
                c2_coeff_swap = reduce(lambda x, y : x*y,
                                       map(lambda x: dfact(x - 1), exps))
                c2_coeff_list.append(c2_coeff_swap)
                partial_eqs.append(swap_eqs)

            # Now let's build the equations separating the cases in which
            # we need to take the ratios
            c_pow_half = reduce(lambda x, y: x + y, exps)//2

            # Storing the equations involving powers of c_0^2
            for elem_i in range(len(partial_eqs)):
                swap_eqs = (partial_eqs[elem_i] -
                            c2_coeff_list[elem_i] * self.c2_sym ** c_pow_half)
                self.eq_s_c.append(swap_eqs)
            
            partial_eqs_ratio = []
            if len(partial_eqs) == 1:
                partial_eqs[0] = (partial_eqs[0] -
                                  c2_coeff_list[0] * self.c2_sym ** c_pow_half)
            else:
                for elem_i in range(len(partial_eqs) - 1):
                    c2_coeff_ratio = c2_coeff_list[elem_i]//c2_coeff_list[elem_i + 1]
                    swap_eqs = (partial_eqs[elem_i] -
                                c2_coeff_ratio * partial_eqs[elem_i + 1])
                    partial_eqs_ratio.append(swap_eqs)
                partial_eqs = partial_eqs_ratio

            for elem in partial_eqs:
                self.eq_s.append(elem)

        # Now we get the solutions of the system of equations
        # First we select the only w's involved - pruning
        # Can we do it better?
        self.sol_c = linsolve(self.eq_s, self.w_sym_pr)
        self.w_sol_c = {}
        for fset in self.sol_c:
            for elem_i in range(len(fset)):
                self.w_sol_c[str(self.w_sym_pr[elem_i])] = fset[elem_i]

        # Finally we need to solve a non-linear equation for computing the
        # speed of sound
        self.eq_c2 = 0
        exps = self.exps_list[2][0]
        
        for len_2 in self.LV.unique_len:
            coeff = 0
            for elem in self.LV.lv_len_2[len_2]:
                coeff += reduce(lambda x, y: x*y,
                                map(lambda x, y : x**y, elem, exps))
            self.eq_c2 += coeff * self.w_sol_c[str(self.w_sym[len_2])]
        
        c2_coeff_swap = reduce(lambda x, y : x*y,
                               map(lambda x: dfact(x - 1), exps))
        c_pow_half = reduce(lambda x, y: x + y, exps)//2
        self.eq_c2 -= c2_coeff_swap * self.c2_sym ** c_pow_half
        self.c2 = max(solve(self.eq_c2, self.c2_sym))

        # Subsitution of the solution for the speed of sound
        for len_2 in self.LV.unique_len:
            self.w_sol_c[str(self.w_sym[len_2])] = \
                self.w_sol_c[str(self.w_sym[len_2])].subs(self.c2_sym, self.c2)
    
        
