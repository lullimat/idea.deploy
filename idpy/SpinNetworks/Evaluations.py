__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2023 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
__credits__ = ["Matteo Lulli", "Emanuele Zappala", "Antonino Marciano"]
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
__maintainer__ = "Matteo Lulli, Emanuele Zappala"
__email__ = "matteo.lulli@gmail.com"
__status__ = "Development"

import sympy as sp
from functools import reduce
import math

# Always: A = \sqrt{q}, where A is the variable of Kauffman's polynomial, 
# and q is used in recoupling theory        
from idpy.Utils.ManageData import ManageData

def FromLabelToStrSym(label):
    _k, _n, _letter = str(label[0][0]), str(label[0][1]), str(label[1])
    return _letter + '_{' + _k + '}^{' + _n + '}'

class Q:
    def __init__(self, r = -1, phase = 1):
        if not isinstance(r, int):
            raise Exception("Parameter 'r' must be an integer")
            
        self.r = r
        self.phase = phase
        self.exponent = sp.I * sp.pi / r
        
    def __call__(self):
        if self.r == -1 and self.phase == 1:
            return (-1,)
        else:
            return sp.exp(self.exponent) * self.phase, self.r
        
class Delta_n(ManageData):
    """
    x is our variable
    x -> -q gives the value in Section 3.3 of Kauffman-Lins
    Next steps: 
    1 - redefine the class as the parent class of everything else
    2 - need to separate the result into a prefactor for making negative 
    powers disappear q^{-k(n)} \\tilde{\\Delta}_n 
    """
    def __init__(self):
        self.q = sp.Symbol('q')
        
    def __call__(self, n, q = None):
        _expr = self._delta_n(n)
        
        if q is None:
            return _expr
        else:
            return _expr.subs(self.q, q[0])
        
    def _delta_n_new(self, n):
        _expr = ((-1) ** n) * (self.q ** (n + 1) - self.q ** (-n - 1)) / (self.q - self.q ** (-1))
        return _expr
    
    def _delta_n(self, n):
        _expr = 0
        for _k in range(n + 1):
            _expr += (-self.q) ** (n - 2 * _k)
        return _expr    
    
class DeltaFactorial(Delta_n):
    def __init__(self):
        Delta_n.__init__(self)
        self.Zero = sp.core.numbers.Zero()
        
    def CallNew(self, n, q = None):
        if q is not None and len(q):
            if n > q[1] - 2:
                return self.Zero
        
        _prod = 1
        for _i in range(1, n + 1):
            _prod *= Delta_n.__call__(self, _i)

        return _prod if q is None else _prod.subs(self.q, q[0])
            
    def __call__(self, n, q = None):
        if q is not None and len(q) > 1:
            if n > q[1] - 2:
                return self.Zero
        
        _prod = sp.core.numbers.One()
        for _i in range(1, n + 1):
            _prod *= Delta_n.__call__(self, _i)
            _prod = sp.expand(_prod)
            
        return _prod if q is None else _prod.subs(self.q, q[0])
        
class QFactorial(DeltaFactorial):
    def __init__(self):
        DeltaFactorial.__init__(self)

    def __call__(self, n, q = None):
        if q is not None and len(q):
            if n > q[1] - 1:
                return self.Zero
        
        return ((-1) ** ((n - 1) * n // 2)) * \
                    DeltaFactorial.__call__(self, n - 1, q = q)

def CheckCompatibility(a, b, c):
    cond1 = ((a + b + c) %2) == 0
    cond2 = (a + b - c) >= 0
    cond3 = (b + c - a) >= 0
    cond4 = (c + a - b) >= 0
    return cond1 and cond2 and cond3 and cond4

def CheckQCompatibility(a,b,c,r):
    cond5 = (a+b+c) <= 2*r - 4 
    return cond5 and CheckCompatibility(a, b, c)

'''
Keep on thinking...
We should implement a flag for avoiding the check when we perform the summation
'''
def CheckVertex(a, b, c, q = None):
    if q is None or (q is not None and len(q) == 1):
        return CheckCompatibility(a, b, c)
    if q is not None and len(q) > 1:
        return CheckQCompatibility(a, b, c, q[1])
    
def ThetaHat(a, b, c, q = None):
    return sp.core.numbers.One() if CheckVertex(a, b, c, q) else sp.core.numbers.Zero()

'''
m = (a + b - c) / 2
n = (b + c - a) / 2
p = (a + c - b) / 2
'''                               
class Theta(DeltaFactorial):
    def __init__(self):
        DeltaFactorial.__init__(self)
        self.m, self.n, self.p = 0, 0, 0
    
    '''
    See p.57 KL Corollary 2
    '''
    def __call__(self, a, b, c, q = None):
        if not CheckVertex(a, b, c, q):
            ##print("Ahia!")
            return self.Zero
        
        self.GetMNP(a, b, c)
        _prod_num = 1
        ## 1
        _expr = \
            DeltaFactorial.__call__(self, self.m + self.n + self.p, 
                                    q = q)
        if _expr == self.Zero:
            return self.Zero
        _prod_num *= _expr
        ## 2
        _expr = \
            DeltaFactorial.__call__(self, self.n - 1, q = q)
        if _expr == self.Zero:
            return self.Zero
        _prod_num *= _expr
        ## 3
        _expr = \
            DeltaFactorial.__call__(self, self.m - 1, q = q)
        if _expr == self.Zero:
            return self.Zero
        _prod_num *= _expr
        ## 4
        _expr = \
            DeltaFactorial.__call__(self, self.p - 1, q = q)
        if _expr == self.Zero:
            return self.Zero
        _prod_num *= _expr        
        '''
        The denominator cannot be zero alone by theorem
        '''
        _prod_den = \
            (DeltaFactorial.__call__(self, self.m + self.p - 1, q = q) * 
             DeltaFactorial.__call__(self, self.n + self.p - 1, q = q) * 
             DeltaFactorial.__call__(self, self.m + self.n - 1, q = q))

        return _prod_num / _prod_den
        
    def GetMNP(self, a, b, c):
        self.m = (a + b - c) // 2
        self.n = (b + c - a) // 2
        self.p = (a + c - b) // 2
            
class Tetra(QFactorial):
    '''
    p.98 KL
    '''
    def __init__(self):
        QFactorial.__init__(self)
        
        self.a1, self.a2, self.a3, self.a4 = 0, 0, 0, 0        
        self.b1, self.b2, self.b3 = 0, 0, 0
                
        self.m, self.M = 0, 0
        
    def __call__(self, A, B, C, D, E, F, q = None):
        if not CheckVertex(A, E, D, q):
            ##print("tt aed!")
            return self.Zero
        if not CheckVertex(B, C, E, q):
            #3print("tt bce!")
            return self.Zero
        if not CheckVertex(A, B, F, q):
            ##print("tt abf!")
            return self.Zero
        if not CheckVertex(C, D, F, q):
            ##print("tt cdf!")
            return self.Zero
        
        self.SetA(A, B, C, D, E, F)
        self.SetB(A, B, C, D, E, F)
        self.SetMm()
        
        _IFact = self.SetIFact()
        if _IFact == self.Zero:
            return self.Zero
                
        _sum = self.Zero
        for _s in range(self.m, self.M + 1):
            _swap_num = self.GetNum(_s)
            if _swap_num == self.Zero:
                continue
            _sum += _swap_num / self.GetDen(_s)
            
        if _sum == self.Zero:
            return self.Zero
        
        '''
        Delay the unchecked computations as much as possible
        '''
        _EFact = self.SetEFact(A, B, C, D, E, F)        
        _expr = (_IFact/_EFact) * _sum 
        
        if q is None:
            return _expr
        else:
            return _expr.subs(self.q, q[0])
    
    def GetNum(self, s):
        return ((-1) ** s) * QFactorial.__call__(self, s + 1)
    
    def GetDen(self, s):
        _prod = 1
        for _a in self.a_s:
            _prod *= QFactorial.__call__(self, s - _a)
        for _b in self.b_s:
            _prod *= QFactorial.__call__(self, _b - s)
        return _prod
    
    def SetEFact(self, A, B, C, D, E, F):
        _expr = \
            QFactorial.__call__(self, A) * \
            QFactorial.__call__(self, B) * \
            QFactorial.__call__(self, C) * \
            QFactorial.__call__(self, D) * \
            QFactorial.__call__(self, E) * \
            QFactorial.__call__(self, F)
        return _expr
    
    def SetIFact(self):
        _expr = 1
        for _a in self.a_s:
            for _b in self.b_s:
                _swap_expr = QFactorial.__call__(self, _b - _a)
                if _swap_expr == self.Zero:
                    return self.Zero
                _expr *= _swap_expr
                
        return _expr
        
    def SetA(self, A, B, C, D, E, F):
        self.a1 = (A + D + E) // 2
        self.a2 = (B + C + E) // 2
        self.a3 = (A + B + F) // 2
        self.a4 = (C + D + F) // 2
        self.a_s = [self.a1, self.a2, self.a3, self.a4]        
        
    def SetB(self, A, B, C, D, E, F):
        self.b1 = (B + D + E + F) // 2
        self.b2 = (A + C + E + F) // 2
        self.b3 = (A + B + C + D) // 2
        self.b_s = [self.b1, self.b2, self.b3]
        
    def SetMm(self):
        self.m = max(self.a_s)
        self.M = min(self.b_s)

class SixJWigner:
    def __init__(self):
        self.tt = Tetra()
        self.th = Theta()
        self.Zero = sp.core.numbers.Zero()        
    
    def __call__(self, a, b, c, d, i, j, q = None):
        _num_tt = self.tt(a, b, c, d, i, j, q)
        
        if _num_tt == self.Zero:
            return self.Zero
        
        _den = \
            self.th(b, a, j, q) * self.th(c, d, j, q) * \
            self.th(b, c, i, q) * self.th(a, d, i, q)
        
        _expr = _num_tt / sp.sqrt(_den)
        return _expr

class SixJ:
    def __init__(self):
        self.tt = Tetra()
        self.dn = Delta_n()
        self.th = Theta()
        self.Zero = sp.core.numbers.Zero()
        
    def __call__(self, a, b, c, d, i, j, q = None):
        _num_tt = self.tt(a, b, c, d, i, j, q)
        if _num_tt == self.Zero:
            return self.Zero
        _num_dn = self.dn(i, q)
        if _num_dn == self.Zero:
            return self.Zero
            
        _expr = _num_tt * _num_dn / self.th(a, d, i, q) / self.th(b, c, i, q)
        return _expr
    
class SixJKL(SixJ):
    def __init__(self):
        SixJ.__init__(self)
        
    def _W(self, i, q = None):
        return (sp.I ** (2 * i)) * sp.sqrt(((-1) ** (2 * i)) * self.dn(2 * i, q = q))

    def _Whalf(self, i, q = None):
        return (sp.I ** i) * sp.sqrt(((-1) ** i) * self.dn(i, q = q))
    
    def __call__(self, a, b, c, d, i, j, q = None):
        _symbol = [[a, b, j], [c, d, i]]
        display(sp.Matrix(_symbol))
        _expr_new = SixJ.__call__(self, a, b, c, d, i, j, q = q)
        if _expr_new == self.Zero:
            return self.Zero
                
        ##_w_ih = self._W(sp.Rational(i, 2), q = q)
        ##_w_jh = self._W(sp.Rational(j, 2), q = q)
        
        _w_ih = self._Whalf(i, q = q)
        _w_jh = self._Whalf(j, q = q)
        
        ##_expr2 = _expr / sp.sqrt(self.dn(i + 1, q = q)) / sp.sqrt(self.dn(j + 1, q = q))
        ##_expr2 /= (sp.I ** (i//2)) / (sp.I ** (j//2))
        
        #display(sp.simplify(_expr_new))
        #display(sp.simplify(_w_ih).evalf())
        #display(sp.simplify(_w_jh).evalf())
        #_expr_new /= _w_i / _w_j
        
        ##print(_expr.evalf(), "--------", _expr2.evalf())
        ## sp.simplify(_expr / _w_i / _w_j)
        
        return sp.simplify(_expr_new / (_w_ih * _w_jh))


def Get6JSymbol(params = ['a', 'b', 'c', 'd', 'e', 'f']):
    if len(params) != 6:
        raise Exception("There must be 6 values for the 6j symbol")    
    _str = \
        r'\left\{\begin{array}{ccc}' + \
        params[0] + ' & ' + params[1] + ' & ' + params[4] + r' \\' + \
        params[2] + ' & ' + params[3] + ' & ' + params[5] + \
        r'\end{array}\right\}'    
    return sp.Symbol(_str)

def GetTetSymbol(params = ['a', 'b', 'c', 'd', 'e', 'f']):
    if len(params) != 6:
        raise Exception("There must be 6 values for the Tetrahedron symbol")    
    _str = \
        r'\left[\begin{array}{ccc}' + \
        params[0] + ' & ' + params[1] + ' & ' + params[4] + r' \\' + \
        params[2] + ' & ' + params[3] + ' & ' + params[5] + \
        r'\end{array}\right]'
    return sp.Symbol(_str)    

def GetThetaSymbol(params = ['a', 'b', 'c']):
    if len(params) != 3:
        raise Exception("There must be 3 values for the theta-net")        
    _str = \
        r'\theta\left(' + \
        params[0] + ',' + params[1] + ',' + params[2] + \
        r'\right)'
    return sp.Symbol(_str)

def GetDNSymbol(params = ['a']):
    if len(params) != 1:
        raise Exception("There must be 1 values for the quantum dimension")    
    _str = \
        r'\Delta_{' + params[0] + '}'
    return sp.Symbol(_str)

def GetExtremaStr(extrema_labels=(('a', 'b'), ('c', 'd'))):
    sum_0, sum_1 = \
        extrema_labels[0][0] + '+' + extrema_labels[0][1],\
        extrema_labels[1][0] + '+' + extrema_labels[1][1]
    
    diff_00, diff_01 = \
        extrema_labels[0][0] + '-' + extrema_labels[0][1],\
        extrema_labels[0][1] + '-' + extrema_labels[0][0]
    
    diff_10, diff_11 = \
        extrema_labels[1][0] + '-' + extrema_labels[1][1],\
        extrema_labels[1][1] + '-' + extrema_labels[1][0]

    supremum = r'\min(' + sum_0 + ',' + sum_1 + ')'
    infimum = r'\max(\max(' + diff_00 + ',' + diff_01 + '),' + \
               '\max(' + diff_10 + ',' + diff_11 + '))'
    
    return supremum, infimum

def GetSumSymbol(params = ['a'], extrema = ['a_m', 'a_M']):
    if len(params) != 1:
        raise Exception("There must be 1 values for the summation")
    _str = \
        r'\sum_{' + params[0] + ' = ' + extrema[0] + '}^{' + extrema[1] + '}'
    return sp.Symbol(_str)

def GetMinSymbol(label):
    return r'm_{' + str(label[0][0]) + '}^{' + str(label[0][1]) + '}'

def GetMaxSymbol(label):
    return r'M_{' + str(label[0][0]) + '}^{' + str(label[0][1]) + '}'

class Evaluation:
    def __init__(self):
        self.list_6j, self.list_tet = [], []
        self.list_th, self.list_dn = [], []
        self.list_th_m1, self.list_dn_m1 = [], []        
        self.list_kdelta, self.list_sums = [], []
        self.sums_indices = {}
        self.list_sums_extrema = []
        '''
        lists dictionary
        '''
        self.elists = \
            {'6j': self.list_6j, 'tet': self.list_tet, 
             'th': self.list_th, 'dn': self.list_dn, 
             'th_m1': self.list_th_m1, 'dn_m1': self.list_dn_m1, 
             'kdelta': self.list_kdelta, 'sums': self.list_sums, 
             'sums_extrema': self.list_sums_extrema}
        
        self.SymbolsFunctions = \
            {'6j': Get6JSymbol, 
             'tet': GetTetSymbol, 
             'th': GetThetaSymbol, 
             'dn': GetDNSymbol, 
             'sums': GetSumSymbol}
        '''
        declare recoupling objects
        '''
        self.EvaluationFunctions = \
            {'6j': SixJ(), 'tet': Tetra(), 
             'th': Theta(), 'dn': Delta_n()}

        self.EvaluationFunctionsM1 = \
            {'th': Theta(), 'dn': Delta_n()}
        
        self.from_labels_to_colors = \
            lambda labels: tuple(self.colors[x][0] for x in labels)
        
        self.evaluation_wrapper = lambda functions, ls, q: \
            tuple(map(lambda x, y: x(*self.from_labels_to_colors(y), q), 
                      functions, ls))
                
    '''
    Promote colors to lists of length 1
    If there is more than a possible color there would be a list of all possible
    values
    '''
    
    '''
    We are using the 
    '''
    def FindPInLabelsExtrema(self, summed_index, sum_extrema):
        found = False
        for couple in sum_extrema:
            if summed_index in couple:
                found = True
                break
        return found    
        
    
    '''
    to be fixed
    '''
    def GetColorSumExtrema(self, label_i, q=None):
        ## sum_index = self.list_sums.index((label,))
        ## sum_bounds_labels = self.list_sums_extrema[label_i]

        upper_bound, lower_bound = math.inf, 0

        ## We could not do any better...
        if q is None or len(q) == 1:
            for neigh_labels in self.list_sums_extrema[label_i]:
                upper_bound = min(upper_bound, self.colors[neigh_labels[0]][0] + self.colors[neigh_labels[1]][0])
                lower_bound = max(lower_bound, abs(self.colors[neigh_labels[0]][0] - self.colors[neigh_labels[1]][0]))

                if lower_bound > upper_bound:
                    return 0, 1

        elif q is not None and len(q) > 1:
            for neigh_labels in self.list_sums_extrema[label_i]:
                color_sum = self.colors[neigh_labels[0]][0] + self.colors[neigh_labels[1]][0]
                color_diff = self.colors[neigh_labels[0]][0] - self.colors[neigh_labels[1]][0]
                upper_bound = min(upper_bound, min(2 * q[1] - 4 - color_sum, color_sum))
                lower_bound = max(lower_bound, abs(color_diff))

                if lower_bound > upper_bound:
                    return 0, 1

        return upper_bound, lower_bound    

    def GetColorSumExtremaOld(self, label_i, q=None):
        ## sum_index = self.list_sums.index((label,))
        ## sum_bounds_labels = self.list_sums_extrema[label_i]

        ## We could not do any better...
        upper_bound, lower_bound = math.inf, 0
        for neigh_labels in self.list_sums_extrema[label_i]:
            color_a, color_b = \
                self.colors[neigh_labels[0]][0], \
                self.colors[neigh_labels[1]][0]

            color_sum = color_a + color_b
            if q is None or len(q) == 1:
                upper_bound = min(upper_bound, color_sum)
            elif q is not None and len(q) > 1:
                upper_bound = min(upper_bound, min(2 * q[1] - 4 - color_sum, color_sum))

            lower_bound = max(lower_bound, abs(color_a - color_b))

            if lower_bound > upper_bound:
                return 0, 1
            
        return upper_bound, lower_bound

    def GetColorSumExtremaOld2(self, label_i, q=None):
        ## sum_index = self.list_sums.index((label,))
        sum_bounds_labels = \
            self.list_sums_extrema[label_i]

        upper_bound, lower_bound = 1e6, 0
        for neigh_labels in sum_bounds_labels:
            color_a, color_b = \
                self.colors[neigh_labels[0]][0], \
                self.colors[neigh_labels[1]][0]

            upper_bound = min(upper_bound, color_a + color_b)
            lower_bound = max(lower_bound, abs(color_a - color_b))
            
        return upper_bound, lower_bound

    def GetColorSumExtremaOld3(self, label, q=None):
        ## sum_index = self.list_sums.index((label,))
        sum_bounds_labels = \
            self.list_sums_extrema[
                self.sums_indices[label]
            ]

        upper_bound, lower_bound = 1e6, 0
        for neigh_labels in sum_bounds_labels:
            color_a, color_b = \
                self.colors[neigh_labels[0]][0], \
                self.colors[neigh_labels[1]][0]

            upper_bound = min(upper_bound, color_a + color_b)
            lower_bound = max(lower_bound, abs(color_a - color_b))
            
        return upper_bound, lower_bound

    def GetColorSumExtremaOld4(self, label, q=None):
        ## sum_index = self.list_sums.index((label,))
        sum_index = self.sums_indices[label]
        sum_bounds_labels = self.list_sums_extrema[sum_index]
        sum_bounds_colors = \
            tuple((self.colors[labels[0]][0], self.colors[labels[1]][0]) 
                  for labels in sum_bounds_labels)
        
        ## print("sum_bounds_colors:", sum_bounds_colors)
        
        upper_bounds, lower_bounds = [], []
        for colors in sum_bounds_colors:
            upper_bounds += [colors[0] + colors[1]]
            lower_bounds += [abs(colors[0] - colors[1])]
            
        return min(upper_bounds), max(lower_bounds)

    def IsFactorEmpty(self, factor):
        return sum([len(factor[key]) for key in factor]) == 0        
        
    def GetFactors(self):
        ## These are the 'p' labels, i.e. those that are summed
        sums_labels = [elem[0] for elem in self.list_sums]

        n6j, ntet, nth = \
            len(self.list_6j), len(self.list_tet), len(self.list_th)
        ndn, nth_m1, ndn_m1 = \
            len(self.list_dn), len(self.list_th_m1), len(self.list_dn_m1)

        list_6j_sums, list_tet_sums, list_th_sums = \
            [[] for i in range(n6j)], \
            [[] for i in range(ntet)], \
            [[] for i in range(nth)]

        list_dn_sums, list_th_m1_sums, list_dn_m1_sums = \
            [[] for i in range(ndn)], \
            [[] for i in range(nth_m1)], \
            [[] for i in range(ndn_m1)]

        elists_sums = {'6j': list_6j_sums, 'tet': list_tet_sums, 
                       'th': list_th_sums, 
                       'dn': list_dn_sums, 'th_m1': list_th_m1_sums, 
                       'dn_m1': list_dn_m1_sums}

        for label in sums_labels:
            for name in elists_sums:
                for i, elem_tuple in enumerate(self.elists[name]):
                    if label in elem_tuple:
                        elists_sums[name][i] += [label]

        prefactor = {'6j': [], 'tet': [], 'th': [], 
                     'dn': [], 'th_m1': [], 'dn_m1': []}
        not_prefactor = {'6j': [], 'tet': [], 'th': [], 
                         'dn': [], 'th_m1': [], 'dn_m1': []}
        
        for list_name in elists_sums:
            ## split_name = list_name.split('_')
            ## f_name = split_name[0] 
            for i, labels in enumerate(elists_sums[list_name]):
                if len(labels) == 0:
                    prefactor[list_name] += [self.elists[list_name][i]]
                else:
                    not_prefactor[list_name] += [self.elists[list_name][i]]
        
        self.prefactor, self.not_prefactor = prefactor, not_prefactor
    
    def GetPrefactorEvaluation(self, q=None):
        self.GetFactors()
        evaluation = 1
        
        for list_name in self.prefactor:
            _split_name = list_name.split('_')
            _f_name = _split_name[0]
            N_tuples = len(self.prefactor[list_name])
            first_tuple = 0 if N_tuples == 0 else len(self.prefactor[list_name][0])
            #print(list_name, self.prefactor[list_name], N_tuples, first_tuple)
            
            if N_tuples > 0 and first_tuple > 0:                
                if len(_split_name) > 1 and _split_name[1] == 'm1':
                    _is_m1 = True
                else:
                    _is_m1 = False

                functions = [self.EvaluationFunctions[_f_name]] * N_tuples                    
                swap_evaluations = \
                    self.evaluation_wrapper(functions, self.prefactor[list_name], q=q)
                
                #print(self.from_labels_to_colors(self.prefactor[list_name][0]))
                #print(swap_evaluations)
                   
                ## print(_f_name, _is_m1, swap_evaluations, self.prefactor[_f_name], N_tuples)
                if not _is_m1:
                    evaluation *= reduce(lambda x, y: x * y, swap_evaluations)
                else:
                    evaluation /= reduce(lambda x, y: x * y, swap_evaluations)
                
        return evaluation
    
    def GetNotPrefactorEvaluation(self, q=None):
        self.GetFactors()
        evaluation = 1
        
        ## Same as befor but on self.not_prefactor (!!!)
        for list_name in self.not_prefactor:
            _split_name = list_name.split('_')
            _f_name = _split_name[0]
            N_tuples = len(self.not_prefactor[list_name])
            first_tuple = 0 if N_tuples == 0 else len(self.not_prefactor[list_name][0])
            #print(list_name, self.not_prefactor[list_name], N_tuples, first_tuple)
            
            if N_tuples > 0 and first_tuple > 0:                
                if len(_split_name) > 1 and _split_name[1] == 'm1':
                    _is_m1 = True
                else:
                    _is_m1 = False

                functions = [self.EvaluationFunctions[_f_name]] * N_tuples                    
                swap_evaluations = \
                    self.evaluation_wrapper(functions, self.not_prefactor[list_name], q=q)
                
                #print(self.from_labels_to_colors(self.not_prefactor[list_name][0]))
                #print(swap_evaluations)
                   
                ## print(_f_name, _is_m1, swap_evaluations, self.not_prefactor[_f_name], N_tuples)
                if not _is_m1:
                    evaluation *= reduce(lambda x, y: x * y, swap_evaluations)
                else:
                    evaluation /= reduce(lambda x, y: x * y, swap_evaluations)
                
        return evaluation
        
    def CleanDeltas(self):
        pop_indices, labels_list = [], []
        for i, labels in enumerate(self.list_kdelta):
            if labels[0] == labels[1]:
                pop_indices += [i]
                labels_list += [[labels[0]]]
                
        pop_indices.sort(reverse=True)
        for i in pop_indices:
            self.list_kdelta.pop(i)      
        
    def GetSymbolicExpression(self):
        _expr = sp.core.numbers.One()
        '''
        run across all lists and multiply/divide
        '''
        for labels, boundary_labels in zip(self.elists['sums'], self.elists['sums_extrema']):
            _elem_syms = \
                [FromLabelToStrSym(label) for label in labels]
            
            _bounds_str = [GetMinSymbol(labels[0]), GetMaxSymbol(labels[0])]
            _expr *= self.SymbolsFunctions['sums'](_elem_syms, _bounds_str)
        
        for list_name in self.elists:
            if list_name != 'sums' and list_name != 'sums_extrema':
                _split_name = list_name.split('_')
                _f_name = _split_name[0]
                if len(_split_name) > 1 and _split_name[1] == 'm1':
                    _is_m1 = True
                else:
                    _is_m1 = False

                for labels in self.elists[list_name]:
                    _elem_syms = \
                        [FromLabelToStrSym(label) for label in labels]
                    ## print(labels, _f_name, _is_m1, _elem_syms)

                    if _is_m1:
                        _expr /= self.SymbolsFunctions[_f_name](_elem_syms)
                    else:
                        _expr *= self.SymbolsFunctions[_f_name](_elem_syms)
                    
        return _expr
    
    def Push6j(self, labels):
        if len(labels) != 6:
            raise Exception("There must be 6 values for the 6j symbol")
            
        self.list_6j += [labels]
        
    def PushTet(self, labels):
        if len(labels) != 6:
            raise Exception("There must be 6 values for the tetrahedron symbol")
        
        self.list_tet += [labels]
        
    def PushTh(self, labels):
        if len(labels) != 3:
            raise Exception("There must be 3 values for the theta-net")
        
        self.list_th += [labels]

    def PushDn(self, label):
        if len(label) != 1:
            raise Exception("There must be 1 value for the quantum dimension")
        
        self.list_dn += [label]

    def PushThM1(self, labels):
        if len(labels) != 3:
            raise Exception("There must be 3 values for the theta-net")
        
        self.list_th_m1 += [labels]

    def PushDnM1(self, label):
        if len(label) != 1:
            raise Exception("There must be 1 value for the quantum dimension")
        
        self.list_dn_m1 += [label]
        
    def PushKDelta(self, label):
        if len(label) != 2:
            raise Exception("There must be 2 values for the Kronecker delta")
            
        self.list_kdelta += [label]
        
    def PushSum(self, label):
        if len(label) != 1:
            raise Exception("There must be only one index for the sum")
        
        self.list_sums += [label]

    def SetSumIndices(self):
        for index, label in enumerate(self.list_sums):
            self.sums_indices[label[0]] = index
        
    def PushSumExtrema(self, extrema):
        if len(extrema) != 2:
            raise Exception("There must be two couples of labels")
            
        self.list_sums_extrema += [extrema]
        
    def FindSumsAdjacencyList(self):
        n_sums = len(self.elists['sums'])
        adj_list = []
        for i in range(n_sums):
            adj_list += [[]]
            for j in range(n_sums):
                if self.FindPInLabelsExtrema(self.elists['sums'][j][0], 
                                             self.elists['sums_extrema'][i]):
                    adj_list[-1] += [j]
                    
        return adj_list
        
    def GetLambda(self):
        pass
