__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2021 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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
from functools import reduce

from idpy.Utils.IdpySymbolic import GetTaylorDerivativesDict, SymmetricTensor

class HermiteGen:
    def __init__(self, sym_x = sp.Symbol('x')):
        self.sym_x = sym_x
        self.sym = sp.exp(- self.sym_x ** 2 / 2) / sp.sqrt(2 * sp.pi)
        
class Hermite:
    def __init__(self, d = 1, root_sym = 'x'):
        self.d, self.sym_list, self.gen_list, self.gen_f_list = d, [], [], []
        for _d_i in range(self.d):
            self.sym_list += [sp.Symbol(root_sym + '_' + str(_d_i))]
            self.gen_list += [HermiteGen(sym_x = self.sym_list[-1])]
            self.gen_f_list += [self.gen_list[-1].sym]
            
        self.gen_f = reduce(lambda x, y: x * y, self.gen_f_list)
        
    def GetH(self, n = None):
        if n == 0:
            return 1
        else:
            _der_dict = GetTaylorDerivativesDict(self.gen_f, self.sym_list, n)
            _swap_dict = {}
            for _tuple in _der_dict:
                _swap_dict[_tuple] = sp.simplify(((-1) ** n) * _der_dict[_tuple] / self.gen_f)
            return SymmetricTensor(c_dict = _swap_dict, d = self.d, rank = n)


def HermiteWProd(A, B, Ws):
    if len(A) != len(B) or len(A) != len(Ws) or len(B) != len(Ws):
        raise Exception("Parameters 'A', 'B' and 'Ws' must all have the same length!")
    _prod = 0
    for _i in range(len(A)):
        _prod += A[_i] * B[_i] * Ws[_i]
    return _prod    
