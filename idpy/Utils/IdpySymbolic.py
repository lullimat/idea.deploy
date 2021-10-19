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

'''
Provides some functions and classes for symbolic manipulations
'''

import sympy as sp
from functools import reduce

'''
Function MergeTuples:

The term 'tuple' needs to be understood as mathematical jargon and not as the python class.
Here 'tuples' are represented by lists
'''
def MergeTuples(_t_in, _t_list):
    _t_list_swap = []
    for _t in _t_list:
        _t_list_swap += [_t_in + _t]
    return _t_list_swap


'''
Function TaylorTuples:

It returns all the possible 'independent' tuples to be used as indices of the '_n'-th
derivative provided the list of the coordinates in d dimensions '_x_list'
'''
def TaylorTuples(_x_list, _n):
    if _n == 0:
        return ()
    
    if _n == 1:
        return _x_list
        
    if _n == 2:        
        _swap_x_tuples = [(_,) for _ in _x_list]
        _swap_tuple = []
        
        for _x_i in range(len(_x_list)):
            _swap_tuple += MergeTuples((_x_list[_x_i],), _swap_x_tuples[_x_i:])
        return _swap_tuple
    
    if _n > 2:
        _swap_tuple = []
        
        for _x_i in range(len(_x_list)):
            _swap_tuple += MergeTuples((_x_list[_x_i],), TaylorTuples(_x_list[_x_i:], _n - 1))
        return _swap_tuple

'''
Function GetDerivativeTuple:

_f: sympy object or function
_sym_list: list of symbols with respect to which to perform the derivatives
_tuple: list of indices indicating the symbols
'''
def GetDerivativeTuple(_f, _sym_list, _tuple):
    _swap_res = _f
    for _ in _tuple:
        _swap_res = _swap_res.diff(_sym_list[_])
    return _swap_res


'''
Function GetTaylorDerivatives:

returns the list of all the independent derivatives of order '_n'
for the function '_f' given the coordinates '_sym_list'

_f: sympy object or function
_sym_list: list of symbols with respect to which to perform the derivatives
_n: order of the derivatives
'''
def GetTaylorDerivatives(_f, _sym_list, _n):
    _taylor_tuples = TaylorTuples(list(range(len(_sym_list))), _n)
    _swap_res = []
    for _tuple in _taylor_tuples:
        if type(_tuple) != tuple:
            _swap_res += [GetDerivativeTuple(_f, _sym_list, [_tuple])]
        else:
            _swap_res += [GetDerivativeTuple(_f, _sym_list, _tuple)]
          
    if len(_swap_res) == 0:
        _swap_res += [_f]
    
    return _swap_res, _taylor_tuples


'''
Function GetTaylorDerivativesDict:

returns a dictionary of all the independent derivatives of order '_n'
for the function '_f' given the coordinates '_sym_list'.
The dictionary is indexed by the list provided by TaylorTuples

_f: sympy object or function
_sym_list: list of symbols with respect to which to perform the derivatives
_n: order of the derivatives
'''
def GetTaylorDerivativesDict(_f, _sym_list, _n):
    _swap_res, _taylor_tuples = GetTaylorDerivatives(_f, _sym_list, _n)
    _swap_dict = {}
    
    if len(_taylor_tuples):
        _k = 0
        for _tuple in _taylor_tuples:
            _swap_dict[_tuple] = _swap_res[_k]
            _k += 1
        return _swap_dict
    else:
        return {'_': _swap_res[0]}


'''
class SymmetricTensor:

Provides a class for managing symmetric tensors.
So far, it has been built for managing n-th order partial derivatives

Still under active development
'''
class SymmetricTensor:
    '''
    This class assumes that only the indices i <= j <= k ... are passed'''
    def __init__(self, c_dict = None, d = None, rank = None):
        self.d, self.rank = d, rank
        self.c_dict = c_dict
        
    def __getitem__(self, _index):
        if isinstance(_index, slice):
            return self.c_dict[_index]
        else:
            if type(_index) == tuple:
                _index = list(_index)
                _index.sort()
                _index = tuple(_index)
                
            return self.c_dict[_index]
        
    def __mul__(self, _b):
        pass
