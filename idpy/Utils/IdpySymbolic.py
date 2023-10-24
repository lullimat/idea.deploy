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
import numpy as np
from functools import reduce
from idpy.Utils.Geometry import FlipVector, IsSameVector
from idpy.Utils.Statements import AllTrue
from idpy.Utils.Combinatorics import GetUniquePermutations

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
    def __init__(self, c_dict = None, list_values = None, list_ttuples = None,
                 d = None, rank = None):

        if c_dict is None and list_values is None and list_ttuples is None:
            raise Exception("Missing arguments: either 'c_dict' or 'list_values' and 'list_ttuples'")
        elif c_dict is None and (list_values is None or list_ttuples is None):
            raise Exception("Missing argument: either 'list_values' or 'list_ttuples'")        
        if c_dict is not None and (list_values is not None or list_ttuples is not None):
            raise Exception("Arguments conflict: either 'c_dict' or 'list_values' and 'list_ttuples'")
        
        self.d, self.rank = d, rank
        if c_dict is not None:
            self.c_dict = c_dict
        else:
            self.c_dict = dict(zip(list_ttuples, list_values))
            
        self.shape = self.set_shape()

    def set_shape(self):
        _key_0 = list(self.c_dict)[0]
        _shape = (0 if not hasattr(self.c_dict[_key_0], 'shape') else
                  self.c_dict[_key_0].shape)
        return _shape
        
    def __getitem__(self, _index):
        if isinstance(_index, slice):
            return self.c_dict[_index]
        else:
            if type(_index) == tuple:
                _index = list(_index)
                _index.sort()
                _index = tuple(_index)
                
            return self.c_dict[_index]

    '''
    Implements a full contraction among fully-symmetric tensors
    '''
    def __mul__(self, _b):
        if _b.d != self.d:
            raise Exception('Dimensionalities of the two SymmetricTensor differ!',
                            self.d, _b.d)

        if _b.rank != self.rank:
            """
            Do I need to manage the case where each component of the symmetric tensor is still
            a tenorial quantity, an object of the class sympy.Matrix ?
            """
            if False:
                raise Exception('Ranks of the two SymmetricTensor differ!',
                                self.rank, _b.rank)
            
            if self.rank > _b.rank:
                A, B = self, _b
            else:
                A, B = _b, self

            rank_diff = A.rank - B.rank
            list_ttuples_diff = TaylorTuples(list(range(self.d)), rank_diff)
            list_ttuples_B = TaylorTuples(list(range(self.d)), B.rank)

            """
            I need to sum over all tuples, including the symmetric ones
            """
            contraction_dict = {}
            for ttuple_diff in list_ttuples_diff:

                partial_sum = 0
                for ttuple_B in list_ttuples_B:
                    print(ttuple_diff, ttuple_B)
                    elems_tuple, count_elems = np.unique(ttuple_B, return_counts=True)
                    """
                    Now we cycle on all possible symmetric realization of the ttuple_B
                    """
                    elems_list = list(zip(elems_tuple, count_elems))
                    elems_list = elems_list[1:] if len(elems_list) > 1 else elems_list
                    print("elems_list:", elems_list)
                    """
                    The solution for the equal rank case below looks less readable
                    But it seems more elegant
                    """

                    for symm_ttuple_B in GetUniquePermutations(elems_list, B.rank):
                        symm_ttuple_B = tuple(symm_ttuple_B)
                        partial_sum += A[ttuple_diff + symm_ttuple_B] * B[symm_ttuple_B]
                        print("\t", symm_ttuple_B, partial_sum)
                    print()

                contraction_dict[ttuple_diff] = partial_sum
                print()
            return SymmetricTensor(c_dict = contraction_dict, d=self.d, rank=rank_diff)

        '''
        This routine is written in order to handle the contraction of objects from the class
        SymmetricTensor such that each component can be an object of the sympy class Matrix
        This is used to contain all the values of the Hermite polynomials associated to different
        stencil vectors
        '''
        if _b.rank == self.rank:
            _largest_shape = 0
            _shapes = [self.shape, _b.shape]
            _shapes_types = [type(self.shape), type(_b.shape)]
            _largest_shape = None

            _product, _symt_out = None, False
            if int in _shapes_types and tuple in _shapes_types:
                _product = lambda x, y: x * y
                for _i, _ in enumerate(_shapes_types): 
                    if _ == tuple:
                        _largest_shape = _shapes[_i]
            elif AllTrue([_ == tuple for _ in _shapes_types]):
                _product = lambda x, y: sp.matrix_multiply_elementwise(x, y)
                _symt_out = True
                if self.shape != _b.shape:
                    raise Exception("Cannot perform the element-wise product")
            '''
            Full contraction
            '''
            _contraction = \
                sp.Matrix([0] * _largest_shape[0]) \
                if _largest_shape is not None else 0
            
            ##for _tuple in TaylorTuples(list(range(self.d)), self.rank):
            for _tuple in self.c_dict:
                _is_symmetric_tuple = True
                if type(_tuple) == tuple and len(_tuple):
                    _flip_tuple = FlipVector(_tuple)
                    _is_symmetric_tuple = IsSameVector(_tuple, _flip_tuple)
                '''
                need to check the shapes in case of sympy matrices, 
                or if one of the two is a a scalar and the apply the elemntwise product
                even though I do not need it for now...
                '''
                if _is_symmetric_tuple:
                    _contraction += _product(self[_tuple], _b[_tuple])
                else:
                    """
                    Need to double check whether multiplying by the factorial of the
                    rank is the correct procedure
                    """
                    _contraction += sp.factorial(self.rank) * _product(self[_tuple], _b[_tuple])

            return (_contraction if not _symt_out else
                    SymmetricTensor(c_dict = {0: _contraction}, d = self.d, rank = 0))

        """
        else:
            _product, _swap_dict = 1, {}
            for _tuple in self.c_dict:
                _swap_dict[_tuple] = _b * self[_tuple]
            return SymmetricTensor(c_dict = _swap_dict, d = self.d, rank = self.rank)
        """

    def __add__(self, _b):
        if _b.__class__.__name__ != self.__class__.__name__:
            raise Exception('Summation is only defined between SymmetricTensor(s)')
        
        if _b.d != self.d:
            raise Exception('Dimensionalities of the two SymmetricTensor differ!',
                            self.d, _b.d)
        if _b.rank != self.rank:
            raise Exception('Ranks of the two SymmetricTensor differ!',
                            self.rank, _b.rank)

        _largest_shape = 0
        _shapes = [self.shape, _b.shape]
        _shapes_types = [type(self.shape), type(_b.shape)]
        _largest_shape = None

        if int in _shapes_types and tuple in _shapes_types:
            _product = lambda x, y: x * y
            for _i, _ in enumerate(_shapes_types): 
                if _ == tuple:
                    _largest_shape = _shapes[_i]
        elif AllTrue([_ == tuple for _ in _shapes_types]):
            if self.shape != _b.shape:
                raise Exception("Cannot perform the element-wise product")
        
        '''
        Summation
        '''
        _sum_dict = {}
        for _key in self.c_dict:
            _sum_dict[_key] = self[_key] + _b[_key]

        return SymmetricTensor(c_dict = _sum_dict, d = self.d, rank = self.rank)

    def __sub__(self, _b):
        if _b.__class__.__name__ != self.__class__.__name__:
            raise Exception('Summation is only defined between SymmetricTensor(s)')
        
        if _b.d != self.d:
            raise Exception('Dimensionalities of the two SymmetricTensor differ!',
                            self.d, _b.d)
        if _b.rank != self.rank:
            raise Exception('Ranks of the two SymmetricTensor differ!',
                            self.rank, _b.rank)

        _largest_shape = 0
        _shapes = [self.shape, _b.shape]
        _shapes_types = [type(self.shape), type(_b.shape)]
        _largest_shape = None

        if int in _shapes_types and tuple in _shapes_types:
            _product = lambda x, y: x * y
            for _i, _ in enumerate(_shapes_types): 
                if _ == tuple:
                    _largest_shape = _shapes[_i]
        elif AllTrue([_ == tuple for _ in _shapes_types]):
            if self.shape != _b.shape:
                raise Exception("Cannot perform the element-wise product")
        
        '''
        Subtraction
        '''
        _sum_dict = {}
        for _key in self.c_dict:
            _sum_dict[_key] = self[_key] - _b[_key]

        return SymmetricTensor(c_dict = _sum_dict, d = self.d, rank = self.rank)
    
def GetASymmetricTensor(dim, order, root_sym = 'A'):
    _taylor_indices = TaylorTuples(list(range(dim)), order)
    _swap_dict = {}
    for _i, _index_tuple in enumerate(_taylor_indices):
        _lower_indices = reduce(lambda x, y: str(x) + str(y), _index_tuple)
        _swap_dict[_index_tuple] = sp.Symbol(root_sym + "_" + _lower_indices)
    return SymmetricTensor(c_dict = _swap_dict, d = dim, rank = order)
