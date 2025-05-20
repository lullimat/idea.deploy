__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2023 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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
from idpy.Utils.Combinatorics import GetUniquePermutations, SplitTuplePerm, cycle_list

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
    Implements the tensor product without the full-symmetry assumption -> JSymmetricTensor
    '''
    def __or__(self, b):
        if isinstance(b, SymmetricTensor):
            if self.d != b.d:
                raise Exception("The two fully symmetric tensors must have the same dimensionality")
            
            taylor_indices_0 = TaylorTuples(list(range(self.d)), self.rank)
            taylor_indices_1 = TaylorTuples(list(range(self.d)), b.rank)

            j_c_dict = {}
            for tt0 in taylor_indices_0:
                tt0_tuple = tt0 if isinstance(tt0, tuple) else (tt0,)
                for tt1 in taylor_indices_1:
                    tt1_tuple = tt1 if isinstance(tt1, tuple) else (tt1,)
                    j_c_dict[tt0_tuple + tt1_tuple] = self[tt0] * b[tt1]

            return JSymmetricTensor(c_dict=j_c_dict, d=self.d, rank=self.rank + b.rank, ranks=[self.rank, b.rank])


    '''
    Implements the tensor products between fully-symmetric tensors
    - yields a higher ranking tensor as output with the same dimension
    '''
    def __xor__(self, b):
        if b.d != self.d:
            raise Exception('Dimensionalities of the two SymmetricTensor differ!',
                            self.d, b.d)

        A, B = self, b
        new_rank = A.rank + B.rank
        """
        - I need to check if the symmetric tensor contains scalars or sympy arrays
        - once I know which product function to use, then I need to cycle over all possible
        indices of a fully symmetric tensor of rank 'new_rank' and take the products
        - I need to associate the first self.rank indices to self and the remaining to b
        """
        _shapes = [A.shape, B.shape]
        _shapes_types = [type(A.shape), type(B.shape)]
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
            if A.shape != B.shape:
                raise Exception("Cannot perform the element-wise product")
        elif AllTrue([_ == int for _ in _shapes_types]):
            _product = lambda x, y: x * y
        
        ttuples_new = TaylorTuples(list(range(A.d)), new_rank)
        swap_dict = {}
        for t in ttuples_new:
            t_A, t_B = tuple(t[:A.rank]), tuple(t[A.rank:])
            t_A = t_A if len(t_A) > 1 else t_A[0]
            t_B = t_B if len(t_B) > 1 else t_B[0]
            
            swap_dict[t] = _product(A[t_A], B[t_B])
        
        return SymmetricTensor(d=A.d, rank=new_rank, c_dict=swap_dict)
        
        
    '''
    Implements a full contraction among fully-symmetric tensors
    '''
    def __mul__(self, _b):
        if _b.__class__.__name__ != self.__class__.__name__:
            """
            Here we assume a multiplication by a scalar
            """
            contraction_dict = {key: self.c_dict[key] * _b for key in self.c_dict}
            return SymmetricTensor(c_dict = contraction_dict, d=self.d, rank=self.rank)
        else:
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
                    ttuple_diff_index = ttuple_diff
                    if type(ttuple_diff) != tuple:
                        ttuple_diff = (ttuple_diff,)

                    partial_sum = 0                
                    for ttuple_B in list_ttuples_B:
                        ## print(ttuple_diff, ttuple_B)
                        elems_tuple, count_elems = np.unique(ttuple_B, return_counts=True)
                        """
                        Now we cycle on all possible symmetric realization of the ttuple_B
                        """
                        elems_list = \
                            [(v, c) for v, c in zip(elems_tuple, count_elems)
                            if (v == 0 and c == B.rank) or (v > 0)]
                        
                        ## print("elems_list:", elems_list, partial_sum)
                        
                        for symm_ttuple_B in GetUniquePermutations(elems_list, B.rank):
                            if B.rank == 1:
                                symm_ttuple_B_left = tuple(symm_ttuple_B,)
                                symm_ttuple_B_right = symm_ttuple_B[0]
                            else:
                                symm_ttuple_B_left = tuple(symm_ttuple_B)
                                symm_ttuple_B_right = symm_ttuple_B_left

                            swap_sum = A[ttuple_diff + symm_ttuple_B_left] * B[symm_ttuple_B_right]
                            partial_sum += swap_sum
                            ## print("\t", symm_ttuple_B, swap_sum, partial_sum)
                        ## print()

                    contraction_dict[ttuple_diff_index] = partial_sum
                    ## print()
                    
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
                        if _ == tuple and len(_shapes[_i]):
                            _largest_shape = _shapes[_i]
                elif AllTrue([_ == tuple for _ in _shapes_types]):
                    _product = lambda x, y: sp.matrix_multiply_elementwise(x, y)
                    _symt_out = True
                    _largest_shape=self.shape
                    if self.shape != _b.shape:
                        raise Exception("Cannot perform the element-wise product")
                elif AllTrue([_ == int for _ in _shapes_types]):
                    _product = lambda x, y: x * y

                '''
                Full contraction
                '''
                # print("_largest_shape:", _largest_shape, "_shapes_types:", _shapes_types)
                _contraction = \
                    sp.Matrix([0] * _largest_shape[0]) \
                    if _largest_shape is not None else 0
                
                ##for _tuple in TaylorTuples(list(range(self.d)), self.rank):
                """
                Need to loop over all the tuples of the dict:
                then deconstruct the tuple and generate all the symmetric ones
                sum over the list
                """
                for _tuple in self.c_dict:
                    values, counts = np.unique(_tuple, return_counts=True)
                    perm_elems_list = \
                        [(v, c) for v, c in zip(values, counts)
                        if (v == 0 and c == self.rank) or (v > 0)]
                    
                    for p_tuple in GetUniquePermutations(perm_elems_list, self.rank):
                        ##print(p_tuple, len(p_tuple))
                        p_tuple = tuple(p_tuple) if len(p_tuple) > 1 else p_tuple[0]
                        ##print(self[p_tuple])
                        ##print(_b[p_tuple])
                        _contraction += _product(self[p_tuple], _b[p_tuple])
                        
                    if False:
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
                            rank is the correct procedure: it is not
                            """
                            _Contraction += sp.factorial(self.rank) * _product(self[_tuple], _b[_tuple])

                return (_contraction if not _symt_out else
                        SymmetricTensor(c_dict = {0: _contraction}, d = self.d, rank = 0))

        """
        else:
            _product, _swap_dict = 1, {}
            for _tuple in self.c_dict:
                _swap_dict[_tuple] = _b * self[_tuple]
            return SymmetricTensor(c_dict = _swap_dict, d = self.d, rank = self.rank)
        """

    ## Need to move this function to the JointSymmetricTensor
    def PartialContraction(self, _b, n_indices):
        if _b.__class__.__name__ != self.__class__.__name__:
            raise Exception("the two object must belong to the same class!", self.__class__.__name__)
        else:                    
            if _b.d != self.d:
                raise Exception('Dimensionalities of the two SymmetricTensor differ!',
                                self.d, _b.d)
            
            """
            In case of full contraction of one of the tensors call __mul__ method
            """
            if self.rank == n_indices or _b.rank == n_indices:
                return self.__mul__(_b)

            """
            Manage the remaining case
            """
            if self.rank > n_indices and _b.rank > n_indices:
                rank_diff_self, rank_diff_b = self.rank - n_indices, _b.rank - n_indices
                list_tuples_diff_self = TaylorTuples(list(range(self.d)), rank_diff_self)
                list_tuples_diff_b = TaylorTuples(list(range(self.d)), rank_diff_b)
                list_tuples_contraction = TaylorTuples(list(range(self.d)), n_indices)

                is_rd_self_1 = rank_diff_self == 1
                is_rd_b_1 = rank_diff_b == 1
                is_rcontraction_1 = n_indices == 1

                contraction_dict = {}
                for ttuple_self in list_tuples_diff_self:
                    for ttuple_b in list_tuples_diff_b:
                        tuple_prefix = ttuple_self if not is_rd_self_1 else (ttuple_self, )
                        tuple_postfix = ttuple_b if not is_rd_b_1 else (ttuple_b, )
                        # print(tuple_prefix, tuple_postfix)

                        partial_sum = 0
                        for ttuple_contraction in list_tuples_contraction:
                            ttuple_contraction = ttuple_contraction if not is_rcontraction_1 else (ttuple_contraction, )

                            # print(tuple_prefix, tuple_postfix, ttuple_contraction)

                            tuple_sum_self = tuple_prefix + ttuple_contraction
                            tuple_sum_b = ttuple_contraction + tuple_postfix

                            partial_sum += self[tuple_sum_self] * _b[tuple_sum_b]

                        tuple_result = tuple_prefix + tuple_postfix
                        contraction_dict[tuple_result] = partial_sum

                rank_result = rank_diff_self + rank_diff_b                
                return SymmetricTensor(c_dict = contraction_dict, d=self.d, rank=rank_result)

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

"""
class JSymmetricTensor
- for rank 1 tensors need to pass the argument ranks = [1], without a second entry
"""
class JSymmetricTensor:
    def __init__(self, c_dict = None, list_values = None, list_ttuples = None,
                 d = None, rank = None, ranks = None):
        
        if c_dict is None and list_values is None and list_ttuples is None:
            raise Exception("Missing arguments: either 'c_dict' or 'list_values' and 'list_ttuples'")
        elif c_dict is None and (list_values is None or list_ttuples is None):
            raise Exception("Missing argument: either 'list_values' or 'list_ttuples'")        
        if c_dict is not None and (list_values is not None or list_ttuples is not None):
            raise Exception("Arguments conflict: either 'c_dict' or 'list_values' and 'list_ttuples'")
        if ranks is None or sum(ranks) != rank:
            raise Exception("Missing arguments: 'ranks' needs to be a list of two values adding to 'rank'!")

        self.d, self.rank, self.ranks = d, rank, ranks
        if c_dict is not None:
            self.c_dict = c_dict
        else:
            self.c_dict = dict(zip(list_ttuples, list_values))
            
        # self.shape = self.set_shape()

    def set_shape(self):
        _key_0 = list(self.c_dict)[0]
        _shape = (0 if not hasattr(self.c_dict[_key_0], 'shape') else
                  self.c_dict[_key_0].shape)
        return _shape
        
    def __getitem__(self, _index):
        if isinstance(_index, slice):
            return self.c_dict[_index]
        else:
            if isinstance(_index, tuple):
                _index_0 = list(_index)[:self.ranks[0]]
                _index_1 = list(_index)[self.ranks[0]:]
                _index_0.sort()
                _index_1.sort()
                _index = tuple(_index_0 + _index_1)
                
            return self.c_dict[_index]
        
    def GetFullySymmetric(self):
        taylor_indices = TaylorTuples(list(range(self.d)), self.rank)
        c_dict = {tt: self[tt] for tt in taylor_indices}
        return SymmetricTensor(d=self.d, rank=self.rank, c_dict=c_dict)
        
    ## So complicated...?
    def add__mah(self, b):
        if isinstance(b, JSymmetricTensor):
            ## check the two tensors are of the same kind
            if self.d != b.d or self.rank != b.rank or self.ranks != b.ranks:
                raise Exception("The two tensors are not of the same dimension/rank/partial-ranks")
            add_c_dict = {}
            taylor_indices_0 = TaylorTuples(list(range(self.d)), self.ranks[0])

            if len(self.ranks) > 1:
                
                for tt0 in taylor_indices_0:
                    tt0 = tt0 if isinstance(tt0, tuple) else (tt0,)
                    for tt1 in taylor_indices_1:
                        tt1 = tt1 if isinstance(tt1, tuple) else (tt1,)
                        add_c_dict[tt0 + tt1] = self[tt0 + tt1] + b[tt0 + tt1]
                # print(taylor_indices_0, taylor_indices_1, add_c_dict)
            else:
                for tt0 in taylor_indices_0:
                    add_c_dict[tt0] = self[tt0] + b[tt0]

            return JSymmetricTensor(d=self.d, rank=self.rank, ranks=self.ranks, c_dict=add_c_dict)
        else:
            raise Exception("Can only add Joint-Symmetric Tensors")

    def __add__(self, b):
        if isinstance(b, JSymmetricTensor) or isinstance(b, SymmetricTensor):
            ## check the two tensors are of the same kind
            if self.d != b.d or self.rank != b.rank:
                raise Exception("The two tensors are not of the same dimension/rank")

            add_c_dict = {tt: self[tt] + b[tt] for tt in self.c_dict}
            return JSymmetricTensor(d=self.d, rank=self.rank, ranks=self.ranks, c_dict=add_c_dict)
        else:
            raise Exception("Can only add Joint/Symmetric Tensors")

    def __mul__(self, b):
        if isinstance(b, SymmetricTensor):
            """
            - For each 0-multi-index we can build a SymmetricTensor for the 1-multi-index part
            - at this point the contraction would be given by calling __mul__ between this sub-tensor and b
            """
            taylor_indices_0 = TaylorTuples(list(range(self.d)), self.ranks[0])
            taylor_indices_1 = TaylorTuples(list(range(self.d)), self.ranks[1])
            
            res_c_dict = {}
            for tt0 in taylor_indices_0:
                tt0_tuple = tt0 if isinstance(tt0, tuple) else (tt0,)
                ## building sub tensor
                c_dict_swap = {}
                for tt1 in taylor_indices_1:
                    tt1_tuple = tt1 if isinstance(tt1, tuple) else (tt1,)
                    c_dict_swap[tt1] = self[tt0_tuple + tt1_tuple]
                sub_tensor = SymmetricTensor(c_dict = c_dict_swap, d = self.d, rank = self.ranks[1])
                # print(sub_tensor.c_dict)

                contracion = sub_tensor * b
                ## the contraction might be a scalar
                if isinstance(contracion, SymmetricTensor):
                    for tt_c in contracion.c_dict:
                        tt_c_tuple = tt_c if isinstance(tt_c, tuple) else (tt_c,)
                        res_c_dict[tt0_tuple + tt_c_tuple] = contracion[tt_c]
                else:
                    res_c_dict[tt0] = contracion

            first_elem_index = list(res_c_dict.keys())[0]
            new_full_rank = len(first_elem_index) if isinstance(first_elem_index, tuple) else 1
            new_1_rank = new_full_rank - self.ranks[0]

            if new_1_rank > 0:
                return JSymmetricTensor(res_c_dict, d=self.d, rank=new_full_rank, ranks=[self.ranks[0], new_1_rank])
            else:
                return SymmetricTensor(res_c_dict, d=self.d, rank=new_full_rank)
            
        elif not isinstance(b, SymmetricTensor) and not isinstance(b, JSymmetricTensor):
            ## In this case we assume multiplication by a scalar
            mul_c_dict = {tt: self[tt] * b for tt in self.c_dict}
            return JSymmetricTensor(d=self.d, rank=self.rank, ranks=self.ranks, c_dict=mul_c_dict)
        
    def __sub__(self, b):
        if isinstance(b, JSymmetricTensor) or isinstance(b, SymmetricTensor):
            if self.d != b.d or self.rank != b.rank:
                raise Exception("Mismathcing dimension/rank!!!")
            
            sub_c_dict = {tt: self[tt] - b[tt] for tt in self.c_dict}
            return JSymmetricTensor(d=self.d, rank=self.rank, ranks=self.ranks, c_dict=sub_c_dict)
        else:
            raise Exception("Can only subtract Joint/Symmetric Tensors")

def GetAJSymmetricTensor(d, rank, ranks, root_sym = 'A'):
    taylor_indices_0 = TaylorTuples(list(range(d)), ranks[0])
    taylor_indices_1 = TaylorTuples(list(range(d)), ranks[1])
    swap_dict = {}
    for tt0 in taylor_indices_0:
        tt0 = tt0 if isinstance(tt0, tuple) else (tt0,)
        for tt1 in taylor_indices_1:
            tt1 = tt1 if isinstance(tt1, tuple) else (tt1,)
            full_index = tt0 + tt1            
            lower_indices = reduce(lambda x, y: str(x) + ',' + str(y), full_index)
            swap_dict[full_index] = sp.Symbol(root_sym + "_{" + lower_indices + "}")

    return JSymmetricTensor(c_dict = swap_dict, d = d, rank = rank, ranks=ranks)

def GetASymmetricTensor(dim, order, root_sym = 'A'):
    _taylor_indices = TaylorTuples(list(range(dim)), order)
    _swap_dict = {}
    for _i, _index_tuple in enumerate(_taylor_indices):
        _lower_indices = reduce(lambda x, y: str(x) + ',' + str(y), _index_tuple)
        _swap_dict[_index_tuple] = sp.Symbol(root_sym + "_{" + _lower_indices + "}")
    return SymmetricTensor(c_dict = _swap_dict, d = dim, rank = order)

def GetFullyIsotropicTensor(d=None, rank=None):
    if rank % 2:
        raise Exception("rank must be even!")

    ttuples = TaylorTuples(list(range(d)), 2)
    values = [1 if t[0] == t[1] else 0 for t in ttuples]
    lead_kr_2 = \
        SymmetricTensor(d=d, rank=2, list_values=values, list_ttuples=ttuples)
    
    if rank == 2:
        return lead_kr_2
        
    if rank > 2:        
        root_index_list = list(range(rank))
        index_lists = [root_index_list]
        last_perm = root_index_list
        
        for i in range(rank - 2):
            last_perm = cycle_list(last_perm, 1)
            index_lists += [last_perm]

        follow_kr_rankm2 = GetFullyIsotropicTensor(d=d, rank=rank-2)
        
        tuples_map = \
            lambda in_tuple: \
            map(lambda perm: \
                SplitTuplePerm(in_tuple=in_tuple, 
                               perm=perm, 
                               split_point=2), 
                index_lists)

        summands = lambda in_tuple:\
            map(lambda out_tuple: \
                lead_kr_2[out_tuple[0]] * follow_kr_rankm2[out_tuple[1]], 
                tuples_map(in_tuple))

        sum_results = \
            lambda in_tuple: reduce(lambda x, y: x + y, summands(in_tuple))        

        components = TaylorTuples(list(range(d)), rank)
        swap_dict = {}
        for full_tuple in components:
            ## print(full_tuple)
            swap_dict[full_tuple] = sum_results(full_tuple)
                    
        return SymmetricTensor(d=d, rank=rank, c_dict=swap_dict)

def GetGeneralizedKroneckerDelta(d=None, rank=None):
    # if rank % 2:
    #     raise Exception("rank must be even!")

    if rank == 2:
        return GetFullyIsotropicTensor(d=d, rank=rank)
    elif rank == 1 or rank > 2:
        ttuples = TaylorTuples(list(range(d)), rank)
        values = []
        
        for t in ttuples:
            v, c = np.unique(t, return_counts=True)
            values += [1 if c[0] == rank else 0]
        
        gen_kr = \
            SymmetricTensor(d=d, rank=rank, list_values=values, list_ttuples=ttuples)

        return gen_kr 
        
def GetPiTensor(d=None, half_rank=None):
    # Building the half_rank = 1 case
    ttuples = TaylorTuples(list(range(d)), 2)
    values = [1 if t[0] == t[1] else 0 for t in ttuples]
    lead_kr_2 = SymmetricTensor(d=d, rank=2, list_values=values, list_ttuples=ttuples)

    # The final result needs to be a JSymmetricTensor - at least from half_rank>=2
    if half_rank == 1:
        return lead_kr_2
    
    if half_rank > 1:
        root_index_list = list(range(2 * half_rank))
        index_lists = [root_index_list]
        last_perm = root_index_list

        for i in range(half_rank - 1):
            last_perm = cycle_list(last_perm, half_rank)
            index_lists += [last_perm]

        # tuple_map = lambda in_tuple: map(lambda perm: SplitTuplePerm(in_tuple=in_tuple, perm=perm, split_point=2), index_lists_1)

        follow_Pi_hrankm1 = GetPiTensor(d, half_rank=half_rank-1)

        tuples_map = lambda in_tuple: map(lambda perm: SplitTuplePerm(in_tuple=in_tuple, perm=perm, split_point=half_rank), index_lists)
        summands = lambda in_tuple: map(lambda out_tuple: lead_kr_2[(out_tuple[0][0], out_tuple[1][0])] * follow_Pi_hrankm1[out_tuple[0][1:] + out_tuple[1][1:]], tuples_map(in_tuple))
        sum_results = lambda in_tuple: reduce(lambda x, y: x + y, summands(in_tuple))

        taylor_indices_0 = TaylorTuples(list(range(d)), half_rank)
        taylor_indices_1 = TaylorTuples(list(range(d)), half_rank)

        swap_dict = {}
        for tt0 in taylor_indices_0:
            for tt1 in taylor_indices_1:
                swap_dict[tt0 + tt1] = sum_results(tt0 + tt1) / half_rank

        return JSymmetricTensor(d=d, rank=2*half_rank, ranks=[half_rank, half_rank], c_dict=swap_dict)
