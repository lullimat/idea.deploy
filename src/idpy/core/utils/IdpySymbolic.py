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

from idpy.core.utils.Geometry import FlipVector, IsSameVector
from idpy.core.utils.Statements import AllTrue
from idpy.core.utils.Combinatorics import GetUniquePermutations, SplitTuplePerm, cycle_list


def _is_device_array(x):
    """IdpyMemory arrays expose a non-None .lang tag."""
    return getattr(x, 'lang', None) is not None


def _is_array_like(x):
    return isinstance(x, np.ndarray) or _is_device_array(x)

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
    
#########################################

def GetTaylorExpansion(_f, _vars, _delta, _order):
    _expr = 0
    for _i in range(_order + 1):
        for _der in GetTaylorDerivatives(_f, _vars, _i)[0]:
            _expr += ((_delta ** _i) / sp.factorial(_i)) * _der
        
    return _expr

def GetDerivativeCut(_expr, _index, _max_order):
    '''
    Getting the highest power of _index
    '''
    _hp = sp.LM(_expr, _index).exp
    for _i in range(_max_order + 1, _hp + 1):
        _expr = _expr.subs((_index ** _i), 0)
        
    return _expr

#########################################

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
                 d = None, rank = None, dtype=np.float64):

        if c_dict is None and list_values is None and list_ttuples is None:
            raise Exception("Missing arguments: either 'c_dict' or 'list_values' and 'list_ttuples'")
        elif c_dict is None and (list_values is None or list_ttuples is None):
            raise Exception("Missing argument: either 'list_values' or 'list_ttuples'")        
        if c_dict is not None and (list_values is not None or list_ttuples is not None):
            raise Exception("Arguments conflict: either 'c_dict' or 'list_values' and 'list_ttuples'")
        
        self.dtype = dtype
        self.d, self.rank = d, rank
        if c_dict is not None:
            self.c_dict = c_dict
        else:
            self.c_dict = dict(zip(list_ttuples, list_values))

        # Array-backed fields: numpy ndarray and/or Idpy device arrays (hasattr lang)
        self.has_np_arrays = False
        _has_shaped = False
        for key in self.c_dict:
            _val = self.c_dict[key]
            if _is_array_like(_val):
                self.has_np_arrays = True
                _has_shaped = True
                break
            if isinstance(_val, sp.MatrixBase):
                _has_shaped = True
                break
        if _has_shaped:
            self.shape = self.set_shape()
            for key in self.c_dict:
                if getattr(self.c_dict[key], 'shape', None) != self.shape:
                    raise ValueError(
                        "the shapes of the array objects in the c_dict are not the same"
                    )
        else:
            self.shape = 0

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

    def __setitem__(self, _index, value):
        if isinstance(_index, tuple):
            _index = list(_index)
            if hasattr(self, "ranks"):  # JSymmetricTensor
                _i0 = _index[:self.ranks[0]]
                _i1 = _index[self.ranks[0]:]
                _i0.sort(); _i1.sort()
                _index = tuple(_i0 + _i1)
            else:  # SymmetricTensor
                _index.sort()
                _index = tuple(_index)
        self.c_dict[_index] = value

    def __repr__(self):
        header = f"SymmetricTensor(d={self.d}, rank={self.rank})"
        lines = [header, "c_dict:"]
        for key in sorted(self.c_dict.keys(), key=lambda k: (0 if isinstance(k, int) else 1, repr(k))):
            lines.append(f"  {key}: {self.c_dict[key]!r}")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("SymmetricTensor(...)")
            return
        p.text(repr(self))
        
    '''
    Implements the tensor product without the full-symmetry assumption -> JSymmetricTensor
    '''
    def __or__(self, b):
        if not isinstance(b, SymmetricTensor):
            return NotImplemented
        if self.d != b.d:
            raise Exception("The two fully symmetric tensors must have the same dimensionality")

        def _index_tuples(rank, d):
            if rank == 0:
                return [()]
            raw = TaylorTuples(list(range(d)), rank)
            return [t if isinstance(t, tuple) else (t,) for t in raw]

        def _canonical_index(tt, rank):
            if rank == 0:
                return 0
            if rank == 1:
                return tt[0] if isinstance(tt, tuple) else tt
            return tt if isinstance(tt, tuple) else (tt,)

        def _component(tensor, tt):
            return tensor.c_dict[_canonical_index(tt, tensor.rank)]

        def _joint_key(tt0, tt1, r0, r1):
            if r0 == 0 and r1 == 0:
                return 0
            if r0 == 0:
                return _canonical_index(tt1, r1)
            if r1 == 0:
                return _canonical_index(tt0, r0)
            t0 = tt0 if isinstance(tt0, tuple) else (tt0,)
            t1 = tt1 if isinstance(tt1, tuple) else (tt1,)
            return t0 + t1

        j_c_dict = {}
        for tt0 in _index_tuples(self.rank, self.d):
            for tt1 in _index_tuples(b.rank, self.d):
                key = _joint_key(tt0, tt1, self.rank, b.rank)
                j_c_dict[key] = _component(self, tt0) * _component(b, tt1)

        # if self.rank == 0 and b.rank > 0:
        #     ranks = [b.rank, 0]
        # elif b.rank == 0:
        #     ranks = [self.rank, 0]
        # else:
        ranks = [self.rank, b.rank]

        if self.rank == 0 and b.rank == 0:
            return SymmetricTensor(d=self.d, rank=0, c_dict={0: self[0] * b[0]})

        return JSymmetricTensor(
            c_dict=j_c_dict,
            d=self.d,
            rank=self.rank + b.rank,
            ranks=ranks,
        )

        # taylor_indices_0 = TaylorTuples(list(range(self.d)), self.rank)
        # taylor_indices_1 = TaylorTuples(list(range(self.d)), b.rank)

        # j_c_dict = {}
        # for tt0 in taylor_indices_0:
        #     tt0_tuple = tt0 if isinstance(tt0, tuple) else (tt0,)
        #     for tt1 in taylor_indices_1:
        #         tt1_tuple = tt1 if isinstance(tt1, tuple) else (tt1,)
        #         j_c_dict[tt0_tuple + tt1_tuple] = self[tt0] * b[tt1]

        # return JSymmetricTensor(c_dict=j_c_dict, d=self.d, rank=self.rank + b.rank, ranks=[self.rank, b.rank])


    '''
    Implements the tensor products between fully-symmetric tensors
    - yields a higher ranking tensor as output with the same dimension
    '''
    def __xor__(self, b):
        if b.d != self.d:
            raise Exception(
                'Dimensionalities of the two SymmetricTensor differ!',
                self.d, b.d,
            )

        A, B = self, b
        new_rank = A.rank + B.rank

        # --- choose elementwise product by backend ---
        if A.has_np_arrays or B.has_np_arrays:
            if not (A.has_np_arrays and B.has_np_arrays):
                raise ValueError(
                    "SymmetricTensor.__xor__: both operands must be "
                    "np.ndarray-backed (or both symbolic)"
                )
            if A.shape != B.shape:
                raise ValueError(
                    f"SymmetricTensor.__xor__: shape mismatch {A.shape} vs {B.shape}"
                )
            _product = lambda x, y: x * y
            out_dtype = np.result_type(A.dtype, B.dtype)

        else:
            _shapes_types = [type(A.shape), type(B.shape)]
            if int in _shapes_types and tuple in _shapes_types:
                # mixed scalar / sympy-array shape bookkeeping (legacy)
                _product = lambda x, y: x * y
                out_dtype = A.dtype
            elif AllTrue([_ == tuple for _ in _shapes_types]):
                # both sympy-matrix-like (shape is a tuple, not np arrays)
                if A.shape != B.shape:
                    raise Exception("Cannot perform the element-wise product")
                _product = lambda x, y: sp.matrix_multiply_elementwise(x, y)
                out_dtype = A.dtype
            elif AllTrue([_ == int for _ in _shapes_types]):
                # pure scalars
                _product = lambda x, y: x * y
                out_dtype = A.dtype
            else:
                raise TypeError(
                    f"SymmetricTensor.__xor__: unsupported shape types {_shapes_types}"
                )

        if _product is None:
            raise RuntimeError("SymmetricTensor.__xor__: no product rule selected")

        ttuples_new = TaylorTuples(list(range(A.d)), new_rank)
        swap_dict = {}
        for t in ttuples_new:
            t_A, t_B = tuple(t[:A.rank]), tuple(t[A.rank:])
            t_A = t_A if len(t_A) > 1 else t_A[0]
            t_B = t_B if len(t_B) > 1 else t_B[0]
            swap_dict[t] = _product(A[t_A], B[t_B])

        return SymmetricTensor(
            d=A.d, rank=new_rank, c_dict=swap_dict, dtype=out_dtype
        )

    # def __xor__(self, b):
    #     if b.d != self.d:
    #         raise Exception('Dimensionalities of the two SymmetricTensor differ!',
    #                         self.d, b.d)

    #     A, B = self, b
    #     new_rank = A.rank + B.rank
    #     """
    #     - I need to check if the symmetric tensor contains scalars or sympy arrays
    #     - once I know which product function to use, then I need to cycle over all possible
    #     indices of a fully symmetric tensor of rank 'new_rank' and take the products
    #     - I need to associate the first self.rank indices to self and the remaining to b
    #     """
    #     _shapes = [A.shape, B.shape]
    #     _shapes_types = [type(A.shape), type(B.shape)]
    #     _largest_shape = None

    #     _product, _symt_out = None, False
    #     if int in _shapes_types and tuple in _shapes_types:
    #         _product = lambda x, y: x * y
    #         for _i, _ in enumerate(_shapes_types): 
    #             if _ == tuple:
    #                 _largest_shape = _shapes[_i]
    #     elif AllTrue([_ == tuple for _ in _shapes_types]):
    #         _product = lambda x, y: sp.matrix_multiply_elementwise(x, y)
    #         _symt_out = True
    #         if A.shape != B.shape:
    #             raise Exception("Cannot perform the element-wise product")
    #     elif AllTrue([_ == int for _ in _shapes_types]):
    #         _product = lambda x, y: x * y
        
    #     ttuples_new = TaylorTuples(list(range(A.d)), new_rank)
    #     swap_dict = {}
    #     for t in ttuples_new:
    #         t_A, t_B = tuple(t[:A.rank]), tuple(t[A.rank:])
    #         t_A = t_A if len(t_A) > 1 else t_A[0]
    #         t_B = t_B if len(t_B) > 1 else t_B[0]
            
    #         swap_dict[t] = _product(A[t_A], B[t_B])
        
    #     return SymmetricTensor(d=A.d, rank=new_rank, c_dict=swap_dict)
        
        
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

            # rank-0 × rank-0: single scalar in c_dict[0]
            if self.rank == 0 and _b.rank == 0:
                val = self[0] * _b[0]
                return val

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

# Helper returning a zero SymmetricTensor
def ZeroSymmetricTensor(d, rank, shape=None, dtype=np.float64):
    if rank == 0:
        if shape is None:
            return SymmetricTensor(d=d, rank=0, c_dict={0: 0})
        else:
            return SymmetricTensor(d=d, rank=0, c_dict={0: np.zeros(shape, dtype=dtype)})
    elif rank > 0:
        c_dict = {}
        for tt in TaylorTuples(list(range(d)), rank):
            if shape is not None:
                c_dict[tt] = np.zeros(shape, dtype=dtype)
            else:
                c_dict[tt] = 0
        return SymmetricTensor(d=d, rank=rank, c_dict=c_dict)
    else:
        raise ValueError("rank must be non-negative")

"""
class JSymmetricTensor
- for rank 1 tensors need to pass the argument ranks = [1], without a second entry
"""
class _ConvolutionView:
    def __init__(self, tensor, *, reverse_shift=False, periodic=True):
        self.tensor = tensor
        self._reverse_shift = reverse_shift
        self.periodic = periodic

    @property
    def H(self):
        return _ConvolutionView(
            self.tensor, reverse_shift=not self._reverse_shift, periodic=self.periodic
        )

    @property
    def bnd(self):
        return _ConvolutionView(self.tensor, reverse_shift=self._reverse_shift, periodic=False)

    @property
    def reverse_shift(self):
        return _ConvolutionView(self.tensor, reverse_shift=True, periodic=self.periodic)

    def __matmul__(self, b):
        return self.tensor._matmul_impl(
            b, reverse_shift=self._reverse_shift, periodic=self.periodic
        )


class JSymmetricTensor:
    def __init__(self, c_dict = None, list_values = None, list_ttuples = None,
                 d = None, rank = None, ranks = None, dtype=np.float64):
        
        if c_dict is None and list_values is None and list_ttuples is None:
            raise Exception("Missing arguments: either 'c_dict' or 'list_values' and 'list_ttuples'")
        elif c_dict is None and (list_values is None or list_ttuples is None):
            raise Exception("Missing argument: either 'list_values' or 'list_ttuples'")        
        if c_dict is not None and (list_values is not None or list_ttuples is not None):
            raise Exception("Arguments conflict: either 'c_dict' or 'list_values' and 'list_ttuples'")
        if ranks is None or sum(ranks) != rank:
            raise Exception("Missing arguments: 'ranks' needs to be a list of two values adding to 'rank'!")
        self.dtype = dtype

        self.d, self.rank, self.ranks = d, rank, ranks
        if c_dict is not None:
            self.c_dict = c_dict
        else:
            self.c_dict = dict(zip(list_ttuples, list_values))

        ## Need to check if the c_dict contains array-backed objects
        ## and gate on whether they have the same shape
        self.has_np_arrays = False
        _has_shaped = False
        for key in self.c_dict:
            _val = self.c_dict[key]
            if _is_array_like(_val):
                self.has_np_arrays = True
                _has_shaped = True
                break
            if isinstance(_val, sp.MatrixBase):
                _has_shaped = True
                break
        if _has_shaped:
            self.shape = self.set_shape()
            for key in self.c_dict:
                if getattr(self.c_dict[key], 'shape', None) != self.shape:
                    raise ValueError(
                        "the shapes of the array objects in the c_dict are not the same"
                    )
        else:
            self.shape = 0


    def set_shape(self):
        _key_0 = list(self.c_dict)[0]
        _shape = (0 if not hasattr(self.c_dict[_key_0], 'shape') else
                  self.c_dict[_key_0].shape)
        return _shape

    @property
    def H(self):
        return _ConvolutionView(self, reverse_shift=True, periodic=True)

    @property
    def reverse_shift(self):
        return _ConvolutionView(self, reverse_shift=True, periodic=True)

    @property
    def bnd(self):
        return _ConvolutionView(self, reverse_shift=False, periodic=False)
        
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

    def __setitem__(self, _index, value):
        if isinstance(_index, tuple):
            _index = list(_index)
            if hasattr(self, "ranks"):  # JSymmetricTensor
                _i0 = _index[:self.ranks[0]]
                _i1 = _index[self.ranks[0]:]
                _i0.sort(); _i1.sort()
                _index = tuple(_i0 + _i1)
            else:  # SymmetricTensor
                _index.sort()
                _index = tuple(_index)
        self.c_dict[_index] = value            

    def __repr__(self):
        header = f"JSymmetricTensor(d={self.d}, rank={self.rank}, ranks={self.ranks})"
        lines = [header, "c_dict:"]
        for key in sorted(self.c_dict.keys(), key=lambda k: (0 if isinstance(k, int) else 1, repr(k))):
            lines.append(f"  {key}: {self.c_dict[key]!r}")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("JSymmetricTensor(...)")
            return
        p.text(repr(self))
        
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
            def _index_tuples(rank, d):
                if rank == 0:
                    return [()]
                raw = TaylorTuples(list(range(d)), rank)
                return [t if isinstance(t, tuple) else (t,) for t in raw]

            def _canonical_index(tt, rank):
                if rank == 0:
                    return 0
                if rank == 1:
                    return tt[0] if isinstance(tt, tuple) else tt
                return tt if isinstance(tt, tuple) else (tt,)

            def _joint_key(tt0, tt1, r0, r1):
                if r0 == 0 and r1 == 0:
                    return 0
                if r0 == 0:
                    return _canonical_index(tt1, r1)
                if r1 == 0:
                    return _canonical_index(tt0, r0)
                t0 = tt0 if isinstance(tt0, tuple) else (tt0,)
                t1 = tt1 if isinstance(tt1, tuple) else (tt1,)
                return t0 + t1

            # multiply by scalar SymmetricTensor (rank 0)
            if b.rank == 0:
                s = b[0]
                mul_c_dict = {k: self.c_dict[k] * s for k in self.c_dict}
                if self.ranks[1] == 0:
                    return SymmetricTensor(c_dict=mul_c_dict, d=self.d, rank=self.ranks[0])
                if self.ranks[0] == 0:
                    return SymmetricTensor(c_dict=mul_c_dict, d=self.d, rank=self.ranks[1])
                return JSymmetricTensor(
                    c_dict=mul_c_dict, d=self.d, rank=self.rank, ranks=self.ranks
                )

            # 0|B : nonzero block is the second factor
            if self.ranks[0] == 0:
                sub = SymmetricTensor(
                    c_dict=dict(self.c_dict), d=self.d, rank=self.ranks[1]
                )
                return sub * b

            # A|0 : nonzero block is the first factor
            if self.ranks[1] == 0:
                sub = SymmetricTensor(
                    c_dict=dict(self.c_dict), d=self.d, rank=self.ranks[0]
                )
                return sub * b

            # general A|B with both partial ranks > 0
            res_c_dict = {}
            for tt0 in _index_tuples(self.ranks[0], self.d):
                c_dict_swap = {}
                for tt1 in _index_tuples(self.ranks[1], self.d):
                    key = _joint_key(tt0, tt1, self.ranks[0], self.ranks[1])
                    c_dict_swap[_canonical_index(tt1, self.ranks[1])] = self.c_dict[key]

                sub_tensor = SymmetricTensor(
                    c_dict=c_dict_swap, d=self.d, rank=self.ranks[1]
                )
                contracion = sub_tensor * b

                if isinstance(contracion, SymmetricTensor):
                    for tt_c in contracion.c_dict:
                        out_key = _joint_key(
                            tt0, tt_c, self.ranks[0], contracion.rank
                        )
                        res_c_dict[out_key] = contracion[tt_c]
                else:
                    res_c_dict[_canonical_index(tt0, self.ranks[0])] = contracion

            if not res_c_dict:
                raise Exception("Empty contraction result in JSymmetricTensor.__mul__")

            first_elem_index = list(res_c_dict.keys())[0]
            new_full_rank = (
                len(first_elem_index)
                if isinstance(first_elem_index, tuple)
                else 1
            )
            new_1_rank = new_full_rank - self.ranks[0]

            if new_1_rank > 0:
                return JSymmetricTensor(
                    c_dict=res_c_dict,
                    d=self.d,
                    rank=new_full_rank,
                    ranks=[self.ranks[0], new_1_rank],
                )
            return SymmetricTensor(
                c_dict=res_c_dict, d=self.d, rank=new_full_rank
            )

        elif not isinstance(b, SymmetricTensor) and not isinstance(b, JSymmetricTensor):
            mul_c_dict = {tt: self.c_dict[tt] * b for tt in self.c_dict}
            return JSymmetricTensor(
                d=self.d, rank=self.rank, ranks=self.ranks, c_dict=mul_c_dict
            )

        return NotImplemented

    def __mul__old(self, b):
        if isinstance(b, SymmetricTensor):
            """
            - For each 0-multi-index we can build a SymmetricTensor for the 1-multi-index part
            - at this point the contraction would be given by calling __mul__ between this sub-tensor and b
            """
            # scalar factor (rank 0)
            if b.rank == 0:
                s = b[0]
                mul_c_dict = {k: self.c_dict[k] * s for k in self.c_dict}
                if self.ranks[1] == 0:
                    return SymmetricTensor(c_dict=mul_c_dict, d=self.d, rank=self.ranks[0])
                return JSymmetricTensor(
                    c_dict=mul_c_dict, d=self.d, rank=self.rank, ranks=self.ranks
                )

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

    def _convolve_step(self, a, b, *, reverse_shift=False, periodic=True):
        if not _is_array_like(a) or not _is_array_like(b):
            raise TypeError("_convolve_step expects array-like inputs")

        device_a, device_b = _is_device_array(a), _is_device_array(b)
        if device_a or device_b:
            if not (device_a and device_b):
                raise TypeError(
                    "device convolution requires both kernel and field to be Idpy arrays"
                )
            if not periodic:
                raise NotImplementedError(
                    "device open-boundary convolution is not implemented yet"
                )
            # Kernel may be multi-d (tap layout); field must be flat length-V
            return self._convolve_step_device(a, b, reverse_shift=reverse_shift)

        if getattr(a, 'ndim', None) != getattr(b, 'ndim', None):
            raise ValueError(
                f"dimensionality mismatch: kernel ndim={getattr(a, 'ndim', None)}, "
                f"field ndim={getattr(b, 'ndim', None)}"
            )
        if not periodic and any(ks > fs for ks, fs in zip(a.shape, b.shape)):
            raise ValueError(
                f"kernel shape {a.shape} cannot exceed field shape {b.shape} with open boundaries"
            )

        # Cache nonzero taps per kernel layout (numpy path)
        if not hasattr(self, "_conv_taps_cache"):
            self._conv_taps_cache = {}
        key = (a.shape, a.dtype.str, a.tobytes())
        taps = self._conv_taps_cache.get(key)
        if taps is None:
            center = tuple(s // 2 for s in a.shape)
            nz = np.nonzero(a)
            if nz[0].size == 0:
                offsets = np.zeros((0, a.ndim), dtype=np.int64)
                coeffs = np.zeros((0,), dtype=a.dtype)
            else:
                offsets = np.array(list(zip(*nz)), dtype=np.int64) - np.array(center, dtype=np.int64)
                coeffs = a[nz]
            self._conv_taps_cache[key] = (offsets, coeffs)
            taps = (offsets, coeffs)

        offsets, coeffs = taps
        if reverse_shift:
            offsets = -offsets

        out = np.zeros_like(b, dtype=np.result_type(a.dtype, b.dtype))

        if periodic:
            axes = tuple(range(b.ndim))
            for off, c in zip(offsets, coeffs):
                out += c * np.roll(b, shift=tuple(int(x) for x in off), axis=axes)
            return out

        # open boundaries (zero outside)
        for off, c in zip(offsets, coeffs):
            src = []
            dst = []
            valid = True
            for n, s in zip(b.shape, off):
                s = int(s)
                if s >= 0:
                    if s >= n:
                        valid = False
                        break
                    src.append(slice(0, n - s))
                    dst.append(slice(s, n))
                else:
                    if -s >= n:
                        valid = False
                        break
                    src.append(slice(-s, n))
                    dst.append(slice(0, n + s))
            if valid:
                out[tuple(dst)] += c * b[tuple(src)]

        return out

    def _extract_conv_taps(self, a, *, reverse_shift=False):
        """Host-side tap extraction (works for numpy or via D2H)."""
        a_np = np.asarray(a.D2H() if _is_device_array(a) and hasattr(a, 'D2H') else a)
        center = tuple(s // 2 for s in a_np.shape)
        nz = np.nonzero(a_np)
        if nz[0].size == 0:
            offsets = np.zeros((0, a_np.ndim), dtype=np.int64)
            coeffs = np.zeros((0,), dtype=a_np.dtype)
        else:
            offsets = np.array(list(zip(*nz)), dtype=np.int64) - np.array(center, dtype=np.int64)
            coeffs = a_np[nz]
        if reverse_shift:
            offsets = -offsets
        return offsets, coeffs, a_np.shape

    def _convolve_step_device(self, a, b, *, reverse_shift=False):
        from idpy.physics.stencils.IdpyConvolution import (
            convolve_periodic, _resolve_tenet, get_convolution_shape,
        )
        if getattr(b, 'ndim', 1) != 1:
            raise TypeError(
                "device convolution requires a flat length-V field; "
                "use pack_lattice_flat(field) and "
                "set_convolution_shape(shape, tenet=...)"
            )
        tenet = _resolve_tenet(b)
        offsets, coeffs, _kshape = self._extract_conv_taps(
            a, reverse_shift=reverse_shift,
        )
        shape = get_convolution_shape(tenet=tenet)
        if shape is None:
            raise ValueError(
                "flat device field requires "
                "set_convolution_shape(shape, tenet=...) or shape on convolve"
            )
        return convolve_periodic(
            b, offsets, coeffs, shape=shape, tenet=tenet,
        )

    def _convolve(self, b, *, reverse_shift=False, periodic=True):
        def _as_tuple_index(index):
            if isinstance(index, tuple):
                return index
            if index == 0 and self.rank == 0:
                return ()
            return (index,)

        def _canonical_index(index_tuple, rank):
            if rank == 0:
                return 0
            if rank == 1:
                return index_tuple[0]
            return index_tuple

        sample = next(iter(b.c_dict.values()))
        on_device = _is_device_array(sample)

        c_dict_out = {}
        for full_tt in self.c_dict:
            full_tt_tuple = _as_tuple_index(full_tt)
            tt_0_tuple = full_tt_tuple[:self.ranks[0]]
            tt_1_tuple = full_tt_tuple[self.ranks[0]:]

            tt_0 = _canonical_index(tt_0_tuple, self.ranks[0])
            tt_1 = _canonical_index(tt_1_tuple, self.ranks[1])

            term = self._convolve_step(
                self[full_tt], b[tt_1],
                reverse_shift=reverse_shift, periodic=periodic,
            )
            if tt_0 not in c_dict_out:
                c_dict_out[tt_0] = term
            elif on_device:
                from idpy.core import IdpyMemory
                from idpy.physics.stencils.IdpyConvolution import _tenet_of
                prev = c_dict_out[tt_0]
                t_prev, t_term = _tenet_of(prev), _tenet_of(term)
                if t_prev is not None and t_term is not None and t_prev is not t_term:
                    raise ValueError(
                        "device convolution accumulate: operands on different tenets"
                    )
                tenet = t_prev or t_term
                if tenet is None:
                    raise RuntimeError(
                        "device convolution accumulate: cannot resolve tenet "
                        "from operands"
                    )
                s = np.asarray(
                    prev.D2H() if hasattr(prev, 'D2H') else prev
                )
                t = np.asarray(term.D2H() if hasattr(term, 'D2H') else term)
                acc = np.ascontiguousarray(s + t)
                c_dict_out[tt_0] = IdpyMemory.OnDevice(acc, tenet=tenet)
            else:
                c_dict_out[tt_0] = c_dict_out[tt_0] + term

        if on_device:
            return SymmetricTensor(
                d=self.d, rank=self.ranks[0],
                c_dict=c_dict_out, dtype=self.dtype,
            )

        output_tensor = ZeroSymmetricTensor(
            d=self.d, rank=self.ranks[0], shape=b.shape, dtype=self.dtype,
        )
        for key, val in c_dict_out.items():
            output_tensor[key] = val
        return output_tensor

    def _validate_convolution_operand(self, b):
        if not isinstance(b, (JSymmetricTensor, SymmetricTensor)):
            raise ValueError(
                "only JSymmetricTensor and SymmetricTensor are supported for __matmul__"
            )

        if not self.has_np_arrays or not b.has_np_arrays:
            raise ValueError(
                "only JSymmetricTensor and SymmetricTensor with array-backed "
                "components are supported for __matmul__"
            )

        if isinstance(b, JSymmetricTensor) and b.ranks[0] != self.ranks[1]:
            raise ValueError("only full contraction is supported for __matmul__")
        elif isinstance(b, SymmetricTensor) and b.rank != self.ranks[1]:
            raise ValueError("only full contraction is supported for __matmul__")

    def _matmul_impl(self, b, *, reverse_shift=False, periodic=True):
        self._validate_convolution_operand(b)
        return self._convolve(b, reverse_shift=reverse_shift, periodic=periodic)

    def __matmul__(self, b):
        ## The idea of this method is that both self and b
        ## contain (for now) np.array objects in their c_dict
        ## in this case self acts a convolution stencil while
        ## the convolution operation needs to be performed either with
        ## or without periodic boundary conditions
        return self._matmul_impl(b, reverse_shift=False, periodic=True)

    def trace_symmetric(self, symbolic_flag=False):
        """
        Trace over the Sym^n basis for a JSymmetricTensor with ranks [n, n]:
            tr(A) = sum_t m(t) * A[t|t],
        where t runs over canonical symmetric tuples (TaylorTuples) and
            m(t) = n! / prod_i c_i!
        is the multiplicity of tuple t (c_i are repeated-index counts in t).
        """
        if len(self.ranks) != 2 or self.ranks[0] != self.ranks[1]:
            raise Exception("trace_symmetric requires a JSymmetricTensor with ranks=[n, n]")

        n = self.ranks[0]
        ttuples = TaylorTuples(list(range(self.d)), n)

        # scalar identity case (n=0)
        if n == 0:
            key = list(self.c_dict.keys())[0]
            return self.c_dict[key]

        if symbolic_flag:
            import sympy as sp
            from collections import Counter

            tr = sp.Integer(0)
            n_fact = sp.factorial(n)

            for t in ttuples:
                t_tuple = t if isinstance(t, tuple) else (t,)
                counts = Counter(t_tuple).values()
                mult = n_fact / sp.prod(sp.factorial(c) for c in counts)  # exact Rational/Integer
                tr += mult * self[t_tuple + t_tuple]

            return sp.simplify(tr)
        else:
            import math
            from collections import Counter

            tr = 0.0
            n_fact = math.factorial(n)

            for t in ttuples:
                t_tuple = t if isinstance(t, tuple) else (t,)
                counts = Counter(t_tuple).values()
                denom = 1
                for c in counts:
                    denom *= math.factorial(c)
                mult = n_fact / denom
                tr += mult * self[t_tuple + t_tuple]

            return tr

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

def GetZeroSymmetricTensor(d=None, rank=None):
    if rank == 0:
        return SymmetricTensor(d=d, rank=0, c_dict={0: 0})

    ttuples = TaylorTuples(list(range(d)), rank)
    return SymmetricTensor(d=d, rank=rank, list_ttuples=ttuples,
                           list_values=[0] * len(ttuples))

def GetASymmetricTensor(dim, order, root_sym = 'A'):
    if order == 0:
        return SymmetricTensor(d=dim, rank=order, c_dict={0: sp.Symbol(root_sym)})
    elif order > 0:
        _taylor_indices = TaylorTuples(list(range(dim)), order)
        _swap_dict = {}
        for _i, _index_tuple in enumerate(_taylor_indices):
            _lower_indices = reduce(lambda x, y: str(x) + ',' + str(y), _index_tuple)
            _swap_dict[_index_tuple] = sp.Symbol(root_sym + "_{" + _lower_indices + "}")
        return SymmetricTensor(c_dict = _swap_dict, d = dim, rank = order)

def GetFullyIsotropicTensor(d=None, rank=None):
    if rank % 2:
        raise Exception("rank must be even!")

    if rank == 0:
        return SymmetricTensor(d=d, rank=0, c_dict={0: 1})

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
        
def GetPiTensor(d=None, half_rank=None, symbolic_flag=False):
    if half_rank == 0:
        return SymmetricTensor(d=d, rank=0, c_dict={0: 1})

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

        follow_Pi_hrankm1 = GetPiTensor(d, half_rank=half_rank-1, symbolic_flag=symbolic_flag)

        tuples_map = lambda in_tuple: map(lambda perm: SplitTuplePerm(in_tuple=in_tuple, perm=perm, split_point=half_rank), index_lists)
        summands = lambda in_tuple: map(lambda out_tuple: lead_kr_2[(out_tuple[0][0], out_tuple[1][0])] * follow_Pi_hrankm1[out_tuple[0][1:] + out_tuple[1][1:]], tuples_map(in_tuple))
        sum_results = lambda in_tuple: reduce(lambda x, y: x + y, summands(in_tuple))

        taylor_indices_0 = TaylorTuples(list(range(d)), half_rank)
        taylor_indices_1 = TaylorTuples(list(range(d)), half_rank)

        scale = sp.Rational(1, half_rank) if symbolic_flag else (1.0 / half_rank)

        swap_dict = {}
        for tt0 in taylor_indices_0:
            for tt1 in taylor_indices_1:
                swap_dict[tt0 + tt1] = sum_results(tt0 + tt1) * scale

        return JSymmetricTensor(d=d, rank=2*half_rank, ranks=[half_rank, half_rank], c_dict=swap_dict)

def FromTuplesToPows(dim, order):
    if order == 0:
        return [(0,) * dim]
    else:
        fully_sym_comp = TaylorTuples(list(range(dim)), order)
        fully_sym_pows = [np.unique(c, return_counts=True) for c in fully_sym_comp]

        pows_list = [[0] * dim for _ in range(len(fully_sym_comp))]
        for i, (pows, counts) in enumerate(fully_sym_pows):
            for idx, p in zip(pows, counts):
                pows_list[i][idx] += int(p)

        pows_list = [tuple(pows) for pows in pows_list]
        return pows_list, fully_sym_comp