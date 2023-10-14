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

import numpy as np
from sympy import factorial

def SwapElem(array, i):
    array[i], array[i + 1] = array[i + 1], array[i]
    
def NSwapDownElem(array, i, n):
    for k in range(n):
        SwapElem(array, i + k)
        
def NSwapDownElemSeq(array, i, n):
    _seq = []
    for k in range(n):
        SwapElem(array, i + k)
        _seq += [array.copy()]
    return _seq

def FindOneAtFirstGap(seq):
    _last_one, _last_zero = 0, 0
    if seq[0] == 0:
        return -1, -1
    
    '''
    Find last one
    '''
    _found_flag = False
    for i, elem in enumerate(seq):
        if elem == 0 and i > 0:
            _last_one = i - 1
            _found_flag = True
            break
            
    if not _found_flag:
        _last_one = len(seq) - 1

    '''
    Find last zero
    '''
    if _last_one < len(seq) - 1:
        _found_flag = False
        for i, elem in enumerate(seq[_last_one + 1:]):
            if elem > 0 and i > 0:
                _last_zero = _last_one + i
                _found_flag = True
                break

        if not _found_flag:
            _last_zero = len(seq) - 1
    else:
        _last_zero = -1
            
    return _last_one, _last_zero

def JustDoIt(seq):
    _a, _b = FindOneAtFirstGap(seq)
    _out_list = []
    
    if _a != -1 and _b != -1:
        _seq_list = NSwapDownElemSeq(seq, _a, _b - _a)
                    
        #_seq_list_swap = []
        for elem in _seq_list:
            _seq_list_swap = \
                JustDoIt(elem.copy())
            
            _out_list += [elem.copy()] + _seq_list_swap
            
    return _out_list

def Multinomial(n, below: list):
    _multinomial = factorial(n)
    for i in range(len(below)):
        _multinomial /= factorial(below[i])
        
    return _multinomial    

def Multinomial2(n, i, k):
    return factorial(n) // factorial(i) // factorial(k)

def PrintMultinomial(n, below: list):
    list_str = ''
    return '$$\\binom{a}{' + list_str + '}$$'

'''
Need to compute all the possible combinations of all the possible sum decompositions
of an integer M over the set of indices of the array of cycles. This creates a sequence
for laying down cycles on the graph such that the classical coloring compatibility is
autmatically satisfied. We also need to take care of the quantum case: one can check
whether or not a given combination of loops is q-compatible, however, the global exit
condition is not clear yet.

1. Generate all unique permutations of 1s and 0s
2. Generate all unique permutations of 2s and 0s using the left-over number
    of 0s from the previous step; for each element of the previous step generate
    a new sequence by "intersecting" the two lists, i.e., 
    [0, 1, 0, 0] 'intersect' [2, 0, 0], [0, 2, 0], [0, 0, 2] gives
    [2, 1, 0, 0], [0, 1, 2, 0], [0, 1, 0, 2] (need to write a good function for this)

n. Publish a post on my website
n + 1. Make a Youtube video
'''

"""
sum = 1 -> [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], ...
            (7, (1, 6))

sum = 2 -> [1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0], ...
            (7, (2, 5))
           [2, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], ...
            (7, (1, 6))
            
([(1, 2), (2, 1), (3, 1)], 6) -> (1, 1, 2, 3, 0, 0)

(1, 1, 0, 0, 0, 0) -> (0, 4) -> (2, 0, 0, 0) -> [(2, 0, 0, 0), (0, 2, 0, 0), ...]

(2, 0, 0, 0) -> (0, 3) -> (3, 0, 0) -> [(3, 0, 0), (0, 3, 0), (0, 0, 3)]

"""


def GetBaseString(n: int, x: int, y: int):
    return np.array(([n] * x) + ([0] * y))

def GetSinglePermutations(n: int, x: int, y: int):
    return [GetBaseString(n, x, y)] + JustDoIt(GetBaseString(n, x, y))

def GetUniquePermutations(elems: list, N: int):
    total_len = 0
    for elem in elems:
        total_len += elem[1]
    if total_len > N:
        raise Exception("The total number of elements is larger than N!")
    
    _elem_0 = elems[0]
    _local_seqs = GetSinglePermutations(*_elem_0, N - _elem_0[1])
    ##print(_local_seqs)

    _swap_insertions = []    
    
    if len(elems) > 1:
        _insertions = GetUniquePermutations(elems[1:], N - _elem_0[1])    

        for _seq in _local_seqs:
            for _insertion in _insertions:
                _local_copy = _seq.copy()
                _local_copy[_local_copy == 0] = _insertion
                _swap_insertions += [_local_copy]
    else:
        _swap_insertions = _local_seqs
                    
    return _swap_insertions

'''
This function finds all the sum decompositions, a part from the trivial ones:
- the identity
'''
def FindNonTrivialCombinations(n: int, nm1=None):
    _sequences = []
    _i_start = 1 if nm1 is None else nm1
    for i in range(_i_start, n // 2 + 1):
        k = n - i
        _sequences += [(i, k)]
        
        if k - i >= 1:
            for _seq in FindNonTrivialCombinations(k, i):
                _sequences += [(i,) + _seq]
        
    return _sequences

def FindSumCombinations(n: int):
    return FindNonTrivialCombinations(n) + [(n,)]

def FindSumCombinationsTuples(n: int, cutoff: int):
    _tuples_lists = []
    for combination in FindSumCombinations(n):
        if len(combination) <= cutoff:
            elems, count = np.unique(combination, return_counts=True)
            _tuples_lists += [list(zip(elems, count))]
    return _tuples_lists
