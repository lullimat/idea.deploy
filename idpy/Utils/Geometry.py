__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2022 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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

from functools import reduce

def IndexFromPos(pos, dim_strides):
    index = pos[0]
    for i in range(1, len(pos)):
        index += pos[i] * dim_strides[i - 1]
    return index

def PosFromIndex(index, dim_strides): 
    pos = [index%dim_strides[0]]
    pos += [(index//dim_strides[stride_i]) % (dim_strides[stride_i + 1]//dim_strides[stride_i]) \
            if stride_i < len(dim_strides) - 1 else \
            index//dim_strides[stride_i] \
            for stride_i in range(len(dim_strides))]
    return tuple(pos)

def GetDimStrides(_dim_sizes):
    return [reduce(lambda x, y: x * y, _dim_sizes[0: i + 1]) 
                   for i in range(len(_dim_sizes) - 1)]

def GetLen2Pos(pos):
    return reduce(lambda x, y: x + y, map(lambda x: x ** 2, pos))

from idpy.Utils.Statements import AllTrue

def IsOppositeVector(A, B):
    return AllTrue(list(map(lambda x, y: x == -y, A, B)))

def IsSameVector(A, B):
    return AllTrue(list(map(lambda x, y: x == y, A, B)))

import numpy as np
def FlipVector(A):
    return tuple(np.flip(A))

'''
Here U and V are supposed to be iterables and prod a function of two arguments
'''
def ProjectionVAlongU(V, U, prod):
    return (prod(V, U) / prod(U, U))

'''
Here U is supposed to be multiplyable by a scalar: sympy.Matrix works
'''
def ProjectVAlongU(V, U, prod):
    return U * ProjectionVAlongU(V, U, prod)
