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
import math

def IndexFromPos(pos, dim_strides):
    index = pos[0]
    for i in range(1, len(pos)):
        index += pos[i] * dim_strides[i - 1]
    return index

def PosFromIndex(index, dim_strides):
    if len(dim_strides):
        pos = [index%dim_strides[0]]
        pos += [(index//dim_strides[stride_i]) % (dim_strides[stride_i + 1]//dim_strides[stride_i]) \
                for stride_i in range(len(dim_strides) - 1)]
        pos += [index//dim_strides[len(dim_strides) - 1]]
    else:
        pos = [index]
    return tuple(pos)

def PosFromIndexOOld(index, dim_strides):
    if len(dim_strides):
        pos = [index%dim_strides[0]]
        pos += [(index//dim_strides[stride_i]) % (dim_strides[stride_i + 1]//dim_strides[stride_i]) \
                if stride_i < len(dim_strides) - 1 else \
                index//dim_strides[stride_i] \
                for stride_i in range(len(dim_strides))]
    else:
        pos = [index]
    return tuple(pos)    

def PosFromIndexOld(index, dim_strides): 
    pos = [index%dim_strides[0]]
    pos += [(index//dim_strides[stride_i]) % (dim_strides[stride_i + 1]//dim_strides[stride_i]) \
            if stride_i < len(dim_strides) - 1 else \
            index//dim_strides[stride_i] \
            for stride_i in range(len(dim_strides))]
    return tuple(pos)

def GetDimStridesLambda(_dim_sizes):
    return [reduce(lambda x, y: x * y, _dim_sizes[0: i + 1]) 
                   for i in range(len(_dim_sizes) - 1)]

def GetDimStrides(_dim_sizes):
    return [math.prod(_dim_sizes[0: i + 1]) for i in range(len(_dim_sizes) - 1)]

def GetLen2PosLambda(pos):
    return reduce(lambda x, y: x + y, map(lambda x: x ** 2, pos))

def GetLen2Pos(pos):
    return sum(map(lambda x: x ** 2, pos))    

def GetDiffPos(A, B):
    return tuple(map(lambda x, y: x - y, A, B))

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

def TriDet(triangle):
    A, B, C = triangle[0], triangle[1], triangle[2]
    _det = \
        (A[0] - C[0]) * (B[1] - A[1]) - \
        (A[0] - B[0]) * (C[1] - A[1])
    
    _det = None if abs(_det) < 1e-6 else _det
    
    return _det

def EScalarProduct(v1, v2):
    return reduce(lambda x, y: x + y, map(lambda a, b: a * b, v1, v2))

def ENorm2(pos):
    return EScalarProduct(pos, pos)

def ENorm(pos):
    return math.sqrt(ENorm2(pos))

def GetRelCosine(rel, ref):
    output = EScalarProduct(rel, ref) / ENorm(rel) / ENorm(ref)
    output = 1 if (abs(output) - 1) > 0 else output
    return output

def GetRelSine(rel, ref):
    tri = [(0,) * len(ref), ref, rel]
    output = TriDet(tri) / ENorm(rel) / ENorm(ref)
    output = 1 if (abs(output) - 1) > 0 else output
    return output

def FindTwoPiAngle(rel, ref):
    cosine, sine = GetRelCosine(rel, ref), GetRelSine(rel, ref)
    ##print('cosine, sine', cosine, sine)
    '''
    First Quadrant
    '''
    if cosine >= 0 and sine >= 0:
        return math.asin(sine)
    '''
    Second Quadrant
    '''
    if cosine < 0 and sine >= 0:
        return math.acos(cosine)
    '''
    Third Quadrant
    '''
    if cosine < 0 and sine < 0:
        return math.pi - math.asin(sine)
    '''
    Fourth Quadrant
    '''
    if cosine >= 0 and sine < 0:
        return 2 * math.pi + math.asin(sine)


