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

def PerezNouiAmplitude(A, B):    
    _mod2_A = (A.conjugate() * A).real
    _mod2_B = (B.conjugate() * B).real    
    
    _max = max(_mod2_A, _mod2_B)
    _A_star_B = A.conjugate() * B
    
    return _A_star_B / (_max) if _max > 0 else 0

def PerezNouiProbability(A, B):    
    _mod2_A = (A.conjugate() * A).real
    _mod2_B = (B.conjugate() * B).real    
    
    _max = max(_mod2_A, _mod2_B)
    _A_star_B = A.conjugate() * B
    _mod2_A_star_B = (_A_star_B.conjugate() * _A_star_B).real
    
    return _mod2_A_star_B / (_max ** 2) if _max > 0 else 0
