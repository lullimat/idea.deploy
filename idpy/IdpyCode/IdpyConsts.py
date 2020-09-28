__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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
Provides classes containing language specific definitions
'''

from collections import defaultdict

from idpy.Utils.IsModuleThere import AreModulesThere
from . import CUDA_T, OCL_T, IDPY_T

class AddrQualif:
    '''
    class AddrQualif:
    class containing the address qualifiers for the different languages
    '''
    def __init__(self):
        '''
        self.qualifiers: double nested defaultdict (allowing keys 
        repetitions)
        - first entry: language
        - second entry: common naming (when available)
        '''
        self.qualifiers = defaultdict(
            lambda: defaultdict(dict)
        )
        
        self.qualifiers[CUDA_T] = {'const': """const""",
                                   'local': """local""",
                                   'restrict': """__restrict__""",
                                   'shared': """__shared__""",
                                   'device': """__device__""",
                                   'global': ''}

        self.qualifiers[OCL_T] = {'global': """__global""",
                                  'const': """__const""",
                                  'local': """__local""",
                                  'restrict': '',
                                  'restrict': '',
                                  'shared': '',
                                  'device': ''}

    def __getitem__(self, lang):
        return self.qualifiers[lang]


class KernQualif:
    '''
    class KernQualif:
    class containing the kernels qualifiers for the different languages
    '''
    def __init__(self):
        '''
        self.qualifiers: double nested defaultdict (allowing keys 
        repetitions)
        - first entry: language
        - second entry: common naming (when available)
        '''
        self.qualifiers = {CUDA_T: """__global__ void""", OCL_T: """__kernel void"""}

    def __getitem__(self, lang):
        return self.qualifiers[lang]



class FuncQualif:
    '''
    class FuncQualif:
    class containing the function qualifiers for the different languages
    '''
    def __init__(self):
        self.qualifiers = defaultdict(dict)
        self.qualifiers = {CUDA_T: """__device__""", OCL_T: """ """}

    def __getitem__(self, lang):
        return self.qualifiers[lang]


