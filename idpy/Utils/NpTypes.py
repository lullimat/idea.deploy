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
Provides a minimal interface for handling numpy types
'''

import numpy as np

class NpTypes:
    '''
    class NpTypes:
    NpTypes: handles a dictionary with numpy types and offers "C"-like 
    key-words
    -- CType(key)
    '''
    
    def __init__(self):
        # C Types
        self.C = {'double': np.float64, 'float': np.float32,
                  'int': np.int32, 'unsigned int': np.uint32,
                  'long long int': np.int64,
                  'unsigned long': np.uint64,
                  'unsigned long long int': np.uint64,
                  'char': np.byte, 'unsigned char': np.ubyte}

        self.NP = {value: key for (key, value) in self.C.items()}

    def ToList(self):
        return [key for key in self.C], [key for key in self.NP]
