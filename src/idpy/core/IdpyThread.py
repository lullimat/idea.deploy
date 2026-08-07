__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2024 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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

"""
Referenced pages:

https://discourse.jupyter.org/t/running-a-cell-in-a-background-thread/12267/8
https://stackoverflow.com/questions/21284319/can-i-make-one-method-of-class-run-in-background
https://stackoverflow.com/questions/36937667/python-threads-can-only-be-started-once
https://stackoverflow.com/questions/6904487/how-to-pass-arguments-to-a-thread

use ultiprocess because multiprocessing did not work
https://pypi.org/project/multiprocess/
"""

import multiprocess

class IdpyThread(multiprocess.Process):    
    def __init__(self, target_function, *args, **kwargs):
        multiprocess.Process.__init__(self)
        self.target_function = target_function
        self.args, self.kwargs = args, kwargs

        self.manager = multiprocess.Manager()
        self.output = self.manager.list([None])
        
    def run(self):
        self.target_function(self.output, *self.args, **self.kwargs)

    def get_output(self):
        self.join()
        return self.output[0]

def IDThreadFunc(f):
    def wrapper(output, *argc, **kwarg):
        output[0] = f(*argc, **kwarg)
    return wrapper