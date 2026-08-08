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
Provides a simple class for timing
'''

import time

class SimpleTiming:
    def __init__(self):
        self.start, self.end = None, None
        self.start_f, self.end_f = False, True
        self.lapse = {}

    def Start(self):
        if self.end_f and not self.start_f:
            self.start = time.time()
            self.start_f = True
            self.end_f = False
        else:
            raise Exception("Timing already started!")

    def End(self):
        if self.start_f and not self.end_f:
            self.end = time.time()
            self.start_f = False
            self.end_f = True
        else:
            raise Exception("Timing already stopped!")

    def GetElapsedTime(self):
        self.lapse['time_s'] = self.end - self.start
        self.lapse['time_ms'] = self.lapse['time_s'] * 1e3
        self.lapse['time_us'] = self.lapse['time_s'] * 1e6
        self.lapse['time_ns'] = self.lapse['time_s'] * 1e9
        
        _n_sec_min, _n_min_hrs = 60, 60
        _n_sec_hrs, _n_hrs_day = _n_min_hrs * _n_sec_min, 24
        _n_sec_day = _n_hrs_day * _n_sec_hrs

        self.lapse['d'] = int(self.lapse['time_s'])//_n_sec_day
        self.lapse['h'] = \
            (int(self.lapse['time_s'])//_n_sec_hrs)%_n_hrs_day
        self.lapse['m'] = \
            ((int(self.lapse['time_s'])//_n_sec_min)%_n_min_hrs)
        self.lapse['s'] = int(self.lapse['time_s']) % _n_sec_min
        
        self.lapse['ms'] = \
            int((self.lapse['time_s'] - int(self.lapse['time_s'])) * 1e3)
        self.lapse['us'] = \
            int((self.lapse['time_s'] -
                 (int(self.lapse['time_s']) + self.lapse['ms'] * 1e-3)) * 1e6)
        '''
        This one is just an algebraic exercise
        Can we trust the python timing to the nano second?
        '''
        self.lapse['ns'] = \
            int((self.lapse['time_s'] -
                 (int(self.lapse['time_s']) +
                  self.lapse['ms'] * 1e-3 + self.lapse['us'] * 1e-6)) * 1e9)

        return self.lapse

    def Get_s(self):
        return self.GetElapsedTime()['time_s']

    def Get_ms(self):
        return self.GetElapsedTime()['time_ms']

    def Get_us(self):
        return self.GetElapsedTime()['time_us']

    def Get_ns(self):
        return self.GetElapsedTime()['time_ns']
    
    def PrintElapsedTime(self):
        _lapse = self.GetElapsedTime()
        print(_lapse['d'], "d, ",
              _lapse['h'], "h, ",
              _lapse['m'], "m, ", 
              _lapse['s'], "s")
