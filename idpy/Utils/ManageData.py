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
Provides general dumping/reading facility through the package dill
'''

# Acknowledgement
# How to dump lambdas: https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
from collections import defaultdict
import dill, os

class ManageData:
    '''
    class ManageData(dump_file):
    -- PushData(data, key) : inserts data in the dictionary under the key value
    -- WhichData: returns a list with the entries
    -- PullData(key): returns the data structure associated to the key
    -- Dump
    -- Read
    -- CleanDump
    '''
    def __init__(self, dump_file = 'ManageData'):
        self.data_dictionary = defaultdict(dict)
        self.dump_file = dump_file

    def ChangeDumpFile(self, dump_file = 'ManageData'):
        self.dump_file = dump_file

    def PushData(self, data = None, key = None):
        if data is None or key is None:
            raise Exception("Please insert valid data and key")
        self.data_dictionary[key] = data

    def DelData(self, key = None):
        if key is None:
            raise Exception("Please insert a valid 'key' argument")
        del self.data_dictionary[key]

    def WhichData(self):
        entries = []
        for elem in self.data_dictionary:
            entries.append(elem)
        return entries

    def IsThereKey(self, key):
        return key in self.data_dictionary

    def PullData(self, key):
        if key in self.WhichData():
            return self.data_dictionary[key]
        else:
            raise Exception("key not present in the data base")
        
    def Dump(self):
        file_out = open(self.dump_file, 'wb')
        ##chars_n = file_out.write(dill.dumps(self.__dict__))
        chars_n = file_out.write(dill.dumps(self.data_dictionary))
        file_out.close()
        return chars_n
        
    def Read(self):
        if os.path.isfile(self.dump_file):
            file_in = open(self.dump_file, 'rb')
            data_dill = file_in.read()
            file_in.close()
            ##self.__dict__ = dill.loads(data_dill)
            self.data_dictionary = dill.loads(data_dill)
            return True
        else:
            return False

    def CleanDump(self):
        if os.path.isfile(self.dump_file):
            os.remove(self.dump_file)
