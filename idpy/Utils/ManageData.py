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

'''
Provides general dumping/reading facility through the package dill
'''

# Acknowledgement
# How to dump lambdas: https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
from collections import defaultdict
import dill, os
import h5py, json
import numpy as np

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

    def Dump(self, kind = 'dill', indent = None):
        if kind not in ['hdf5', 'dill', 'json']:
            raise Exception("Parameter 'kind' must be in ['hdf5','dill','json']")
        if kind == 'hdf5':
            self.DumpHDF5()        
        if kind == 'dill':
            self.DumpDill()
        if kind == 'json':
            self.DumpJson(indent = indent)

    def Read(self, kind = 'dill'):
        if kind not in ['hdf5', 'dill', 'json']:
            raise Exception("Parameter 'kind' must be in ['hdf5','dill','json']")        
        if kind == 'hdf5':
            return self.ReadHDF5()        
        if kind == 'dill':
            return self.ReadDill()
        if kind == 'json':
            return self.ReadJson()

    def DumpJson(self, indent = None):
        file_out = open(self.dump_file, 'w')
        file_out.write(json.dumps(self.data_dictionary, indent = indent))
        file_out.close()
        
    def DumpDill(self):
        file_out = open(self.dump_file, 'wb')
        ##chars_n = file_out.write(dill.dumps(self.__dict__))
        chars_n = file_out.write(dill.dumps(self.data_dictionary))
        file_out.close()
        return chars_n

    def DumpHDF5(self):
        if not self.dump_file.lower().endswith(('.hdf5', '.h5')):
            file_out = self.dump_file + '.hdf5'
        else:
            file_out = self.dump_file
            
        _out_f = h5py.File(file_out, "a")
        _grp = _out_f.create_group(self.__class__.__name__)
            
        for _key in self.data_dictionary:
            if _key not in _grp:
                print(_key)
                if type(self.data_dictionary[_key]) is not dict:
                    _grp.create_dataset(_key, data = self.data_dictionary[_key])
                else:
                    _grp.create_dataset(_key,
                                        data = np.string_(str(self.data_dictionary[_key])))
            
        _out_f.close()
        
    def ReadDill(self):
        if os.path.isfile(self.dump_file):
            file_in = open(self.dump_file, 'rb')
            data_dill = file_in.read()
            file_in.close()
            self.data_dictionary = dill.loads(data_dill)
            return True
        else:
            return False

    def ReadJson(self):
        if os.path.isfile(self.dump_file):
            file_in = open(self.dump_file, 'r')
            data_json = json.load(file_in)
            file_in.close()
            self.data_dictionary = defaultdict(dict, data_json)
            return True
        else:
            return False

    def ReadHDF5(self, full_key = None):
        if full_key is None:
            raise Exception("Missing parameter 'full_key'")

        if not self.dump_file.lower().endswith(('.hdf5', '.h5')):
            file_name = self.dump_file + '.hdf5'
        else:
            file_name = self.dump_file
        
        _in_f = h5py.File(file_name, "r")
        _sims_class_name = list(_in_f.keys())[0]
        if _sims_class_name != self.__class__.__name__:
            raise Exception("The file you are reading has been created by another class!")

        if not full_key in _in_f:
            raise Exception("Key", full_key, "cannot be found in", file_name)

        _swap_data = np.array(_in_f.get(full_key))
        _in_f.close()
        return _swap_data

    def CleanDump(self):
        if os.path.isfile(self.dump_file):
            os.remove(self.dump_file)
