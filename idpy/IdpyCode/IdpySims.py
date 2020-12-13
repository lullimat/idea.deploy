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

import warnings
import numpy as np
import sympy as sp
import threading, h5py
from collections import defaultdict

from idpy.IdpyCode import GetParamsClean

class IdpySims(threading.Thread):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'params_dict'):
            self.params_dict = {}

        self.kwargs = GetParamsClean(kwargs, [self.params_dict],
                                     needed_params = ['daemon_flag'])
        
        if 'daemon_flag' in self.params_dict:
            self.daemon_flag = self.params_dict['daemon_flag']
        else:
            self.daemon_flag = False
        
        threading.Thread.__init__(self)
        self.sims_vars = {}
        self.sims_idpy_memory = {}
        '''
        sims_dump_vars, sims_dump_idpy_memory:
        lists contaiing the dictionary key to be dumped
        if empty all key are dumped: these are to be set in the
        child class: likely specified in a child class of a general
        class
        '''
        self.sims_dump_vars, self.sims_dump_idpy_memory = [], []
        self.sims_not_dump_vars, self.sims_not_dump_idpy_memory = [], []
        self.sims_dump_vars_flag, self.sims_dump_idpy_memory_flag = True, True
        self.aux_idpy_memory, self.aux_vars = [], []
        
    def CleanAuxilliary(self):
        for key in self.aux_vars:
            if key in self.sims_vars:
                del self.sims_vars[key]

        for key in self.aux_idpy_memory:
            if key in self.sims_idpy_memory:
                del self.sims_idpy_memory[key]

    def DumpSnapshot(self, file_name = None, custom_types = None):
        if file_name is None:
            raise Exception("Parameter file_name must not be None")
        
        if custom_types is None:
            raise Exception("Parameter custom_types must not be None")            
        
        _out_f = h5py.File(file_name, "a")
        _grp  = _out_f.create_group(self.__class__.__name__)

        '''
        dumping vars
        '''
        if self.sims_dump_vars_flag:
            _grp_vars = _grp.create_group("vars")

            for key in self.sims_vars:
                '''
                old logic: key in self.sims_dump_vars and \
                '''
                if (len(self.sims_dump_vars) == 0 and \
                    len(self.sims_not_dump_vars) == 0) or \
                   (key not in self.sims_not_dump_vars):
                    _type = type(self.sims_vars[key]).__module__.split(".")[0]
                    '''
                    Checking dump exclusion according to the data type
                    print(key, _type, len(self.sims_dump_vars), key not in self.sims_not_dump_vars)
                    '''
                    if _type == np.__name__ or _type == 'builtins':
                        _grp_vars.create_dataset(key, data = self.sims_vars[key])
                    else:
                        print("Key: ", key, "not builtin/numpy: not dumped!")
        
        '''
        dumping idpy_memory
        '''
        if self.sims_dump_idpy_memory_flag:
            _grp_idpy_memory = _grp.create_group("idpy_memory")

            for key in self.sims_idpy_memory:
                if self.sims_idpy_memory[key] is None:
                    _out_f.close()
                    os.remove(file_name)
                    raise Exception("Cannot dump ", key, ": not allocated")
                else:
                    if len(self.sims_dump_idpy_memory) == 0 or \
                       key in self.sims_dump_idpy_memory:
                        _grp_idpy_memory.create_dataset(
                            key, data = self.sims_idpy_memory[key].D2H()
                        )

        '''
        dumping custom_types
        '''
        _grp_custom_types = _grp.create_group("custom_types")
        for key, value in custom_types.Push().items():
            _grp_custom_types.create_dataset(str(key),
                                             data = np.string_(value))
            
        _out_f.close()

    def ReadSnapshotData(self, file_name = None, full_key = None):
        if file_name is None:
            raise Exception("Parameter file_name must not be None")
        if full_key is None:
            raise Exception("Parameter full_key must not be None")

        _in_f = h5py.File(file_name, "r")
        _sims_class_name = list(_in_f.keys())[0]
        if  _sims_class_name != self.__class__.__name__:
            print("File class: ", _sims_class_name)
            print("Present class: ", self.__class__.__name__)
            raise Exception("The file you are reading has been created by another class!")

        if not full_key in _in_f:
            raise Exception("Key", full_key, "cannot be found in", file_name)
        _swap_data = np.array(_in_f.get(full_key))
        _in_f.close()
        return _swap_data

    def CheckSnapshotData(self, file_name = None, full_key = None):
        if file_name is None:
            raise Exception("Parameter file_name must not be None")
        if full_key is None:
            raise Exception("Parameter full_key must not be None")

        _in_f = h5py.File(file_name, "r")
        _sims_class_name = list(_in_f.keys())[0]
        if _sims_class_name != self.__class__.__name__:
            print("File class: ", _sims_class_name)
            print("Present class: ", self.__class__.__name__)
            raise Exception("The file you are reading has been created by another class!")

        return full_key in _in_f

    def ReadSnapshotFull(self):
        '''
        ReadSnapshotFull:
        the idea is to use the ReadSnapshotData applied to all cases
        '''
        pass
