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

import numpy as np
import os

from idpy.Utils.NpTypes import NpTypes
from idpy.Utils.CustomTypes import CustomTypes

from idpy.IdpyCode import GetParamsClean, GetTenet
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T
from idpy.IdpyCode import idpy_langs_sys, idpy_tenet_types

if idpy_langs_sys[OCL_T]:
    import pyopencl as cl

from idpy.IdpyCode.IdpySims import IdpySims
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyFunction, IdpyLoop

NPT = NpTypes()

'''
Class for different congruential pseudo-random number generators.
It is a child class of idpy.IdpyCode.IdpySims.IdpySims

CRNGS(**kwargs):
kwargs must be a dictionary.
Necessary Arguments
- 'lang': language type. Possible values ['CUDA_T', 'OCL_T'], used for setting the 64-bits type that differ between CUDA and OpenCL.

Optional Arguments
- 'kind': type of congruential generator. Defaults to 'MINSTD'. Possible values are
-- 'MINSTD': 32-bits generator with ID_MAXRAND the Mersenne prime 2^31 - 1, the absence of a constant in the definition implies that 0 does not belong to the sequence. Temporary 64-bits variables are used in the implementation.
-- 'NUMREC': 32-bits generator from "Numerical Recipes" with ID_MAXRAND 2^32 - 1, 
-- 'MMIX': 64-bits generator with ID_MAXRAND 2^64 - 1

- 'init_from': how to initialize the PRNG
-- 'numpy': using the numpy random number generator
-- 'urandom': using the system random number pool urandom

- 'init_seed': seed for the initialization from numpy. Defaults to 1.
- 'n_prngs': number of random number generators. Defaults to 1.
- 'optimizer_flag': boolean flag to disable/enable optimizer. Defaults to 'False'
- 'block_size': size of the block of threads. Defaults to 128.

- 'tenet': tenet object. If not passed the class automatically instantiates a new tenet object. If the CRNG object is instantiated in another class then the two tenets MUST be the same, otherwise the memory allocated in the CRNG object cannot be accessed together with the memory allocated by the upper class.

If no 'tenet' object is passed then the following are Necessary Arguments
- 'cl_kind': type of OpenCL device, either 'gpu' or 'cpu'. Defaults to 'gpu'
- 'device': device number. Defaults to 0.
'''

class CRNGS(IdpySims):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'params_dict'):
            self.params_dict = {}

        self.InitCRNGS(args, kwargs)
        
        '''
        Setting variables and constants
        '''
        self.sims_vars['kind'] = self.params_dict['kind']
        self.sims_vars['fp64'] = self.params_dict['fp64']        

        '''
        Setting custom types: handling here the different
        type namings for 64-bits unsigned integers for 
        CUDA and OpenCL
        '''
        if self.sims_vars['kind'] == 'MINSTD':
            if self.params_dict['lang'] == CUDA_T:
                self.custom_types = CustomTypes({'CRNGType': 'unsigned int',
                                                 'UINT64': 'unsigned long long int'})
            if self.params_dict['lang'] == OCL_T:
                self.custom_types = CustomTypes({'CRNGType': 'unsigned int',
                                                 'UINT64': 'unsigned long'})                
        if self.sims_vars['kind'] == 'NUMREC':
            self.custom_types = CustomTypes({'CRNGType': 'unsigned int'})
        if self.sims_vars['kind'] == 'MMIX':
            if self.params_dict['lang'] == CUDA_T:
                self.custom_types = CustomTypes({'CRNGType': 'unsigned long long int'})
            if self.params_dict['lang'] == OCL_T:
                self.custom_types = CustomTypes({'CRNGType': 'unsigned long'})

        '''
        Setting ID_RANDMAX
        '''
        if self.sims_vars['kind'] == 'MINSTD':
            self.sims_vars['ID_RANDMAX'] = int('7fffffff', 16)
        if self.sims_vars['kind'] == 'NUMREC':
            self.sims_vars['ID_RANDMAX'] = int('ffffffff', 16)
        if self.sims_vars['kind'] == 'MMIX':
            self.sims_vars['ID_RANDMAX'] = int('ffffffffffffffff', 16)

        
        self.sims_vars['init_from'] = self.params_dict['init_from']
        self.sims_vars['init_seed'] = self.params_dict['init_seed']
        self.sims_vars['n_prngs'] = self.params_dict['n_prngs']

        self.constants = {'ID_RANDMAX': self.sims_vars['ID_RANDMAX'],
                          'N_PRNGS': self.sims_vars['n_prngs']}
        '''
        Setting device function CRNG through a macro
        '''
        if self.sims_vars['kind'] == 'MINSTD':
            self.F_CRNG = F_MINSTD
            self.constants['F_CRNG'] = 'F_MINSTD'
            
        if self.sims_vars['kind'] == 'NUMREC':
            self.F_CRNG = F_NUMREC
            self.constants['F_CRNG'] = 'F_NUMREC'
            
        if self.sims_vars['kind'] == 'MMIX':
            self.F_CRNG = F_MMIX
            self.constants['F_CRNG'] = 'F_MMIX'

        
        '''
        Memory to be allocated
        '''
        self.sims_idpy_memory = {'seeds': None,
                                 'output': None}
        '''
        Init seeds
        '''
        self.GridAndBlocks()
        self.InitSeeds()
        
    def InitCRNGS(self, args, kwargs):
        self.kwargs = GetParamsClean(kwargs, [self.params_dict], 
                                     needed_params = ['n_prngs', 
                                                      'kind', 
                                                      'init_from', 
                                                      'init_seed',
                                                      'tenet',
                                                      'block_size',
                                                      'optimizer_flag',
                                                      'lang',
                                                      'cl_kind',
                                                      'device'])

        '''
        Necessary parameters:
        'lang': Given the difference in the unsigned 64 bits types between
        CUDA and OpenCL I need to check the language always
        in order to define the macros/constants properly
        '''        
        for _ in ['lang']:
            if _ not in self.params_dict:
                raise Exception("Missing Parameter \'" + _ + "\'")
                
        if 'kind' not in self.params_dict:
            self.params_dict['kind'] = 'MINSTD'        
            
        if 'init_from' not in self.params_dict:
            self.params_dict['init_from'] = 'numpy'
        elif self.params_dict['init_from'] not in ['numpy', 'urandom']:
            raise Exception("\'init_from\' must be either 'numpy' or 'urandom'")
            
        if 'init_seed' not in self.params_dict:
            self.params_dict['init_seed'] = 1

        if 'n_prngs' not in self.params_dict:
            self.params_dict['n_prngs'] = 1
        if 'optimizer_flag' not in self.params_dict:
            self.optimizer_flag = False
        else:
            self.optimizer_flag = self.params_dict['optimizer_flag']

        '''
        I can pass the tenet from outside, 
        otherwise the class gets it
        '''
        if 'tenet' in self.params_dict:
            self.tenet = self.params_dict['tenet']
        else:
            self.tenet = GetTenet(self.params_dict)

        '''
        Check if OpenCL device supports 64-bits variables
        The flag 'fp64' is set True by default because only OpenCL devices
        might bring up the issue
        '''
        self.params_dict['fp64'] = True
        
        if idpy_langs_sys[OCL_T] and isinstance(self.tenet, idpy_tenet_types[OCL_T]):
            if self.tenet.device.get_info(cl.device_info.DOUBLE_FP_CONFIG) == 0:
                self.params_dict['fp64'] = False
                
                if self.params_dict['kind'] in ['MINSTD', 'MMIX']:
                    print("The device",
                          self.tenet.device.get_info(cl.device_info.NAME),
                          "does not support 64-bits integers needed by",
                          self.params_dict['kind'])
                    print("Switching CRNG kind to 'NUMREC'")
                    self.params_dict['kind'] = 'NUMREC'                    

        IdpySims.__init__(self, *args, **self.kwargs)

    def GridAndBlocks(self):
        '''
        looks pretty general
        '''
        _block_size = None
        if 'block_size' in self.params_dict:
            _block_size = self.params_dict['block_size']
        else:
            _block_size = 128

        _grid = ((self.sims_vars['n_prngs'] + _block_size - 1)//_block_size, 1, 1)
        _block = (_block_size, 1, 1)

        self.sims_vars['grid'], self.sims_vars['block'] = _grid, _block

    def End(self):
        self.tenet.End()

    def InitSeeds(self):
        _swap_seeds = None
        
        if self.sims_vars['init_from'] == 'numpy':
            np.random.seed(self.sims_vars['init_seed'])
            _swap_seeds = np.random.randint(low = 1, high = self.sims_vars['ID_RANDMAX'],
                                            size = self.sims_vars['n_prngs'],
                                            dtype = NPT.C[self.custom_types['CRNGType']])
            print(_swap_seeds.dtype)
            
        elif self.sims_vars['init_from'] == 'urandom':
            _swap_seeds = np.zeros(self.sims_vars['n_prngs'],
                                   dtype = NPT.C[self.custom_types['CRNGType']])
            '''
            Very slow...(?)
            '''
            for _seed_i in range(self.sims_vars['n_prngs']):
                _swap_seeds[_seed_i] = \
                    int.from_bytes(os.urandom(4), byteorder = 'big',
                                   signed = False)

        self.sims_idpy_memory['seeds'] = \
            IdpyMemory.OnDevice(_swap_seeds, tenet = self.tenet)

        del _swap_seeds
        '''
        If 'kind' = MINSTD need to remove most significant bit
        '''
        if self.sims_vars['kind'] == 'MINSTD':
            '''
            Kernel for initializing seeds
            '''
            _K_MINSTDSeedsFit = K_MINSTDSeedsFit(custom_types = self.custom_types.Push(),
                                                 constants = self.constants,
                                                 optimizer_flag = self.optimizer_flag)
            Idea = _K_MINSTDSeedsFit(tenet = self.tenet,
                                     grid = self.sims_vars['grid'],
                                     block = self.sims_vars['block'])

            Idea.Deploy([self.sims_idpy_memory['seeds']])

    def __call__(self, reps = 1):
        _K_NSteps = K_NSteps(custom_types = self.custom_types.Push(),
                             constants = self.constants,
                             optimizer_flag = self.optimizer_flag,
                             f_classes = [self.F_CRNG])

        Idea = _K_NSteps(tenet = self.tenet,
                         grid = self.sims_vars['grid'],
                         block = self.sims_vars['block'])

        Idea.Deploy([self.sims_idpy_memory['seeds'], np.int32(reps)])
        return self.sims_idpy_memory['seeds'].D2H()

    def Norm(self, reps = 1, rand_type = None):
        '''
        Setting the type macro checking device architecture
        '''
        self.constants['F_CRNGFunction'] = 'F_Norm'
        if rand_type is None:
            if self.sims_vars['fp64']:
                self.custom_types.Set({'RANDType': 'double'})
            else:
                self.custom_types.Set({'RANDType': 'float'})
        else:
            if rand_type['RANDType'] == 'double' and not self.sims_vars['fp64']:
                print("The device",
                      self.tenet.device.get_info(cl.device_info.NAME),
                      "custom type RANDType demoted to float")
                self.custom_types.Set({'RANDType': 'float'})
            else:
                self.custom_types.Set(rand_type)
                
        '''
        Check output allocation
        '''
        if self.sims_idpy_memory['output'] is None:
            self.sims_idpy_memory['output'] = \
                IdpyMemory.Zeros(self.sims_vars['n_prngs'], tenet = self.tenet,
                                 dtype = NPT.C[self.custom_types['RANDType']])
        '''
        Init the generic kernel
        '''
        _K_OutputFunction = \
            K_OutputFunction(custom_types = self.custom_types.Push(),
                             constants = self.constants,
                             optimizer_flag = self.optimizer_flag,
                             f_classes = [self.F_CRNG, F_Norm])

        
        Idea = _K_OutputFunction(tenet = self.tenet,
                                 grid = self.sims_vars['grid'],
                                 block = self.sims_vars['block'])

        Idea.Deploy([self.sims_idpy_memory['output'],
                     self.sims_idpy_memory['seeds'],
                     np.int32(reps)])

        return self.sims_idpy_memory['output'].D2H()
                
        
'''
Generic Kernels
'''
class K_NSteps(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'CRNGType * seeds': ['global', 'restrict'],
                       'int reps': ['const']}
        self.kernels[IDPY_T] = """ 
        if(g_tid < N_PRNGS){
            CRNGType l_seed = seeds[g_tid];
            for(int i=0; i<reps; i++){
                F_CRNG(&l_seed);
            }
            seeds[g_tid] = l_seed;
        }
        """

class K_OutputFunction(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'RANDType * output': ['global', 'restrict'],
                       'CRNGType * seeds': ['global', 'restrict'],
                       'int reps': ['const']}
        self.kernels[IDPY_T] = """ 
        if(g_tid < N_PRNGS){
            CRNGType l_seed = seeds[g_tid];
            for(int i=0; i<reps; i++){
               output[g_tid] = F_CRNGFunction(&l_seed);
            }
            seeds[g_tid] = l_seed;
        }
        """

'''
Generic device functions:
need to add:
- Normalized (0, 1]
- Gaussian
- Integers in an interval
The idea is to parametrize the device function
call by using a macro
'''
class F_Norm(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': []}
        self.functions[IDPY_T] = """
        F_CRNG(l_seed);
        return ((RANDType) (*l_seed)) / (ID_RANDMAX - 1);
        """        

class F_Gaussian(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': [],
                       'RANDType mean': ['const'], 'RANDType var': ['const']}
        self.functions[IDPY_T] = """
        F_CRNG(l_seed);
        return ((RANDType) *l_seed) / ((RANDType) (ID_RANDMAX - 1));
        """

class F_Integers(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': [],
                       'RANDType mean': ['const'], 'RANDType var': ['const']}
        self.functions[IDPY_T] = """
        F_CRNG(l_seed);
        return ((RANDType) *l_seed) / ((RANDType) (ID_RANDMAX - 1));
        """

'''
With F_CustomHITORMISS the idea is to be able to generate any probability
distribution by brute force hit-or-miss which should be fairly
parametrizable in terms of a specific device function, depending on a few
external parameters set through macros, that can be passed to this one
'''
class F_CustomHITORMISS(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': [],
                       'RANDType mean': ['const'], 'RANDType var': ['const']}
        self.functions[IDPY_T] = """
        F_CRNG(l_seed);
        return ((RANDType) *l_seed) / ((RANDType) (ID_RANDMAX - 1));
        """
    
'''
Specific Device Functions
'''
class F_MINSTD(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'void'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        '''
        l_seed: local seed
        the idea is to manually read and write the seed from the global memory
        at the beginning and at the end of each kernel so that internal loops
        can take advantage of it rather than reapting useless storage operations
        '''
        self.params = {'CRNGType * l_seed': []}
        
        self.functions[IDPY_T] = """
        UINT64 swap = (16807LL) * (*l_seed);
        *l_seed = (swap & 0x7fffffff) + (swap >> 31);
        if((*l_seed) & 0x80000000) *l_seed = ((*l_seed) & 0x70000000) + 1;
        return;
        """

class F_NUMREC(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'void'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        '''
        l_seed: local seed
        the idea is to manually read and write the seed from the global memory
        at the beginning and at the end of each kernel so that internal loops
        can take advantage of it rather than reapting useless storage operations
        '''
        self.params = {'CRNGType * l_seed': []}
        
        self.functions[IDPY_T] = """
        *l_seed = (*l_seed) * 1664525U + 1013904223U;
        return;
        """

class F_MMIX(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'void'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        '''
        l_seed: local seed
        the idea is to manually read and write the seed from the global memory
        at the beginning and at the end of each kernel so that internal loops
        can take advantage of it rather than reapting useless storage operations
        '''
        self.params = {'CRNGType * l_seed': []}
        
        self.functions[IDPY_T] = """
        *l_seed = (*l_seed) * 6364136223846793005LLU + 1442695040888963407LLU;
        return;
        """

'''
MINSTD Seeds Kernel
'''
class K_MINSTDSeedsFit(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'CRNGType * seeds': ['global', 'restrict']}
        self.kernels[IDPY_T] = """ 
        if(g_tid < N_PRNGS){
            seeds[g_tid] = (seeds[g_tid] & ID_RANDMAX);
        }
        """
