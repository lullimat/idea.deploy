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
import sympy as sp
import os

from idpy.Utils.NpTypes import NpTypes
from idpy.Utils.CustomTypes import CustomTypes

from idpy.IdpyCode import GetParamsClean, GetTenet
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyCode import CUDA_T, OCL_T, CTYPES_T, IDPY_T
from idpy.IdpyCode import idpy_langs_sys, idpy_tenet_types

from idpy.IdpyCode.IdpyUnroll import _codify_newl, _codify_comment
from idpy.IdpyCode.IdpyUnroll import _codify_assignment, _codify_add_assignment
from idpy.IdpyCode.IdpyUnroll import _codify_declaration_const_check
from idpy.IdpyCode.IdpyUnroll import _check_declared_variables_constants
from idpy.IdpyCode.IdpyUnroll import _check_needed_variables_constants
from idpy.IdpyCode.IdpyUnroll import _check_lambda_args, _array_value
from idpy.IdpyCode.IdpyUnroll import _codify_assignments, _codify_assignment_type_check

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
        Setting number of streams
        '''
        if 'n_streams' not in self.params_dict:
            self.sims_vars['n_streams'] = 1
        else:
            self.sims_vars['n_streams'] = self.params_dict['n_streams']
            
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
        if self.sims_vars['kind'] == 'MINSTD' or self.sims_vars['kind'] == 'NUMREC':
            if self.params_dict['lang'] == CUDA_T or self.params_dict['lang'] == CTYPES_T:
                self.custom_types = CustomTypes({'CRNGType': 'unsigned int',
                                                 'UINT64': 'unsigned long long int'})
            if self.params_dict['lang'] == OCL_T:
                self.custom_types = CustomTypes({'CRNGType': 'unsigned int',
                                                 'UINT64': 'unsigned long'})                
        """
        - To be deleted...
        if self.sims_vars['kind'] == 'NUMREC':
            self.custom_types = CustomTypes({'CRNGType': 'unsigned int'})
        """
        if self.sims_vars['kind'] == 'MMIX':
            if self.params_dict['lang'] == CUDA_T or self.params_dict['lang'] == CTYPES_T:
                self.custom_types = CustomTypes({'CRNGType': 'unsigned long long int'})
            if self.params_dict['lang'] == OCL_T:
                self.custom_types = CustomTypes({'CRNGType': 'unsigned long'})

        '''
        Setting ID_RANDMAX
        '''        
        if self.sims_vars['kind'] == 'MINSTD':
            self.sims_vars['ID_RANDMAX'] = int('7fffffff', 16)
            self.sims_vars['ID_RANDMAX_STR'] = 'ID_RANDMAX_MINSTD'             
        if self.sims_vars['kind'] == 'NUMREC':
            self.sims_vars['ID_RANDMAX'] = int('ffffffff', 16)
            self.sims_vars['ID_RANDMAX_STR'] = 'ID_RANDMAX_NUMREC'             
        if self.sims_vars['kind'] == 'MMIX':
            self.sims_vars['ID_RANDMAX'] = int('ffffffffffffffff', 16)
            self.sims_vars['ID_RANDMAX_STR'] = 'ID_RANDMAX_MMIX'            
        
        self.sims_vars['init_from'] = self.params_dict['init_from']
        self.sims_vars['init_seed'] = self.params_dict['init_seed']
        self.sims_vars['n_prngs'] = self.params_dict['n_prngs']

        self.constants = {'ID_RANDMAX': self.sims_vars['ID_RANDMAX'], 
                          self.sims_vars['ID_RANDMAX_STR']: self.sims_vars['ID_RANDMAX'],
                          'N_PRNGS': self.sims_vars['n_prngs'],
                          'TWOPI': 2. * np.pi}
        '''
        Setting device function CRNG through a macro
        '''
        if self.sims_vars['kind'] == 'MINSTD':
            if self.params_dict['fp64']:
                self.F_CRNG = F_MINSTD
                self.constants['F_CRNG'] = 'F_MINSTD'
            else:
                self.F_CRNG = F_MINSTD32
                self.constants['F_CRNG'] = 'F_MINSTD32'
                
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
                                                      'n_streams',
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
            self.optimizer_flag = True
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
                
                if self.params_dict['kind'] in ['MMIX']:
                    print("The device",
                          self.tenet.device.get_info(cl.device_info.NAME),
                          "does not support 64-bits integers needed by",
                          self.params_dict['kind'])
                    print("Switching CRNG kind to 'MINSTD'")
                    self.params_dict['kind'] = 'MINSTD'                    

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

        _n_streams = self.sims_vars['n_streams']
        _idpy_memory_swap = None
        for _i in range(_n_streams):
            if self.sims_vars['init_from'] == 'numpy':
                np.random.seed(self.sims_vars['init_seed'] + _i)
                _swap_seeds = np.random.randint(low = 1, high = self.sims_vars['ID_RANDMAX'],
                                                size = self.sims_vars['n_prngs'],
                                                dtype = NPT.C[self.custom_types['CRNGType']])

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


            _idpy_memory_swap = \
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

                Idea.Deploy([_idpy_memory_swap])

            if _n_streams > 1:
                self.sims_idpy_memory['seeds_' + str(_i)] = _idpy_memory_swap
            else:
                self.sims_idpy_memory['seeds'] = _idpy_memory_swap
                

    def __call__(self, reps = 1):
        _K_NSteps = K_NSteps(custom_types = self.custom_types.Push(),
                             constants = self.constants,
                             optimizer_flag = self.optimizer_flag,
                             f_classes = [self.F_CRNG])

        Idea = _K_NSteps(tenet = self.tenet,
                         grid = self.sims_vars['grid'],
                         block = self.sims_vars['block'])

        _n_streams = self.sims_vars['n_streams']
        if _n_streams > 1:
            for _i in range(_n_streams):            
                Idea.Deploy([self.sims_idpy_memory['seeds_' + str(_i)], np.int32(reps)])
                
            return np.array([self.sims_idpy_memory['seeds_' + str(_i)].D2H()
                             for _i in range(_n_streams)])
        else:
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
        type_str = self.custom_types.Push()['RANDType']
        output_name = 'output_' + type_str
        if output_name not in self.sims_idpy_memory:
            self.sims_idpy_memory[output_name] = \
                IdpyMemory.Zeros(self.sims_vars['n_prngs'], tenet = self.tenet,
                                 dtype = NPT.C[self.custom_types['RANDType']])
        self.sims_idpy_memory['output'] = self.sims_idpy_memory[output_name]
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

        _n_streams = self.sims_vars['n_streams']
        if _n_streams > 1:
            _outputs = []
            for _i in range(_n_streams):
                Idea.Deploy([self.sims_idpy_memory['output'],
                             self.sims_idpy_memory['seeds_' + str(_i)],
                             np.int32(reps)])
                _outputs += [np.copy(self.sims_idpy_memory['output'].D2H())]
                
            return np.array(_outputs)
        else:
            Idea.Deploy([self.sims_idpy_memory['output'],
                         self.sims_idpy_memory['seeds'],
                         np.int32(reps)])            
            return self.sims_idpy_memory['output'].D2H()


    def GaussianSingle(self, mean = 0, var = 1, reps = 1, rand_type = None):
        '''
        Setting the type macro checking device architecture
        '''
        self.constants['F_CRNGGaussFunction'] = 'F_GaussianCosSingle'
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
        type_str = self.custom_types.Push()['RANDType']
        output_name = 'output_' + type_str
        if output_name not in self.sims_idpy_memory:
            self.sims_idpy_memory[output_name] = \
                IdpyMemory.Zeros(self.sims_vars['n_prngs'], tenet = self.tenet,
                                 dtype = NPT.C[self.custom_types['RANDType']])
        self.sims_idpy_memory['output'] = self.sims_idpy_memory[output_name]
        '''
        Init the generic kernel
        '''
        _K_OutputGaussianSingle = \
            K_OutputGaussianSingle(custom_types = self.custom_types.Push(),
                                   constants = self.constants,
                                   optimizer_flag = self.optimizer_flag,
                                   f_classes = [self.F_CRNG, F_Norm,
                                                F_GaussianCosSingle])

        
        Idea = _K_OutputGaussianSingle(tenet = self.tenet,
                                       grid = self.sims_vars['grid'],
                                       block = self.sims_vars['block'])

        _n_streams = self.sims_vars['n_streams']
        _mean = NPT.C[self.custom_types['RANDType']](mean)
        _var = NPT.C[self.custom_types['RANDType']](var)
        
        if _n_streams > 1:
            _outputs = []
            for _i in range(_n_streams):
                Idea.Deploy([self.sims_idpy_memory['output'],
                             self.sims_idpy_memory['seeds_' + str(_i)],
                             _mean, _var, np.int32(reps)])
                _outputs += [np.copy(self.sims_idpy_memory['output'].D2H())]
                
            return np.array(_outputs)
        else:
            Idea.Deploy([self.sims_idpy_memory['output'],
                         self.sims_idpy_memory['seeds'],
                         _mean, _var, np.int32(reps)])            
            return self.sims_idpy_memory['output'].D2H()

        
    def Gaussian(self, mean = 0, var = 1, reps = 1, rand_type = None):
        if self.sims_vars['n_streams'] % 2 != 0:
            raise Exception("'n_streams' must be 2!")
        '''
        Setting the type macro checking device architecture
        '''
        self.constants['F_CRNGGaussFunction'] = 'F_GaussianCos'
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
        type_str = self.custom_types.Push()['RANDType']
        output_name = 'output_' + type_str
        if output_name not in self.sims_idpy_memory:
            self.sims_idpy_memory[output_name] = \
                IdpyMemory.Zeros(self.sims_vars['n_prngs'], tenet = self.tenet,
                                 dtype = NPT.C[self.custom_types['RANDType']])
        self.sims_idpy_memory['output'] = self.sims_idpy_memory[output_name]
        '''
        Init the generic kernel
        '''
        _K_OutputGaussian = \
            K_OutputGaussian(custom_types = self.custom_types.Push(),
                             constants = self.constants,
                             optimizer_flag = self.optimizer_flag,
                             f_classes = [self.F_CRNG, F_Norm, F_GaussianCos])

        
        Idea = _K_OutputGaussian(tenet = self.tenet,
                                 grid = self.sims_vars['grid'],
                                 block = self.sims_vars['block'])

        _mean = NPT.C[self.custom_types['RANDType']](mean)
        _var = NPT.C[self.custom_types['RANDType']](var)

        _n_streams, _outputs = self.sims_vars['n_streams'], []
        for _i in range(0, _n_streams, 2):
            Idea.Deploy([self.sims_idpy_memory['output'],
                         self.sims_idpy_memory['seeds_' + str(_i)],
                         self.sims_idpy_memory['seeds_' + str(_i + 1)],
                         _mean, _var,
                         np.int32(reps)])
            _outputs += [np.copy(self.sims_idpy_memory['output'].D2H())]
            
        return np.array(_outputs)


    def Integers(
        self, reps = 1, int_range=1, 
        rand_type={'RANDType': 'unsigned int'}):

        if self.sims_vars['kind'] == 'MMIX':
            raise Exception("This method only works with 32-bits generators")

        self.custom_types.Set(rand_type)

        '''
        Setting the type macro checking device architecture
        '''
        self.constants['F_CRNGFunction'] = 'F_RandomIntegerUnbiasedLemireMACRO' 
        self.constants['CRNG_Integers_range'] = int_range              
        '''
        Check output allocation
        '''
        type_str = self.custom_types.Push()['RANDType']
        output_name = 'output_' + type_str
        if output_name not in self.sims_idpy_memory:
            self.sims_idpy_memory[output_name] = \
                IdpyMemory.Zeros(self.sims_vars['n_prngs'], tenet = self.tenet,
                                 dtype = NPT.C[self.custom_types['RANDType']])

        self.sims_idpy_memory['output'] = self.sims_idpy_memory[output_name]
        '''
        Init the generic kernel
        '''
        _K_OutputFunction = \
            K_OutputFunction(custom_types = self.custom_types.Push(),
                             constants = self.constants,
                             optimizer_flag = self.optimizer_flag,
                             f_classes = [self.F_CRNG, F_RandomIntegerUnbiasedLemireMACRO])

        
        Idea = _K_OutputFunction(tenet = self.tenet,
                                 grid = self.sims_vars['grid'],
                                 block = self.sims_vars['block'])

        _n_streams = self.sims_vars['n_streams']
        if _n_streams > 1:
            _outputs = []
            for _i in range(_n_streams):
                Idea.Deploy([self.sims_idpy_memory['output'],
                             self.sims_idpy_memory['seeds_' + str(_i)],
                             np.int32(reps)])
                _outputs += [np.copy(self.sims_idpy_memory['output'].D2H())]
                
            return np.array(_outputs)
        else:
            Idea.Deploy([self.sims_idpy_memory['output'],
                         self.sims_idpy_memory['seeds'],
                         np.int32(reps)])
            return self.sims_idpy_memory['output'].D2H()

'''
Meta-Programming functions
'''
def _codify_declare_MINSTD32_swap(declared_variables = None, declared_constants = None,
                                  root_hi = 'hi', root_lo = 'lo'):
    _check_declared_variables_constants(declared_variables, declared_constants)
    _swap_code = """"""
    _swap_code += _codify_comment("Possibly declaring hi & lo variables for MINSTD 32-bits")
    _swap_code += \
        _codify_declaration_const_check(root_hi, 0, 'CRNGType',
                                        declared_variables,
                                        declared_constants)
    _swap_code += \
        _codify_declaration_const_check(root_lo, 0, 'CRNGType',
                                        declared_variables,
                                        declared_constants)
    _swap_code += _codify_newl

    return _swap_code

def _codify_MINSTD32(declared_variables = None, declared_constants = None,
                     root_seed = 'lseed', root_hi = 'hi', root_lo = 'lo'):
    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_needed_variables_constants([root_seed], declared_variables, declared_constants)

    _swap_code = """"""
    _swap_code += \
        _codify_declare_MINSTD32_swap(
            declared_variables = declared_variables,
            declared_constants = declared_constants,
            root_hi = root_hi, root_lo = root_lo
        )

    _swap_code += _codify_comment("MINSTD PRNG; 32-bits implementation")
    _swap_code += _codify_assignment(root_lo,
                                     '16807*(' + root_seed + '&0xffff)')
    _swap_code += _codify_assignment(root_hi,
                                     '16807*(' + root_seed + '>>16)')
    _swap_code += _codify_add_assignment(root_lo, '(' + root_hi + '&0x7fff)<<16')
    _swap_code += _codify_add_assignment(root_lo, root_hi + '>>15')

    if False:
        _swap_code += \
            _codify_assignment(
                root_seed, '((lo & 0x80000000)>>31) * (lo - 0x7fffffff)'
            )
        _swap_code += \
            _codify_add_assignment(
                root_seed, '((lo ^ 0x80000000)>>31) * lo'
            )

    if True:
        _swap_code += \
            _codify_assignment(
                root_seed,
                root_lo + ' - ((-((' + root_lo + '&0x80000000)>>31))&0x7fffffff)'
            )
        _swap_code += _codify_newl

    return _swap_code

def _codify_declare_MINSTD_swap(declared_variables = None, declared_constants = None,
                                root_swap = 'minstd_swap'):
    _check_declared_variables_constants(declared_variables, declared_constants)    
    _swap_code = """"""
    _swap_code += _codify_comment("Possibly declaring swap variable for MINSTD 64-bits")
    _swap_code += \
        _codify_declaration_const_check(root_swap, 0, 'UINT64',
                                        declared_variables,
                                        declared_constants)
    _swap_code += _codify_newl

    return _swap_code

def _codify_MINSTD(declared_variables = None, declared_constants = None,
                   root_seed = 'lseed', root_swap = 'minstd_swap'):
    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_needed_variables_constants([root_seed], declared_variables, declared_constants)
    
    _swap_code = """"""
    _swap_code += \
        _codify_declare_MINSTD_swap(
            declared_variables = declared_variables,
            declared_constants = declared_constants,
            root_swap = root_swap
        )

    _swap_code += _codify_comment("MINSTD PRNG; 64-bits implementation")        
    _swap_code += _codify_assignment(root_swap, '(16807LL) * ' + root_seed)
    _swap_code += \
        _codify_assignment(
            root_seed,
            '(' + root_swap + '&0x7fffffff) + (' + root_swap + '>>31)'
        )

    _swap_code += \
        'if(' + root_seed + '&0x80000000){' + _codify_newl + \
        _codify_assignment(
            root_seed, '(' + root_seed + '&0x7fffffff) + 1'
        ) + _codify_newl + \
        '}'
    _swap_code += _codify_newl

    return _swap_code

def _codify_NUMREC(declared_variables = None, declared_constants = None,
                   root_seed = 'lseed'):
    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_needed_variables_constants([root_seed], declared_variables, declared_constants)

    _swap_code = """"""
    _swap_code += _codify_comment("Numerical Recipes PRNG; 32-bits implementation")    
    _swap_code += _codify_assignment(root_seed, root_seed + ' * 1664525U + 1013904223U')
    _swap_code += _codify_newl

    return _swap_code

def _codify_MMIX(declared_variables = None, declared_constants = None,
                 root_seed = 'lseed'):
    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_needed_variables_constants([root_seed], declared_variables, declared_constants)

    _swap_code = """"""
    _swap_code += _codify_comment("MMIX PRNG; 64-bits implementation")    
    _swap_code += \
        _codify_assignment(
            root_seed,
            root_seed + ' * 6364136223846793005 + 1442695040888963407'
        )
    _swap_code += _codify_newl

    return _swap_code

_codify_CRNGS = {'MMIX': _codify_MMIX, 'NUMREC': _codify_NUMREC,
                 'MINSTD': _codify_MINSTD, 'MINSTD32': _codify_MINSTD32}
_codify_CRNGS_list = list(_codify_CRNGS.keys())

def _codify_flat(declared_variables = None, declared_constants = None,
                 root_seed = 'lseed', root_flat = 'lflat',
                 crng_kind = None, rand_type = 'RANDType',
                 const_out = False):
    
    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_needed_variables_constants([root_seed, 'ID_RANDMAX_' + str(crng_kind)],
                                      declared_variables, declared_constants)
    if crng_kind is None:
        raise Exception("Missing argument 'crng_kind' !")
    if crng_kind not in _codify_CRNGS_list:
        raise Exception("Argument 'crng_kind' must be in the list", _codify_CRNGS_list)
    
    _swap_code = """"""
    _swap_code += _codify_comment("Generating flat distribution random number")
    _swap_code += _codify_newl
    _swap_code += _codify_comment("Invoking the PRNG")
    _swap_code += \
        _codify_CRNGS[crng_kind](
            declared_variables = declared_variables,
            declared_constants = declared_constants,
            root_seed = root_seed
        )
    _swap_code += _codify_comment("Possibly declaring swap flat variable")
    _swap_code += \
        _codify_declaration_const_check(
            root_flat,
            '((' + rand_type + ')' + root_seed + ' / (ID_RANDMAX_' + str(crng_kind) + '))',
            rand_type,
            declared_variables,
            declared_constants,
            const_out
        )
    _swap_code += _codify_newl
    
    return _swap_code

def _codify_flat_integers_biased(
        declared_variables = None, declared_constants = None, 
        min_int = None, max_int = None,
        root_seed = 'lseed', root_int = 'rand_int', assignment_type = None, 
        crng_kind = None, rand_type = 'RANDType', const_out = False):

    _check_declared_variables_constants(declared_variables, declared_constants)
    """
    _check_needed_variables_constants([root_seed, 'ID_RANDMAX_' + str(crng_kind)],
                                      declared_variables, declared_constants)
    """

    if crng_kind is None:
        raise Exception("Missing argument 'crng_kind' !")
    if crng_kind not in _codify_CRNGS_list:
        raise Exception("Argument 'crng_kind' must be in the list", _codify_CRNGS_list)

    if min_int is None or max_int is None:
        raise Exception("Missing argument(s) 'max_int', 'min_int'")

    _swap_code = """"""
    _swap_code += _codify_comment("Generating an integer in the range [min_int, max_int - 1]")
    _swap_code += _codify_comment("WARNING: This method will yield a biased result!")
    _swap_code += _codify_comment("Invoking the PRNG")
    _swap_code += \
        _codify_CRNGS[crng_kind](
            declared_variables=declared_variables, 
            declared_constants=declared_constants, 
            root_seed=root_seed
            )
    _swap_code += _codify_comment("Generating an integer in the range [0,max_int-min_int-1] and then shift")
    _swap_code += \
        _codify_assignment_type_check(
            root_int, str(min_int) + " + " + root_seed + 
            " % (" + str(max_int) + "-" + str(min_int) + ")",
            rand_type,
            declared_variables,
            declared_constants,
            const_out,
            assignment_type            
            )

    return _swap_code

## Following the implementation from
## https://www.pcg-random.org/posts/bounded-rands.html
def _codify_flat_integers_unbiased(
        declared_variables = None, declared_constants = None, 
        min_int = None, max_int = None,
        root_seed = 'lseed', root_int = 'rand_int', assignment_type = None, 
        crng_kind = 'NUMREC', rand_type = 'RANDType', const_out = False):

    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_needed_variables_constants([root_seed, 'ID_RANDMAX'],
                                      declared_variables, declared_constants)

    if crng_kind is None:
        raise Exception("Missing argument 'crng_kind' !")
    if crng_kind not in _codify_CRNGS_list:
        raise Exception("Argument 'crng_kind' must be in the list", _codify_CRNGS_list)

    if min_int is None or max_int is None:
        raise Exception("Missing argument(s) 'max_int', 'min_int'")

    _swap_code = """"""
    _swap_code += _codify_comment("This is described as the Lemire's method: https://www.pcg-random.org/posts/bounded-rands.html")
    _swap_code += _codify_comment("Generating an integer in the range [min_int, max_int - 1]")
    _swap_code += _codify_comment("1. Invoke the PRNG")
    _swap_code += \
        _codify_CRNGS[crng_kind](
            declared_variables=declared_variables, 
            declared_constants=declared_constants, 
            root_seed=root_seed
            )
    _swap_code += _codify_comment("2. Multiply the random number by the range")
    _swap_code += \
        _codify_declaration_const_check(
                'm', 
                '((UINT64)(' + root_seed + ')) * \
                 ((UINT64)(' + str(max_int) + ' - ' + str(min_int) + '))', 
                _type = 'UINT64',
                declared_variables = None,
                declared_constants = None,
                declare_const_flag = False
            )

    _swap_code += _codify_comment("Generating an integer in the range [0,max_int-min_int-1] and then shift")
    _swap_code += \
        _codify_assignment_type_check(
            "t", 
            "-(RANDType)(max_int-min_int-1) % ",
            rand_type,
            declared_variables,
            declared_constants,
            const_out,
            assignment_type        
            )

    return _swap_code

def _codify_flat_mean_var(declared_variables = None, declared_constants = None,
                          root_seed = 'lseed', root_flat = 'lflat',
                          assignment_type = None,
                          crng_kind = None, rand_type = 'RANDType',
                          mean = sp.Rational(1, 2), var = sp.Rational(1, 12),
                          const_out = False):
    
    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_needed_variables_constants([root_seed, 'ID_RANDMAX_' + str(crng_kind)],
                                      declared_variables, declared_constants)
    if crng_kind is None:
        raise Exception("Missing argument 'crng_kind' !")
    if crng_kind not in _codify_CRNGS_list:
        raise Exception("Argument 'crng_kind' must be in the list", _codify_CRNGS_list)
    
    _swap_code = """"""
    _swap_code += _codify_comment("Generating flat distribution random number")
    _swap_code += _codify_comment("Possibly declaring swap flat variable")
    _swap_code += _codify_comment("Invoking the PRNG")
    _swap_code += \
        _codify_CRNGS[crng_kind](
            declared_variables = declared_variables,
            declared_constants = declared_constants,
            root_seed = root_seed
        )
    _swap_code += _codify_comment("Generating flat distribution variable")

    _shift = (-sp.Rational(1, 2) + mean).evalf()
    _delta = (sp.sqrt(12 * var)).evalf()

    _swap_code += \
        _codify_assignment_type_check(
            root_flat,
            '((' + rand_type + ')' + root_seed + ' / (ID_RANDMAX_' + str(crng_kind) +') + (' + str(_shift) + \
            ')) * ' + str(_delta),
            rand_type,
            declared_variables,
            declared_constants,
            const_out,
            assignment_type
        )
    _swap_code += _codify_newl        
    
    return _swap_code

'''
Generating one Gaussian number from two flat numbers belonging to the same sequence
'''
def _codify_gaussian_seq(declared_variables = None, declared_constants = None,
                         mean = 0, var = 1, which_box_m = 'cos',root_seed = 'lseed',
                         root_gauss = 'lgauss', assignment_type = None,
                         root_flat = 'lflat', crng_kind = None, rand_type = 'RANDType',
                         const_out = False):

    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_needed_variables_constants([root_seed, 'TWOPI'],
                                      declared_variables, declared_constants)
    
    if crng_kind is None:
        raise Exception("Missing argument 'crng_kind' !")
    if crng_kind not in ['MINSTD', 'MINSTD32', 'NUMREC', 'MMIX']:
        raise Exception(
            "Argument 'crng_kind' must be in the list ['MINSTD', 'MINSTD32', 'NUMREC', 'MMIX']"
        )

    if which_box_m not in ['cos', 'sin']:
        raise Exception("Argument 'which_box_m' must be in ['cos', 'sin']")

    _swap_code = """"""
    _swap_code += \
        _codify_comment("One Gaussian pseudo-random number from a single uniform distribution")
    for _i in range(2):
        _swap_code += \
            _codify_flat(declared_variables = declared_variables,
                         declared_constants = declared_constants,
                         root_seed = root_seed, root_flat = root_flat + '_' + str(_i),
                         crng_kind = crng_kind, rand_type = rand_type)

    _swap_code += _codify_comment("Possibly declaring swap gaussian variable")    
    _swap_code += \
        _codify_comment("Applying Box-Mueller Transform")

    if which_box_m == 'cos':
        _swap_code += \
            _codify_assignment_type_check(
                root_gauss,
                'sqrt((' + rand_type + ')(-2. * ' + str(var) +
                ' * log((' + rand_type + ') ' +
                root_flat + '_0))) * cos((' + rand_type + ')(TWOPI * ' +
                root_flat + '_1)) + ' + str(mean),
                rand_type,
                declared_variables,
                declared_constants,
                const_out,
                assignment_type
            )
    else:
        _swap_code += \
            _codify_assignment_type_check(
                root_gauss,
                'sqrt((' + rand_type + ')(-2. * ' + str(var) +
                ' * log((' + rand_type + ') ' +
                root_flat + '_0))) * sin((' + rand_type + ')(TWOPI * ' +
                root_flat + '_1)) + ' + str(mean),
                rand_type,
                declared_variables,
                declared_constants,
                const_out,
                assignment_type
            )
        
    return _swap_code

'''
Generating two Gaussian numbers from two flat numbers from two independent sequences
'''
def _codify_gaussian_p(declared_variables = None, declared_constants = None,
                       mean = 0, var = 1,
                       root_seeds = ['lseed_0', 'lseed_1'],
                       root_gauss = 'lgauss', assignment_type = None,
                       root_flat = 'lflat', crng_kind = None, rand_type = 'RANDType',
                       const_out = False, single_out = True):
    
    _check_declared_variables_constants(declared_variables, declared_constants)

    _check_needed_variables_constants(root_seeds + ['TWOPI'],
                                      declared_variables, declared_constants)
    
    if crng_kind is None:
        raise Exception("Missing argument 'crng_kind' !")
    if crng_kind not in ['MINSTD', 'MINSTD32', 'NUMREC', 'MMIX']:
        raise Exception(
            "Argument 'crng_kind' must be in the list ['MINSTD', 'MINSTD32', 'NUMREC', 'MMIX']"
        )

    _swap_code = """"""
    _swap_code += \
        _codify_comment("Two Gaussian pseudo-random number from two indep. uniform distributions")
    for _i, _seed in enumerate(root_seeds):
        _swap_code += \
            _codify_flat(declared_variables = declared_variables,
                         declared_constants = declared_constants,
                         root_seed = _seed, root_flat = root_flat + '_' + str(_i),
                         crng_kind = crng_kind, rand_type = rand_type)

    _swap_code += _codify_comment("Possibly declaring swap gaussian variable")    
    _swap_code += \
        _codify_comment("Applying Box-Mueller Transform")

    _swap_code += \
        _codify_assignment_type_check(
            root_gauss + ('_0' if not single_out else ''),
            'sqrt((' + rand_type + ')(-2. * ' + str(var) +
            ' * log((' + rand_type + ') ' +
            root_flat + '_0))) * cos((' + rand_type + ')(TWOPI * ' +
            root_flat + '_1)) + ' + str(mean),
            rand_type,
            declared_variables,
            declared_constants,
            const_out,
            assignment_type
        )

    if not single_out:
        _swap_code += \
            _codify_assignment_type_check(
                root_gauss + '_1',
                'sqrt((' + rand_type + ')(-2. * ' + str(var) +
                ' * log((' + rand_type + ') ' +
                root_flat + '_0))) * sin((' + rand_type + ')(TWOPI * ' +
                root_flat + '_1)) + ' + str(mean),
                rand_type,
                declared_variables,
                declared_constants,
                const_out,
                assignment_type                
            )
        
    return _swap_code

'''
Function implementing the reads from an array:
the lambda_ordering should only have one argument: need to 'prune' a more complex
lambda before passing
'''
def _codify_read_seeds(declared_variables = None, declared_constants = None,
                       seeds_array = 'seeds_array', root_seed = 'lseed',
                       n_reads = 1, lambda_ordering = None, use_ptrs = False):

    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_lambda_args(1, lambda_ordering)
    _check_needed_variables_constants([seeds_array],
                                      declared_variables, declared_constants)
    _swap_code = """"""

    for _i in range(n_reads):
        _swap_code += \
            _codify_declaration_const_check(
                root_seed + '_' + str(_i),
                _array_value(seeds_array, lambda_ordering(_i), use_ptrs),
                'CRNGType',
                declared_variables,
                declared_constants
            )            
    
    return _swap_code
'''
Function implementing the reads from an array:
the lambda_ordering should only have one argument: need to 'prune' a more complex
lambda before passing
'''    
def _codify_write_seeds(declared_variables = None, declared_constants = None,
                        seeds_array = 'seeds_array', root_seed = 'lseed',
                        n_writes = 1, lambda_ordering = None, use_ptrs = False):

    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_lambda_args(1, lambda_ordering)
    _check_needed_variables_constants([seeds_array],
                                      declared_variables, declared_constants)
    _swap_code = """"""

    for _i in range(n_writes):
        _swap_code += \
            _codify_assignment(
                _array_value(seeds_array, lambda_ordering(_i), use_ptrs),
                root_seed + '_' + str(_i)
            )            
    
    return _swap_code

'''
This method should declare all that is needed and check for the seeds array
In particular I can use this method no matter the approach if I just parametrize it
in terms of the number of random variables that I need.
How the random numbers are used is method-specific.
Hence, I need to pass the variances
The desired random number can be constants by default
The specified 'rand_vars_type' must be compatible with 'RANDType'
'''
def _codify_n_random_vars(
        declared_variables = None, declared_constants = None,
        seeds_array = 'seeds_array', root_seed = 'lseed',
        rand_vars = None, rand_vars_type = 'RANDType',
        assignment_type = None,
        lambda_ordering = None, use_ptrs = False,
        variances = None, means = None, distribution = 'flat',
        generator = 'MINSTD32', parallel_streams = 1,
        output_const = True, which_box_m = None, 
        read_seeds_flag=True, write_seeds_flag=True
):    
    if distribution not in ['flat', 'gaussian']:
        raise Exception("Parameter 'distribution' must be in ['flat', 'gaussian']")

    if variances is None:
        raise Exception("Missing argument 'variances'")
    if type(variances) != list:
        raise Exception("Argument 'variances' must be a list")

    if means is None:
        raise Exception("Missing argument 'means'")
    if type(means) != list:
        raise Exception("Argument 'means' must be a list")
    
    if rand_vars is None:
        raise Exception("Missing argument 'rand_vars'")
    if type(rand_vars) != list:
        raise Exception("Argument 'rand_vars' must be a list")
    _n_rand_vars = len(rand_vars)

    if len(variances) != _n_rand_vars:
        raise Exception("Number of variances and random variables must coincide")

    '''
    Need to check which is the best option and possibly change it as the default one
    '''
    if parallel_streams > 1:
        if distribution == 'flat' and parallel_streams < _n_rand_vars:
            raise Exception("Not enough streams for 'flat' distribution: expected", _n_rand_vars)
        if distribution == 'gaussian' and parallel_streams < _n_rand_vars:
            raise Exception("Not enough streams for 'gaussian' distribution: expected at least", _n_rand_vars)

    if generator not in _codify_CRNGS_list:
        raise Exception("Parameter 'generator' must be in", _codify_CRNGS_list)

    if lambda_ordering is None:
        raise Exception("Missing paramter 'lambda_ordering'")

    '''
    Setting the flag for the generation of Gaussian numbers using double of single stream each
    '''
    _gaussian_parallel = None
    if distribution == 'gaussian':
        _gaussian_parallel = True if parallel_streams == 2 * _n_rand_vars else False
        if which_box_m is None and not _gaussian_parallel:
            which_box_m = 'cos'
    
    _check_declared_variables_constants(declared_variables, declared_constants)
    _check_needed_variables_constants([seeds_array],
                                      declared_variables, declared_constants)

    _swap_code = """"""

    if read_seeds_flag:
        _swap_code += _codify_comment("Reading prng seeds")
        _swap_code += \
            _codify_read_seeds(
                declared_variables = declared_variables,
                declared_constants = declared_constants,
                seeds_array = seeds_array, root_seed = root_seed,
                n_reads = parallel_streams,
                lambda_ordering = lambda_ordering,
                use_ptrs = use_ptrs
            )
        _swap_code += _codify_newl
        _swap_code += _codify_newl    

    if distribution == 'flat':
        for _i, _rv in enumerate(rand_vars):
            _swap_code += _codify_comment("--------------------------------")
            _swap_code += _codify_comment("GENERATING VALUE FOR " + str(_rv))
            _i_seed = _i if parallel_streams > 1 else 0
            _swap_code += \
                _codify_flat_mean_var(
                    declared_variables = declared_variables,
                    declared_constants = declared_constants,
                    root_seed = root_seed + '_' + str(_i_seed),
                    root_flat = _rv,
                    assignment_type = assignment_type,
                    crng_kind = generator,
                    rand_type = rand_vars_type,
                    mean = means[_i],
                    var = variances[_i],
                    const_out = output_const
                )
            _swap_code += _codify_newl
        _swap_code += _codify_newl

    if distribution == 'gaussian':
        if _gaussian_parallel:
            for _i, _rv in enumerate(rand_vars):
                _swap_code += _codify_comment("--------------------------------")
                _swap_code += _codify_comment("GENERATING VALUE FOR " + str(_rv))
                _swap_code += \
                    _codify_gaussian_p(
                        declared_variables = declared_variables,
                        declared_constants = declared_constants,
                        mean = means[_i],
                        var = variances[_i],
                        root_seeds = ['lseed_' + str(_i * 2),
                                      'lseed_' + str(_i * 2 + 1)],
                        root_gauss = _rv,
                        assignment_type = assignment_type,                        
                        root_flat = 'lflat',
                        crng_kind = generator,
                        rand_type = rand_vars_type,
                        const_out = output_const,
                        single_out = True
                    )
                _swap_code += _codify_newl
            _swap_code += _codify_newl

        if not _gaussian_parallel:
            for _i, _rv in enumerate(rand_vars):
                _i_seed = _i if parallel_streams > 1 else 0
                _swap_code += _codify_comment("--------------------------------")
                _swap_code += _codify_comment("GENERATING VALUE FOR " + str(_rv))
                _swap_code += \
                    _codify_gaussian_seq(
                        declared_variables = declared_variables,
                        declared_constants = declared_constants,
                        mean = means[_i],
                        var = variances[_i],
                        which_box_m = which_box_m,
                        root_seed = root_seed + '_' + str(_i_seed),
                        root_gauss = _rv,
                        assignment_type = assignment_type,                        
                        root_flat = 'lflat',
                        crng_kind = generator,
                        rand_type = rand_vars_type,
                        const_out = output_const
                    )
                _swap_code += _codify_newl
            _swap_code += _codify_newl
                
    if write_seeds_flag:
        _swap_code += _codify_comment("Writing back prng seeds")                
        _swap_code += \
            _codify_write_seeds(
                declared_variables = declared_variables,
                declared_constants = declared_constants,
                seeds_array = seeds_array, root_seed = root_seed,
                n_writes = parallel_streams,
                lambda_ordering = lambda_ordering,
                use_ptrs = use_ptrs
            )
    
    return _swap_code


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

class K_OutputGaussian(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'RANDType * output': ['global', 'restrict'],
                       'CRNGType * seeds_0': ['global', 'restrict'],
                       'CRNGType * seeds_1': ['global', 'restrict'],
                       'RANDType mean': ['const'], 'RANDType var': ['const'],
                       'int reps': ['const']}
        self.kernels[IDPY_T] = """ 
        if(g_tid < N_PRNGS){
            CRNGType l_seed_0 = seeds_0[g_tid], l_seed_1 = seeds_1[g_tid];
            for(int i=0; i<reps; i++){
               output[g_tid] = F_CRNGGaussFunction(&l_seed_0, &l_seed_1, mean, var);
            }
            seeds_0[g_tid] = l_seed_0;
            seeds_1[g_tid] = l_seed_1;
        }
        """

class K_OutputGaussianSingle(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'RANDType * output': ['global', 'restrict'],
                       'CRNGType * seeds': ['global', 'restrict'],
                       'RANDType mean': ['const'], 'RANDType var': ['const'],
                       'int reps': ['const']}
        self.kernels[IDPY_T] = """ 
        if(g_tid < N_PRNGS){
            CRNGType l_seed = seeds[g_tid];
            for(int i=0; i<reps; i++){
               output[g_tid] = F_CRNGGaussFunction(&l_seed, mean, var);
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
## https://www.pcg-random.org/posts/bounded-rands.html
class F_RandomIntegerUnbiasedLemire(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': [], 'CRNGType range': []}
        self.functions[IDPY_T] = """
        CRNGType t = (-range) % range;
        UINT64 m = 0;
        CRNGType l = 0;
        do {
            F_CRNG(l_seed);
            m = ((UINT64) (*l_seed)) * ((UINT64) (range));
            l = ((CRNGType)(m));
        } while (l < t);
        return ((CRNGType) (m >> 32));
        """

class F_RandomIntegerUnbiasedLemireMACRO(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': []}
        self.functions[IDPY_T] = """
        CRNGType t = (-CRNG_Integers_range) % CRNG_Integers_range;
        UINT64 m = 0;
        CRNGType l = 0;
        do {
            F_CRNG(l_seed);
            m = ((UINT64) (*l_seed)) * ((UINT64) (CRNG_Integers_range));
            l = ((CRNGType)(m));
        } while (l < t);
        return ((CRNGType) (m >> 32));
        """        

class F_RandomIntegerUnbiasedLemire_BUG(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': [], 'CRNGType range': []}
        self.functions[IDPY_T] = """
        F_CRNG(l_seed);
        UINT64 m = ((UINT64) (*l_seed)) * ((UINT64) range);
        CRNGType l = ((CRNGType) m);
        if(l < range){
            CRNGType t = -range;
            if(t >= range){
                t -= range;
                if(t >= range){
                    t %= range;
                }
            }
            while(l < t){
                F_CRNG(l_seed);
                m = ((UINT64) (*l_seed)) * ((UINT64) range);
                CRNGType l = ((CRNGType) m);
            }
        }
        return m >> 32;
        """
        
class F_RandomIntegerUnbiasedLemireMACRO_BUG(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': []}
        self.functions[IDPY_T] = """
        F_CRNG(l_seed);
        UINT64 m = ((UINT64) (*l_seed)) * ((UINT64) CRNG_Integers_range);
        CRNGType l = ((CRNGType) m);
        if(l < CRNG_Integers_range){
            CRNGType t = -CRNG_Integers_range;
            if(t >= CRNG_Integers_range){
                t -= CRNG_Integers_range;
                if(t >= CRNG_Integers_range){
                    t %= CRNG_Integers_range;
                }
            }
            while(l < t){
                F_CRNG(l_seed);
                m = ((UINT64) (*l_seed)) * ((UINT64) CRNG_Integers_range);
                CRNGType l = ((CRNGType) m);
            }
        }
        return m >> 32;
        """        
        
class F_Norm(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': []}
        self.functions[IDPY_T] = """
        F_CRNG(l_seed);
        return ((RANDType) (*l_seed)) / (ID_RANDMAX);
        """        

class F_GaussianCos(IdpyFunction):
    '''
    Using Box-Mueller
    '''
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed_0': [], 'CRNGType * l_seed_1': [],
                       'RANDType mean': ['const'], 'RANDType var': ['const']}
        self.functions[IDPY_T] = """
        RANDType u0 = F_Norm(l_seed_0), u1 = F_Norm(l_seed_1);
        return sqrt((RANDType)(-2. * var * log((RANDType) u0))) * cos((RANDType)(TWOPI * u1)) + mean;
        """

class F_GaussianSin(IdpyFunction):
    '''
    Using Box-Mueller
    '''
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed_0': [], 'CRNGType * l_seed_1': [],
                       'RANDType mean': ['const'], 'RANDType var': ['const']}
        self.functions[IDPY_T] = """
        RANDType u0 = F_Norm(l_seed_0), u1 = F_Norm(l_seed_1);
        return sqrt((RANDType)(-2. * var * log((RANDType) u0))) * sin((RANDType)(TWOPI * u1)) + mean;
        """

class F_GaussianCosSingle(IdpyFunction):
    '''
    Using Box-Mueller
    a single stream should generate two reasonably uncorrelated random numbers
    '''
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': [],
                       'RANDType mean': ['const'], 'RANDType var': ['const']}
        self.functions[IDPY_T] = """
        RANDType u0 = F_Norm(l_seed); 
        RANDType u1 = F_Norm(l_seed);
        return sqrt((RANDType)(-2. * var * log((RANDType) u0))) * cos((RANDType)(TWOPI * u1)) + mean;
        """

class F_GaussianSinSingle(IdpyFunction):
    '''
    Using Box-Mueller:
    a single stream should generate two reasonably uncorrelated random numbers
    '''
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': [],
                       'RANDType mean': ['const'], 'RANDType var': ['const']}
        self.functions[IDPY_T] = """
        RANDType u0 = F_Norm(l_seed); 
        RANDType u1 = F_Norm(l_seed);
        return sqrt((RANDType)(-2. * var * log((RANDType) u0))) * sin((RANDType)(TWOPI * u1)) + mean;
        """
        
class F_Integers(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'RANDType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'CRNGType * l_seed': [],
                       'RANDType mean': ['const'], 'RANDType var': ['const']}
        self.functions[IDPY_T] = """
        F_CRNG(l_seed);
        return ((RANDType) *l_seed) / ((RANDType) (ID_RANDMAX));
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
        return ((RANDType) *l_seed) / ((RANDType) (ID_RANDMAX));
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
        if((*l_seed) & 0x80000000) *l_seed = ((*l_seed) & 0x7fffffff) + 1;
        return;
        """

class F_MINSTD32(IdpyFunction):
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
        CRNGType lo = 16807*((*lseed)&0xffff);
        CRNGType hi = 16807*((*lseed)>>16);
        lo += (hi&0x7fff)<<16;
        lo += hi>>15;
        *lseed = lo - ((-((lo&0x80000000)>>31))&0x7fffffff);
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
