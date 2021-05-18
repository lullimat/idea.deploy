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
Provides the simulations classes fo the Lattice Boltzmann method
'''

from functools import reduce
from collections import defaultdict
import sympy as sp
import numpy as np
import h5py
import time

from idpy.Utils.CustomTypes import CustomTypes
from idpy.Utils.NpTypes import NpTypes
from idpy.IdpyCode.IdpySims import IdpySims
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T
from idpy.IdpyCode import idpy_langs_sys, idpy_langs_list

from idpy.IdpyCode import GetTenet, GetParamsClean, CheckOCLFP
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyFunction
from idpy.IdpyCode.IdpyCode import IdpyMethod, IdpyLoop

if idpy_langs_sys[OCL_T]:
    import pyopencl as cl
    from idpy.OpenCL.OpenCL import OpenCL
    from idpy.OpenCL.OpenCL import Tenet as CLTenet
if idpy_langs_sys[CUDA_T]:
    from idpy.CUDA.CUDA import CUDA
    from idpy.CUDA.CUDA import Tenet as CUTenet

XIStencils = defaultdict(
    lambda: defaultdict(dict)
)

FStencils = defaultdict(
    lambda: defaultdict(dict)
)

XIStencils['D2Q9'] = {'XIs': ((0, 0),
                              (1, 0), (0, 1), (-1, 0), (0, -1),
                              (1, 1), (-1, 1), (-1, -1), (1, -1)),
                      'Ws': (4/9,
                             1/9, 1/9, 1/9, 1/9,
                             1/36, 1/36, 1/36, 1/36),
                      'c2': 1/3}

XIStencils['D3Q19'] = {'XIs': ((0, 0, 0),
                               (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                               (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
                               (1, 0, 1), (-1, 0, 1), (-1, 0, -1), (1, 0, -1),
                               (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1)),
                      'Ws': (1/3,
                             1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
                             1/36, 1/36, 1/36, 1/36,
                             1/36, 1/36, 1/36, 1/36,
                             1/36, 1/36, 1/36, 1/36),
                      'c2': 1/3}

FStencils['D2E4'] = {'Es': ((1, 0), (0, 1), (-1, 0), (0, -1),
                            (1, 1), (-1, 1), (-1, -1), (1, -1)),
                     'Ws': (1/3, 1/3, 1/3, 1/3,
                            1/12, 1/12, 1/12, 1/12)}

FStencils['D2E6'] = {'Es': ((1, 0), (0, 1), (-1, 0), (0, -1),
                            (1, 1), (-1, 1), (-1, -1), (1, -1)),
                     'Ws': (1/3, 1/3, 1/3, 1/3,
                            1/12, 1/12, 1/12, 1/12)}

FStencils['D2E8'] = {'Es': ((1, 0), (0, 1), (-1, 0), (0, -1),
                            (1, 1), (-1, 1), (-1, -1), (1, -1)),
                     'Ws': (1/3, 1/3, 1/3, 1/3,
                            1/12, 1/12, 1/12, 1/12)}

FStencils['D3E4'] = {'Es': ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                            (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
                            (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1),
                            (1, 0, 1), (1, 0, -1), (-1, 0, -1), (-1, 0, 1)),
                     'Ws': (1/6, 1/6, 1/6, 1/6, 1/6, 1/6,
                            1/12, 1/12, 1/12, 1/12, 1/12, 1/12,
                            1/12, 1/12, 1/12, 1/12, 1/12, 1/12)}

FStencils['D3E6'] = {'Es': ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                            (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
                            (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1),
                            (1, 0, 1), (1, 0, -1), (-1, 0, -1), (-1, 0, 1),
                            (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1),
                            (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1),
                            (2, 0, 0), (0, 2, 0), (-2, 0, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)),
                     'Ws': (2/15, 2/15, 2/15, 2/15, 2/15, 2/15,
                            1/15, 1/15, 1/15, 1/15,
                            1/15, 1/15, 1/15, 1/15,
                            1/15, 1/15, 1/15, 1/15,
                            1/60, 1/60, 1/60, 1/60,
                            1/60, 1/60, 1/60, 1/60,
                            1/120, 1/120, 1/120, 1/120, 1/120, 1/120)}

FStencils['D3E8'] = {'Es': ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                            (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
                            (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1),
                            (1, 0, 1), (1, 0, -1), (-1, 0, -1), (-1, 0, 1),
                            (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1),
                            (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1),
                            (2, 0, 0), (0, 2, 0), (-2, 0, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2),
                            (2, 1, 0), (2, -1, 0), (2, 0, 1), (2, 0, -1),
                            (1, 2, 0), (-1, 2, 0), (0, 2, 1), (0, 2, -1),
                            (-2, 1, 0), (-2, -1, 0), (-2, 0, 1), (-2, 0, -1),
                            (1, -2, 0), (-1, -2, 0), (0, -2, 1), (0, -2, -1),
                            (1, 0, 2), (-1, 0, 2), (0, 1, 2), (0, -1, 2),
                            (1, 0, -2), (-1, 0, -2), (0, 1, -2), (0, -1, -2),
                            (2, 1, 1), (2, -1, 1), (2, -1, -1), (2, 1, -1),
                            (1, 2, 1), (-1, 2, 1), (-1, 2, -1), (1, 2, -1),
                            (-2, 1, 1), (-2, -1, 1), (-2, -1, -1), (-2, 1, -1),
                            (1, -2, 1), (-1, -2, 1), (-1, -2, -1), (1, -2, -1),
                            (1, 1, 2), (-1, 1, 2), (-1, -1, 2), (1, -1, 2),
                            (1, 1, -2), (-1, 1, -2), (-1, -1, -2), (1, -1, -2),
                            (2, 2, 0), (-2, 2, 0), (-2, -2, 0), (2, -2, 0),
                            (2, 0, 2), (-2, 0, 2), (-2, 0, -2), (2, 0, -2),
                            (0, 2, 2), (0, -2, 2), (0, -2, -2), (0, 2, -2)),
                     'Ws': (4/45, 4/45, 4/45, 4/45, 4/45, 4/45,
                            1/21, 1/21, 1/21, 1/21,
                            1/21, 1/21, 1/21, 1/21,
                            1/21, 1/21, 1/21, 1/21,
                            2/105, 2/105, 2/105, 2/105,
                            2/105, 2/105, 2/105, 2/105,
                            5/504, 5/504, 5/504, 5/504, 5/504, 5/504,
                            1/315, 1/315, 1/315, 1/315,
                            1/315, 1/315, 1/315, 1/315,
                            1/315, 1/315, 1/315, 1/315,
                            1/315, 1/315, 1/315, 1/315,
                            1/315, 1/315, 1/315, 1/315,
                            1/315, 1/315, 1/315, 1/315,
                            1/630, 1/630, 1/630, 1/630,
                            1/630, 1/630, 1/630, 1/630,
                            1/630, 1/630, 1/630, 1/630,
                            1/630, 1/630, 1/630, 1/630,
                            1/630, 1/630, 1/630, 1/630,
                            1/630, 1/630, 1/630, 1/630,
                            1/5040, 1/5040, 1/5040, 1/5040,
                            1/5040, 1/5040, 1/5040, 1/5040,
                            1/5040, 1/5040, 1/5040, 1/5040)}

LBMTypes = CustomTypes({'PopType': 'double', 
                        'NType': 'double', 
                        'UType': 'double',
                        'PsiType': 'double',
                        'SCFType': 'double',
                        'SType': 'int',
                        'WType': 'double',
                        'LengthType': 'double',
                        'FlagType': 'unsigned char'})

NPT = NpTypes()

RootLB_Dump_def_name = 'RootLB_dump.hdf5'

def IndexFromPos(pos, dim_strides):
    index = pos[0]
    for i in range(1, len(pos)):
        index += pos[i] * dim_strides[i - 1]
    return index

def PosFromIndex(index, dim_strides): 
    pos = [index%dim_strides[0]]
    pos += [(index//dim_strides[stride_i]) % (dim_strides[stride_i + 1]//dim_strides[stride_i]) \
            if stride_i < len(dim_strides) - 1 else \
            index//dim_strides[stride_i] \
            for stride_i in range(len(dim_strides))]
    return tuple(pos)

def ComputeCenterOfMass(lbm, c_i = ''):
    first_flag = False
    if 'cm_coords' not in lbm.sims_idpy_memory:
        lbm.sims_idpy_memory['cm_coords'] = \
            IdpyMemory.Zeros(lbm.sims_vars['V'],
                             dtype = NPT.C[lbm.custom_types['NType']],
                             tenet = lbm.tenet)
        lbm.aux_idpy_memory.append('cm_coords')
        first_flag = True

    _mass = IdpyMemory.Sum(lbm.sims_idpy_memory['n' + c_i])
        
    _K_CenterOfMass = K_CenterOfMass(custom_types = lbm.custom_types.Push(),
                                     constants = lbm.constants,
                                     f_classes = [F_PosFromIndex],
                                     optimizer_flag = lbm.optimizer_flag)

    Idea = _K_CenterOfMass(tenet = lbm.tenet, grid = lbm.sims_vars['grid'],
                           block = lbm.sims_vars['block'])

    _cm_coords = ()
    for direction in range(lbm.sims_vars['DIM']):        
        Idea.Deploy([lbm.sims_idpy_memory['cm_coords'],
                     lbm.sims_idpy_memory['n' + c_i],
                     lbm.sims_idpy_memory['dim_sizes'],
                     lbm.sims_idpy_memory['dim_strides'],
                     NPT.C[lbm.custom_types['SType']](direction)])
        _cm_coords += (IdpyMemory.Sum(lbm.sims_idpy_memory['cm_coords'])/_mass, )

    return _mass, _cm_coords


def CheckCenterOfMassDeltaPConvergence(lbm):
    _first_flag = False
    if 'cm_conv' not in lbm.aux_vars:
        lbm.sims_vars['cm_conv'] = []
        lbm.aux_vars.append('cm_conv')

        lbm.sims_vars['cm_coords'] = []
        lbm.aux_vars.append('cm_coords')        

        lbm.sims_vars['delta_p'] = []
        lbm.aux_vars.append('delta_p')

        lbm.sims_vars['p_in'], lbm.sims_vars['p_out'] = \
                        [], []
        lbm.aux_vars.append('p_out')
        lbm.aux_vars.append('p_in')

        lbm.sims_vars['is_centered_seq'] = []
        lbm.aux_vars.append('is_centered_seq')
        
        _first_flag = True
        
    _p_in, _p_out, _delta_p = lbm.DeltaPLaplace()
    print("p_in: ", _p_in, "p_out: ", _p_out, "delta_p: ", _delta_p)
    print()

    _chk, _break_f = [], False
    if not _first_flag:
        _delta_delta_p = _delta_p - lbm.sims_vars['delta_p'][-1]
        _delta_p_in = _p_in - lbm.sims_vars['p_in'][-1]
        _delta_p_out = _p_out - lbm.sims_vars['p_out'][-1]
        
        _chk += [not lbm.sims_vars['is_centered']]
        _chk += [abs(_delta_p) < 1e-9]
        _chk += [abs(_delta_delta_p / _delta_p) < 1e-5]

        _break_f = OneTrue(_chk)        

        print("Center of mass: ", lbm.sims_vars['cm_coords'])
        print("delta delta_p: ", _delta_delta_p,
              "delta p_in: ", _delta_p_in,
              "delta p_out: ", _delta_p_out)
        
        print(_chk)
        print()

    lbm.sims_vars['cm_conv'].append(np.copy(lbm.sims_vars['cm_coords']))
    lbm.sims_vars['delta_p'].append(_delta_p)
    lbm.sims_vars['p_in'].append(_p_in)
    lbm.sims_vars['p_out'].append(_p_out)
    lbm.sims_vars['is_centered_seq'].append(lbm.sims_vars['is_centered'])
    
    return _break_f

def CheckUConvergence(lbm):
    first_flag = False

    if 'old_u' not in lbm.sims_idpy_memory:
        lbm.sims_idpy_memory['old_u'] = \
            IdpyMemory.Zeros(lbm.sims_vars['V'] *
                             lbm.sims_vars['DIM'],
                             dtype = NPT.C[lbm.custom_types['UType']],
                             tenet = lbm.tenet)
        lbm.aux_idpy_memory.append('old_u')

        lbm.sims_idpy_memory['delta_u'] = \
            IdpyMemory.Zeros(lbm.sims_vars['V'],
                             dtype = NPT.C[lbm.custom_types['UType']],
                             tenet = lbm.tenet)
        lbm.aux_idpy_memory.append('delta_u')

        lbm.sims_idpy_memory['max_u'] = \
            IdpyMemory.Zeros(lbm.sims_vars['V'],
                             dtype = NPT.C[lbm.custom_types['UType']],
                             tenet = lbm.tenet)
        lbm.aux_idpy_memory.append('max_u')
        
        lbm.sims_vars['u_conv'], lbm.sims_vars['max_u'] = [], []
        lbm.aux_vars.append('u_conv')
        lbm.aux_vars.append('max_u')
        
        first_flag = True
        

    _K_CheckU = K_CheckU(custom_types = lbm.custom_types.Push(),
                         constants = lbm.constants,
                         optimizer_flag = lbm.optimizer_flag)

    _K_CheckU(tenet = lbm.tenet, grid = lbm.sims_vars['grid'],
              block = lbm.sims_vars['block']).Deploy([lbm.sims_idpy_memory['delta_u'],
                                                      lbm.sims_idpy_memory['old_u'],
                                                      lbm.sims_idpy_memory['max_u'],
                                                      lbm.sims_idpy_memory['u']], idpy_stream = None)


    u_conv = IdpyMemory.Sum(lbm.sims_idpy_memory['delta_u'])/lbm.sims_vars['V']
    max_u = np.sqrt(IdpyMemory.Max(lbm.sims_idpy_memory['max_u'])/lbm.sims_vars['c2'])
    lbm.sims_vars['u_conv'].append(u_conv)
    lbm.sims_vars['max_u'].append(max_u)

    _u_threshold = 1e-12 if NPT.C[lbm.custom_types['UType']] == np.float64 else 1e-5

    print('u_conv: ', u_conv, 'max_u: ', max_u)
    print("Conv!", u_conv < _u_threshold, first_flag)
    
    if not first_flag and u_conv < _u_threshold:
        break_f = True
    else:
        break_f = False
        
    return break_f
        

def AllTrue(list_swap):
    return reduce(lambda x, y: x and y, list_swap)

def OneTrue(list_swap):
    return reduce(lambda x, y: x or y, list_swap)

def InitDimSizesStridesVolume(dim_sizes, custom_types):
    '''
    InitDimSizesStridesVolume:
    returns the dimension, lists of sizes, strides and volumes
    all quantities are cast either in np scalars or arrays
    of the corrensponding stencils type 'SType'
    '''
    D = len(dim_sizes)
    dim_sizes = np.array(dim_sizes, dtype = NPT.C[custom_types['SType']])
    dim_strides = np.array([reduce(lambda x, y: x*y, dim_sizes[0:i+1]) 
                            for i in range(len(dim_sizes) - 1)], 
                           dtype = NPT.C[custom_types['SType']])
    dim_center = np.array(list(map(lambda x: x//2, dim_sizes)),
                               dtype = NPT.C[custom_types['SType']])
    V = int(reduce(lambda x, y: x * y, dim_sizes))
    return D, dim_sizes, dim_strides, dim_center, V

def InitStencilWeights(xi_stencil, custom_types):
    '''
    InitStencilWeights:
    returns the number of vectors Q 
    the lists of vectors coordinates XI_list
    the list of weights W_list 
    and the square of the sound speed c2
    all types ar cast to the appropriate np type
    '''
    Q = len(xi_stencil['XIs'])
    XI_list = [x for xi in xi_stencil['XIs'] for x in xi]
    XI_list = np.array(XI_list, NPT.C[custom_types['SType']])
    W_list = np.array(xi_stencil['Ws'], NPT.C[custom_types['WType']])
    c2 = xi_stencil['c2']
    return Q, XI_list, W_list, c2

def InitFStencilWeights(f_stencil, custom_types):
    '''
    InitFStencilWeights:
    returns the number of vectors Q 
    the lists of vectors coordinates XI_list
    the list of weights W_list 
    and the square of the sound speed c2
    all types are cast to the appropriate np type
    '''
    EQ = len(f_stencil['Es'])
    E_list = [x for e in f_stencil['Es'] for x in e]
    E_list = np.array(E_list, NPT.C[custom_types['SType']])
    EW_list = np.array(f_stencil['Ws'], NPT.C[custom_types['WType']])
    return EQ, E_list, EW_list


class RootLB(IdpySims):
    '''
    class RootLB:
    root LB class. I will try to use inheritance for the more complicated ones
    This class is in some sense "virtual" since it cannot be declared,
    and the methods cannot be used unless the variable custom_types is passed
    from the child class
    '''
    def __init__(self, *args, **kwargs):

        if not hasattr(self, 'params_dict'):
            self.params_dict = {}
            
        self.kwargs = \
            GetParamsClean(kwargs, [self.params_dict],
                           needed_params = ['custom_types', 'dim_sizes', 'xi_stencil'])

        if 'dim_sizes' not in self.params_dict:
            raise Exception("Missing parameter 'dim_sizes'")
        if 'xi_stencil' not in self.params_dict:
            raise Exception("Missing parameter 'xi_stencil'")

        custom_types = None
        if 'custom_types' in self.params_dict:
            custom_types = self.params_dict['custom_types']
        else:
            custom_types = LBMTypes
                
        IdpySims.__init__(self, *args, **self.kwargs)

        dim_sizes, xi_stencil = self.params_dict['dim_sizes'], self.params_dict['xi_stencil']

        self.sims_vars['DIM'], self.sims_vars['dim_sizes'], \
            self.sims_vars['dim_strides'], self.sims_vars['dim_center'], \
            self.sims_vars['V'] = InitDimSizesStridesVolume(dim_sizes, custom_types)

        self.sims_vars['Q'], self.sims_vars['XI_list'], \
            self.sims_vars['W_list'], self.sims_vars['c2'] = \
                InitStencilWeights(xi_stencil, custom_types)

        self.constants = {'V': self.sims_vars['V'],
                          'Q': self.sims_vars['Q'],
                          'DIM': self.sims_vars['DIM'],
                          'CM2': 1/self.sims_vars['c2'],
                          'CM4': 1/(self.sims_vars['c2'] ** 2)}

        '''
        memory to be allocated
        '''
        self.sims_idpy_memory = {'pop': None,
                                 'dim_sizes': None,
                                 'dim_strides': None,
                                 'dim_center': None,
                                 'XI_list': None,
                                 'W_list': None}
        
    def InitMemory(self, tenet, custom_types): 
        self.sims_idpy_memory['pop'] = \
            IdpyMemory.Zeros(shape = self.sims_vars['V'] * self.sims_vars['Q'],
                             dtype = NPT.C[custom_types['PopType']],
                             tenet = tenet)
        
        self.sims_idpy_memory['dim_sizes'] = \
            IdpyMemory.OnDevice(self.sims_vars['dim_sizes'],
                                tenet = tenet)
        
        self.sims_idpy_memory['dim_strides'] = \
            IdpyMemory.OnDevice(self.sims_vars['dim_strides'],
                                tenet = tenet)

        self.sims_idpy_memory['dim_center'] = \
            IdpyMemory.OnDevice(self.sims_vars['dim_center'],
                                tenet = tenet)
        
        self.sims_idpy_memory['XI_list'] = \
            IdpyMemory.OnDevice(self.sims_vars['XI_list'], tenet = tenet)
        
        self.sims_idpy_memory['W_list'] = \
            IdpyMemory.OnDevice(self.sims_vars['W_list'], tenet = tenet)

    def DumpPopSnapshot(self, file_name = RootLB_Dump_def_name):
        IdpySims.DumpSnapshot(self, file_name = file_name,
                              custom_types = self.custom_types)

class ShanChenMultiPhase(RootLB):
    def __init__(self, *args, **kwargs):
        self.SetupRoot(*args, **kwargs)
        
        self.InitVars()
        self.GridAndBlocks()
        
        for name in ['n', 'u']:
            self.sims_idpy_memory[name] = None

        if not self.params_dict['empty_sim']:
            self.InitMemory()

        self.init_status = {'n': False,
                            'u': False,
                            'pop': False}


    def MainLoop(self, time_steps, convergence_functions = []):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_ComputeMoments = K_ComputeMoments(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             optimizer_flag = self.optimizer_flag)

        _K_ComputePsi = K_ComputePsi(custom_types = self.custom_types.Push(),
                                     constants = self.constants,
                                     psi_code = self.params_dict['psi_code'],
                                     optimizer_flag = self.optimizer_flag)

        _K_Collision_ShanChenGuoMultiPhase = \
            K_Collision_ShanChenGuoMultiPhase(custom_types = self.custom_types.Push(),
                                              constants = self.constants,
                                              f_classes = [F_PosFromIndex,
                                                           F_IndexFromPos],
                                              optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = K_StreamPeriodic(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             f_classes = [F_PosFromIndex,
                                                          F_IndexFromPos],
                                             optimizer_flag = self.optimizer_flag)
        
        self._MainLoop = \
            IdpyLoop(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeMoments(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['n', 'u', 'pop',
                                                                              'XI_list', 'W_list']),
                        (_K_ComputePsi(tenet = self.tenet,
                                       grid = self.sims_vars['grid'],
                                       block = self.sims_vars['block']), ['psi', 'n']),

                        (_K_Collision_ShanChenGuoMultiPhase(tenet = self.tenet,
                                                            grid = self.sims_vars['grid'],
                                                            block = self.sims_vars['block']),
                         ['pop', 'u', 'n', 'psi',
                          'XI_list', 'W_list',
                          'E_list', 'EW_list',
                          'dim_sizes', 'dim_strides']),
                        
                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['pop_swap', 'pop',
                                                                              'XI_list', 'dim_sizes',
                                                                              'dim_strides']),
                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step = 0
        for step in time_steps:
            print(step, step - old_step)
            '''
            Very simple timing, reasonable for long executions
            '''
            self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break
        
    def ComputeMoments(self):
        if not self.init_status['pop']:
            raise Exception("Populations are not initialized")

        _K_ComputeMoments = \
            K_ComputeMoments(custom_types = self.custom_types.Push(),
                             constants = self.constants, f_classes = [],
                             optimizer_flag = self.optimizer_flag)

        Idea = _K_ComputeMoments(tenet = self.tenet,
                                 grid = self.sims_vars['grid'],
                                 block = self.sims_vars['block'])

        Idea.Deploy([self.sims_idpy_memory['n'],
                     self.sims_idpy_memory['u'],
                     self.sims_idpy_memory['pop'],
                     self.sims_idpy_memory['XI_list'],
                     self.sims_idpy_memory['W_list']])

        self.init_status['n'] = True
        self.init_status['u'] = True
        
        
    def InitPopulations(self):
        if not AllTrue([self.init_status['n'], self.init_status['u']]):
            raise Exception("Fields u and n are not initialized")

        _K_InitPopulations = \
            K_InitPopulations(custom_types = self.custom_types.Push(),
                              constants = self.constants, f_classes = [],
                              optimizer_flag = self.optimizer_flag)

        Idea = _K_InitPopulations(tenet = self.tenet,
                                  grid = self.sims_vars['grid'],
                                  block = self.sims_vars['block'])
        
        Idea.Deploy([self.sims_idpy_memory['pop'],
                     self.sims_idpy_memory['n'],
                     self.sims_idpy_memory['u'],
                     self.sims_idpy_memory['XI_list'],
                     self.sims_idpy_memory['W_list']])
        
        self.init_status['pop'] = True

    def DeltaPLaplace(self, inside = None, outside = None):
        if 'n_in_n_out' not in self.sims_idpy_memory:
            self.sims_idpy_memory['n_in_n_out'] = \
                IdpyMemory.Zeros(2, dtype = NPT.C[self.custom_types['NType']],
                                 tenet = self.tenet)
            self.aux_idpy_memory.append('n_in_n_out')

        _K_NInNOut = K_NInNOut(custom_types = self.custom_types.Push(),
                               constants = self.constants,
                               f_classes = [],
                               optimizer_flag = self.optimizer_flag)

        if inside is None:
            _mass, _inside = ComputeCenterOfMass(self)
            _inside = list(_inside)
            self.sims_vars['cm_coords'] = np.array(_inside)
            self.sims_vars['mass'] = _mass

            """
            for i in range(len(_inside)):
                _inside[i] = int(round(_inside[i]))
            """
                
            '''
            Check center
            '''
            _chk = False
            for d in range(self.sims_vars['DIM']):
                if abs(self.sims_vars['dim_center'][d] - _inside[d]) > 1e-3:
                    _chk = _chk or True

            self.sims_vars['is_centered'] = True
            if _chk:
                print("Center of mass: ", _inside,
                      "; Center of the system: ", self.sims_vars['dim_center'])
                self.sims_vars['is_centered'] = False

            
            _inside = IndexFromPos(self.sims_vars['dim_center'].tolist(),
                                   self.sims_vars['dim_strides'])
            _inside = NPT.C['unsigned int'](_inside)
        else:
            _inside = NPT.C['unsigned int'](inside)

        if outside is None:
            _outside = self.sims_vars['V'] - 1
            _outside = NPT.C['unsigned int'](_outside)
        else:
            _outside = NPT.C['unsigned int'](outside)

        
        Idea = _K_NInNOut(tenet = self.tenet,
                          grid = self.sims_vars['grid'],
                          block = self.sims_vars['block'])
            
        Idea.Deploy([self.sims_idpy_memory['n_in_n_out'],
                     self.sims_idpy_memory['n'],
                     _inside, _outside])

        _swap_innout = self.sims_idpy_memory['n_in_n_out'].D2H()

        self.sims_vars['n_in_n_out'] = _swap_innout
        
        _p_in = self.PBulk(_swap_innout[0])
        _p_out = self.PBulk(_swap_innout[1])

        return _p_in, _p_out, _p_in - _p_out


    def PBulk(self, n):
        if 'psi_f' not in self.sims_vars:
            self.sims_vars['psi_f'] = \
                sp.lambdify(self.sims_vars['n_sym'], self.sims_vars['psi_sym'])
            self.sims_not_dump_vars += ['psi_f']
            
        _p = (n * self.sims_vars['c2'] +
              0.5 * self.sims_vars['SC_G'] * self.sims_vars['e2_val'] * (self.sims_vars['psi_f'](n)) ** 2)

        return _p


    def InitCylinderInterface(self, n_g, n_l, R, direction = 0,
                              full_flag = True):
        '''
        Record init values
        '''
        self.sims_vars['init_type'] = 'cylinder'
        self.sims_vars['n_g'], self.sims_vars['n_l'] = n_g, n_l
        self.sims_vars['full_flag'], self.sims_vars['R'] = full_flag, R
        self.sims_vars['direction'] = direction
        
        _K_InitCylinderInterface = \
            K_InitCylinderInterface(custom_types = self.custom_types.Push(),
                                    constants = self.constants,
                                    f_classes = [F_PosFromIndex,
                                                 F_PointDistance],
                                    optimizer_flag = self.optimizer_flag)

        n_g = NPT.C[self.custom_types['NType']](n_g)
        n_l = NPT.C[self.custom_types['NType']](n_l)
        R = NPT.C[self.custom_types['LengthType']](R)
        full_flag = NPT.C[self.custom_types['FlagType']](full_flag)
        direction = NPT.C[self.custom_types['SType']](direction)

        
        Idea = _K_InitCylinderInterface(tenet = self.tenet,
                                        grid = self.sims_vars['grid'],
                                        block = self.sims_vars['block'])

        Idea.Deploy([self.sims_idpy_memory['n'],
                     self.sims_idpy_memory['u'],
                     self.sims_idpy_memory['dim_sizes'],
                     self.sims_idpy_memory['dim_strides'],
                     self.sims_idpy_memory['dim_center'],
                     n_g, n_l, R, direction, full_flag])

        self.init_status['n'] = True
        self.init_status['u'] = True
        '''
        Finally, initialize populations...
        Need to do it here: already forgot once to do it outside
        '''
        self.InitPopulations()

        
    def InitRadialInterface(self, n_g, n_l, R, full_flag = True):
        '''
        Record init values
        '''
        self.sims_vars['init_type'] = 'radial'
        self.sims_vars['n_g'], self.sims_vars['n_l'] = n_g, n_l
        self.sims_vars['full_flag'], self.sims_vars['R'] = full_flag, R
        
        _K_InitRadialInterface = \
            K_InitRadialInterface(custom_types = self.custom_types.Push(),
                                  constants = self.constants,
                                  f_classes = [F_PosFromIndex,
                                               F_PointDistanceCenterFirst],
                                  optimizer_flag = self.optimizer_flag)

        n_g = NPT.C[self.custom_types['NType']](n_g)
        n_l = NPT.C[self.custom_types['NType']](n_l)
        R = NPT.C[self.custom_types['LengthType']](R)
        full_flag = NPT.C[self.custom_types['FlagType']](full_flag)
        
        Idea = _K_InitRadialInterface(tenet = self.tenet,
                                      grid = self.sims_vars['grid'],
                                      block = self.sims_vars['block'])

        Idea.Deploy([self.sims_idpy_memory['n'],
                     self.sims_idpy_memory['u'],
                     self.sims_idpy_memory['dim_sizes'],
                     self.sims_idpy_memory['dim_strides'],
                     self.sims_idpy_memory['dim_center'],
                     n_g, n_l, R, full_flag])
        
        self.init_status['n'] = True
        self.init_status['u'] = True
        '''
        Finally, initialize populations...
        Need to do it here: already forgot once to do it outside
        '''
        self.InitPopulations()

        
    def InitFlatInterface(self, n_g, n_l, width, direction = 0,
                          full_flag = True):
        '''
        Record init values
        '''
        self.sims_vars['init_type'] = 'flat'
        self.sims_vars['n_g'], self.sims_vars['n_l'] = n_g, n_l
        self.sims_vars['full_flag'] = full_flag
        self.sims_vars['width'], self.sims_vars['direction'] = width, direction
        
        _K_InitFlatInterface = \
            K_InitFlatInterface(custom_types = self.custom_types.Push(),
                                constants = self.constants,
                                f_classes = [F_PosFromIndex,
                                             F_NFlatProfile],
                                optimizer_flag = self.optimizer_flag)

        n_g = NPT.C[self.custom_types['NType']](n_g)
        n_l = NPT.C[self.custom_types['NType']](n_l)
        width = NPT.C[self.custom_types['LengthType']](width)
        direction = NPT.C[self.custom_types['SType']](direction)
        full_flag = NPT.C[self.custom_types['FlagType']](full_flag)
        
        Idea = _K_InitFlatInterface(tenet = self.tenet,
                                    grid = self.sims_vars['grid'],
                                    block = self.sims_vars['block'])

        Idea.Deploy([self.sims_idpy_memory['n'],
                     self.sims_idpy_memory['u'],
                     self.sims_idpy_memory['dim_sizes'],
                     self.sims_idpy_memory['dim_strides'],
                     self.sims_idpy_memory['dim_center'],
                     n_g, n_l, width, direction, full_flag])

        self.init_status['n'] = True
        self.init_status['u'] = True
        '''
        Finally, initialize populations...
        Need to do it here: already forgot once to do it outside
        '''
        self.InitPopulations()


    def InitVars(self):
        '''
        sims_vars
        '''
        self.sims_vars['QE'], self.sims_vars['E_list'], self.sims_vars['EW_list'] = \
            InitFStencilWeights(f_stencil = self.params_dict['f_stencil'],
                                custom_types = self.custom_types)
        self.sims_vars['SC_G'] = self.params_dict['SC_G']
        if 'e2_val' not in self.params_dict:
            self.sims_vars['e2_val'] = 1
        else:
            self.sims_vars['e2_val'] = self.params_dict['e2_val']

        self.sims_vars['psi_sym'] = self.params_dict['psi_sym']
        self.sims_vars['n_sym'] = sp.symbols('n')

        '''
        constants
        '''
        self.constants['QE'] = self.sims_vars['QE']
        self.constants['SC_G'] = self.params_dict['SC_G']
        self.constants['OMEGA'] = 1./self.params_dict['tau']


    def InitMemory(self):
        RootLB.InitMemory(self, tenet = self.tenet,
                          custom_types = self.custom_types)
        
        self.sims_idpy_memory['n'] = \
            IdpyMemory.Zeros(shape = self.sims_vars['V'],
                             dtype = NPT.C[self.custom_types['NType']],
                             tenet = self.tenet)

        self.sims_idpy_memory['psi'] = \
            IdpyMemory.Zeros(shape = self.sims_vars['V'],
                             dtype = NPT.C[self.custom_types['PsiType']],
                             tenet = self.tenet)        

        self.sims_idpy_memory['u'] = \
            IdpyMemory.Zeros(shape = self.sims_vars['V'] * \
                             self.sims_vars['DIM'],
                             dtype = NPT.C[self.custom_types['UType']],
                             tenet = self.tenet)

        self.sims_idpy_memory['pop_swap'] = \
            IdpyMemory.Zeros(shape = self.sims_vars['V'] * self.sims_vars['Q'],
                             dtype = NPT.C[self.custom_types['PopType']],
                             tenet = self.tenet)

        self.sims_idpy_memory['E_list'] = \
            IdpyMemory.OnDevice(self.sims_vars['E_list'], tenet = self.tenet)

        self.sims_idpy_memory['EW_list'] = \
            IdpyMemory.OnDevice(self.sims_vars['EW_list'], tenet = self.tenet)

    def GridAndBlocks(self):
        '''
        looks pretty general
        '''
        _block_size = None
        if 'block_size' in self.params_dict:
            _block_size = self.params_dict['block_size']
        else:
            _block_size = 128

        _grid = ((self.sims_vars['V'] + _block_size - 1)//_block_size, 1, 1)
        _block = (_block_size, 1, 1)

        self.sims_vars['grid'], self.sims_vars['block'] = _grid, _block
        
        
    def SetupRoot(self, *args, **kwargs):
        '''
        looks pretty general
        '''
        if not hasattr(self, 'params_dict'):
            self.params_dict = {}

        self.kwargs = GetParamsClean(kwargs, [self.params_dict],
                                     needed_params = ['lang', 'cl_kind', 'device',
                                                      'custom_types', 'block_size',
                                                      'f_stencil', 'psi_code', 'SC_G',
                                                      'tau', 'optimizer_flag', 'e2_val',
                                                      'psi_sym', 'empty_sim'])

        if 'f_stencil' not in self.params_dict:
            raise Exception("Missing 'f_stencil'")

        if 'tau' not in self.params_dict:
            raise Exception("Missing 'tau'")
        
        if 'SC_G' not in self.params_dict:
            raise Exception("Missing 'SC_G'")

        if 'psi_code' not in self.params_dict:
            raise Exception("Missing 'psi_code', e.g. psi_code = 'exp(-1/ln)'")

        if 'lang' not in self.params_dict:
            raise Exception("Param lang = CUDA_T | OCL_T is needed")

        if 'psi_sym' not in self.params_dict:
            raise Exception("Missing sympy expression for the pseudo-potential, parameter 'psi_sym'")

        if 'optimizer_flag' in self.params_dict:
            self.optimizer_flag = self.params_dict['optimizer_flag']
        else:
            self.optimizer_flag = True

        if 'empty_sim' not in self.params_dict:
            self.params_dict['empty_sim'] = False

        self.tenet = GetTenet(self.params_dict)
        if 'custom_types' in self.params_dict:
            self.custom_types = self.params_dict['custom_types']
        else:
            self.custom_types = LBMTypes

        self.custom_types = \
            CheckOCLFP(tenet = self.tenet, custom_types = self.custom_types)

        print(self.custom_types.Push())
            
        RootLB.__init__(self, *args, custom_types = self.custom_types,
                         **self.kwargs)
        
    def End(self):
        self.tenet.End()

                           
        
'''
Functions
'''

class F_FetchU(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'void'):
        super().__init__(custom_types = custom_types, f_type = f_type)
        self.params = {'UType * lu': ['global', 'const'],
                       'UType * u': ['global', 'const'],
                       'unsigned int g_tid': ['const']}

        self.functions[IDPY_T] = """
        for(int d=0; d<DIM; d++){
            lu[d] = u[tid + d*V];
        }
        """

class F_FetchUDotU(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'UType'):
        super().__init__(custom_types = custom_types, f_type = f_type)
        self.params = {'UType * lu': ['global', 'const'],
                       'UType * u': ['global', 'const'],
                       'unsigned int g_tid': ['const']}

        self.functions[IDPY_T] = """
        UType u_dot_u = 0.;
        for(int d=0; d<DIM; d++){
            lu[d] = u[tid + d*V]; u_dot_u = lu[d] * lu[d];
        }
        return u_dot_u;
        """

class F_PopsFromRhoU(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'void'):
        super().__init__(custom_types = custom_types, f_type = f_type)
        self.params = {'PopType * pop': ['global'],
                       'RhoType * rho': ['global', 'const'],
                       'UType * u': ['global', 'const'],
                       'SType * XI_list': ['global', 'const'],
                       'WType * W_list': ['global', 'const'],
                       'unsigned int g_tid': ['const']}

        self.functions[CUDA_T] = """
        UType lu[DIM], u_dot_u = 0.;
        for(int d=0; d<DIM; d++){
            lu[d] = u[g_tid + d * V]; u_dot_u += lu[d] * lu[d];
        }
        // Cycle over the populations
        for(int q=0; q<Q; q++){
            UType u_dot_xi = 0.;
            for(int d=0; d<DIM; d++){
                u_dot_xi += lu[d] * XI_list[d + q * DIM];
            }
            PopType leq_pop = 1.;

            pop[g_tid + q * V] = leq_pop;
        }
        return;
        """

class F_PosFromIndex(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'void'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType * pos': [],
                       'SType * dim_sizes': ['global', 'const'],
                       'SType * dim_strides': ['global', 'const'],
                       'unsigned int index': ['const']}

        self.functions[IDPY_T] = """
        pos[0] = index % dim_strides[0];
        for(int d=1; d<DIM; d++){
            pos[d] = (index / dim_strides[d - 1]) % dim_sizes[d];
        }
        return;
        """

class F_PosFromIndexDIM(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'void'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType * pos': [],
                       'SType * dim_sizes': ['global', 'const'],
                       'SType * dim_strides': ['global', 'const'],
                       'unsigned int index': ['const'],
                       'int dim': ['const']}

        self.functions[IDPY_T] = """
        if(dim == 1) pos[0] = index;
        else{
            pos[0] = index % dim_strides[0];
            for(int d=1; d<dim; d++){
                pos[d] = (index / dim_strides[d - 1]) % dim_sizes[d];
            }
        }
        return;
        """

class F_IndexFromPos(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'SType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType * pos': ['const'],
                       'SType * dim_strides': ['global', 'const']}

        self.functions[IDPY_T] = """
        SType index = pos[0];
        for(int d=1; d<DIM; d++){
            index += pos[d] * dim_strides[d - 1];
        }
        return index;
        """
        
class F_NFlatProfile(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'NType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType x': ['const'],
                       'SType x0': ['const'],
                       'LengthType w': ['const']}

        self.functions[IDPY_T] = """
        return tanh((LengthType)(x - (x0 - 0.5 * w))) - tanh((LengthType)(x - (x0 + 0.5 * w)));
        """

class F_PointDistanceCenterFirst(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'LengthType'):
        IdpyFunction.__init__(self, custom_types = custom_types,
                              f_type = f_type)
        self.params = {'SType * a': ['global', 'const'],
                       'SType * b': ['const']}

        self.functions[IDPY_T] = """
        LengthType dist = 0;
        for(int d=0; d<DIM; d++){
            dist += (a[d] - b[d]) * (a[d] - b[d]);
        }
        return sqrt((LengthType) dist);
        """

class F_PointDistance(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'LengthType'):
        IdpyFunction.__init__(self, custom_types = custom_types,
                              f_type = f_type)
        self.params = {'SType * a': ['const'],
                       'SType * b': ['const']}

        self.functions[IDPY_T] = """
        LengthType dist = 0;
        for(int d=0; d<DIM; d++){
            dist += (a[d] - b[d]) * (a[d] - b[d]);
        }
        return sqrt((LengthType) dist);
        """

'''
Kernels
'''
class K_CenterOfMass(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')
        self.params = {'NType * cm_coords': ['global', 'restrict'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const'],
                       'SType direction': ['const']}

        self.kernels[IDPY_T] = """ 
        if(g_tid < V){
            SType g_tid_pos[DIM];
            F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);
            cm_coords[g_tid] = n[g_tid] * g_tid_pos[direction];            
        }
        """

class K_CheckU(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'UType * delta_u': ['global', 'restrict'],
                       'UType * old_u': ['global', 'restrict'],
                       'UType * max_u': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
            UType ldiff = 0., lu_norm = 0.;

            for(int d=0; d<DIM; d++){
                UType lu_now = u[g_tid + d * V];
                lu_norm += lu_now * lu_now;
                ldiff += (UType) fabs(lu_now - old_u[g_tid + d * V]);
                old_u[g_tid + d * V] = lu_now;
            }

            delta_u[g_tid] = ldiff;
            max_u[g_tid] = lu_norm;
        }
        """

class K_Collision_ShanChenGuoMultiPhase(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'PsiType * psi': ['global', 'restrict', 'const'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'WType * W_list': ['global', 'restrict', 'const'],
                       'SType * E_list': ['global', 'restrict', 'const'],
                       'WType * EW_list': ['global', 'restrict', 'const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const']}
        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
            // Getting thread position
            SType g_tid_pos[DIM];
            F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);

            // Computing Shan-Chen Force
            SCFType F[DIM]; SType neigh_pos[DIM];
            for(int d=0; d<DIM; d++){F[d] = 0.;}

            PsiType lpsi = psi[g_tid];

            for(int qe=0; qe<QE; qe++){
                // Compute neighbor position
                for(int d=0; d<DIM; d++){
                    neigh_pos[d] = ((g_tid_pos[d] + E_list[d + qe*DIM] + dim_sizes[d]) % dim_sizes[d]);
                }
                // Compute neighbor index
                SType neigh_index = F_IndexFromPos(neigh_pos, dim_strides);
                // Get the pseudopotential value
                PsiType npsi = psi[neigh_index];
                // Add partial contribution
                for(int d=0; d<DIM; d++){F[d] += E_list[d + qe*DIM] * EW_list[qe] * npsi;}
            }
            for(int d=0; d<DIM; d++){F[d] *= -SC_G * lpsi;}

            // Local density and velocity for Guo velocity shift and equilibrium
            NType ln = n[g_tid]; UType lu[DIM];

            // Guo velocity shift & Copy to global memory
            for(int d=0; d<DIM; d++){ 
                lu[d] = u[g_tid + V*d] + 0.5 * F[d]/ln;
                u[g_tid + V*d] = lu[d];
            }

            // Compute square norm of Guo shifted velocity
            UType u_dot_u = 0.;
            for(int d=0; d<DIM; d++){u_dot_u += lu[d]*lu[d];}

            // Cycle over the populations: equilibrium + Guo
            for(int q=0; q<Q; q++){
                UType u_dot_xi = 0., F_dot_xi = 0., F_dot_u = 0.; 
                for(int d=0; d<DIM; d++){
                    u_dot_xi += lu[d] * XI_list[d + q*DIM];
                    F_dot_xi += F[d] * XI_list[d + q*DIM];
                    F_dot_u  += F[d] * lu[d];
                }

                PopType leq_pop = 1., lguo_pop = 0.;

                // Equilibrium population
                leq_pop += + u_dot_xi*CM2 + 0.5*u_dot_xi*u_dot_xi*CM4;
                leq_pop += - 0.5*u_dot_u*CM2;
                leq_pop = leq_pop * ln * W_list[q];

                // Guo population
                lguo_pop += + F_dot_xi*CM2 + F_dot_xi*u_dot_xi*CM4;
                lguo_pop += - F_dot_u*CM2;
                lguo_pop = lguo_pop * W_list[q];

                pop[g_tid + q*V] = \
                    pop[g_tid + q*V]*(1. - OMEGA) + leq_pop*OMEGA + (1. - 0.5 * OMEGA) * lguo_pop;

             }
        }
        """

class K_ComputePsi(IdpyKernel):
    def __init__(self, custom_types = None, constants = {}, f_classes = [], psi_code = None,
                 optimizer_flag = None):
        if psi_code is None:
            raise Exception("Missing argument psi_code")

        self.psi_code = psi_code
        
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'PsiType * psi': ['global', 'restrict'],
                       'NType * n': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
            NType ln = n[g_tid];
            psi[g_tid] = """ + self.psi_code + """; }"""

class K_StreamPeriodic(IdpyKernel):
    def __init__(self, custom_types = None, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * dst': ['global', 'restrict'],
                       'PopType * src': ['global', 'restrict', 'const'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
            SType dst_pos[DIM], src_pos[DIM];
            F_PosFromIndex(dst_pos, dim_sizes, dim_strides, g_tid);

            // Zero-th population
            dst[g_tid] = src[g_tid];

            // Gather Populations from neighbors
            for(int q=1; q<Q; q++){
                for(int d=0; d<DIM; d++){
                    src_pos[d] = ((dst_pos[d] - XI_list[d + q * DIM] + dim_sizes[d]) % dim_sizes[d]);
                }

                SType src_index = F_IndexFromPos(src_pos, dim_strides);
                dst[g_tid + q * V] = src[src_index + q * V];
            }
        
        }
        """


class K_ComputeMoments(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'NType * n': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'PopType * pop': ['global', 'restrict', 'const'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'WType * W_list': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
            UType lu[DIM];
            for(int d=0; d<DIM; d++){ lu[d] = 0.; }

            NType ln = 0.;
            for(int q=0; q<Q; q++){
                PopType lpop = pop[g_tid + q * V];
                ln += lpop;
                for(int d=0; d<DIM; d++){
                    lu[d] += lpop * XI_list[d + q * DIM];
                }
            }
            n[g_tid] = ln;
            for(int d=0; d<DIM; d++){ u[g_tid + d * V] = lu[d]/ln;}
        }
        """
        
class K_InitPopulations(IdpyKernel):
    def __init__(self, custom_types = None, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'PopType * pop': ['global', 'restrict'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'UType * u': ['global', 'restrict', 'const'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'WType * W_list': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
            NType ln = n[g_tid];

            UType lu[DIM], u_dot_u = 0.;
            // Copying the velocity
            for(int d=0; d<DIM; d++){
                lu[d] = u[g_tid + d * V]; u_dot_u += lu[d];
            }

            // Loop over the populations
            for(int q=0; q<Q; q++){

                UType u_dot_xi = 0.;
                for(int d=0; d<DIM; d++){
                    u_dot_xi += lu[d] * XI_list[d + q*DIM];
                }

                PopType leq_pop = 1.;
                leq_pop += u_dot_xi * CM2;
                leq_pop += 0.5 * u_dot_xi * u_dot_xi * CM4;
                leq_pop -= 0.5 * u_dot_u * CM2;
                leq_pop = leq_pop * ln * W_list[q];
                pop[g_tid + q * V] = leq_pop;

            }
        
        }
        """
            
class K_InitFlatInterface(IdpyKernel):
    '''
    class K_InitFlatInterface:
    need to add a tuning of the launch grid
    so that in extreme cases each thread cycles on more
    than a single point
    '''
    def __init__(self, custom_types = None, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'NType * n': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const'],
                       'SType * dim_center': ['global', 'restrict', 'const'],
                       'NType n_g': ['const'], 'NType n_l': ['const'],
                       'LengthType width': ['const'],
                       'SType direction': ['const'],
                       'FlagType full_flag': ['const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
            SType g_tid_pos[DIM];
            F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);

            NType delta_n = full_flag * (n_l - n_g) + (1 - full_flag) * (n_g - n_l);

            n[g_tid] = 0.5 * (n_g + n_l) + \
            0.5 * delta_n * (F_NFlatProfile(dim_center[direction], g_tid_pos[direction], width) - 1.);

            for(int d=0; d<DIM; d++){
            u[g_tid + d * V] = 0.;
            }
        }
        """

class K_InitRadialInterface(IdpyKernel):
    '''
    class K_InitRadialInterface:
    need to add a tuning of the launch grid
    so that in extreme cases each thread cycles on more
    than a single point
    '''
    def __init__(self, custom_types = None, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'NType * n': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const'],
                       'SType * dim_center': ['global', 'restrict', 'const'],
                       'NType n_g': ['const'], 'NType n_l': ['const'],
                       'LengthType R': ['const'],
                       'FlagType full_flag': ['const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
        SType g_tid_pos[DIM];
        F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);
        LengthType r = F_PointDistanceCenterFirst(dim_center, g_tid_pos);

        NType delta_n = full_flag * (n_l - n_g) + (1 - full_flag) * (n_g - n_l);

        n[g_tid] = 0.5 * (n_g + n_l) - \
        0.5 * delta_n * tanh((LengthType)(r - R));

        for(int d=0; d<DIM; d++){
        u[g_tid + d * V] = 0.;
        }
        
        }
        """

class K_InitCylinderInterface(IdpyKernel):
    '''
    class K_InitRadialInterface:
    need to add a tuning of the launch grid
    so that in extreme cases each thread cycles on more
    than a single point
    '''
    def __init__(self, custom_types = None, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'NType * n': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const'],
                       'SType * dim_center': ['global', 'restrict', 'const'],
                       'NType n_g': ['const'], 'NType n_l': ['const'],
                       'LengthType R': ['const'],
                       'SType direction': ['const'],
                       'FlagType full_flag': ['const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
        SType g_tid_pos[DIM], dim_center_proj[DIM];
        F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);
        for(int d=0; d<DIM; d++) dim_center_proj[d] = dim_center[d];

        g_tid_pos[direction] = 0.;
        dim_center_proj[direction] = 0.;
        
        LengthType r = F_PointDistance(dim_center_proj, g_tid_pos);

        NType delta_n = full_flag * (n_l - n_g) + \
        (1 - full_flag) * (n_g - n_l);

        n[g_tid] = 0.5 * (n_g + n_l) - \
        0.5 * delta_n * tanh((LengthType)(r - R));

        for(int d=0; d<DIM; d++){
        u[g_tid + d * V] = 0.;
        }
        
        }
        """
        
class K_SetPopulationSpike(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop': ['global', 'restrict'],
                       'SType pos_index': ['const']}
        self.kernels[IDPY_T] = """
        if(g_tid < V){
            for(int q=0; q<Q; q++){
                if(g_tid == pos_index){
                    pop[g_tid + q * V] = 1.;
                }else{
                    pop[g_tid + q * V] = 0.;
                }
            }
        }
        """

class K_NInNOut(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes)

        self.SetCodeFlags('g_tid')
        self.params = {'NType * n_in_n_out': ['global'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'unsigned int inside': ['const'],
                       'unsigned int outside': ['const']}

        self.kernels[IDPY_T] = """ 
        if(g_tid < V){
            if(g_tid == inside) n_in_n_out[0] = n[inside];
            if(g_tid == outside) n_in_n_out[1] = n[outside];
        }
        """

        
'''
IdpyMethods
'''
class M_SwapPop(IdpyMethod):
    def __init__(self, tenet = None):
        IdpyMethod.__init__(self, tenet = tenet)

    def Deploy(self, swap_list = None, idpy_stream = None):
        swap_list[0], swap_list[1] = swap_list[1], swap_list[0]

        return IdpyMethod.PassIdpyStream(self, idpy_stream = idpy_stream)

