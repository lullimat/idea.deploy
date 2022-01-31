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

import numpy as np

from idpy.IdpyCode import IdpyMemory
from idpy.LBM.LBM import NPT

from idpy.LBM.LBMKernelsMeta import K_ComputeMomentsMeta, K_CheckUMeta

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
    lbm.sims_vars['delta_p'].append(float(_delta_p))
    lbm.sims_vars['p_in'].append(float(_p_in))
    lbm.sims_vars['p_out'].append(float(_p_out))
    lbm.sims_vars['is_centered_seq'].append(lbm.sims_vars['is_centered'])
    
    return _break_f

def CheckMinMaxAveN(lbm):
    _u_swap = lbm.sims_idpy_memory['n'].D2H()
    print(np.mean(_u_swap), np.amax(_u_swap), np.amin(_u_swap))
    return False

def CheckUConvergenceSCMP(lbm):
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

    lbm.ComputeMomentsMeta()

    _K_CheckUMeta = \
        K_CheckUMeta(
            custom_types = lbm.custom_types.Push(),
            constants = lbm.constants, f_classes = [],
            optimizer_flag = lbm.optimizer_flag,
            use_ptrs = lbm.params_dict['use_ptrs'], 
            ordering_lambda_u = lbm.sims_vars['ordering']['u']
        )

    _K_CheckUMeta(tenet = lbm.tenet, grid = lbm.sims_vars['grid'],
                  block = lbm.sims_vars['block']).Deploy(
                      [lbm.sims_idpy_memory['delta_u'],
                       lbm.sims_idpy_memory['old_u'],
                       lbm.sims_idpy_memory['max_u'],
                       lbm.sims_idpy_memory['u']],
                      idpy_stream = None)

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
