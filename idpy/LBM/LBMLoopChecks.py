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

from idpy.LBM.LBMKernels import K_CheckU

from idpy.IdpyCode import IdpyMemory
from idpy.LBM.LBM import NPT

from idpy.Utils.Statements import AllTrue, OneTrue

def GetSnapshotN(lbm):
    first_flag = False

    if 'snapshots_n_k' not in lbm.sims_idpy_memory:
        lbm.sims_idpy_memory['snapshots_n_k'] = 0
        first_flag = True

    _n_swap = np.copy(lbm.sims_idpy_memory['n'].D2H())
    _n_swap = _n_swap.reshape(np.flip(lbm.sims_vars['dim_sizes']))
    _dim = len(lbm.sims_vars['dim_sizes'])

    if _dim == 3:
        _n_swap = _n_swap[lbm.sims_vars['dim_sizes'][2]//2,:,:] 

    _k_fig = lbm.sims_idpy_memory['snapshots_n_k']
    _fig = plt.figure()
    plt.imshow(_n_swap, origin = 'lower')
    plt.savefig('./snapshot_' + ('%010d' % (_k_fig)) + '.png', dpi = 150)
    plt.close()

    lbm.sims_idpy_memory['snapshots_n_k'] += 1
    
    return False

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
