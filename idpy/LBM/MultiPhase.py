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

import sympy as sp
import numpy as np

from idpy.IdpyCode import GetTenet, GetParamsClean, CheckOCLFP, SwitchToFP32

from idpy.IdpyCode import IDPY_T, OCL_T, CUDA_T
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyMethod, IdpyLoop, IdpyLoopProfile
from idpy.IdpyCode.IdpyUnroll import _get_seq_macros, _get_seq_vars, _codify_newl
from idpy.IdpyCode.IdpyUnroll import _get_cartesian_coordinates_macro

from idpy.LBM.LBM import RootLB, LBMTypes, InitFStencilWeights, NPT
from idpy.LBM.LBMKernels import F_PosFromIndex, F_IndexFromPos
from idpy.LBM.LBMKernels import K_InitPopulations, K_ComputeMoments, K_CenterOfMass
from idpy.LBM.LBMKernels import K_StreamPeriodic, K_HalfWayBounceBack, M_SwapPop
from idpy.LBM.LBMKernels import K_ComputeMomentsWalls, K_NInNOut
from idpy.LBM.LBMKernels import K_SetPopNUXInletDutyCycle, K_SetPopNUOutletNoGradient

'''
Importing LBM Meta Kernels
'''
from idpy.LBM.LBMKernelsMeta import K_InitPopulationsMeta, K_StreamPeriodicMeta
from idpy.LBM.LBMKernelsMeta import K_IsotropyFilter

from idpy.Utils.Statements import AllTrue, OneTrue
'''
Importing IdpyFunction's
'''
from idpy.LBM.MultiPhaseKernels import F_NFlatProfile, F_NFlatProfilePeriodic
from idpy.LBM.MultiPhaseKernels import F_NFlatProfilePeriodicR, F_PointDistanceCenterFirst
from idpy.LBM.MultiPhaseKernels import F_PointDistance, F_NSingleFlatProfile
from idpy.LBM.MultiPhaseKernels import K_Collision_ShanChenGuoMultiPhaseWallsGravity

'''
Importing IdpyKernel's
'''
from idpy.LBM.MultiPhaseKernels import K_Collision_ShanChenGuoMultiPhase
from idpy.LBM.MultiPhaseKernels import K_Collision_ShanChenGuoMultiPhaseWalls
from idpy.LBM.MultiPhaseKernels import K_ComputePsi, K_ComputePsiWalls, K_InitFlatInterface
from idpy.LBM.MultiPhaseKernels import K_InitSingleFlatInterface
from idpy.LBM.MultiPhaseKernels import K_InitRadialInterface, K_InitCylinderInterface
from idpy.LBM.MultiPhaseKernels import K_Collision_ShanChenMultiPhase
from idpy.LBM.MultiPhaseKernels import K_Collision_ShanChenMultiPhaseWalls

from idpy.LBM.MultiPhaseKernelsMeta import K_ComputeDensityPsiMeta
from idpy.LBM.MultiPhaseKernelsMeta import K_ComputeVelocityAfterForceSCMPMeta
from idpy.LBM.MultiPhaseKernelsMeta import K_ForceCollideStreamSCMPMeta
from idpy.LBM.MultiPhaseKernelsMeta import K_ForceCollideSCMPMeta
from idpy.LBM.MultiPhaseKernelsMeta import K_ForceCollideStreamSCMP_MRTMeta
from idpy.LBM.MultiPhaseKernelsMeta import K_ForceCollideStreamWallsSCMPMeta
from idpy.LBM.MultiPhaseKernelsMeta import K_ComputeDensityPsiWallsMeta
from idpy.LBM.MultiPhaseKernelsMeta import K_ComputeVelocityAfterForceSCMPWallsMeta

'''
Fluctuating Hydrodynamics kernels
'''
from idpy.LBM.MultiPhaseKernelsMeta import K_ForceGross2011CollideStreamSCMPMeta
from idpy.LBM.Fluctuations import GrossShanChenMultiPhase

from idpy.Utils.CustomTypes import CustomTypes

'''
Importing Congruential Pseudo-Random numbers generator
'''
from idpy.PRNGS.CRNGS import CRNGS

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

    def GetWalls(self, walls, psi_w):
        _psi_w_swap = np.array(psi_w, dtype=NPT.C[self.custom_types['PsiType']])
        self.sims_idpy_memory['psi'].H2D(np.ravel(_psi_w_swap))

        RootLB.GetWalls(self, walls)


    def MainLoopSimple(self, time_steps, convergence_functions = [], profiling = False):
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

        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeMoments(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']),
                         ['n', 'u', 'pop',
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
                                           block = self.sims_vars['block']),
                         ['pop_swap', 'pop',
                          'XI_list',
                          'dim_sizes',
                          'dim_strides']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None

    def MainLoopSimpleSC(self, time_steps, convergence_functions = [], profiling = False):
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

        _K_Collision_ShanChenMultiPhase = \
            K_Collision_ShanChenMultiPhase(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = K_StreamPeriodic(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             f_classes = [F_PosFromIndex,
                                                          F_IndexFromPos],
                                             optimizer_flag = self.optimizer_flag)

        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeMoments(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']),
                         ['n', 'u', 'pop',
                          'XI_list', 'W_list']),
                        (_K_ComputePsi(tenet = self.tenet,
                                       grid = self.sims_vars['grid'],
                                       block = self.sims_vars['block']), ['psi', 'n']),

                        (_K_Collision_ShanChenMultiPhase(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'u', 'n', 'psi',
                          'XI_list', 'W_list',
                          'E_list', 'EW_list',
                          'dim_sizes', 'dim_strides']),
                        
                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']),
                         ['pop_swap', 'pop',
                          'XI_list',
                          'dim_sizes',
                          'dim_strides']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None            

    def MainLoopSimpleWalls(self, time_steps, convergence_functions = [], profiling = False):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_ComputeMomentsWalls = \
            K_ComputeMomentsWalls(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                optimizer_flag = self.optimizer_flag)

        _K_ComputePsiWalls = \
            K_ComputePsiWalls(
                    custom_types = self.custom_types.Push(),
                    constants = self.constants,
                    psi_code = self.params_dict['psi_code'],
                    optimizer_flag = self.optimizer_flag)

        _K_Collision_ShanChenGuoMultiPhaseWalls = \
            K_Collision_ShanChenGuoMultiPhaseWalls(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = \
            K_StreamPeriodic(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _K_HalfWayBounceBack = \
            K_HalfWayBounceBack(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeMomentsWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['n', 'u', 'pop',
                          'XI_list', 'W_list', 'walls']),

                        (_K_ComputePsiWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), ['psi', 'n', 'walls']),

                        (_K_Collision_ShanChenGuoMultiPhaseWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'u', 'n', 'psi',
                          'XI_list', 'W_list',
                          'E_list', 'EW_list',
                          'dim_sizes', 'dim_strides', 'walls']),
                        
                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']),
                         ['pop_swap', 'pop',
                          'XI_list',
                          'dim_sizes',
                          'dim_strides']),

                        (_K_HalfWayBounceBack(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), 
                        ['pop_swap', 'XI_list', 'dim_sizes', 'dim_strides', 
                        'walls', 'xi_opposite']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None

    def MainLoopSimpleWallsSC(self, time_steps, convergence_functions = [], profiling = False):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_ComputeMomentsWalls = \
            K_ComputeMomentsWalls(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                optimizer_flag = self.optimizer_flag)

        _K_ComputePsiWalls = \
            K_ComputePsiWalls(
                    custom_types = self.custom_types.Push(),
                    constants = self.constants,
                    psi_code = self.params_dict['psi_code'],
                    optimizer_flag = self.optimizer_flag)

        _K_Collision_ShanChenMultiPhaseWalls = \
            K_Collision_ShanChenMultiPhaseWalls(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = \
            K_StreamPeriodic(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _K_HalfWayBounceBack = \
            K_HalfWayBounceBack(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeMomentsWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['n', 'u', 'pop',
                          'XI_list', 'W_list', 'walls']),

                        (_K_ComputePsiWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), ['psi', 'n', 'walls']),

                        (_K_Collision_ShanChenMultiPhaseWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'u', 'n', 'psi',
                          'XI_list', 'W_list',
                          'E_list', 'EW_list',
                          'dim_sizes', 'dim_strides', 'walls']),
                        
                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']),
                         ['pop_swap', 'pop',
                          'XI_list',
                          'dim_sizes',
                          'dim_strides']),

                        (_K_HalfWayBounceBack(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), 
                        ['pop_swap', 'XI_list', 'dim_sizes', 'dim_strides', 
                        'walls', 'xi_opposite']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None            

    def MainLoopSimpleWallsInletOutlet(
        self, time_steps, convergence_functions = [], profiling = False,
        n_in=1, u_in=0, tau_in=100, max_mult=2):

        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        self.constants['N_IN'] = n_in
        self.constants['U_IN'] = u_in
        self.constants['TAU_IN'] = tau_in
        self.constants['MAX_MULT'] = max_mult

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_SetPopNUXInletDutyCycle = \
            K_SetPopNUXInletDutyCycle(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                optimizer_flag = self.optimizer_flag)

        _K_SetPopNUOutletNoGradient = \
            K_SetPopNUOutletNoGradient(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes=[F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)            

        _K_ComputeMomentsWalls = \
            K_ComputeMomentsWalls(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                optimizer_flag = self.optimizer_flag)            

        _K_ComputePsiWalls = \
            K_ComputePsiWalls(
                    custom_types = self.custom_types.Push(),
                    constants = self.constants,
                    psi_code = self.params_dict['psi_code'],
                    optimizer_flag = self.optimizer_flag)

        _K_Collision_ShanChenGuoMultiPhaseWalls = \
            K_Collision_ShanChenGuoMultiPhaseWalls(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = \
            K_StreamPeriodic(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _K_HalfWayBounceBack = \
            K_HalfWayBounceBack(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_SetPopNUXInletDutyCycle(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'n', 'u',
                          'XI_list', 'W_list', 'walls', 'idloop_k']),

                        (_K_SetPopNUOutletNoGradient(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'n', 'u',
                          'XI_list', 'W_list', 'walls', 
                          'dim_sizes', 'dim_strides']),                        

                        (_K_ComputeMomentsWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['n', 'u', 'pop',
                          'XI_list', 'W_list', 'walls']),

                        (_K_ComputePsiWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), ['psi', 'n', 'walls']),

                        (_K_Collision_ShanChenGuoMultiPhaseWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'u', 'n', 'psi',
                          'XI_list', 'W_list',
                          'E_list', 'EW_list',
                          'dim_sizes', 'dim_strides', 'walls']),

                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']),
                         ['pop_swap', 'pop',
                          'XI_list',
                          'dim_sizes',
                          'dim_strides']),

                        (_K_HalfWayBounceBack(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), 
                        ['pop_swap', 'XI_list', 'dim_sizes', 'dim_strides', 
                        'walls', 'xi_opposite']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None

    def MainLoopSimpleWallsInletOutletSC(
        self, time_steps, 
        convergence_functions = [], convergence_functions_args = [], 
        profiling = False,
        n_in=1, u_in=0, tau_in=100, max_mult=2):

        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        self.constants['N_IN'] = n_in
        self.constants['U_IN'] = u_in
        self.constants['TAU_IN'] = tau_in
        self.constants['MAX_MULT'] = max_mult

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_SetPopNUXInletDutyCycle = \
            K_SetPopNUXInletDutyCycle(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes=[F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _K_SetPopNUOutletNoGradient = \
            K_SetPopNUOutletNoGradient(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes=[F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)            

        _K_ComputeMomentsWalls = \
            K_ComputeMomentsWalls(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                optimizer_flag = self.optimizer_flag)            

        _K_ComputePsiWalls = \
            K_ComputePsiWalls(
                    custom_types = self.custom_types.Push(),
                    constants = self.constants,
                    psi_code = self.params_dict['psi_code'],
                    optimizer_flag = self.optimizer_flag)

        _K_Collision_ShanChenMultiPhaseWalls = \
            K_Collision_ShanChenMultiPhaseWalls(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = \
            K_StreamPeriodic(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _K_HalfWayBounceBack = \
            K_HalfWayBounceBack(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_SetPopNUOutletNoGradient(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'n', 'u',
                          'XI_list', 'W_list', 'walls', 
                          'dim_sizes', 'dim_strides']),

                        (_K_SetPopNUXInletDutyCycle(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'n', 'u',
                          'XI_list', 'W_list', 'walls', 
                          'dim_sizes', 'dim_strides', 'idloop_k']),                                               

                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']),
                         ['pop_swap', 'pop', 'XI_list',
                          'dim_sizes', 'dim_strides']),

                        (_K_HalfWayBounceBack(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), 
                        ['pop_swap', 'XI_list', 'dim_sizes', 'dim_strides', 'walls', 'xi_opposite']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop']),

                        (_K_ComputeMomentsWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['n', 'u', 'pop',
                          'XI_list', 'W_list', 'walls']),

                        (_K_ComputePsiWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), ['psi', 'n', 'walls']),

                        (_K_Collision_ShanChenMultiPhaseWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'u', 'n', 'psi',
                          'XI_list', 'W_list',
                          'E_list', 'EW_list',
                          'dim_sizes', 'dim_strides', 'walls']),
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_i, c_f in enumerate(convergence_functions):
                    checks.append(c_f(self, **convergence_functions_args[c_i]))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None                        

    def MainLoopSimpleWallsGravity(
        self, time_steps, convergence_functions = [], 
        g=0, profiling = False):

        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_ComputeMomentsWalls = \
            K_ComputeMomentsWalls(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                optimizer_flag = self.optimizer_flag)

        _K_ComputePsiWalls = \
            K_ComputePsiWalls(
                    custom_types = self.custom_types.Push(),
                    constants = self.constants,
                    psi_code = self.params_dict['psi_code'],
                    optimizer_flag = self.optimizer_flag)

        _K_Collision_ShanChenGuoMultiPhaseWallsGravity = \
            K_Collision_ShanChenGuoMultiPhaseWallsGravity(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = \
            K_StreamPeriodic(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        _K_HalfWayBounceBack = \
            K_HalfWayBounceBack(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex, F_IndexFromPos],
                optimizer_flag = self.optimizer_flag)

        '''
        Explicit casting to numpy type
        '''
        adding_g_dict = {'g': NPT.C[self.custom_types['ForceType']](g)}

        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile
        
        self._MainLoop = \
            _loop_class(
                [{**self.sims_idpy_memory, **adding_g_dict}],
                [
                    [
                        (_K_ComputeMomentsWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['n', 'u', 'pop',
                          'XI_list', 'W_list', 'walls']),

                        (_K_ComputePsiWalls(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), ['psi', 'n', 'walls']),

                        (_K_Collision_ShanChenGuoMultiPhaseWallsGravity(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']),
                         ['pop', 'u', 'n', 'psi',
                          'XI_list', 'W_list',
                          'E_list', 'EW_list', 'g',
                          'dim_sizes', 'dim_strides', 'walls']),
                        
                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']),
                         ['pop_swap', 'pop',
                          'XI_list',
                          'dim_sizes',
                          'dim_strides']),

                        (_K_HalfWayBounceBack(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']), 
                        ['pop_swap', 'XI_list', 'dim_sizes', 'dim_strides', 
                        'walls', 'xi_opposite']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None

    '''
    Need to pass: psi, tau, and forcing stencil overriding the initial choice
    '''
    def MainLoop(self, time_steps, convergence_functions = [], profiling = False):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")

        _K_ComputeDensityPsiMeta = \
            K_ComputeDensityPsiMeta(
                custom_types = self.custom_types.Push(), 
                constants = self.constants, f_classes = [], 
                XIStencil = self.params_dict['xi_stencil'], 
                use_ptrs = self.params_dict['use_ptrs'],
                ordering_lambda = self.sims_vars['ordering']['pop'], 
                psi_code = self.params_dict['psi_code']
            )

        _K_ForceCollideStreamSCMPMeta = \
            K_ForceCollideStreamSCMPMeta(
                custom_types = self.custom_types.Push(), 
                constants = self.constants, f_classes = [],
                optimizer_flag = self.optimizer_flag,
                XIStencil = self.params_dict['xi_stencil'], 
                SCFStencil = self.params_dict['f_stencil'],
                use_ptrs = self.params_dict['use_ptrs'],
                ordering_lambda_pop = self.sims_vars['ordering']['pop'],
                ordering_lambda_u = self.sims_vars['ordering']['u'],
                collect_mul = False, pressure_mode = 'compute',
                root_dim_sizes = self.params_dict['root_dim_sizes'],
                root_strides = self.params_dict['root_strides'],
                root_coord = self.params_dict['root_coord']
            )
        
        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile

        ### return _K_ComputeDensityPsiMeta, _K_ForceCollideStreamSCMPMeta
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeDensityPsiMeta(tenet = self.tenet,
                                                  grid = self.sims_vars['grid'],
                                                  block = self.sims_vars['block']),
                         ['n', 'psi', 'pop']),
                        
                        (_K_ForceCollideStreamSCMPMeta(tenet = self.tenet,
                                                       grid = self.sims_vars['grid'],
                                                       block = self.sims_vars['block']),
                         ['pop_swap', 'pop', 'n', 'psi']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None

    def MainLoopIsoFilter(self, time_steps, convergence_functions = [], profiling = False):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")

        _K_ComputeDensityPsiMeta = \
            K_ComputeDensityPsiMeta(
                custom_types = self.custom_types.Push(), 
                constants = self.constants, f_classes = [], 
                XIStencil = self.params_dict['xi_stencil'], 
                use_ptrs = self.params_dict['use_ptrs'],
                ordering_lambda = self.sims_vars['ordering']['pop'], 
                psi_code = self.params_dict['psi_code']
            )

        _K_ForceCollideSCMPMeta = \
            K_ForceCollideSCMPMeta(
                custom_types = self.custom_types.Push(), 
                constants = self.constants, f_classes = [],
                optimizer_flag = self.optimizer_flag,
                XIStencil = self.params_dict['xi_stencil'], 
                SCFStencil = self.params_dict['f_stencil'],
                use_ptrs = self.params_dict['use_ptrs'],
                ordering_lambda_pop = self.sims_vars['ordering']['pop'],
                ordering_lambda_u = self.sims_vars['ordering']['u'],
                collect_mul = False, pressure_mode = 'compute',
                root_dim_sizes = self.params_dict['root_dim_sizes'],
                root_strides = self.params_dict['root_strides'],
                root_coord = self.params_dict['root_coord']
            )

        _K_IsotropyFilter = \
            K_IsotropyFilter(
                custom_types = self.custom_types.Push(), 
                constants = self.constants, f_classes = [],
                optimizer_flag = self.optimizer_flag,
                XIStencil = self.params_dict['xi_stencil'],
                use_ptrs = self.params_dict['use_ptrs'],
                search_depth=6,
                declare_const_dict = {'new_pop': True},
                output_pop = 'pop_iso',
                ordering_lambda_pop = self.sims_vars['ordering']['pop'],
                collect_mul = False
            )

        _K_StreamPeriodicMeta = \
            K_StreamPeriodicMeta(
                custom_types = self.custom_types.Push(),
                constants = self.constants, f_classes = [], 
                pressure_mode = 'compute',
                optimizer_flag = self.optimizer_flag,
                XIStencil = self.params_dict['xi_stencil'],
                collect_mul = False,
                stream_mode = 'push',
                root_dim_sizes = self.params_dict['root_dim_sizes'],
                root_strides = self.params_dict['root_strides'], 
                use_ptrs = self.params_dict['use_ptrs'],
                root_coord = self.params_dict['root_coord'],
                ordering_lambda = self.sims_vars['ordering']['pop']
            )
        
        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile

        ### return _K_ComputeDensityPsiMeta, _K_ForceCollideStreamSCMPMeta
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [                        
                        (_K_ComputeDensityPsiMeta(tenet = self.tenet,
                                                  grid = self.sims_vars['grid'],
                                                  block = self.sims_vars['block']),
                         ['n', 'psi', 'pop']),
                        
                        (_K_ForceCollideSCMPMeta(tenet = self.tenet,
                                                       grid = self.sims_vars['grid'],
                                                       block = self.sims_vars['block']),
                         ['pop', 'n', 'psi']),

                        (_K_IsotropyFilter(
                            tenet = self.tenet,
                            grid = self.sims_vars['grid'],
                            block = self.sims_vars['block']
                        ), ['pop', 'pop']),                        

                        (_K_StreamPeriodicMeta(
                            tenet=self.tenet,
                            grid=self.sims_vars['grid'],
                            block=self.sims_vars['block']
                        ), ['pop_swap', 'pop']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None
        

    def MainLoopWalls(self, time_steps, convergence_functions = [], profiling = False):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")

        _K_ComputeDensityPsiWallsMeta = \
            K_ComputeDensityPsiWallsMeta(
                custom_types = self.custom_types.Push(), 
                constants = self.constants, f_classes = [], 
                XIStencil = self.params_dict['xi_stencil'], 
                use_ptrs = self.params_dict['use_ptrs'],
                ordering_lambda = self.sims_vars['ordering']['pop'], 
                psi_code = self.params_dict['psi_code']
            )

        _K_ForceCollideStreamWallsSCMPMeta = \
            K_ForceCollideStreamWallsSCMPMeta(
                custom_types = self.custom_types.Push(), 
                constants = self.constants, f_classes = [],
                optimizer_flag = None, 
                XIStencil = self.params_dict['xi_stencil'], 
                SCFStencil = self.params_dict['f_stencil'], 
                use_ptrs = self.params_dict['use_ptrs'], 
                ordering_lambda_pop = self.sims_vars['ordering']['pop'], 
                ordering_lambda_u = self.sims_vars['ordering']['u'],
                collect_mul = False, stream_mode = 'push', pressure_mode = 'compute',
                root_dim_sizes = self.params_dict['root_dim_sizes'], 
                root_strides = self.params_dict['root_strides'], 
                root_coord = self.params_dict['root_coord'], 
                walls_array_var = 'walls'
            )
        
        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile

        ### return _K_ComputeDensityPsiMeta, _K_ForceCollideStreamSCMPMeta
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeDensityPsiWallsMeta(tenet = self.tenet,
                                                  grid = self.sims_vars['grid'],
                                                  block = self.sims_vars['block']),
                         ['n', 'psi', 'pop', 'walls']),
                        
                        (_K_ForceCollideStreamWallsSCMPMeta(tenet = self.tenet,
                                                            grid = self.sims_vars['grid'],
                                                            block = self.sims_vars['block']),
                         ['pop_swap', 'pop', 'n', 'psi', 'walls']),

                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None                    

    def MainLoopMRT(self, time_steps, relaxation_matrix = None, omega_syms_vals = None,
                    convergence_functions = [], profiling = False):
        
        if relaxation_matrix is None:
            raise Exception("Missing argument 'relaxation_matrix'")
        if omega_syms_vals is None:
            raise Exception("Missing argument 'omega_syms_vals'")
        
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")

        _K_ComputeDensityPsiMeta = \
            K_ComputeDensityPsiMeta(
                custom_types = self.custom_types.Push(), 
                constants = self.constants, f_classes = [], 
                XIStencil = self.params_dict['xi_stencil'], 
                use_ptrs = self.params_dict['use_ptrs'],
                ordering_lambda = self.sims_vars['ordering']['pop'], 
                psi_code = self.params_dict['psi_code']
            )

        _K_ForceCollideStreamSCMP_MRTMeta = \
            K_ForceCollideStreamSCMP_MRTMeta(
                custom_types = self.custom_types.Push(), 
                constants = self.constants, f_classes = [],
                optimizer_flag = self.optimizer_flag,
                XIStencil = self.params_dict['xi_stencil'], 
                SCFStencil = self.params_dict['f_stencil'],
                use_ptrs = self.params_dict['use_ptrs'],
                relaxation_matrix = relaxation_matrix,
                omega_syms_vals = omega_syms_vals,                
                ordering_lambda_pop = self.sims_vars['ordering']['pop'],
                ordering_lambda_u = self.sims_vars['ordering']['u'],
                collect_mul = False, pressure_mode = 'compute',
                root_dim_sizes = self.params_dict['root_dim_sizes'],
                root_strides = self.params_dict['root_strides'],
                root_coord = self.params_dict['root_coord']
            )
        
        _loop_class = IdpyLoop if not profiling else IdpyLoopProfile

        ### return _K_ComputeDensityPsiMeta, _K_ForceCollideStreamSCMPMeta
        
        self._MainLoop = \
            _loop_class(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeDensityPsiMeta(tenet = self.tenet,
                                                  grid = self.sims_vars['grid'],
                                                  block = self.sims_vars['block']),
                         ['n', 'psi', 'pop']),
                        
                        (_K_ForceCollideStreamSCMP_MRTMeta(tenet = self.tenet,
                                                           grid = self.sims_vars['grid'],
                                                           block = self.sims_vars['block']),
                         ['pop_swap', 'pop', 'n', 'psi']),
                        
                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        for step in time_steps[1:]:
            print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break

        if profiling:
            return _profiling_results
        else:
            return None

    def MainLoopGross2011SRT(self, time_steps, 
                             convergence_functions = [], convergence_functions_args = [],
                             profiling = False,
                             kBT = None, n0 = None, print_flag = True):

        if kBT is None:
            raise Exception("Missing parameter 'kBT'")
        elif kBT < 0:
            raise Exception("Paramter 'kBT' must be positive")
        
        if n0 is None:
            raise Exception("Missing parameter 'n0'")

        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")    

        _compile_flag = True
        if 'LoopGross2011SRT_once' not in self.sims_vars:
            self.sims_vars['LoopGross2011SRT_once'] = True
            self.sims_vars['LoopGross2011SRT_n0'] = n0
            self.sims_vars['LoopGross2011SRT_kBT'] = kBT
        else:
            if (self.sims_vars['LoopGross2011SRT_n0'] == n0 and
                self.sims_vars['LoopGross2011SRT_kBT'] == kBT):
                _compile_flag = False
            else:
                self.sims_vars['LoopGross2011SRT_n0'] = n0
                self.sims_vars['LoopGross2011SRT_kBT'] = kBT

        '''
        Trying to save time on the generation and compilation of the meta kernels
        another possibility would be that of saving the kernel objects in 
        sims_vars, so that at least the algebra is saved but then I do not pass kBT as
        a parameter
        '''
        if _compile_flag:
            _K_ComputeDensityPsiMeta = \
                K_ComputeDensityPsiMeta(
                    custom_types = self.custom_types.Push(), 
                    constants = self.constants, f_classes = [], 
                    XIStencil = self.params_dict['xi_stencil'], 
                    use_ptrs = self.params_dict['use_ptrs'],
                    ordering_lambda = self.sims_vars['ordering']['pop'], 
                    psi_code = self.params_dict['psi_code']
                )

            _K_ForceGross2011CollideStreamSCMPMeta = \
                K_ForceGross2011CollideStreamSCMPMeta(
                    custom_types = self.custom_types.Push(), 
                    constants = self.constants, 
                    f_classes = [], optimizer_flag = self.optimizer_flag,
                    XIStencil = self.params_dict['xi_stencil'], 
                    SCFStencil = self.params_dict['f_stencil'], 
                    use_ptrs = self.params_dict['use_ptrs'],
                    ordering_lambda_pop = self.sims_vars['ordering']['pop'], 
                    ordering_lambda_u = self.sims_vars['ordering']['u'],
                    ordering_lambda_prng = \
                    lambda _i: self.sims_vars['ordering']['crng']('g_tid', _i), 
                    kBT = kBT, n0 = n0,
                    distribution =  self.params_dict['prng_distribution'],
                    generator = self.params_dict['prng_kind'],
                    parallel_streams = self.constants['N_PRNG_STREAMS'],
                    collect_mul = False, pressure_mode = 'compute',
                    root_dim_sizes = self.params_dict['root_dim_sizes'],
                    root_strides = self.params_dict['root_strides'],
                    root_coord = self.params_dict['root_coord']
                )

            _loop_class = IdpyLoop if not profiling else IdpyLoopProfile

            ### return _K_ComputeDensityPsiMeta, _K_ForceCollideStreamSCMPMeta

            self._MainLoop = \
                _loop_class(
                    [{**self.sims_idpy_memory, **self.crng.sims_idpy_memory}],
                    [
                        [
                            (_K_ComputeDensityPsiMeta(tenet = self.tenet,
                                                      grid = self.sims_vars['grid'],
                                                      block = self.sims_vars['block']),
                             ['n', 'psi', 'pop']),

                            (_K_ForceGross2011CollideStreamSCMPMeta(
                                tenet = self.tenet,
                                grid = self.sims_vars['grid'],
                                block = self.sims_vars['block']
                            ),
                             ['pop_swap', 'seeds', 'pop', 'n', 'psi']),

                            (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop'])
                        ]
                    ]
                )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step, _profiling_results = 0, {}
        '''
        Deaclaring a list for the time steps
        '''
        self.sims_vars['time_steps'] = []

        for step in time_steps[1:]:
            if print_flag:
                print("Step:", step)
            elif (step % (2 ** 20)) == 0:
                print("Step:", step)
            '''
            Very simple timing, reasonable for long executions
            '''
            _profiling_results[step] = self._MainLoop.Run(range(step - old_step))
            self.sims_vars['time_steps'] += [step]

            old_step = step
            if len(convergence_functions):
                checks = []
                if len(convergence_functions_args):
                    for c_i, c_f in enumerate(convergence_functions):
                        checks.append(c_f(self, **convergence_functions_args[c_i]))
                else:
                    for c_f in convergence_functions:
                        checks.append(c_f(self))

                if OneTrue(checks):
                    break

        self.sims_vars['time_steps'] = np.array(self.sims_vars['time_steps'])

        if profiling:
            return _profiling_results
        else:
            return None ## _K_ForceGross2011CollideStreamSCMPMeta            
        
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

    def ComputeMomentsMeta(self):
        if not self.init_status['pop']:
            raise Exception("Populations are not initialized")

        '''
        Need to fix the code: temporary for comparison fields
        '''

        if 'walls' not in self.sims_idpy_memory:
            _K_Psi = \
                K_ComputeDensityPsiMeta(
                    custom_types = self.custom_types.Push(), 
                    constants = self.constants, f_classes = [], 
                    XIStencil = self.params_dict['xi_stencil'], 
                    use_ptrs = self.params_dict['use_ptrs'],
                    ordering_lambda = self.sims_vars['ordering']['pop'], 
                    psi_code = self.params_dict['psi_code']
                )        

            _K_Velocity = \
                K_ComputeVelocityAfterForceSCMPMeta(
                    custom_types = self.custom_types.Push(),
                    constants = self.constants, f_classes = [],
                    optimizer_flag = self.optimizer_flag,
                    XIStencil = self.params_dict['xi_stencil'], 
                    SCFStencil = self.params_dict['f_stencil'],
                    use_ptrs = self.params_dict['use_ptrs'],
                    ordering_lambda_pop = self.sims_vars['ordering']['pop'],
                    ordering_lambda_u= self.sims_vars['ordering']['u'],
                    collect_mul = False, pressure_mode = 'compute',
                    root_dim_sizes = self.params_dict['root_dim_sizes'],
                    root_strides = self.params_dict['root_strides'],
                    root_coord = self.params_dict['root_coord']
                )

        else:
            _K_Psi = \
                K_ComputeDensityPsiWallsMeta(
                    custom_types = self.custom_types.Push(), 
                    constants = self.constants, f_classes = [], 
                    XIStencil = self.params_dict['xi_stencil'], 
                    use_ptrs = self.params_dict['use_ptrs'],
                    ordering_lambda = self.sims_vars['ordering']['pop'], 
                    psi_code = self.params_dict['psi_code']
                )        

            _K_Velocity = \
                K_ComputeVelocityAfterForceSCMPWallsMeta(
                    custom_types = self.custom_types.Push(),
                    constants = self.constants, f_classes = [],
                    optimizer_flag = self.optimizer_flag,
                    XIStencil = self.params_dict['xi_stencil'], 
                    SCFStencil = self.params_dict['f_stencil'],
                    use_ptrs = self.params_dict['use_ptrs'],
                    ordering_lambda_pop = self.sims_vars['ordering']['pop'],
                    ordering_lambda_u= self.sims_vars['ordering']['u'],
                    collect_mul = False, pressure_mode = 'compute',
                    root_dim_sizes = self.params_dict['root_dim_sizes'],
                    root_strides = self.params_dict['root_strides'],
                    root_coord = self.params_dict['root_coord'], 
                    walls_array_var = 'walls'
                )                

        IdeaPsi = _K_Psi(tenet = self.tenet,
                         grid = self.sims_vars['grid'],
                         block = self.sims_vars['block'])            

        IdeaVelocity = \
            _K_Velocity(tenet = self.tenet,
                        grid = self.sims_vars['grid'],
                        block = self.sims_vars['block'])

        _stream = IdeaPsi.Deploy([self.sims_idpy_memory['n'],
                                  self.sims_idpy_memory['psi'],
                                  self.sims_idpy_memory['pop']] + \
                                  (
                                      [self.sims_idpy_memory['walls']] 
                                      if 'walls' in self.sims_idpy_memory else
                                      []
                                  )
                                 )

        _stream = [_stream] if self.params_dict['lang'] == OCL_T else None        
        
        IdeaVelocity.Deploy([self.sims_idpy_memory['n'],
                             self.sims_idpy_memory['u'],
                             self.sims_idpy_memory['pop'],
                             self.sims_idpy_memory['psi']] + \
                             (
                                 [self.sims_idpy_memory['walls']] 
                                 if 'walls' in self.sims_idpy_memory else
                                 []
                             ),
                            idpy_stream = _stream)

        self.init_status['n'] = True
        self.init_status['u'] = True


    def InitPopulations(self):
        if not AllTrue([self.init_status['n'], self.init_status['u']]):
            raise Exception("Fields u and n are not initialized")

        _K_InitPopulationsMeta = \
            K_InitPopulationsMeta(
                custom_types = self.custom_types.Push(),
                constants = self.constants, f_classes = [],
                optimizer_flag = self.optimizer_flag,
                XIStencil = self.params_dict['xi_stencil'],
                use_ptrs = self.params_dict['use_ptrs'], 
                ordering_lambda_pop = self.sims_vars['ordering']['pop'], 
                ordering_lambda_u = self.sims_vars['ordering']['u'],
                pressure_mode = 'registers', order = 2
            )
        
        Idea = _K_InitPopulationsMeta(tenet = self.tenet,
                                      grid = self.sims_vars['grid'],
                                      block = self.sims_vars['block'])
        
        Idea.Deploy([self.sims_idpy_memory['pop'],
                     self.sims_idpy_memory['n'],
                     self.sims_idpy_memory['u']])
        
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
                                             F_NFlatProfilePeriodic],
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

    def InitSingleFlatInterface(self, n_g, n_l, direction=0, full_flag=True):
        '''
        Record init values
        '''
        self.sims_vars['init_type'] = 'flat'
        self.sims_vars['n_g'], self.sims_vars['n_l'] = n_g, n_l
        self.sims_vars['full_flag'] = full_flag
        self.sims_vars['direction'] = direction
        
        _K_InitSingleFlatInterface = \
            K_InitSingleFlatInterface(
                custom_types = self.custom_types.Push(),
                constants = self.constants,
                f_classes = [F_PosFromIndex,
                             F_NSingleFlatProfile],
                optimizer_flag = self.optimizer_flag)

        n_g = NPT.C[self.custom_types['NType']](n_g)
        n_l = NPT.C[self.custom_types['NType']](n_l)
        direction = NPT.C[self.custom_types['SType']](direction)
        full_flag = NPT.C[self.custom_types['FlagType']](full_flag)
        
        Idea = \
            _K_InitSingleFlatInterface(
                tenet = self.tenet,
                grid = self.sims_vars['grid'],
                block = self.sims_vars['block']
                )

        Idea.Deploy([self.sims_idpy_memory['n'],
                     self.sims_idpy_memory['u'],
                     self.sims_idpy_memory['dim_sizes'],
                     self.sims_idpy_memory['dim_strides'],
                     self.sims_idpy_memory['dim_center'],
                     n_g, n_l, direction, full_flag])

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
        self.sims_vars['tau'] = self.params_dict['tau']
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

        if self.params_dict['fluctuations'] is not None:
            '''
            Setting up the ordering lambdas
            '''
            self.sims_vars['ordering_lambdas']['gpu']['crng'] = \
                lambda _pos, _i: str(_pos) + ' + ' + str(_i) + ' * V'
            self.sims_vars['ordering_lambdas']['cpu']['crng'] = \
                lambda _pos, _i: str(_i) + ' + ' + str(_pos) + ' * N_PRNG_STREAMS'        

        _device_type = self.tenet.GetKind()
        if self.params_dict['set_ordering'] is not None:
            _device_type = self.params_dict['set_ordering']
        else:
            self.params_dict['set_ordering'] = _device_type
            
        self.sims_vars['ordering'] = \
            self.sims_vars['ordering_lambdas'][_device_type]

        '''
        constants
        '''
        self.constants['QE'] = self.sims_vars['QE']
        self.constants['SC_G'] = self.params_dict['SC_G']
        self.constants['OMEGA'] = 1./self.params_dict['tau']

        '''
        Macros for meta-programming kernels
        '''
        if False:
            ## dim_sizes
            _swap_macros_sizes = \
                _get_seq_macros(self.sims_vars['DIM'], self.params_dict['root_dim_sizes'])
            for _i, _ in enumerate(self.sims_vars['dim_sizes']):
                self.constants[_swap_macros_sizes[_i]] = _

            ## dim_strides
            _swap_macros_strides = \
                _get_seq_macros(self.sims_vars['DIM'] - 1, self.params_dict['root_strides'])
            for _i, _ in enumerate(self.sims_vars['dim_strides']):
                self.constants[_swap_macros_strides[_i]] = _

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

        if self.params_dict['fluctuations'] is not None:       
            _n_prngs = 0
            '''
            Gross2011
            '''
            if self.params_dict['fluctuations'] == 'Gross2011':
                _n_rand_mom = self.sims_vars['Q'] - (self.sims_vars['DIM'] + 1)
                if self.params_dict['prng_distribution'] == 'flat':
                    _n_prngs = _n_rand_mom * self.sims_vars['V']
                    self.constants['N_PRNG_STREAMS'] = _n_rand_mom
                    
                if self.params_dict['prng_distribution'] == 'gaussian':
                    if self.params_dict['indep_gaussian']:
                        _n_prngs = 2 * _n_rand_mom * self.sims_vars['V']
                        self.constants['N_PRNG_STREAMS'] = 2 * _n_rand_mom
                    else:
                        _n_prngs = _n_rand_mom * self.sims_vars['V']
                        self.constants['N_PRNG_STREAMS'] = _n_rand_mom
                
            optional_seed = \
                {} if self.params_dict['prng_init_from'] == 'urandom' else \
                {'init_seed': self.params_dict['init_seed']}

                
            self.crng = CRNGS(n_prngs = _n_prngs,
                              kind = self.params_dict['prng_kind'],
                              init_from = self.params_dict['prng_init_from'],
                              lang = self.params_dict['lang'],
                              cl_kind = self.params_dict['cl_kind'],
                              device = self.params_dict['device'],
                              tenet = self.tenet, 
                              **optional_seed)

            self.constants = {**self.constants, **self.crng.constants}
            self.custom_types = \
                CustomTypes(types_dict = {**self.custom_types.Push(),
                                          **self.crng.custom_types.Push()})


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

        self.kwargs = \
            GetParamsClean(
                kwargs, [self.params_dict],
                needed_params = \
                ['lang', 'cl_kind', 'device',
                 'custom_types', 'block_size',
                 'f_stencil', 'psi_code', 'SC_G',
                 'tau', 'optimizer_flag', 'e2_val',
                 'psi_sym', 'empty_sim',
                 'set_ordering', 'use_ptrs',
                 'fluctuations', 'prng_kind', 'init_seed',
                 'prng_init_from', 'prng_distribution',
                 'indep_gaussian', 'fp32_flag']
            )

        if 'f_stencil' not in self.params_dict:
            raise Exception("Missing 'f_stencil'")

        if 'tau' not in self.params_dict:
            raise Exception("Missing 'tau'")
        
        if 'SC_G' not in self.params_dict:
            raise Exception("Missing 'SC_G'")

        if 'psi_code' not in self.params_dict:
            raise Exception("Missing 'psi_code', e.g. psi_code = '(PsiType) exp(-1/ln_0)'")

        if 'lang' not in self.params_dict:
            raise Exception("Param lang = CUDA_T | OCL_T | CTYPES_T is needed")

        if 'psi_sym' not in self.params_dict:
            raise Exception(
                "Missing sympy expression for the pseudo-potential, parameter 'psi_sym'"
            )

        if 'optimizer_flag' in self.params_dict:
            self.optimizer_flag = self.params_dict['optimizer_flag']
        else:
            self.optimizer_flag = True

        if 'set_ordering' not in self.params_dict:
            self.params_dict['set_ordering'] = None
        elif self.params_dict['set_ordering'] not in ['gpu', 'cpu', None]:
            raise Exception("Parameter 'set_ordering' must either be 'gpu' or 'cpu'")

        if 'use_ptrs' not in self.params_dict:
            self.params_dict['use_ptrs'] = True

        if 'empty_sim' not in self.params_dict:
            self.params_dict['empty_sim'] = False

        self.tenet = GetTenet(self.params_dict)
        if 'custom_types' in self.params_dict:
            self.custom_types = self.params_dict['custom_types']
        else:
            self.custom_types = LBMTypes

        if 'fluctuations' not in self.params_dict:
            self.params_dict['fluctuations'] = None

        if 'fp32_flag' not in self.params_dict:
            self.params_dict['fp32_flag'] = False

        self.custom_types = \
            CheckOCLFP(tenet = self.tenet, custom_types = self.custom_types)
        
        if self.params_dict['fp32_flag']:
            self.custom_types = SwitchToFP32(self.custom_types)

        print(self.custom_types.Push())
            
        RootLB.__init__(self, *args, custom_types = self.custom_types,
                         **self.kwargs)
        
    def End(self):
        self.tenet.FreeMemoryDict(memory_dict = self.sims_idpy_memory)
        self.tenet.End()
        ##del self.tenet                 
        
