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
Provides unit tests for the idpy.LBM module
'''

import unittest

import numpy as np
import sys, os, filecmp, inspect
from functools import reduce
from collections import defaultdict

##np.set_printoptions(threshold=sys.maxsize)

_file_abs_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_idea_dot_deploy_path = os.path.dirname(os.path.abspath(_file_abs_path + "../../"))
sys.path.append(_idea_dot_deploy_path)

def AllTrue(list_swap):
    return reduce(lambda x, y: x and y, list_swap)

from idpy.LBM.LBM import RootLB, LBMTypes, NPT
from idpy.LBM.LBM import XIStencils, FStencils
from idpy.LBM.LBM import ShanChenMultiPhase
from idpy.LBM.LBM import K_SetPopulationSpike, K_StreamPeriodic, M_SwapPop


from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T
from idpy.IdpyCode import idpy_langs_sys, idpy_langs_list

from idpy.IdpyCode import IdpyMemory, GetTenet, CheckOCLFP
from idpy.IdpyCode.IdpyCode import IdpyLoop

if idpy_langs_sys[OCL_T]:
        from idpy.OpenCL.OpenCL import OpenCL
if idpy_langs_sys[CUDA_T]:
        from idpy.CUDA.CUDA import CUDA


class TestShanChenMultiPhase(unittest.TestCase):
    def setUp(self):
        self.dim_sizes_2d, self.dim_sizes_3d = (32, 34), (32, 34, 36)
        self.hdf5_name_2d, self.hdf5_name_3d = 'test_hdf5_2d', 'test_hdf5_3d'        

    def test_ShanChenMultiPhase(self):
        sc_multi_phase = ShanChenMultiPhase(lang = OCL_T,
                                            dim_sizes = self.dim_sizes_2d,
                                            xi_stencil = XIStencils['D2Q9'],
                                            f_stencil = FStencils['D2E4'],
                                            psi_code = 'exp(-1./ln)')

        print(sc_multi_phase.sims_idpy_memory)
        print(sc_multi_phase.sims_vars)
        sc_multi_phase.InitFlatInterface(width = 7, n_g = 0.5, n_l = 1.0)
        print(sc_multi_phase.sims_idpy_memory['u'].D2H())
        a = sc_multi_phase.sims_idpy_memory['n'].D2H()
        print(a)
        sc_multi_phase.End()

        self.assertTrue(True)

    def TestStreamPeriodic(self):
        sc_multi_phase = ShanChenMultiPhase(lang = OCL_T,
                                            dim_sizes = self.dim_sizes_2d,
                                            xi_stencil = XIStencils['D2Q9'])
        
        sc_multi_phase.sims_idpy_memory['pop'].SetConst(0)
        sc_multi_phase.sims_idpy_memory['pop_swap'].SetConst(0)

        _SetPopulationSpike = K_SetPopulationSpike(custom_types = sc_multi_phase.custom_types.Push(),
                                                   constants = sc_multi_phase.constants,
                                                   f_classes = [])

        
        Idea = _SetPopulationSpike(tenet = sc_multi_phase.tenet,
                                   grid = sc_multi_phase.sims_vars['grid'],
                                   block = sc_multi_phase.sims_vars['block'])

        Idea.Deploy([sc_multi_phase.sims_idpy_memory['pop'], np.int32(0)])

        _StreamPeriodic = K_StreamPeriodic(custom_types = sc_multi_phase.custom_types.Push(),
                                           constants = sc_multi_phase.constants,
                                           f_classes = [F_IndexFromPos,
                                                        F_PosFromIndex])

        args_list = [sc_multi_phase.sims_idpy_memory['pop_swap'], sc_multi_phase.sims_idpy_memory['pop'],
                     sc_multi_phase.sims_idpy_memory['XI_list'],
                     sc_multi_phase.sims_idpy_memory['dim_sizes'],
                     sc_multi_phase.sims_idpy_memory['dim_strides']]
        
        StreamLoop = IdpyLoop([sc_multi_phase.sims_idpy_memory],
                              [
                                  [
                                      (_StreamPeriodic(tenet = sc_multi_phase.tenet,
                                                       grid = sc_multi_phase.sims_vars['grid'],
                                                       block = sc_multi_phase.sims_vars['block']),
                                       ['pop_swap', 'pop', 'XI_list', 'dim_sizes', 'dim_strides']),
                                      (M_SwapPop(tenet = sc_multi_phase.tenet), ['pop_swap', 'pop'])
                                  ]
                              ])

        StreamLoop.Run(range(1))

        sc_multi_phase.End()
        self.assertTrue(True)


class TestRootLB(unittest.TestCase):
    def setUp(self):
        self.test_vars = defaultdict(dict)
        self.test_vars['dim_sizes_2d'] = (32, 32)
        self.test_vars['dim_sizes_3d'] = (32, 34, 36)
        
        self.dim_sizes_2d, self.dim_sizes_3d = (32, 32), (32, 34, 36)
        self.hdf5_name_2d, self.hdf5_name_3d = 'test_hdf5_2d.hdf5', 'test_hdf5_3d.hdf5'
        self.block_size = 128

    def test_DumpSnapshot(self):
        checks = [True]

        for lang in idpy_langs_list:            
            if idpy_langs_sys[lang]:
                tenet = GetTenet({'lang': lang})

                CT = CheckOCLFP(tenet = tenet,
                                custom_types = LBMTypes)

                print("LBMTypes: ", CT.Push())
                
                if os.path.exists(self.hdf5_name_2d):
                    os.remove(self.hdf5_name_2d)
                if os.path.exists(self.hdf5_name_3d):
                    os.remove(self.hdf5_name_3d)

                root_lb_2d = RootLB(dim_sizes = self.dim_sizes_2d,
                                    xi_stencil = XIStencils['D2Q9'],
                                    custom_types = CT)
                root_lb_2d.InitMemory(tenet, custom_types = CT)
                root_lb_2d.DumpSnapshot(file_name = self.hdf5_name_2d,
                                        custom_types = CT)

                root_lb_3d = RootLB(dim_sizes = self.dim_sizes_3d,
                                    xi_stencil = XIStencils['D3Q19'],
                                    custom_types = CT)
                root_lb_3d.InitMemory(tenet, custom_types = CT)
                root_lb_3d.DumpSnapshot(file_name = self.hdf5_name_3d,
                                        custom_types = CT)
                
                if os.path.exists(self.hdf5_name_2d):
                    os.remove(self.hdf5_name_2d)
                if os.path.exists(self.hdf5_name_3d):
                    os.remove(self.hdf5_name_3d)
                
                tenet.End()
            '''
            lang loop end
            '''

        self.assertTrue(AllTrue(checks))


    def test_RootInitMemory(self):
        checks = []

        for lang in idpy_langs_list:
            if idpy_langs_sys[lang]:
                tenet = GetTenet({'lang': lang})

                CT = CheckOCLFP(tenet = tenet,
                                custom_types = LBMTypes)

                print("LBMTypes: ", CT.Push())

                root_lb_2d = RootLB(dim_sizes = self.dim_sizes_2d,
                                    xi_stencil = XIStencils['D2Q9'],
                                    custom_types = CT)
                root_lb_2d.InitMemory(tenet = tenet, custom_types = CT)

                root_lb_3d = RootLB(dim_sizes = self.dim_sizes_3d,
                                    xi_stencil = XIStencils['D3Q19'],
                                    custom_types = CT)
                root_lb_3d.InitMemory(tenet = tenet, custom_types = CT)

                checks = []
                '''
                two dimensions
                '''
                print()
                print("root_lb_2d.sims_idpy_memory['pop']: ",
                      root_lb_2d.sims_idpy_memory['pop'].D2H(),
                      root_lb_2d.sims_idpy_memory['pop'].dtype)
                
                checks += [AllTrue(list(root_lb_2d.sims_idpy_memory['pop'].D2H() ==
                                        np.zeros(root_lb_2d.sims_idpy_memory['pop'].shape,
                                                 NPT.C[CT['PopType']])))]
                checks += [root_lb_2d.sims_idpy_memory['pop'].dtype == NPT.C[CT['PopType']]]

                print("root_lb_2d.sims_idpy_memory['dim_sizes']: ",
                      root_lb_2d.sims_idpy_memory['dim_sizes'].D2H(),
                      root_lb_2d.sims_idpy_memory['dim_sizes'].dtype)
                checks += [AllTrue(list(root_lb_2d.sims_idpy_memory['dim_sizes'].D2H() ==
                                        np.array(self.dim_sizes_2d,
                                                 dtype = NPT.C[CT['SType']])))]
                checks += [root_lb_2d.sims_idpy_memory['dim_sizes'].dtype == NPT.C[CT['SType']]]

                print("root_lb_2d.sims_idpy_memory['dim_strides']: ",
                      root_lb_2d.sims_idpy_memory['dim_strides'].D2H(),
                      root_lb_2d.sims_idpy_memory['dim_strides'].dtype)
                checks += [AllTrue(list(root_lb_2d.sims_idpy_memory['dim_strides'].D2H() ==
                                        root_lb_2d.sims_vars['dim_strides']))]
                checks += [root_lb_2d.sims_idpy_memory['dim_strides'].dtype == NPT.C[CT['SType']]]

                print("root_lb_2d.sims_idpy_memory['XI_list']: ",
                      root_lb_2d.sims_idpy_memory['XI_list'].D2H(),
                      root_lb_2d.sims_idpy_memory['XI_list'].dtype)
                checks += [AllTrue(list(root_lb_2d.sims_idpy_memory['XI_list'].D2H() ==
                                        root_lb_2d.sims_vars['XI_list']))]
                checks += [root_lb_2d.sims_idpy_memory['XI_list'].dtype == NPT.C[CT['SType']]]

                print("root_lb_2d.sims_idpy_memory['W_list']: ",
                      root_lb_2d.sims_idpy_memory['W_list'].D2H(),
                      root_lb_2d.sims_idpy_memory['W_list'].dtype)
                checks += [AllTrue(list(root_lb_2d.sims_idpy_memory['W_list'].D2H() ==
                                        root_lb_2d.sims_vars['W_list']))]
                checks += [root_lb_2d.sims_idpy_memory['W_list'].dtype == NPT.C[CT['WType']]]
                '''
                three dimensions
                '''
                print()
                print("root_lb_3d.sims_idpy_memory['pop']: ",
                      root_lb_3d.sims_idpy_memory['pop'].D2H(),
                      root_lb_3d.sims_idpy_memory['pop'].dtype)
                
                checks += [AllTrue(list(root_lb_3d.sims_idpy_memory['pop'].D2H() ==
                                        np.zeros(root_lb_3d.sims_idpy_memory['pop'].shape,
                                                 NPT.C[CT['PopType']])))]
                checks += [root_lb_3d.sims_idpy_memory['pop'].dtype == NPT.C[CT['PopType']]]

                print("root_lb_3d.sims_idpy_memory['dim_sizes']: ",
                      root_lb_3d.sims_idpy_memory['dim_sizes'].D2H(),
                      root_lb_3d.sims_idpy_memory['dim_sizes'].dtype)
                checks += [AllTrue(list(root_lb_3d.sims_idpy_memory['dim_sizes'].D2H() ==
                                        np.array(self.dim_sizes_3d,
                                                 dtype = NPT.C[CT['SType']])))]
                checks += [root_lb_3d.sims_idpy_memory['dim_sizes'].dtype == NPT.C[CT['SType']]]

                print("root_lb_3d.sims_idpy_memory['dim_strides']: ",
                      root_lb_3d.sims_idpy_memory['dim_strides'].D2H(),
                      root_lb_3d.sims_idpy_memory['dim_strides'].dtype)
                checks += [AllTrue(list(root_lb_3d.sims_idpy_memory['dim_strides'].D2H() ==
                                        root_lb_3d.sims_vars['dim_strides']))]
                checks += [root_lb_3d.sims_idpy_memory['dim_strides'].dtype == NPT.C[CT['SType']]]
                
                print("root_lb_3d.sims_idpy_memory['XI_list']: ",
                      root_lb_3d.sims_idpy_memory['XI_list'].D2H(),
                      root_lb_3d.sims_idpy_memory['XI_list'].dtype)
                checks += [AllTrue(list(root_lb_3d.sims_idpy_memory['XI_list'].D2H() ==
                                        root_lb_3d.sims_vars['XI_list']))]
                checks += [root_lb_3d.sims_idpy_memory['XI_list'].dtype == NPT.C[CT['SType']]]

                print("root_lb_3d.sims_idpy_memory['W_list']: ",
                      root_lb_3d.sims_idpy_memory['W_list'].D2H(),
                      root_lb_3d.sims_idpy_memory['W_list'].dtype)
                checks += [AllTrue(list(root_lb_3d.sims_idpy_memory['W_list'].D2H() ==
                                        root_lb_3d.sims_vars['W_list']))]
                checks += [root_lb_3d.sims_idpy_memory['W_list'].dtype == NPT.C[CT['WType']]]
                                
                tenet.End()
                                
            '''
            lang loop end
            '''

        self.assertTrue(AllTrue(checks))
        
        
    def test_RootLBVars(self):
        print()
        print("LBMTypes: ")
        print(LBMTypes.Push())
        checks = [True]
        root_lb_2d = RootLB(dim_sizes = self.dim_sizes_2d,
                            xi_stencil = XIStencils['D2Q9'],
                            custom_types = LBMTypes)

        root_lb_3d = RootLB(dim_sizes = self.dim_sizes_3d,
                            xi_stencil = XIStencils['D3Q19'],
                            custom_types = LBMTypes)
        print()
        for key in root_lb_2d.sims_vars:
            print("2D", key, ":", root_lb_2d.sims_vars[key])

        print()
        for key in root_lb_3d.sims_vars:
            print("3D", key, ":", root_lb_3d.sims_vars[key])

        '''
        2d checks
        '''
        checks += [AllTrue(list(root_lb_2d.sims_vars['dim_sizes'] == self.dim_sizes_2d))]
        checks += [root_lb_2d.sims_vars['dim_sizes'].dtype == NPT.C[LBMTypes['SType']]]
        checks += [AllTrue(list(root_lb_2d.sims_vars['dim_strides']\
                                == np.array([self.dim_sizes_2d[0]],
                                            NPT.C[LBMTypes['SType']])))]
        checks += [root_lb_2d.sims_vars['dim_strides'].dtype == NPT.C[LBMTypes['SType']]]
        checks += [root_lb_2d.sims_vars['V'] == self.dim_sizes_2d[0] * self.dim_sizes_2d[1]]

        '''
        3d checks
        '''
        checks += [AllTrue(list(root_lb_3d.sims_vars['dim_sizes'] == self.dim_sizes_3d))]
        checks += [root_lb_3d.sims_vars['dim_sizes'].dtype == NPT.C[LBMTypes['SType']]]
        checks += [AllTrue(list(root_lb_3d.sims_vars['dim_strides'] == \
                                np.array([self.dim_sizes_3d[0],
                                          self.dim_sizes_3d[0] * self.dim_sizes_3d[1]],
                                         NPT.C[LBMTypes['SType']])))]
        checks += [root_lb_3d.sims_vars['dim_strides'].dtype == NPT.C[LBMTypes['SType']]]
        checks += [root_lb_3d.sims_vars['V'] == \
                   self.dim_sizes_3d[0] * self.dim_sizes_3d[1] * self.dim_sizes_3d[2]]

        self.assertTrue(AllTrue(checks))

from idpy.LBM.DQ import LatticeVectors as XI_LatticeVectors
from idpy.LBM.DQ import Weights as XI_Weights

class TestDQ(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_LatticeVectors(self):
        XI = XI_LatticeVectors(x_max = 1)
        W = XI_Weights(XI)
        W.TypicalSolution()
        print()
        print(XI.lv_len_2)
        print(W.w_sol_c)
        self.assertTrue(True)

