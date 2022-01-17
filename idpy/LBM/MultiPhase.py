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

from idpy.IdpyCode import IDPY_T
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyMethod
from idpy.IdpyCode.IdpyUnroll import _get_seq_macros, _get_seq_vars, _codify_newl
from idpy.IdpyCode.IdpyUnroll import _get_cartesian_coordinates_macro

from idpy.LBM.LBM import RootLB

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


    def MainLoop(self, time_steps, convergence_functions = [], profiling = False):
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
        self.tenet.FreeMemoryDict(memory_dict = self.sims_idpy_memory)
        self.tenet.End()
        ##del self.tenet                 
        
'''
Functions
'''        
class F_NFlatProfile(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'NType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType x': ['const'],
                       'SType x0': ['const'],
                       'LengthType w': ['const']}

        self.functions[IDPY_T] = """
        return tanh((LengthType)(x - (x0 - 0.5 * w))) - tanh((LengthType)(x - (x0 + 0.5 * w)));
        """

class F_NFlatProfilePeriodic(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'NType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType x': ['const'],
                       'SType x0': ['const'],
                       'LengthType w': ['const'],
                       'SType d_size': ['const']}

        self.functions[IDPY_T] = """
        SType xm = d_size/2, xp = 0;
        SType delta = x0 - xm;
        if(delta >= 0){
            if(x < x0 - xm) xp = x - (x0 - xm) + d_size;
            else xp = x - (x0 - xm);
        }else{
            if(x < x0 + xm) xp = x - (x0 + xm) + d_size;
            else xp = x - (x0 + xm);
        }
        return tanh((LengthType)(xp - (xm - 0.5 * w))) - tanh((LengthType)(xp - (xm + 0.5 * w)));
        """

class F_NFlatProfilePeriodicR(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'NType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType x': ['const'],
                       'LengthType x0': ['const'],
                       'LengthType w': ['const'],
                       'SType d_size': ['const']}

        self.functions[IDPY_T] = """
        SType xm = d_size/2;
        LengthType xp = 0;
        SType delta = x0 - xm;
        if(delta >= 0){
            if(x < x0 - xm) xp = x - (x0 - xm) + d_size;
            else xp = x - (x0 - xm);
        }else{
            if(x < x0 + xm) xp = x - (x0 + xm) + d_size;
            else xp = x - (x0 + xm);
        }
        return tanh((LengthType)(xp - (xm - 0.5 * w))) - tanh((LengthType)(xp - (xm + 0.5 * w)));
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
            UType u_dot_u = 0., F_dot_u = 0.;
            for(int d=0; d<DIM; d++){u_dot_u += lu[d]*lu[d]; F_dot_u  += F[d] * lu[d];}

            // Cycle over the populations: equilibrium + Guo
            for(int q=0; q<Q; q++){
                UType u_dot_xi = 0., F_dot_xi = 0.; 
                for(int d=0; d<DIM; d++){
                    u_dot_xi += lu[d] * XI_list[d + q*DIM];
                    F_dot_xi += F[d] * XI_list[d + q*DIM];
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
            0.5 * delta_n * (F_NFlatProfilePeriodic(g_tid_pos[direction], dim_center[direction], width, dim_sizes[direction]) - 1.);

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
