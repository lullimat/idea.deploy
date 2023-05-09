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

from idpy.IdpyCode.IdpyCode import IdpyKernel
from idpy.IdpyStencils.IdpyStencils import IdpyStencil
from idpy.LBM.IDShanChenForce import IDShanChenForce
from idpy.LBM.Forcings import GuoForcing
from idpy.LBM.Equilibria import HermiteEquilibria
from idpy.LBM.Collisions import BGK
from idpy.LBM.Fluctuations import GrossShanChenMultiPhase

from idpy.IdpyCode.IdpyUnroll import _codify_comment, _get_cartesian_coordinates_macro
from idpy.IdpyCode.IdpyUnroll import _get_seq_macros, _codify_newl, _codify_assignment
from idpy.IdpyCode.IdpyUnroll import _array_value, _codify_declaration_const_check

from idpy.IdpyCode import IDPY_T

class K_ComputeDensityPsiMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, 
                 XIStencil = None, use_ptrs = False, ordering_lambda = None,
                 psi_code = None, headers_files=['math.h']):
        
        self.expect_lambda_args = 2
        
        if ordering_lambda is None:
            raise Exception("Missing argument 'ordering_lambda'")
        elif ordering_lambda.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion("The number of arguments for 'ordering_lambda' should be ",
                            self.expect_lambda_args)

        if XIStencil is None:
            raise Exception("Missing argument 'XIStencil'")
        if psi_code is None:
            raise Exception("Missing argument 'psi_code'")
        
        
        self.idpy_stencil = IdpyStencil(XIStencil)
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants, 
                            f_classes = f_classes, optimizer_flag = optimizer_flag, 
                            headers_files=headers_files)
        
        '''
        Kernel Parameters declaration
        '''        
        self.SetCodeFlags('g_tid')
        self.params = {'NType * n': ['global', 'restrict'], 
                       'PsiType * psi': ['global', 'restrict'], 
                       'PopType * pop': ['global', 'restrict', 'const']}
        
        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()
        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        self.idpy_stencil.ZeroHermiteCode(declared_variables = self.declared_variables, 
                                          declared_constants = self.declared_constants, 
                                          arrays = ['pop'], arrays_types = ['PopType'],
                                          root_n = 'ln', n_type = 'NType',
                                          lex_index = 'g_tid', keep_read = True,
                                          ordering_lambdas = [ordering_lambda], 
                                          use_ptrs = use_ptrs, 
                                          declare_const_dict = {'arrays_xi': True,
                                                                'moments': True}) + \
        _codify_declaration_const_check('lpsi', psi_code, 'PsiType', 
                                        self.declared_variables, 
                                        self.declared_constants, 
                                        declare_const_flag = True) + \
        _codify_assignment(_array_value('psi', 'g_tid', use_ptrs = use_ptrs), 'lpsi') + \
        _codify_assignment(_array_value('n', 'g_tid', use_ptrs = use_ptrs), 'ln_0') + \
        """
        }
        """

class K_ForceCollideStreamSCMPMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, 
                 XIStencil = None, SCFStencil = None, use_ptrs = False, 
                 ordering_lambda_pop = None, ordering_lambda_u = None,
                 collect_mul = False, stream_mode = 'push', pressure_mode = 'compute',
                 root_dim_sizes = 'L', root_strides = 'STR', root_coord = 'x'):
        
        self.expect_lambda_args = 2

        if ordering_lambda_pop is None:
            raise Exception("Missing argument 'ordering_lambda_pop'")
        elif ordering_lambda_pop.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion(
                "The number of arguments for 'ordering_lambda_pop' should be ",
                self.expect_lambda_args
            )

        if XIStencil is None:
            raise Exception("Missing argument 'XIStencil'")
        if SCFStencil is None:
            raise Exception("Missing argument 'SCFStencil'")

        self.idpy_stencil = IdpyStencil(XIStencil)
        self.idpy_sc_force = IDShanChenForce(SCFStencil)
        self.idpy_guo_forcing = GuoForcing(XIStencil, root_u = 'lu_0', root_f = 'd_psi', 
                                           root_omega = '\\omega')
        self.idpy_equilibria = HermiteEquilibria(XIStencil, root_pop = 'lpop', 
                                                 root_n = 'ln_0', root_u = 'lu_0')
        self.idpy_collision = \
            BGK(root_omega = '\\omega', omega_val = constants['OMEGA'])
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants,
                            f_classes = f_classes, optimizer_flag = optimizer_flag)

        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop_swap': ['global', 'restrict'],
                       'PopType * pop': ['global', 'restrict', 'const'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'PsiType * psi': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()
        self.dim_sizes_macros = _get_seq_macros(self.constants['DIM'], root_dim_sizes)
        self.dim_strides_macros = \
            _get_seq_macros(
                self.constants['DIM'] - 1 if self.constants['DIM'] > 1 else 1, 
                root_strides)

        '''
        Kernel Body
        '''
        # 1. Getting Cartesian coordinates
        # 2. Compute the Shan-Chen force
        # 3. Getting the local populations values and compute the local velocity
        # 4. Guo-shift the velocity
        # 5. Collide with Guo's forcing on local populations
        # 6. Stream in 'push' mode
        
        _swap_kernel_idpy = """"""
        _swap_kernel_idpy += _codify_comment("Getting Cartesian Coordinates")
        _swap_kernel_idpy += \
            _get_cartesian_coordinates_macro(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                _root_coordinate = root_coord,
                _dim_sizes = self.dim_sizes_macros,
                _dim_strides = self.dim_strides_macros,
                _type = 'SType', declare_const_flag = True,
                _lexicographic_index = 'g_tid'
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Computing the Shan-Chen Force")        
        _swap_kernel_idpy += \
            self.idpy_sc_force.ForceCodeMultiPhase(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                pressure_mode = pressure_mode, psi_array = 'psi', 
                use_ptrs = use_ptrs, 
                root_dim_sizes = root_dim_sizes, 
                root_strides = root_strides,
                root_coord = root_coord, 
                lex_index = 'g_tid'
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Loading the local value of the density")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'ln_0', _array_value('n', 'g_tid', use_ptrs), 'NType', 
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                declare_const_flag = True
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += \
            _codify_comment("Loading the local value of the populations and keep it")
        _swap_kernel_idpy += \
            _codify_comment("Computing the local value of the velocity")        
        _swap_kernel_idpy += \
            self.idpy_stencil.VelocityCode(
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                arrays = ['pop'], arrays_types = ['PopType'],
                n_type = 'NType', u_type = 'UType',
                root_n = 'ln', root_u = 'lu', 
                lex_index = 'g_tid', keep_read = True,
                ordering_lambdas = [ordering_lambda_pop], 
                use_ptrs = use_ptrs, 
                declare_const_dict = {'arrays_xi': True, 'moments': False}
            )
        
        _swap_kernel_idpy += _codify_comment("Guo's Velocity shift")
        _swap_kernel_idpy += \
            self.idpy_guo_forcing.ShiftVelocityCode(
                n_sets = ['ln_0'], u_sets = ['lu_0'], f_sets = ['d_psi']
            )
        _swap_kernel_idpy += _codify_newl
               
        '''
        Preparing the variables for collision plus streaming according to the pressure mode
        '''
        _swap_tuples_eq_prod, _swap_tuples_guo_prod = [], []        

        _swap_code, _swap_tuples_eq_prod = \
            self.idpy_equilibria.SetMomentsProductsCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                order = 2, root_n = 'ln_0', root_u = 'lu_0', 
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl

        _swap_code, _swap_tuples_guo_prod = \
            self.idpy_guo_forcing.SetMomentsProductsCode(
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                order = 2, root_f = 'd_psi', root_u = 'lu_0',
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Computing equilibrium, collision and stream")
        _swap_kernel_idpy += \
            self.idpy_collision.SRTCollisionPlusGuoPushStreamCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants, 
                ordering_lambda = ordering_lambda_pop,
                dst_arrays_var = 'pop_swap',
                stencil_obj = self.idpy_stencil, 
                eq_obj = self.idpy_equilibria, 
                guo_obj = self.idpy_guo_forcing,
                neq_pop = 'lpop', pressure_mode = pressure_mode,
                tuples_eq = _swap_tuples_eq_prod, 
                tuples_guo = _swap_tuples_guo_prod,
                pos_type = 'SType', use_ptrs = use_ptrs, collect_mul = collect_mul, 
                root_dim_sizes = root_dim_sizes, root_strides = root_strides, 
                root_coord = root_coord, lex_index = 'g_tid',
                declare_const_dict = {'cartesian_coord_neigh': True}
            )

        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        _swap_kernel_idpy + \
        """
        }
        """

class K_ForceCollideStreamWallsSCMPMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, 
                 XIStencil = None, SCFStencil = None, use_ptrs = False, 
                 ordering_lambda_pop = None, ordering_lambda_u = None,
                 collect_mul = False, stream_mode = 'push', pressure_mode = 'compute',
                 root_dim_sizes = 'L', root_strides = 'STR', root_coord = 'x', 
                 walls_array_var = 'walls'):
        
        self.expect_lambda_args = 2

        if ordering_lambda_pop is None:
            raise Exception("Missing argument 'ordering_lambda_pop'")
        elif ordering_lambda_pop.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion(
                "The number of arguments for 'ordering_lambda_pop' should be ",
                self.expect_lambda_args
            )

        if XIStencil is None:
            raise Exception("Missing argument 'XIStencil'")
        if SCFStencil is None:
            raise Exception("Missing argument 'SCFStencil'")

        self.idpy_stencil = IdpyStencil(XIStencil)
        self.idpy_sc_force = IDShanChenForce(SCFStencil)
        self.idpy_guo_forcing = GuoForcing(XIStencil, root_u = 'lu_0', root_f = 'd_psi', 
                                           root_omega = '\\omega')
        self.idpy_equilibria = HermiteEquilibria(XIStencil, root_pop = 'lpop', 
                                                 root_n = 'ln_0', root_u = 'lu_0')
        self.idpy_collision = \
            BGK(root_omega = '\\omega', omega_val = constants['OMEGA'])
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants,
                            f_classes = f_classes, optimizer_flag = optimizer_flag)

        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop_swap': ['global', 'restrict'],
                       'PopType * pop': ['global', 'restrict', 'const'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'PsiType * psi': ['global', 'restrict', 'const'], 
                       'FlagType * walls': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()
        self.dim_sizes_macros = _get_seq_macros(self.constants['DIM'], root_dim_sizes)
        self.dim_strides_macros = \
            _get_seq_macros(
                self.constants['DIM'] - 1 if self.constants['DIM'] > 1 else 1, 
                root_strides)

        '''
        Kernel Body
        '''
        # 1. Getting Cartesian coordinates
        # 2. Compute the Shan-Chen force
        # 3. Getting the local populations values and compute the local velocity
        # 4. Guo-shift the velocity
        # 5. Collide with Guo's forcing on local populations
        # 6. Stream in 'push' mode
        
        _swap_kernel_idpy = """"""
        _swap_kernel_idpy += _codify_comment("Getting Cartesian Coordinates")
        _swap_kernel_idpy += \
            _get_cartesian_coordinates_macro(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                _root_coordinate = root_coord,
                _dim_sizes = self.dim_sizes_macros,
                _dim_strides = self.dim_strides_macros,
                _type = 'SType', declare_const_flag = True,
                _lexicographic_index = 'g_tid'
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Computing the Shan-Chen Force")        
        _swap_kernel_idpy += \
            self.idpy_sc_force.ForceCodeMultiPhase(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                pressure_mode = pressure_mode, psi_array = 'psi', 
                use_ptrs = use_ptrs, 
                root_dim_sizes = root_dim_sizes, 
                root_strides = root_strides,
                root_coord = root_coord, 
                lex_index = 'g_tid'
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Loading the local value of the density")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'ln_0', _array_value('n', 'g_tid', use_ptrs), 'NType', 
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                declare_const_flag = True
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += \
            _codify_comment("Loading the local value of the populations and keep it")
        _swap_kernel_idpy += \
            _codify_comment("Computing the local value of the velocity")        
        _swap_kernel_idpy += \
            self.idpy_stencil.VelocityCode(
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                arrays = ['pop'], arrays_types = ['PopType'],
                n_type = 'NType', u_type = 'UType',
                root_n = 'ln', root_u = 'lu', 
                lex_index = 'g_tid', keep_read = True,
                ordering_lambdas = [ordering_lambda_pop], 
                use_ptrs = use_ptrs, 
                declare_const_dict = {'arrays_xi': True, 'moments': False}
            )
        
        _swap_kernel_idpy += _codify_comment("Guo's Velocity shift")
        _swap_kernel_idpy += \
            self.idpy_guo_forcing.ShiftVelocityCode(
                n_sets = ['ln_0'], u_sets = ['lu_0'], f_sets = ['d_psi']
            )
        _swap_kernel_idpy += _codify_newl
               
        '''
        Preparing the variables for collision plus streaming according to the pressure mode
        '''
        _swap_tuples_eq_prod, _swap_tuples_guo_prod = [], []        

        _swap_code, _swap_tuples_eq_prod = \
            self.idpy_equilibria.SetMomentsProductsCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                order = 2, root_n = 'ln_0', root_u = 'lu_0', 
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl

        _swap_code, _swap_tuples_guo_prod = \
            self.idpy_guo_forcing.SetMomentsProductsCode(
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                order = 2, root_f = 'd_psi', root_u = 'lu_0',
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Computing equilibrium, collision and stream")
        _swap_kernel_idpy += \
            self.idpy_collision.SRTCollisionPlusGuoPushStreamWallsCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants, 
                ordering_lambda = ordering_lambda_pop,
                dst_arrays_var = 'pop_swap',
                stencil_obj = self.idpy_stencil, 
                eq_obj = self.idpy_equilibria, 
                guo_obj = self.idpy_guo_forcing,
                neq_pop = 'lpop', pressure_mode = pressure_mode,
                tuples_eq = _swap_tuples_eq_prod, 
                tuples_guo = _swap_tuples_guo_prod,
                pos_type = 'SType', use_ptrs = use_ptrs, collect_mul = collect_mul, 
                root_dim_sizes = root_dim_sizes, root_strides = root_strides, 
                root_coord = root_coord, lex_index = 'g_tid', 
                walls_array_var = walls_array_var,
                declare_const_dict = {'cartesian_coord_neigh': True}
            )

        
        self.kernels[IDPY_T] = """
        if(g_tid < V && """ + _array_value(walls_array_var, 'g_tid', use_ptrs) + \
        """ == 1){
        """ + \
        _swap_kernel_idpy + \
        """
        }
        """


class K_ForceCollideStreamSCMP_MRTMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None,
                 XIStencil = None, SCFStencil = None, use_ptrs = False,
                 relaxation_matrix = None, omega_syms_vals = None,
                 ordering_lambda_pop = None, ordering_lambda_u = None,
                 collect_mul = False, stream_mode = 'push', pressure_mode = 'compute',
                 root_dim_sizes = 'L', root_strides = 'STR', root_coord = 'x'):
        
        self.expect_lambda_args = 2

        if ordering_lambda_pop is None:
            raise Exception("Missing argument 'ordering_lambda_pop'")
        elif ordering_lambda_pop.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion(
                "The number of arguments for 'ordering_lambda_pop' should be ",
                self.expect_lambda_args
            )

        if XIStencil is None:
            raise Exception("Missing argument 'XIStencil'")
        if SCFStencil is None:
            raise Exception("Missing argument 'SCFStencil'")

        if relaxation_matrix is None:
            raise Exception("Missing argument 'relaxation_matrix'")
        if omega_syms_vals is None:
            raise Exception("Missing argument 'omega_syms_vals'")        

        self.idpy_stencil = IdpyStencil(XIStencil)
        self.idpy_sc_force = IDShanChenForce(SCFStencil)
        self.idpy_guo_forcing = GuoForcing(XIStencil, root_u = 'lu_0', root_f = 'd_psi', 
                                           root_omega = '\\omega')
        self.idpy_equilibria = HermiteEquilibria(XIStencil, root_pop = 'lpop', 
                                                 root_n = 'ln_0', root_u = 'lu_0')
        self.idpy_collision = \
            BGK(root_omega = '\\omega', omega_val = constants['OMEGA'])
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants,
                            f_classes = f_classes, optimizer_flag = optimizer_flag)

        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop_swap': ['global', 'restrict'],
                       'PopType * pop': ['global', 'restrict', 'const'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'PsiType * psi': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()
        self.dim_sizes_macros = _get_seq_macros(self.constants['DIM'], root_dim_sizes)
        self.dim_strides_macros = \
            _get_seq_macros(
                self.constants['DIM'] - 1 if self.constants['DIM'] > 1 else 1, 
                root_strides)

        '''
        Kernel Body
        '''
        # 1. Getting Cartesian coordinates
        # 2. Compute the Shan-Chen force
        # 3. Getting the local populations values and compute the local velocity
        # 4. Guo-shift the velocity
        # 5. Collide with Guo's forcing on local populations
        # 6. Stream in 'push' mode
        
        _swap_kernel_idpy = """"""
        _swap_kernel_idpy += _codify_comment("Getting Cartesian Coordinates")
        _swap_kernel_idpy += \
            _get_cartesian_coordinates_macro(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                _root_coordinate = root_coord,
                _dim_sizes = self.dim_sizes_macros,
                _dim_strides = self.dim_strides_macros,
                _type = 'SType', declare_const_flag = True,
                _lexicographic_index = 'g_tid'
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Computing the Shan-Chen Force")        
        _swap_kernel_idpy += \
            self.idpy_sc_force.ForceCodeMultiPhase(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                pressure_mode = pressure_mode, psi_array = 'psi', 
                use_ptrs = use_ptrs, 
                root_dim_sizes = root_dim_sizes, 
                root_strides = root_strides,
                root_coord = root_coord, 
                lex_index = 'g_tid'
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Loading the local value of the density")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'ln_0', _array_value('n', 'g_tid', use_ptrs), 'NType', 
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                declare_const_flag = True
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += \
            _codify_comment("Loading the local value of the populations and keep it")
        _swap_kernel_idpy += \
            _codify_comment("Computing the local value of the velocity")        
        _swap_kernel_idpy += \
            self.idpy_stencil.VelocityCode(
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                arrays = ['pop'], arrays_types = ['PopType'],
                n_type = 'NType', u_type = 'UType',
                root_n = 'ln', root_u = 'lu', 
                lex_index = 'g_tid', keep_read = True,
                ordering_lambdas = [ordering_lambda_pop], 
                use_ptrs = use_ptrs, 
                declare_const_dict = {'arrays_xi': True, 'moments': False}
            )
        
        _swap_kernel_idpy += _codify_comment("Guo's Velocity shift")
        _swap_kernel_idpy += \
            self.idpy_guo_forcing.ShiftVelocityCode(
                n_sets = ['ln_0'], u_sets = ['lu_0'], f_sets = ['d_psi']
            )
        _swap_kernel_idpy += _codify_newl
               
        '''
        Preparing the variables for collision plus streaming according to the pressure mode
        '''
        _swap_tuples_eq_prod, _swap_tuples_guo_prod = [], []        

        _swap_code, _swap_tuples_eq_prod = \
            self.idpy_equilibria.SetMomentsProductsCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                order = 2, root_n = 'ln_0', root_u = 'lu_0', 
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl

        _swap_code, _swap_tuples_guo_prod = \
            self.idpy_guo_forcing.SetMomentsProductsCode(
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                order = 2, root_f = 'd_psi', root_u = 'lu_0',
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Computing equilibrium, collision and stream")
        _swap_kernel_idpy += \
            self.idpy_collision.MRTCollisionPlusGuoPushStreamCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants, 
                ordering_lambda = ordering_lambda_pop,
                dst_arrays_var = 'pop_swap',
                stencil_obj = self.idpy_stencil, 
                eq_obj = self.idpy_equilibria, 
                guo_obj = self.idpy_guo_forcing,
                relaxation_matrix = relaxation_matrix,
                omega_syms_vals = omega_syms_vals,                
                neq_pop = 'lpop', pressure_mode = pressure_mode,
                tuples_eq = _swap_tuples_eq_prod, 
                tuples_guo = _swap_tuples_guo_prod,
                pos_type = 'SType', use_ptrs = use_ptrs, collect_mul = collect_mul, 
                root_dim_sizes = root_dim_sizes, root_strides = root_strides, 
                root_coord = root_coord, lex_index = 'g_tid',
                declare_const_dict = {'cartesian_coord_neigh': True}
            )

        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        _swap_kernel_idpy + \
        """
        }
        """
        
class K_ComputeVelocityAfterForceSCMPMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None,
                 XIStencil = None, SCFStencil = None, use_ptrs = False,
                 ordering_lambda_pop = None, ordering_lambda_u = None,
                 collect_mul = False, pressure_mode = 'compute',
                 root_dim_sizes = 'L', root_strides = 'STR', root_coord = 'x'):

        self.expect_lambda_args = 2
        
        if ordering_lambda_pop is None:
            raise Exception("Missing argument 'ordering_lambda_pop'")
        elif ordering_lambda_pop.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion(
                "The number of arguments for 'ordering_lambda_pop' should be ", 
                self.expect_lambda_args
            )

        if ordering_lambda_u is None:
            raise Exception("Missing argument 'ordering_lambda_u'")
        elif ordering_lambda_u.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion(
                "The number of arguments for 'ordering_lambda_u' should be ", 
                self.expect_lambda_args
            )
        
        if XIStencil is None:
            raise Exception("Missing argument 'XIStencil'")
        if SCFStencil is None:
            raise Exception("Missing argument 'SCFStencil'")
        
                        
        self.idpy_stencil = IdpyStencil(XIStencil)
        self.idpy_sc_force = IDShanChenForce(SCFStencil)
        self.idpy_guo_forcing = GuoForcing(XIStencil, root_u = 'lu_0', root_f = 'd_psi', 
                                           root_omega = '\\omega')
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants, 
                            f_classes = f_classes, optimizer_flag = optimizer_flag)
        
        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'NType * n': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'PopType * pop': ['global', 'restrict', 'const'],
                       'PsiType * psi': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()
        self.dim_sizes_macros = _get_seq_macros(self.constants['DIM'], root_dim_sizes)
        self.dim_strides_macros = \
            _get_seq_macros(
                self.constants['DIM'] - 1 if self.constants['DIM'] > 1 else 1, 
                root_strides)

        '''
        Kernel Body
        '''
        _swap_kernel_idpy = """"""
        _swap_kernel_idpy += _codify_comment("Getting Cartesian Coordinates")
        _swap_kernel_idpy += \
            _get_cartesian_coordinates_macro(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                _root_coordinate = root_coord,
                _dim_sizes = self.dim_sizes_macros,
                _dim_strides = self.dim_strides_macros,
                _type = 'SType', declare_const_flag = True,
                _lexicographic_index = 'g_tid'
            )
        _swap_kernel_idpy += _codify_newl

        _swap_kernel_idpy += _codify_comment("Computing the Shan-Chen Force")        
        _swap_kernel_idpy += \
            self.idpy_sc_force.ForceCodeMultiPhase(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                pressure_mode = pressure_mode, psi_array = 'psi', 
                use_ptrs = use_ptrs, 
                root_dim_sizes = root_dim_sizes, 
                root_strides = root_strides,
                root_coord = root_coord, 
                lex_index = 'g_tid'
            )
        _swap_kernel_idpy += _codify_newl

        _swap_kernel_idpy += _codify_comment("Loading the local value of the density")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'ln_0', _array_value('n', 'g_tid', use_ptrs), 'NType', 
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                declare_const_flag = True
            )
        _swap_kernel_idpy += _codify_newl        
        
        _swap_kernel_idpy += _codify_comment("Computing the velocity")
        _swap_kernel_idpy += \
            self.idpy_stencil.VelocityCode(
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                arrays = ['pop'], arrays_types = ['PopType'],
                n_type = 'NType', u_type = 'UType',
                root_n = 'ln', root_u = 'lu', 
                lex_index = 'g_tid', keep_read = True,
                ordering_lambdas = [ordering_lambda_pop], 
                use_ptrs = use_ptrs, 
                declare_const_dict = {'arrays_xi': True, 'moments': False}
            )
        _swap_kernel_idpy += _codify_newl        

        _swap_kernel_idpy += _codify_comment("Guo's Velocity shift")
        _swap_kernel_idpy += \
            self.idpy_guo_forcing.ShiftVelocityCode(
                n_sets = ['ln_0'], u_sets = ['lu_0'], f_sets = ['d_psi']
            )
        _swap_kernel_idpy += _codify_newl        

        _swap_kernel_idpy += _codify_comment("Storing the density")
        _swap_kernel_idpy += \
            _codify_assignment(_array_value('n', 'g_tid', use_ptrs), 'ln_0') + \
            _codify_newl

        _swap_kernel_idpy += _codify_comment("Storing the velocity")        
        for _d in range(self.idpy_stencil.D):
            _swap_kernel_idpy += \
                _codify_assignment(
                    _array_value('u', ordering_lambda_u('g_tid', _d), use_ptrs),
                    'lu_0_' + str(_d)
                )
        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        _swap_kernel_idpy + \
        """
        }
        """

class K_ForceGross2011CollideStreamSCMPMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None,
                 XIStencil = None, SCFStencil = None, use_ptrs = False, 
                 ordering_lambda_pop = None, ordering_lambda_u = None,
                 ordering_lambda_prng = None, kBT = None, n0 = None,
                 collect_mul = False, stream_mode = 'push', pressure_mode = 'compute',
                 root_dim_sizes = 'L', root_strides = 'STR', root_coord = 'x',
                 distribution = 'flat', which_box_m = 'cos', parallel_streams = 1,
                 generator = 'MMIX', headers_files=['math.h']):
        
        self.expect_lambda_args = 2

        if ordering_lambda_pop is None:
            raise Exception("Missing argument 'ordering_lambda_pop'")
        elif ordering_lambda_pop.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion(
                "The number of arguments for 'ordering_lambda_pop' should be ",
                self.expect_lambda_args
            )

        if XIStencil is None:
            raise Exception("Missing argument 'XIStencil'")
        if SCFStencil is None:
            raise Exception("Missing argument 'SCFStencil'")

        self.idpy_stencil = IdpyStencil(XIStencil)
        self.idpy_sc_force = IDShanChenForce(SCFStencil)
        self.idpy_guo_forcing = GuoForcing(XIStencil, root_u = 'lu_0', root_f = 'd_psi', 
                                           root_omega = '\\omega')
        self.idpy_equilibria = HermiteEquilibria(XIStencil, root_pop = 'lpop', 
                                                 root_n = 'ln_0', root_u = 'lu_0')
        self.idpy_collision = \
            BGK(root_omega = '\\omega', omega_val = constants['OMEGA'])

        '''
        Two posssibilities:
        - Store the random populations in separate (constant) variables and add them
        during the collision + streaming step
        - Load the populations as not-constants and add the random values while 
        generating them
        - Hoewever I put it I alwasy need to compute the random moments first and then
        transform back to the populations. So I need at least Q - (D + 1) swap variables
        unless I am ready to generate the right combination to begin with. Which is
        clearly a possibility, but looks like an optimization step
        '''
        self.idpy_gross2011 = \
            GrossShanChenMultiPhase(self.idpy_equilibria)
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants,
                            f_classes = f_classes, optimizer_flag = optimizer_flag, 
                            headers_files=headers_files)

        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop_swap': ['global', 'restrict'],
                       'CRNGType * seeds': ['global', 'restrict'],                       
                       'PopType * pop': ['global', 'restrict', 'const'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'PsiType * psi': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()
        self.dim_sizes_macros = _get_seq_macros(self.constants['DIM'], root_dim_sizes)
        self.dim_strides_macros = \
            _get_seq_macros(
                self.constants['DIM'] - 1 if self.constants['DIM'] > 1 else 1, 
                root_strides)

        '''
        Kernel Body
        '''
        # 1. Getting Cartesian coordinates
        # 2. Compute the Shan-Chen force
        # 3. Getting the local populations values and compute the local velocity
        # 4. Guo-shift the velocity
        # 5. Generate noise moments
        # 6. Collide with Guo's forcing on local populations
        # 7. Stream in 'push' mode and add noise populations (transformed from moments)
        
        _swap_kernel_idpy = """"""
        _swap_kernel_idpy += _codify_comment("Getting Cartesian Coordinates")
        _swap_kernel_idpy += \
            _get_cartesian_coordinates_macro(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                _root_coordinate = root_coord,
                _dim_sizes = self.dim_sizes_macros,
                _dim_strides = self.dim_strides_macros,
                _type = 'SType', declare_const_flag = True,
                _lexicographic_index = 'g_tid'
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Computing the Shan-Chen Force")        
        _swap_kernel_idpy += \
            self.idpy_sc_force.ForceCodeMultiPhase(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                pressure_mode = pressure_mode, psi_array = 'psi', 
                use_ptrs = use_ptrs, 
                root_dim_sizes = root_dim_sizes, 
                root_strides = root_strides,
                root_coord = root_coord, 
                lex_index = 'g_tid'
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Loading the local value of the density")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'ln_0', _array_value('n', 'g_tid', use_ptrs), 'NType', 
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                declare_const_flag = True
            ) + \
            _codify_newl
        
        _swap_kernel_idpy += \
            _codify_comment("Loading the local value of the populations and keep it")
        _swap_kernel_idpy += \
            _codify_comment("Computing the local value of the velocity")        
        _swap_kernel_idpy += \
            self.idpy_stencil.VelocityCode(
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                arrays = ['pop'], arrays_types = ['PopType'],
                n_type = 'NType', u_type = 'UType',
                root_n = 'ln', root_u = 'lu', 
                lex_index = 'g_tid', keep_read = True,
                ordering_lambdas = [ordering_lambda_pop], 
                use_ptrs = use_ptrs, 
                declare_const_dict = {'arrays_xi': True, 'moments': False}
            )
        
        _swap_kernel_idpy += _codify_comment("Guo's Velocity shift")
        _swap_kernel_idpy += \
            self.idpy_guo_forcing.ShiftVelocityCode(
                n_sets = ['ln_0'], u_sets = ['lu_0'], f_sets = ['d_psi']
            )
        _swap_kernel_idpy += _codify_newl
        '''
        Computing the random moments
        '''
        _swap_kernel_idpy += _codify_comment("COMPUTING RANDOM MOMENTS")
        _swap_code, _swap_rpop_list = \
            self.idpy_gross2011.CodifySRTRandomPopulations(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                n0 = n0, kBT = kBT, tau = 1/constants['OMEGA'],
                seeds_array = 'seeds',
                rand_vars_type = 'PopType',
                generator = generator,
                distribution = distribution,
                parallel_streams = parallel_streams,
                which_box_m = which_box_m, 
                lambda_ordering = ordering_lambda_prng, 
                use_ptrs = use_ptrs
            )
        _swap_kernel_idpy += _swap_code
        _swap_kernel_idpy += _codify_newl
               
        '''
        Preparing the variables for collision plus streaming according to the pressure mode
        '''
        _swap_tuples_eq_prod, _swap_tuples_guo_prod = [], []        

        _swap_code, _swap_tuples_eq_prod = \
            self.idpy_equilibria.SetMomentsProductsCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                order = 2, root_n = 'ln_0', root_u = 'lu_0', 
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl
        
        _swap_code, _swap_tuples_guo_prod = \
            self.idpy_guo_forcing.SetMomentsProductsCode(
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                order = 2, root_f = 'd_psi', root_u = 'lu_0',
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Computing equilibrium, collision and stream")
        _swap_kernel_idpy += \
            self.idpy_collision.SRTCollisionPlusGuoPushStreamCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants, 
                ordering_lambda = ordering_lambda_pop,
                dst_arrays_var = 'pop_swap',
                stencil_obj = self.idpy_stencil, 
                eq_obj = self.idpy_equilibria, 
                guo_obj = self.idpy_guo_forcing,
                neq_pop = 'lpop', pressure_mode = pressure_mode,
                tuples_eq = _swap_tuples_eq_prod, 
                tuples_guo = _swap_tuples_guo_prod,
                pos_type = 'SType', use_ptrs = use_ptrs, collect_mul = collect_mul, 
                root_dim_sizes = root_dim_sizes, root_strides = root_strides, 
                root_coord = root_coord, lex_index = 'g_tid',
                declare_const_dict = {'cartesian_coord_neigh': True},
                rnd_pops = _swap_rpop_list
            )

        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        _swap_kernel_idpy + \
        """
        }
        """


