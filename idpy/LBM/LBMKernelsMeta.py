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

from idpy.IdpyCode import IDPY_T
from idpy.IdpyCode.IdpyCode import IdpyKernel
from idpy.IdpyCode.IdpyUnroll import _get_seq_macros, _get_seq_vars, _codify_newl
from idpy.IdpyCode.IdpyUnroll import _get_cartesian_coordinates_macro
from idpy.IdpyCode.IdpyUnroll import _codify_assignment, _array_value
from idpy.IdpyCode.IdpyUnroll import _codify_comment, _codify_declaration_const_check
from idpy.IdpyCode.IdpyUnroll import _codify_add_assignment

from idpy.IdpyStencils.IdpyStencils import IdpyStencil

from idpy.LBM.Equilibria import HermiteEquilibria
from idpy.LBM.Collisions import BGK

class K_ComputeMomentsWallsMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, XIStencil = None, use_ptrs = False,
                 walls_array_var = 'walls',
                 ordering_lambda_pop = None, ordering_lambda_u = None, 
                 headers_files=['math.h']):

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
                        
        self.idpy_stencil = IdpyStencil(XIStencil)
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants, 
                            f_classes = f_classes, optimizer_flag = optimizer_flag, 
                            headers_file=headers_files)
        
        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'NType * n': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'PopType * pop': ['global', 'restrict', 'const'], 
                       'FlagType * walls': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()

        '''
        Kernel Body
        '''        
        self.kernels[IDPY_T] = """
        if(g_tid < V && """ + _array_value(walls_array_var, 'g_tid', use_ptrs) + \
        """ == 1){
        """ + \
        self.idpy_stencil.HydroHermiteCode(
            declared_variables = self.declared_variables,
            declared_constants = self.declared_constants,
            arrays = ['pop'], arrays_types = ['PopType'],
            root_n = 'ln', root_u = 'lu',
            n_type = 'NType', u_type = 'UType',
            lex_index = 'g_tid', keep_read = True,
            ordering_lambdas = [ordering_lambda_pop],
            use_ptrs = use_ptrs,
            declare_const_dict = {'arrays_xi': True, 'moments': False}
        ) + \
        _codify_newl + \
        _codify_newl + \
        _codify_assignment(_array_value('n', 'g_tid', use_ptrs), 'ln_0')

        for _d in range(self.idpy_stencil.D):
            self.kernels[IDPY_T] += \
                _codify_assignment(_array_value('u', ordering_lambda_u('g_tid', _d), use_ptrs),
                                   'lu_0_' + str(_d))

        self.kernels[IDPY_T] += \
        """
        }
        """

class K_ComputeMomentsMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, XIStencil = None, use_ptrs = False,
                 ordering_lambda_pop = None, ordering_lambda_u = None, 
                 headers_files=['math.h']):

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
                        
        self.idpy_stencil = IdpyStencil(XIStencil)
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants, 
                            f_classes = f_classes, optimizer_flag = optimizer_flag, 
                            headers_files=headers_files)
        
        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'NType * n': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'PopType * pop': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()

        '''
        Kernel Body
        '''        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        self.idpy_stencil.HydroHermiteCode(
            declared_variables = self.declared_variables,
            declared_constants = self.declared_constants,
            arrays = ['pop'], arrays_types = ['PopType'],
            root_n = 'ln', root_u = 'lu',
            n_type = 'NType', u_type = 'UType',
            lex_index = 'g_tid', keep_read = True,
            ordering_lambdas = [ordering_lambda_pop],
            use_ptrs = use_ptrs,
            declare_const_dict = {'arrays_xi': True, 'moments': False}
        ) + \
        _codify_newl + \
        _codify_newl + \
        _codify_assignment(_array_value('n', 'g_tid', use_ptrs), 'ln_0')

        for _d in range(self.idpy_stencil.D):
            self.kernels[IDPY_T] += \
                _codify_assignment(_array_value('u', ordering_lambda_u('g_tid', _d), use_ptrs),
                                   'lu_0_' + str(_d))

        self.kernels[IDPY_T] += \
        """
        }
        """
            

class K_StreamPeriodicMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [], 
                 pressure_mode = 'compute',
                 optimizer_flag = None, XIStencil = None, collect_mul = False,
                 stream_mode = None, root_dim_sizes = 'L', root_strides = 'STR', 
                 use_ptrs = False, root_coord = 'x', ordering_lambda = None):
        
        self.expect_lambda_args = 2
        
        if ordering_lambda is None:
            raise Exception("Missing argument 'ordering_lambda'")
        elif ordering_lambda.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion("The number of arguments for 'ordering_lambda' should be ", 
                            self.expect_lambda_args)
        
        if XIStencil is None:
            raise Exception("Missing argument 'XIStencil'")
            
        if stream_mode is None:
            raise Exception("Missing argument 'stream_mode'")
        elif stream_mode not in ['pull', 'push']:
            raise Exception("Argument 'stream_mode' must either be 'pull' or 'push'")
            
        self.idpy_stencil = IdpyStencil(XIStencil)
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants, 
                            f_classes = f_classes, optimizer_flag = optimizer_flag)
        
        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * dst': ['global', 'restrict'], 
                       'PopType * src': ['global', 'restrict', 'const']}
        
        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()
        self.dim_sizes_macros = _get_seq_macros(self.constants['DIM'], root_dim_sizes)
        self.dim_strides_macros = _get_seq_macros(self.constants['DIM'] - 1, root_strides)
        
        '''
        Kernel Body
        '''        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        _codify_comment("Getting the cartesian coordinates of the thread") + \
        _get_cartesian_coordinates_macro(declared_variables = self.declared_variables, 
                                         declared_constants = self.declared_constants, 
                                         _root_coordinate = root_coord, 
                                         _dim_sizes = self.dim_sizes_macros, 
                                         _dim_strides = self.dim_strides_macros,
                                         _type = 'SType', declare_const_flag = True,
                                         _lexicographic_index = 'g_tid') + \
        _codify_newl + \
        _codify_newl + \
        self.idpy_stencil.StreamingCode(declared_variables = self.declared_variables, 
                                        declared_constants = self.declared_constants,
                                        src_arrays_vars = ['src'],
                                        dst_arrays_vars = ['dst'],
                                        pos_type = 'SType', collect_mul = collect_mul,
                                        declare_const_dict = \
                                        {'cartesian_coord_neigh': True}, 
                                        ordering_lambdas = [ordering_lambda], 
                                        stream_mode = stream_mode,
                                        pressure_mode = pressure_mode,
                                        root_dim_sizes = root_dim_sizes,
                                        root_strides = root_strides,
                                        use_ptrs = use_ptrs, root_coord = root_coord, 
                                        lex_index = 'g_tid') + \
        _codify_newl + \
        _codify_newl + \
        """
        }
        """
                                                                       
class K_InitPopulationsMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, XIStencil = None, use_ptrs = False,
                 ordering_lambda_pop = None, ordering_lambda_u = None,
                 order = 2, pressure_mode = 'compute'):

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
                        
        self.idpy_stencil = IdpyStencil(XIStencil)
        self.idpy_equilibria = HermiteEquilibria(XIStencil, root_pop = 'lpop',
                                                 root_n = 'ln_0', root_u = 'lu_0')
        self.D, self.Q = self.idpy_equilibria.D, self.idpy_equilibria.Q
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants, 
                            f_classes = f_classes, optimizer_flag = optimizer_flag)
        
        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop': ['global', 'restrict'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'UType * u': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()

        '''
        Kernel Body
        '''
        _swap_kernel_idpy = """"""
        _swap_kernel_idpy += _codify_comment("Defining density and velocity as constants")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'ln_0', _array_value('n', 'g_tid', use_ptrs), 'NType', 
                declared_variables = self.declared_variables, 
                declared_constants = self.declared_constants, 
                declare_const_flag = True
            ) + \
            _codify_newl

        for _d in range(self.D):
            _swap_kernel_idpy += \
                _codify_declaration_const_check(
                    'lu_0_' + str(_d),
                    _array_value('u', ordering_lambda_u('g_tid', _d), use_ptrs), 'UType', 
                    declared_variables = self.declared_variables, 
                    declared_constants = self.declared_constants, 
                    declare_const_flag = True
                )
        _swap_kernel_idpy += _codify_newl

        _swap_kernel_idpy += \
            _codify_comment("Computing and assigning equilibrium populations")

        _swap_kernel_idpy += \
            _codify_comment("Declaring products if pressure_mode == 'registers'")
        _swap_code, _swap_tuples_eq_prod = \
            self.idpy_equilibria.SetMomentsProductsCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                pressure_mode = pressure_mode, order = order,
                root_n = 'ln_0', root_u = 'lu_0',
                mom_type = 'PopType'
            )
        _swap_kernel_idpy += _swap_code
        _swap_kernel_idpy += _codify_newl

        for _q in range(self.Q):
            _r_hand= \
                self.idpy_equilibria.CodifyEquilibriumSinglePopulation(
                    order = order, i = _q, collect_exprs = _swap_tuples_eq_prod
                )
            _l_hand = _array_value('pop', ordering_lambda_pop('g_tid', _q), use_ptrs)
            _swap_kernel_idpy += _codify_assignment(_l_hand, _r_hand)
            _swap_kernel_idpy += _codify_newl
        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        _swap_kernel_idpy + \
        """
        }
        """

class K_CheckUMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, use_ptrs = False,
                 ordering_lambda_u = None, headers_files=['math.h']):

        self.expect_lambda_args = 2

        if ordering_lambda_u is None:
            raise Exception("Missing argument 'ordering_lambda_u'")
        elif ordering_lambda_u.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion(
                "The number of arguments for 'ordering_lambda_u' should be ", 
                self.expect_lambda_args
            )        
        
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag, headers_files=headers_files)
        self.SetCodeFlags('g_tid')
        self.params = {'UType * delta_u': ['global', 'restrict'],
                       'UType * old_u': ['global', 'restrict'],
                       'UType * max_u': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()

        self.D = self.constants['DIM']
        
        _swap_kernel_idpy = """"""
        _swap_kernel_idpy += _codify_comment("Declaring difference and norm variables")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'ldiff', 0, 'UType',
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                declare_const_flag = False
            )

        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'lu_norm', 0, 'UType',
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                declare_const_flag = False
            )

        _swap_kernel_idpy += _codify_comment("Declaring tmp variable for velocity component")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'lu_now', 0, 'UType',
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                declare_const_flag = False                
            )
        
        
        for _d in range(self.D):
            _swap_kernel_idpy += \
                _codify_assignment(
                    'lu_now',
                    _array_value('u', ordering_lambda_u('g_tid', _d), use_ptrs)
                )
            _swap_kernel_idpy += _codify_newl
            
            _swap_kernel_idpy += _codify_comment("Adding to the norm")
            _swap_kernel_idpy += \
                _codify_add_assignment('lu_norm', 'lu_now * lu_now')
            _swap_kernel_idpy += _codify_newl            

            _swap_kernel_idpy += _codify_comment("Adding to the difference")
            _swap_kernel_idpy += \
                _codify_add_assignment(
                    'ldiff',
                    '(UType) fabs(lu_now - ' + _array_value('old_u',
                                               ordering_lambda_u('g_tid', _d),
                                               use_ptrs) + ')'
                )
            _swap_kernel_idpy += _codify_newl            
            
            _swap_kernel_idpy += _codify_comment("Writing the value in old_u")
            _swap_kernel_idpy += \
                _codify_assignment(
                    _array_value('old_u', ordering_lambda_u('g_tid', _d), use_ptrs),
                    'lu_now'
                )
            _swap_kernel_idpy += _codify_newl            
            
        _swap_kernel_idpy += _codify_comment("Writing difference and norm")
        _swap_kernel_idpy += \
            _codify_assignment(
                _array_value('delta_u', 'g_tid', use_ptrs),
                'ldiff'
            )
        _swap_kernel_idpy += \
            _codify_assignment(
                _array_value('max_u', 'g_tid', use_ptrs),
                'lu_norm'
            )
            
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        _swap_kernel_idpy + \
        """
        }
        """

class K_CheckUMetaWalls(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, use_ptrs = False,
                 walls_array_var = 'walls',
                 ordering_lambda_u = None, headers_files=['math.h']):

        self.expect_lambda_args = 2

        if ordering_lambda_u is None:
            raise Exception("Missing argument 'ordering_lambda_u'")
        elif ordering_lambda_u.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion(
                "The number of arguments for 'ordering_lambda_u' should be ", 
                self.expect_lambda_args
            )        
        
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag, headers_files=headers_files)
        self.SetCodeFlags('g_tid')
        self.params = {'UType * delta_u': ['global', 'restrict'],
                       'UType * old_u': ['global', 'restrict'],
                       'UType * max_u': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict', 'const'], 
                       'FlagType * walls': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()

        self.D = self.constants['DIM']
        
        _swap_kernel_idpy = """"""
        _swap_kernel_idpy += _codify_comment("Declaring difference and norm variables")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'ldiff', 0, 'UType',
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                declare_const_flag = False
            )

        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'lu_norm', 0, 'UType',
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                declare_const_flag = False
            )

        _swap_kernel_idpy += _codify_comment("Declaring tmp variable for velocity component")
        _swap_kernel_idpy += \
            _codify_declaration_const_check(
                'lu_now', 0, 'UType',
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                declare_const_flag = False                
            )
        
        
        for _d in range(self.D):
            _swap_kernel_idpy += \
                _codify_assignment(
                    'lu_now',
                    _array_value('u', ordering_lambda_u('g_tid', _d), use_ptrs)
                )
            _swap_kernel_idpy += _codify_newl
            
            _swap_kernel_idpy += _codify_comment("Adding to the norm")
            _swap_kernel_idpy += \
                _codify_add_assignment('lu_norm', 'lu_now * lu_now')
            _swap_kernel_idpy += _codify_newl            

            _swap_kernel_idpy += _codify_comment("Adding to the difference")
            _swap_kernel_idpy += \
                _codify_add_assignment(
                    'ldiff',
                    '(UType) fabs(lu_now - ' + _array_value('old_u',
                                               ordering_lambda_u('g_tid', _d),
                                               use_ptrs) + ')'
                )
            _swap_kernel_idpy += _codify_newl            
            
            _swap_kernel_idpy += _codify_comment("Writing the value in old_u")
            _swap_kernel_idpy += \
                _codify_assignment(
                    _array_value('old_u', ordering_lambda_u('g_tid', _d), use_ptrs),
                    'lu_now'
                )
            _swap_kernel_idpy += _codify_newl            
            
        _swap_kernel_idpy += _codify_comment("Writing difference and norm")
        _swap_kernel_idpy += \
            _codify_assignment(
                _array_value('delta_u', 'g_tid', use_ptrs),
                'ldiff'
            )
        _swap_kernel_idpy += \
            _codify_assignment(
                _array_value('max_u', 'g_tid', use_ptrs),
                'lu_norm'
            )
            
        self.kernels[IDPY_T] = """
        if(g_tid < V && """ + _array_value(walls_array_var, 'g_tid', use_ptrs) + \
        """ == 1){
        """ + \
        _swap_kernel_idpy + \
        """
        }
        """

class K_MRTCollideStreamMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, 
                 XIStencil = None, use_ptrs = False,
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

        if relaxation_matrix is None:
            raise Exception("Missing argument 'relaxation_matrix'")
        if omega_syms_vals is None:
            raise Exception("Missing argument 'omega_syms_vals'")        

        self.idpy_stencil = IdpyStencil(XIStencil)
        self.idpy_equilibria = HermiteEquilibria(XIStencil, root_pop = 'lpop', 
                                                 root_n = 'ln_0', root_u = 'lu_0')
        self.idpy_collision = BGK()
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants,
                            f_classes = f_classes, optimizer_flag = optimizer_flag)

        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop_swap': ['global', 'restrict'],
                       'PopType * pop': ['global', 'restrict', 'const'],
                       'NType * n': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()
        self.dim_sizes_macros = _get_seq_macros(self.constants['DIM'], root_dim_sizes)
        self.dim_strides_macros = _get_seq_macros(self.constants['DIM'] - 1, root_strides)

        '''
        Kernel Body
        '''
        # 1. Getting Cartesian coordinates
        # 3. Getting the local populations values and compute the local velocity
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
            )
        _swap_kernel_idpy += _codify_newl
        
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
                       
        '''
        Preparing the variables for collision plus streaming according to the pressure mode
        '''
        _swap_tuples_eq_prod, _swap_tuples_guo_prod = [], []        
        _swap_kernel_idpy += \
            _codify_comment("Declaring the variables that are needed by the selected pressure mode")
        _swap_code, _swap_tuples_eq_prod = \
            self.idpy_equilibria.SetMomentsProductsCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                order = 2, root_n = 'ln_0', root_u = 'lu_0', 
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl

        '''
        This call needs '_swap_tuples_eq_prod' otherwise the generated code will not be 
        compilable
        '''
        _swap_kernel_idpy += _codify_comment("Computing equilibrium, collision and stream")
        _swap_kernel_idpy += \
            self.idpy_collision.MRTCollisionPushStreamCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants, 
                ordering_lambda = ordering_lambda_pop,
                dst_arrays_var = 'pop_swap',
                stencil_obj = self.idpy_stencil, 
                eq_obj = self.idpy_equilibria,
                relaxation_matrix = relaxation_matrix,
                omega_syms_vals = omega_syms_vals,                
                neq_pop = 'lpop', pressure_mode = pressure_mode,
                tuples_eq = _swap_tuples_eq_prod, 
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

class K_SRTCollideStreamMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, 
                 XIStencil = None, use_ptrs = False, 
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

        self.idpy_stencil = IdpyStencil(XIStencil)
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
                       'NType * n': ['global', 'restrict', 'const']}

        '''
        Setting the declared variables and constants up to this point
        '''
        self.SetDeclaredVariables()
        self.SetDeclaredConstants()
        self.dim_sizes_macros = _get_seq_macros(self.constants['DIM'], root_dim_sizes)
        self.dim_strides_macros = _get_seq_macros(self.constants['DIM'] - 1, root_strides)

        '''
        Kernel Body
        '''
        # 1. Getting Cartesian coordinates
        # 3. Getting the local populations values and compute the local velocity
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
            )
        _swap_kernel_idpy += _codify_newl
        
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
                       
        '''
        Preparing the variables for collision plus streaming according to the pressure mode
        '''
        _swap_tuples_eq_prod, _swap_tuples_guo_prod = [], []        
        _swap_kernel_idpy += \
            _codify_comment("Declaring the variables that are needed by the selected pressure mode")
        _swap_code, _swap_tuples_eq_prod = \
            self.idpy_equilibria.SetMomentsProductsCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                order = 2, root_n = 'ln_0', root_u = 'lu_0', 
                mom_type = 'PopType', pressure_mode = pressure_mode
            )
        _swap_kernel_idpy += _swap_code + _codify_newl
        
        _swap_kernel_idpy += _codify_comment("Computing equilibrium, collision and stream")
        _swap_kernel_idpy += \
            self.idpy_collision.SRTCollisionPushStreamCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants, 
                ordering_lambda = ordering_lambda_pop,
                dst_arrays_var = 'pop_swap',
                stencil_obj = self.idpy_stencil, 
                eq_obj = self.idpy_equilibria,
                neq_pop = 'lpop', pressure_mode = pressure_mode,
                tuples_eq = _swap_tuples_eq_prod, 
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
        
class K_IsotropyFilter(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, XIStencil = None, use_ptrs = False, search_depth=6,
                 declare_const_dict = {'new_pop': True}, output_pop = 'pop_iso',
                 ordering_lambda_pop = None, collect_mul = False, headers_files=['math.h']):

        self.expect_lambda_args = 2
        
        if ordering_lambda_pop is None:
            raise Exception("Missing argument 'ordering_lambda_pop'")
        elif ordering_lambda_pop.__code__.co_argcount != self.expect_lambda_args:
            raise Excpetion(
                "The number of arguments for 'ordering_lambda_pop' should be ", 
                self.expect_lambda_args
            )

        self.idpy_stencil = IdpyStencil(XIStencil)
        
        IdpyKernel.__init__(self, custom_types = custom_types, constants = constants,
                            f_classes = f_classes, optimizer_flag = optimizer_flag, 
                            headers_files=headers_files)

        '''
        Kernel Parameters declaration
        '''
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop': ['global', 'restrict', 'const'], 
                       'PopType * ' + output_pop: ['global', 'restrict']}
        

        self.SetDeclaredVariables()
        self.SetDeclaredConstants()

        code_body = """"""

        """
        Computing the non-equilibrium moments
        """
        code_body += \
            self.idpy_stencil.MomentsHermiteCode(
                declared_variables = self.declared_variables,
                declared_constants = self.declared_constants,
                arrays = ['pop'], arrays_types = ['PopType'],
                root_neq_mom = 'neq_m', mom_type = 'PopType',
                lex_index = 'g_tid', keep_read = True,
                ordering_lambdas = [ordering_lambda_pop],
                use_ptrs = use_ptrs,
                declare_const_dict = {'arrays_xi': True, 'moments': False}
            )

        """
        Normalize all moments
        """
        code_normalize = """"""
        for q in range(1, self.idpy_stencil.Q):
            code_normalize += "neq_m_0_" + str(q) + " /= neq_m_0_0;"
            code_normalize += _codify_newl

        code_body += code_normalize + _codify_newl

        """
        Define the new populations
        """
        M_dict = self.idpy_stencil.GetInvertibleHermiteSet(search_depth=search_depth)
        M = M_dict['M']

        f_i = sp.Matrix([sp.Symbol('f_{' + str(i) + '}') for i in range(9)])
        D=2
        root_xi_sym = '\\xi'
        xi_sym_list = [sp.Symbol(root_xi_sym + '_' + str(_)) for _ in range(D)]
        
        polys = {'xi_xx': xi_sym_list[0] ** 2, 
                 'xi_xy': xi_sym_list[0] * xi_sym_list[1], 
                 'xi_yy': xi_sym_list[1] ** 2,
                 'xi_xyy': (xi_sym_list[0] ** 1) * (xi_sym_list[1] ** 2),
                 'xi_xxy': (xi_sym_list[0] ** 2) * (xi_sym_list[1] ** 1)}
        
        mom_list = sp.Matrix([sp.Symbol('m_{' + str(i) + '}') for i in range(9)])
        mom_list_i = sp.Matrix([sp.Symbol('m_{' + str(i) + '}^{(i)}') for i in range(9)])
        
        polys_coeffs = {}
        
        for label in polys:
            coeffs_list = []
            for xi in self.idpy_stencil.XIs:
                coeff_swap = polys[label]
                for i in range(len(xi)):
                    coeff_swap = coeff_swap.subs(xi_sym_list[i], xi[i])
                coeffs_list += [coeff_swap]
                
            polys_coeffs[label] = sp.Matrix(coeffs_list).T
        
        eq_xy = (polys_coeffs['xi_xy'] * f_i)[0]
        eq_xx_yy = (polys_coeffs['xi_xx'] * f_i)[0] - (polys_coeffs['xi_yy'] * f_i)[0]
        eq_xyy = (polys_coeffs['xi_xyy'] * f_i)[0]
        eq_xxy = (polys_coeffs['xi_xxy'] * f_i)[0]

        M_H = M * f_i - mom_list      
        M_new = sp.Matrix(M_H[:4] + [eq_xy - mom_list_i[4], eq_xx_yy - mom_list_i[5]] + M_H[6:])
        M_new = sp.Matrix(M_H[:6] + [eq_xxy - mom_list_i[6], eq_xyy - mom_list_i[7]] + M_H[8:])
        M_new = \
            sp.Matrix(M_H[:4] + [eq_xy - mom_list_i[4], eq_xx_yy - mom_list_i[5]] + 
                      [eq_xxy - mom_list_i[6], eq_xyy - mom_list_i[7]] + M_H[8:])
        self.M_H = M_H

        """
        imposing the isotropy conditions for the new populations
        """
        sol_dict_new = sp.solve(M_new, f_i)
        sol_f_new = sp.Matrix([sol_dict_new[_] for _ in sol_dict_new])
        sol_f_new = sol_f_new.subs(mom_list_i[4], mom_list[1] * mom_list[2])
        sol_f_new = sol_f_new.subs(mom_list_i[5], (mom_list[1] ** 2) - (mom_list[2] ** 2))
        sol_f_new = sol_f_new.subs(
            mom_list_i[6], 
            2 * mom_list[4] * mom_list[1] + (mom_list[3] + sp.Rational(1,3)) * mom_list[2] -2 * mom_list[2] * (mom_list[1] ** 2)
        )

        sol_f_new = sol_f_new.subs(
            mom_list_i[7], 
            2 * mom_list[4] * mom_list[2] + (mom_list[5] + sp.Rational(1,3)) * mom_list[1] -2 * mom_list[1] * (mom_list[2] ** 2)
        )        

        self.sol_f_new = sol_f_new

        mom_list = sp.Matrix([sp.Symbol('m_{' + str(i) + '}') for i in range(9)])
        mom_vars = {mom_list[i]: '1.' if i == 0 else 'neq_m_0_' + str(i) for i in range(9)}
        mom_vars_pow = lambda mom, pow: sp.Symbol(((mom_vars[mom] + '*') * pow)[:-1])
        
        evalf_sols = []
        for sol in sol_f_new:
            monomials = sp.Add.make_args(sol)
        
            sol_evalf_tmp = 0
        
            for mono in monomials:
                mono_swap = mono
                for pow_mono in mono_swap.atoms(sp.Pow):
                    base, exp = pow_mono.as_base_exp()
                    mono_swap = mono_swap.subs(base ** exp, mom_vars_pow(base, exp)).evalf()
        
                for mom in mom_list:
                    mono_swap = mono_swap.subs(mom, mom_vars[mom]).evalf()
        
                sol_evalf_tmp += mono_swap
        
            evalf_sols += [sol_evalf_tmp]

        self.evalf_sols = evalf_sols

        """
        Computing the new populations
        """
        code_solutions = """"""
        for q, pop in enumerate(evalf_sols):

            _sx_hnd = "new_pop_" + str(q)
            _dx_hnd = str(pop)
            
            code_solutions += \
                _codify_declaration_const_check(
                    _sx_hnd, _dx_hnd,
                    "PopType",
                    self.declared_variables,
                    self.declared_constants,
                    declare_const_flag = declare_const_dict['new_pop']
                )
        code_solutions += _codify_newl

        code_body += code_solutions

        """
        Rescale and write the new populations
        """
        code_population_write = """"""
        for q in range(self.idpy_stencil.Q):

            _dx_hnd = "new_pop_" + str(q) + " * neq_m_0_0"
            _sx_hnd = \
                _array_value(output_pop, ordering_lambda_pop('g_tid', q), use_ptrs)
            
            code_population_write += _codify_assignment(_sx_hnd, _dx_hnd)

        code_population_write += _codify_newl

        code_body += code_population_write
        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
        """ + \
        code_body + \
        """
        }
        """
