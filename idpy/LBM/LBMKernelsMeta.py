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
from idpy.IdpyCode.IdpyCode import IdpyKernel
from idpy.IdpyCode.IdpyUnroll import _get_seq_macros, _get_seq_vars, _codify_newl
from idpy.IdpyCode.IdpyUnroll import _get_cartesian_coordinates_macro
from idpy.IdpyCode.IdpyUnroll import _codify_assignment, _array_value

from idpy.IdpyStencils.IdpyStencils import IdpyStencil

class K_ComputeMomentsMeta(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None, XIStencil = None, use_ptrs = False,
                 ordering_lambda_pop = None, ordering_lambda_u = None):

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
                            f_classes = f_classes, optimizer_flag = optimizer_flag)
        
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
        _get_cartesian_coordinates_macro(declared_variables = self.declared_variables, 
                                         declared_constants = self.declared_constants, 
                                         _root_coordinate = root_coord, 
                                         _dim_sizes = self.dim_sizes_macros, 
                                         _dim_strides = self.dim_strides_macros,
                                         _type = 'SType', declare_const_flag = True,
                                         _lexicographic_index = 'g_tid') + \
        _codify_newl + \
        _codify_newl + \
        self.idpy_stencil.StreamingCode(self,
                                        declared_variables = self.declared_variables, 
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
                                                                       
