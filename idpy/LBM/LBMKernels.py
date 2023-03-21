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
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyFunction, IdpyMethod

'''
IdpyFunction's
'''
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

'''
IdpyKernel's
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

class K_HalfWayBounceBack(IdpyKernel):
    def __init__(self, custom_types = None, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop': ['global', 'restrict'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const'], 
                       'FlagType * walls': ['global', 'restrict', 'const'], 
                       'SType * xi_opposite': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V && walls[g_tid] == 0){
            SType dst_pos[DIM], src_pos[DIM];
            F_PosFromIndex(src_pos, dim_sizes, dim_strides, g_tid);

            // zero-th population
            pop[g_tid] = 0.;
            for(int q=1; q<Q; q++){
                int q_dest = xi_opposite[q];

                for(int d=0; d<DIM; d++){
                    dst_pos[d] = ((src_pos[d] + XI_list[d + q_dest * DIM] + dim_sizes[d]) % dim_sizes[d]);
                }
                SType dst_index = F_IndexFromPos(dst_pos, dim_strides);

                pop[dst_index + q_dest * V] = pop[g_tid + q * V];
                pop[g_tid + q * V] = 0.;
                // pop[g_tid + q * V] = pop[g_tid + q * V];
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

class K_ComputeMomentsWalls(IdpyKernel):
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
                       'WType * W_list': ['global', 'restrict', 'const'], 
                       'FlagType * walls': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V && walls[g_tid] == 1){
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
                lu[d] = u[g_tid + d * V]; u_dot_u += lu[d] * lu[d];
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

class K_SetPopNUXInletDutyCycle(IdpyKernel):
    def __init__(self, custom_types=None, constants={}, 
                 f_classes=[], optimizer_flag=None):
        IdpyKernel.__init__(self, custom_types=custom_types, constants=constants, 
                            f_classes=f_classes, optimizer_flag=optimizer_flag)
        
        self.SetCodeFlags('g_tid')
        
        self.params = {'PopType * pop': ['global', 'restrict'], 
                       'NType * n': ['global', 'restrict'], 
                       'UType * u': ['global', 'restrict'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'WType * W_list': ['global', 'restrict', 'const'],
                       'FlagType * walls': ['global', 'restrict', 'const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const'],
                       'int idloop_k': ['const']}
        
        self.kernels[IDPY_T] = """
        if(g_tid < V && walls[g_tid] == 2){
            SType dst_pos[DIM], src_pos[DIM];
            // Getting the position of the thread
            F_PosFromIndex(dst_pos, dim_sizes, dim_strides, g_tid);

            // Setting the position of the neighbor
            src_pos[0] = dst_pos[0] + 1;
            for(int d=1; d<DIM; d++){ src_pos[d] = dst_pos[d]; }
            
            // Getting the index of the neighbor
            SType src_index = F_IndexFromPos(src_pos, dim_strides);

            NType ln = n[g_tid] = n[src_index];
            
            UType u_swap = \
                idloop_k <= TAU_IN ? (U_IN / TAU_IN) * idloop_k : \
                idloop_k > TAU_IN && idloop_k <= (MAX_MULT + 1) * TAU_IN ? U_IN : \
                idloop_k > (MAX_MULT + 1) * TAU_IN && idloop_k <= (MAX_MULT + 2) * TAU_IN ? \
                (U_IN / TAU_IN) * ((MAX_MULT + 2) * TAU_IN - idloop_k) : 0.;

            UType lu[DIM], u_dot_u = u_swap * u_swap;
            // Setting the velocity
            lu[0] = u[g_tid + 0 * V] = u_swap;
            for(int d=1; d<DIM; d++){
                lu[d] = u[g_tid + d * V] = 0.;
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

class K_SetPopNUOutletNoGradient(IdpyKernel):
    def __init__(self, custom_types=None, constants={}, 
                 f_classes=[], optimizer_flag=None):
        IdpyKernel.__init__(self, custom_types=custom_types, constants=constants, 
                            f_classes=f_classes, optimizer_flag=optimizer_flag)
        
        self.SetCodeFlags('g_tid')
        
        self.params = {'PopType * pop': ['global', 'restrict'], 
                       'NType * n': ['global', 'restrict'], 
                       'UType * u': ['global', 'restrict'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'WType * W_list': ['global', 'restrict', 'const'],
                       'FlagType * walls': ['global', 'restrict', 'const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const']}
        
        self.kernels[IDPY_T] = """
        if(g_tid < V && walls[g_tid] == 3){
            SType dst_pos[DIM], src_pos[DIM];
            // Getting the position of the thread
            F_PosFromIndex(dst_pos, dim_sizes, dim_strides, g_tid);

            // Setting the position of the neighbor
            src_pos[0] = dst_pos[0] - 1;
            for(int d=1; d<DIM; d++){ src_pos[d] = dst_pos[d]; }
            
            // Getting the index of the neighbor
            SType src_index = F_IndexFromPos(src_pos, dim_strides);
        
            // Copying the neighbor density and velocity
            n[g_tid] = n[src_index];
            for(int d=0; d<DIM; d++){ u[g_tid + d * V] = u[src_index + d * V]; }

            // ...and populations
            for(int q=0; q<Q; q++){
                pop[g_tid + q * V] = pop[src_index + q * V];
            }
        }
        """

class K_CleanPopulationsWalls(IdpyKernel):
    def __init__(self, custom_types = None, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {
            'PopType * pop': ['global', 'restrict'],
            'FlagType * walls': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V && walls[g_tid] == 0){
            for(int q=0; q<Q; q++){
                pop[g_tid + q * V] = 0.;
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
                    pop[g_tid + q * V] = ((PopType)q);
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

    def DeployProfiling(self, swap_list = None, idpy_stream = None):
        swap_list[0], swap_list[1] = swap_list[1], swap_list[0]

        return IdpyMethod.PassIdpyStream(self, idpy_stream = idpy_stream), 0
        
