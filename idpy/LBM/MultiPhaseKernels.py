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
from idpy.IdpyCode.IdpyCode import IdpyFunction, IdpyKernel

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

class F_NSingleFlatProfile(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'NType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType x': ['const'],
                       'SType x0': ['const']}

        self.functions[IDPY_T] = """
        return (tanh((LengthType)(x - x0)) + 1.) * 0.5;
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
class K_Collision_ShanChenMultiPhase(IdpyKernel):
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
            SCFType F[DIM]; SType neigh_pos[DIM]; UType lu_post[DIM];
            for(int d=0; d<DIM; d++){F[d] = lu_post[d] = 0.;}

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

            // Local density and velocity for shift and equilibrium
            NType ln = n[g_tid]; UType lu[DIM];

            // Shan-Chen velocity shift & Copy to global memory
            for(int d=0; d<DIM; d++){ 
                lu[d] = u[g_tid + V*d] + F[d] / ln / OMEGA;
            }

            // Compute square norm of Guo shifted velocity
            UType u_dot_u = 0.;
            for(int d=0; d<DIM; d++){u_dot_u += lu[d]*lu[d];}

            // Cycle over the populations: equilibrium + Shan Chen
            for(int q=0; q<Q; q++){
                UType u_dot_xi = 0.; 
                for(int d=0; d<DIM; d++){
                    u_dot_xi += lu[d] * XI_list[d + q*DIM];
                }

                PopType leq_pop = 1.;

                // Equilibrium population
                leq_pop += + u_dot_xi*CM2 + 0.5*u_dot_xi*u_dot_xi*CM4;
                leq_pop += - 0.5*u_dot_u*CM2;
                leq_pop = leq_pop * ln * W_list[q];

                pop[g_tid + q*V] = \
                    pop[g_tid + q*V]*(1. - OMEGA) + leq_pop*OMEGA;

                for(int d=0; d<DIM; d++){
                   lu_post[d] += pop[g_tid + q*V] * XI_list[d + q*DIM];
                }

             }

            for(int d=0; d<DIM; d++){ 
                u[g_tid + V*d] = 0.5 * (u[g_tid + V*d] + lu_post[d] / ln);
            }

        }
        """

class K_Collision_ShanChenMultiPhaseWalls(IdpyKernel):
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
                       'SType * dim_strides': ['global', 'restrict', 'const'], 
                       'FlagType * walls': ['global', 'restrict', 'const']}
        
        self.kernels[IDPY_T] = """
        if(g_tid < V && walls[g_tid] == 1){
            // Getting thread position
            SType g_tid_pos[DIM];
            F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);

            // Computing Shan-Chen Force
            SCFType F[DIM]; SType neigh_pos[DIM]; UType lu_post[DIM];
            for(int d=0; d<DIM; d++){F[d] = lu_post[d] = 0.;}

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

            // Local density and velocity for shift and equilibrium
            NType ln = n[g_tid]; UType lu[DIM];

            // Shan-Chen velocity shift & Copy to global memory
            for(int d=0; d<DIM; d++){ 
                lu[d] = u[g_tid + V*d] + F[d] / ln / OMEGA;
            }

            // Compute square norm of Guo shifted velocity
            UType u_dot_u = 0.;
            for(int d=0; d<DIM; d++){u_dot_u += lu[d]*lu[d];}

            // Cycle over the populations: equilibrium + Shan Chen
            for(int q=0; q<Q; q++){
                UType u_dot_xi = 0.; 
                for(int d=0; d<DIM; d++){
                    u_dot_xi += lu[d] * XI_list[d + q*DIM];
                }

                PopType leq_pop = 1.;

                // Equilibrium population
                leq_pop += + u_dot_xi*CM2 + 0.5*u_dot_xi*u_dot_xi*CM4;
                leq_pop += - 0.5*u_dot_u*CM2;
                leq_pop = leq_pop * ln * W_list[q];

                pop[g_tid + q*V] = \
                    pop[g_tid + q*V]*(1. - OMEGA) + leq_pop*OMEGA;

                for(int d=0; d<DIM; d++){
                   lu_post[d] += pop[g_tid + q*V] * XI_list[d + q*DIM];
                }

             }

            for(int d=0; d<DIM; d++){ 
                u[g_tid + V*d] = 0.5 * (u[g_tid + V*d] + lu_post[d] / ln);
            }

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

class K_Collision_ShanChenGuoMultiPhaseWalls(IdpyKernel):
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
                       'SType * dim_strides': ['global', 'restrict', 'const'], 
                       'FlagType * walls': ['global', 'restrict', 'const']}
        
        self.kernels[IDPY_T] = """
        if(g_tid < V && walls[g_tid] == 1){
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

class K_Collision_ShanChenGuoMultiPhaseWallsGravity(IdpyKernel):
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
                       'ForceType g': ['const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const'], 
                       'FlagType * walls': ['global', 'restrict', 'const']}
        
        self.kernels[IDPY_T] = """
        if(g_tid < V && walls[g_tid]){
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
            // Adding gravity only along the y direction
            F[1] += g;

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
                 optimizer_flag = None, headers_files=['math.h']):
        if psi_code is None:
            raise Exception("Missing argument psi_code")

        self.psi_code = psi_code
        
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag, headers_files=headers_files)
        self.SetCodeFlags('g_tid')
        self.params = {'PsiType * psi': ['global', 'restrict'],
                       'NType * n': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
            NType ln = n[g_tid];
            psi[g_tid] = """ + self.psi_code + """; }"""


class K_ComputePsiWalls(IdpyKernel):
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
                       'NType * n': ['global', 'restrict', 'const'], 
                       'FlagType * walls': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V && walls[g_tid]){
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
                 optimizer_flag = None, headers_files=['math.h']):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag, headers_files=headers_files)

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
        
class K_InitSingleFlatInterface(IdpyKernel):
    '''
    class K_InitSingleFlatInterface:
    need to add a tuning of the launch grid
    so that in extreme cases each thread cycles on more
    than a single point
    '''
    def __init__(self, custom_types = None, constants = {}, f_classes = [],
                 optimizer_flag = None, headers_files=['math.h']):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag, headers_files=headers_files)

        self.SetCodeFlags('g_tid')

        self.params = {'NType * n': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const'],
                       'SType * dim_center': ['global', 'restrict', 'const'],
                       'NType n_g': ['const'], 'NType n_l': ['const'],
                       'SType direction': ['const'],
                       'FlagType full_flag': ['const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
            SType g_tid_pos[DIM];
            F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);

            NType delta_n = full_flag * (n_l - n_g) + (1 - full_flag) * (n_g - n_l);

            n[g_tid] = n_g + \
            delta_n * F_NSingleFlatProfile(g_tid_pos[direction], dim_center[direction]);

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
                 optimizer_flag = None, headers_files=['math.h']):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag,
                            headers_files=headers_files)

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
        
