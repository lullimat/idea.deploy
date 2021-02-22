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
Provides unit tests for the whole project coding part
Tests are defined in order of dependence
Simulations are tested in their own directories
'''

import unittest

import numpy as np
import sys, os, filecmp, inspect
from functools import reduce
_file_abs_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_idea_dot_deploy_path = os.path.dirname(os.path.abspath(_file_abs_path + "../"))
sys.path.append(_idea_dot_deploy_path)

def AllTrue(list_swap):
    return reduce(lambda x, y: x and y, list_swap)

'''
testing module Utils.ManageData
'''
from idpy.Utils.ManageData import ManageData

class TestManageData(unittest.TestCase):
    def setUp(self):
        self.file_name = 'md_dump_test'
        self.file_name_mv = self.file_name + '_mv'
        
    def test_ManageData_Dump(self):
        md_dump = ManageData(dump_file = self.file_name)
        md_dump.PushData(data = np.random.rand(10), key = 'paperino')
        md_dump.PushData(data = 'pippo, pluto', key = 'pappo')
        chars_n = md_dump.Dump()
        del md_dump
        os.remove(self.file_name)
        self.assertTrue(True)

    def test_ManageData_Read(self):
        md_dump = ManageData(dump_file = self.file_name)
        md_dump.PushData(data = np.random.rand(10), key = 'paperino')
        md_dump.PushData(data = 'pippo, pluto', key = 'pappo')
        md_dump.Dump()
        del md_dump
        
        md_read = ManageData(dump_file = self.file_name)
        read_flag = md_read.Read()
        del md_read
        os.remove(self.file_name)
        self.assertTrue(read_flag)

    def test_ManageData_ReadAndDump(self):
        md_dump = ManageData(dump_file = self.file_name)
        md_dump.PushData(data = np.random.rand(10), key = 'paperino')
        md_dump.PushData(data = 'pippo, pluto', key = 'pappo')
        md_dump.Dump()
        del md_dump
        
        md_read = ManageData(dump_file = self.file_name)
        md_read.Read()
        os.rename(self.file_name, self.file_name_mv)
        md_read.Dump()
        del md_read
        file_cmp = filecmp.cmp(self.file_name, self.file_name_mv)
        os.remove(self.file_name)
        os.remove(self.file_name_mv)
        self.assertTrue(file_cmp)

'''
testing module Utils.IsModuleThere
'''
from idpy.Utils.IsModuleThere import IsModuleThere, AreModulesThere

class TestIsModuleThere(unittest.TestCase):
    def test_IsModuleThere(self):
        self.assertTrue(IsModuleThere('unittest'))
        
    def test_AreModulesThere(self):
        query_list = AreModulesThere(['unittest', 'unittest'])
        answer = AllTrue(query_list)
        self.assertTrue(answer)

'''
testing module Utils.CustomTypes
'''
from idpy.Utils.CustomTypes import CustomTypes

class TestCustomTypes(unittest.TestCase):
    def setUp(self):
        self.types_dict = {'REAL': 'double', 'REAL32': 'float',
                           'integer': 'int', 'SpinType': 'unsigned int',
                           'BigSpin': 'unsigned long long int',
                           'character': 'char', 'uchar': 'unsigned char'}
        
    def test_CustomTypes_Push(self):
        myTypes = CustomTypes(self.types_dict)
        self.assertTrue(myTypes.Push() == self.types_dict)

    def test_CustomTypes_ToList(self):
        myTypes = CustomTypes(self.types_dict)
        test_list = [key for key in self.types_dict]
        self.assertTrue(myTypes.ToList() == test_list)

'''
testing module Utils.NpTypes
'''
from idpy.Utils.NpTypes import NpTypes

class TestNpTypes(unittest.TestCase):
    def setUp(self):
        self.known_c_types = {'double': np.float64, 'float': np.float32,
                              'int': np.int32, 'unsigned int': np.uint32,
                              'long long int': np.int64,
                              'unsigned long': np.uint64,
                              'unsigned long long int': np.uint64,
                              'char': np.byte, 'unsigned char': np.ubyte}
        self.known_np_types = {value: key for (key, value) in self.known_c_types.items()}

    def test_NpTypes_ToList(self):
        np_c = NpTypes()
        lists_out = np_c.ToList()

        list_0 = [key for key in self.known_c_types]
        list_1 = [key for key in self.known_np_types]

        self.assertTrue(AllTrue([list_0 == lists_out[0],
                                 list_1 == lists_out[1]]))

'''
testing module CUDA.CUDA
'''
if IsModuleThere('pycuda'):
    from idpy.CUDA.CUDA import CUDA
    from idpy.CUDA.CUDA import Tenet as CUTenet

    class TestCUDA(unittest.TestCase):
        def test_CUDA_DiscoverGPUs(self):
            cuda = CUDA()
            gpus_list = cuda.DiscoverGPUs()
            print("\n")
            for gpu_i in gpus_list:
                print("CUDA GPU[" + str(gpu_i) + "]")
                for key in gpus_list[gpu_i]:
                    print(key, ": ", gpus_list[gpu_i][key])

            del cuda
            self.assertTrue(True)

        def test_CUDA_ManageTenet(self):
            cuda = CUDA()
            cuda.SetDevice()
            tenet = cuda.GetTenet()
            check_type = isinstance(tenet, CUTenet)
            tenet.End()
            del cuda
            self.assertTrue(check_type)
        

'''
testing module OpenCL.OpenCL
'''
if IsModuleThere('pyopencl'):
    from idpy.OpenCL.OpenCL import OpenCL
    from idpy.OpenCL.OpenCL import Tenet as CLTenet

    class TestOpenCL(unittest.TestCase):
        def test_OpenCL_DiscoverGPUs(self):
            ocl = OpenCL()
            gpus_list = ocl.DiscoverGPUs()
            print("\n")
            for gpu_i in gpus_list:
                print("OpenCL GPU[" + str(gpu_i) + "]")
                for key in gpus_list[gpu_i]:
                    print(key, ": ", gpus_list[gpu_i][key])

            del ocl                    
            self.assertTrue(True)

        def test_OpenCL_DiscoverCPUs(self):
            ocl = OpenCL()
            cpus_list = ocl.DiscoverCPUs()
            print("\n")
            for cpu_i in cpus_list:
                print("OpenCL CPU[" + str(cpu_i) + "]")
                for key in cpus_list[cpu_i]:
                    print(key, ": ", cpus_list[cpu_i][key])

            del ocl                    
            self.assertTrue(True)

        def test_OpenCL_ManageTenet(self):
            ocl = OpenCL()
            ocl.SetDevice()
            tenet = ocl.GetTenet()
            check_type = isinstance(tenet, CLTenet)
            tenet.End()
            del ocl
            self.assertTrue(check_type)

'''
testing variables in IdpyCode.__init__.py
'''

from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T, idpy_langs_sys
from idpy.IdpyCode import idpy_langs_dict, idpy_langs_human_dict
from idpy.IdpyCode import idpy_langs_dict_sym, idpy_langs_list

class TestIdpyCodeInit(unittest.TestCase):
    def test_IdpyCodeInit(self):
        checks = []
        '''
        Checking basic types
        '''
        checks += [CUDA_T == 'pycuda', OCL_T == 'pyopencl', IDPY_T == 'idpy']
        '''
        idpy_langs_dict
        '''
        dict_check = {'CUDA_T': CUDA_T, 'OCL_T': OCL_T}
        checks += [idpy_langs_dict == dict_check]
        '''
        idpy_langs_human_dict
        '''
        dict_check = {CUDA_T: "CUDA", OCL_T: "OpenCL"}
        checks += [idpy_langs_human_dict == dict_check]
        '''
        idpy_langs_dict_sym
        '''
        dict_check = {CUDA_T: "CUDA_T", OCL_T: "OCL_T"}
        checks += [idpy_langs_dict_sym == dict_check]
        '''
        idpy_langs_list
        '''
        list_check = list({'CUDA_T': CUDA_T, 'OCL_T': OCL_T}.values())
        checks += [idpy_langs_list == list_check]
        '''
        idpy_langs_sys
        '''
        checks += [idpy_langs_sys == AreModulesThere(idpy_langs_list)]
        
        self.assertTrue(AllTrue(checks))

'''
testing IdpyCode.IdpyConsts
'''

from idpy.IdpyCode.IdpyConsts import AddrQualif, KernQualif, FuncQualif
class TestIdpyConsts(unittest.TestCase):
    def test_FuncQualif(self):
        checks = []
        fq = FuncQualif()

        dict_check = {CUDA_T: """__device__""", OCL_T: """ """}
        for lang in idpy_langs_list:
            checks += [fq[lang] == dict_check[lang]]

        self.assertTrue(AllTrue(checks))
        
    def test_KernQualif(self):
        checks = []
        kq = KernQualif()

        dict_check = {CUDA_T: """__global__ void""", OCL_T: """__kernel void"""}
        for lang in idpy_langs_list:
            checks += [kq[lang] == dict_check[lang]]

        self.assertTrue(AllTrue(checks))            
        
    def test_AddrQualif(self):
        checks = []
        aq = AddrQualif()
        '''
        CUDA_T values
        '''
        dict_check = {'const': """const""",
                      'local': """local""",
                      'restrict': """__restrict__""",
                      'shared': """__shared__""",
                      'device': """__device__""",
                      'global': ''}
        checks += [aq[CUDA_T] == dict_check]

        '''
        OCL_T values
        '''
        dict_check = {'global': """__global""",
                      'const': """__const""",
                      'local': """__local""",
                      'restrict': '',
                      'restrict': '',
                      'shared': '',
                      'device': ''}
        checks += [aq[OCL_T] == dict_check]
        
        self.assertTrue(AllTrue(checks))

'''
testing IdpyCode.IdpyMemory
'''
import idpy.IdpyCode.IdpyMemory as IdpyMemory

if IsModuleThere('pycuda'):
    from idpy.CUDA.CUDA import CUDA
    from idpy.CUDA.CUDA import Tenet as CUTenet
    
    class TestIdpyArrayCU(unittest.TestCase):
        def setUp(self):
            self.constant = -2
            
        def test_IdpyArray(self):
            cu = CUDA()
            cu.SetDevice()
            tenet = cu.GetTenet()
            rand_mem = IdpyMemory.Array(10, dtype = np.int32, tenet = tenet)
            on_dev_range = IdpyMemory.OnDevice(np.arange(10, dtype = np.int32), tenet = tenet)
            zeros = IdpyMemory.Zeros(10, dtype = np.float32, tenet = tenet)
            i_range = IdpyMemory.Range(10, tenet = tenet)
            const = IdpyMemory.Const(10, dtype = np.int32,
                                     const = self.constant, tenet = tenet)
            print()
            print("rand_mem:\t", rand_mem.D2H(), rand_mem.dtype)
            print("on_dev_range:\t", on_dev_range.D2H(), on_dev_range.dtype)
            print("zeros:\t", zeros.D2H(), zeros.dtype)
            print("i_range:\t", i_range.D2H(), i_range.dtype)
            print("const:\t", const.D2H(), const.dtype)
            int_buffer = np.zeros(10, dtype = np.int32)
            const.D2H(int_buffer)
            print("int_buffer: ", int_buffer, int_buffer.dtype)

            '''
            checks
            '''
            checks = []

            checks += [AllTrue(list(on_dev_range.D2H() == np.arange(10, dtype = np.int32)))]
            checks += [AllTrue(list(zeros.D2H() == np.zeros(10, dtype = np.float32)))]
            checks += [AllTrue(list(i_range.D2H() == np.arange(10, dtype = np.int32)))]

            chk_const = np.zeros(10, dtype = np.int32)
            chk_const.fill(self.constant)
                        
            checks += [AllTrue(list(const.D2H() == chk_const))]
            
            tenet.End()            
            self.assertTrue(AllTrue(checks))

if IsModuleThere('pyopencl'):
    from idpy.OpenCL.OpenCL import OpenCL
    from idpy.OpenCL.OpenCL import Tenet as CLTenet
    
    class TestIdpyArrayCL(unittest.TestCase):
        def setUp(self):
            self.constant = -2
            
        def test_IdpyArray(self):
            ocl = OpenCL()
            ocl.SetDevice()
            tenet = ocl.GetTenet()
            rand_mem = IdpyMemory.Array(10, dtype = np.int32, tenet = tenet)
            on_dev_range = IdpyMemory.OnDevice(np.arange(10, dtype = np.int32), tenet = tenet)
            zeros = IdpyMemory.Zeros(10, dtype = np.float32, tenet = tenet)
            i_range = IdpyMemory.Range(10, tenet = tenet)
            const = IdpyMemory.Const(10, dtype = np.int32,
                                     const = self.constant, tenet = tenet)
            print()
            print("rand_mem:\t", rand_mem.D2H(), rand_mem.dtype)
            print("on_dev_range:\t", on_dev_range.D2H(), on_dev_range.dtype)
            print("zeros:\t", zeros.D2H(), zeros.dtype)
            print("i_range:\t", i_range.D2H(), i_range.dtype)
            print("const:\t", const.D2H(), const.dtype)
            int_buffer = np.zeros(10, dtype = np.int32)
            const.D2H(int_buffer)
            print("int_buffer: ", int_buffer, int_buffer.dtype)
            '''
            checks
            '''
            checks = []

            checks += [AllTrue(list(on_dev_range.D2H() == np.arange(10, dtype = np.int32)))]
            checks += [AllTrue(list(zeros.D2H() == np.zeros(10, dtype = np.float32)))]
            checks += [AllTrue(list(i_range.D2H() == np.arange(10, dtype = np.int32)))]

            chk_const = np.zeros(10, dtype = np.int32)
            chk_const.fill(self.constant)
                        
            checks += [AllTrue(list(const.D2H() == chk_const))]
            
            tenet.End()            
            self.assertTrue(AllTrue(checks))
        
'''
testing IdpyCode.IdpyCode
'''
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyFunction
from idpy.IdpyCode.IdpyCode import IdpyMethod, IdpyLoop

class TestIdpyCode(unittest.TestCase):
    class F_SumTwoArraysVal(IdpyFunction):
        def __init__(self, custom_types = None, f_type = 'SpinType'):
            super().__init__(custom_types = custom_types, f_type = f_type)
            self.params = {'SpinType A': ['const'],
                           'SpinType B': ['const'],
                           'SpinType in_const': ['const']}

            self.functions[IDPY_T] = """
            return A + B + in_const;
            """

    class F_SumTwoArraysRet(IdpyFunction):
        def __init__(self, custom_types = None, f_type = 'SpinType'):
            super().__init__(custom_types = custom_types, f_type = f_type)
            self.params = {'SpinType * A': ['global', 'const'],
                           'SpinType * B': ['global', 'const'],
                           'unsigned int g_tid': ['const']}

            self.functions[IDPY_T] = """
            return A[g_tid] + B[g_tid];
            """

    class F_SumTwoArraysPtr(IdpyFunction):
        def __init__(self, custom_types = None, f_type = 'void'):
            super().__init__(custom_types = custom_types, f_type = f_type)
            self.params = {'SpinType * A': ['global', 'const'],
                           'SpinType * B': ['global', 'const'],
                           'SpinType * C': ['global'],
                           'unsigned int g_tid': ['const']}

            self.functions[IDPY_T] = """
            C[g_tid] += A[g_tid] + B[g_tid];
            return;
            """
        
    class K_SumTwoArrays(IdpyKernel):
        def __init__(self, custom_types = None, constants = {},
                     f_classes = []):
            super().__init__(custom_types = custom_types,
                             constants = constants,
                             f_classes = f_classes)
            self.SetCodeFlags('g_tid')
            self.params = {'SpinType * A': ['global', 'restrict', 'const'],
                           'SpinType * B': ['global', 'restrict', 'const'],
                           'SpinType * C': ['global', 'restrict'],
                           'SpinType in_const': []}
            
            self.kernels[IDPY_T] = """
            if(g_tid < DATA_N){
            C[g_tid] += in_const;
            F_SumTwoArraysPtr(A, B, C, g_tid);
            C[g_tid] += F_SumTwoArraysRet(A, B, g_tid);
            C[g_tid] += F_SumTwoArraysVal(A[g_tid], B[g_tid], in_const);
            }
            """

    class K_SumConst(IdpyKernel):
        def __init__(self, custom_types = None, constants = {},
                     f_classes = []):
            super().__init__(custom_types = custom_types,
                             constants = constants,
                             f_classes = f_classes)
            self.SetCodeFlags('g_tid')
            self.params = {'SpinType * A': ['global', 'restrict'],
                           'SpinType in_const': []}
            self.kernels[IDPY_T] = """
            if(g_tid < DATA_N){
            A[g_tid] += in_const;
            }
            """
            
    class K_SumOne(IdpyKernel):
        def __init__(self, custom_types = None, constants = {},
                     f_classes = []):
            super().__init__(custom_types = custom_types,
                             constants = constants,
                             f_classes = f_classes)
            self.SetCodeFlags('g_tid')
            self.params = {'SpinType * A': ['global', 'restrict']}
            self.kernels[IDPY_T] = """
            if(g_tid < DATA_N){
            A[g_tid] += 1;
            }
            """
        
    class M_SwapArrays(IdpyMethod):
        def __init__(self, tenet = None):
            super().__init__(tenet = tenet)

        def Deploy(self, args_list = None, idpy_stream = None):
            args_list[0], args_list[1] = args_list[1], args_list[0]
            if self.lang == OCL_T:
                if idpy_stream is None:
                    return None
                else:
                    return idpy_stream[0]

    def setUp(self):
        self.n, self.block_size = 10, 128
        self.in_const = 3

    if IsModuleThere('pycuda'):
        from idpy.CUDA.CUDA import CUDA
        from idpy.CUDA.CUDA import Tenet as CUTenet

        def test_IdpyKernelFuncLoopMultStreamCU(self):
            cu = CUDA()
            cu.SetDevice()
            tenet = cu.GetTenet()
            
            grid, block = ((self.n + self.block_size - 1)//self.block_size, 1, 1), (self.block_size, 1, 1)
            
            myTypes = CustomTypes({'SpinType': 'unsigned int'})
            np_c = NpTypes()
            SumTwoArrConst = self.K_SumTwoArrays(custom_types = myTypes.Push(),
                                                 constants = {'DATA_N': self.n},
                                                 f_classes = [self.F_SumTwoArraysPtr,
                                                              self.F_SumTwoArraysRet,
                                                              self.F_SumTwoArraysVal])

            SumConst = self.K_SumConst(custom_types = myTypes.Push(), constants = {'DATA_N': self.n})

            A = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 0, tenet = tenet)

            B = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 1, tenet = tenet)

            C = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 2, tenet = tenet)

            D = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 3, tenet = tenet)

            print()
            print("A: ", A.D2H(), A.dtype)
            print("B: ", B.D2H(), B.dtype)
            print("C: ", C.D2H(), C.dtype)
            print("D: ", D.D2H(), D.dtype)
            '''
            Checking result
            '''
            a, b, c, cc = A.D2H()[0], B.D2H()[0], C.D2H()[0], self.in_const
            d = D.D2H()[0]
            
            for i in range(2):
                '''first stream'''
                c += cc
                c += a + b
                c += a + b
                c += a + b + cc
                a, c = c, a
                '''second stream'''
                d += cc

            # https://stackoverflow.com/questions/5710690/pycuda-passing-variable-by-value-to-kernel
            ##
            mem_dict_0 = {'A': A, 'B': B, 'C': C, 'const': np_c.C[myTypes['SpinType']](self.in_const)}
            mem_dict_1 = {'D': D, 'const': np_c.C[myTypes['SpinType']](self.in_const)}
            SumTwoArrConst_Loop = IdpyLoop(
                [mem_dict_0, mem_dict_1],
                [
                    [
                        (SumTwoArrConst(tenet = tenet, grid = grid, block = block),
                         ['A', 'B', 'C', 'const']),
                        (self.M_SwapArrays(tenet), ['A', 'C'])
                    ],
                    [
                        (SumConst(tenet = tenet, grid = grid, block = block), ['D', 'const']),
                    ]
                ])
            
            print()
            print("SumTwoArrConst_Loop.Run(range(2))")
            SumTwoArrConst_Loop.Run(range(2))

            print("A: ", A.D2H(), A.dtype, a)
            print("B: ", B.D2H(), B.dtype, b)
            print("C: ", C.D2H(), C.dtype, c)
            print("D: ", D.D2H(), D.dtype, d)
            
            checks = []
            check_array = np.full(self.n, a, dtype = np_c.C[myTypes['SpinType']])
            checks += [AllTrue(A.D2H() == check_array)]
            check_array = np.full(self.n, b, dtype = np_c.C[myTypes['SpinType']])
            checks += [AllTrue(B.D2H() == check_array)]
            check_array = np.full(self.n, c, dtype = np_c.C[myTypes['SpinType']])
            checks += [AllTrue(C.D2H() == check_array)]
            check_array = np.full(self.n, d, dtype = np_c.C[myTypes['SpinType']])
            checks += [AllTrue(D.D2H() == check_array)]
            
            tenet.End()
            self.assertTrue(AllTrue(checks))        

        
        def test_IdpyKernelLoopConstCU(self):
            cu = CUDA()
            cu.SetDevice()
            tenet = cu.GetTenet()
            
            grid, block = ((self.n + self.block_size - 1)//self.block_size, 1, 1), (self.block_size, 1, 1)
            
            myTypes = CustomTypes({'SpinType': 'unsigned int'})
            np_c = NpTypes()
            SumConst = self.K_SumConst(custom_types = myTypes.Push(),
                                       constants = {'DATA_N': self.n})
            
            A = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 0, tenet = tenet)
            mem_dict = {'A': A, 'const': np.int32(self.in_const)}
            # https://stackoverflow.com/questions/5710690/pycuda-passing-variable-by-value-to-kernel
            SumOne_Loop = IdpyLoop([mem_dict],
                                   [
                                       [
                                           (SumConst(tenet = tenet,
                                                     grid = grid,
                                                     block = block), ['A', 'const'])
                                       ]
                                   ])

            print()
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Loop.Run(range(1))")
            SumOne_Loop.Run(range(1))            
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Loop.Run(range(8))")
            SumOne_Loop.Run(range(8))            
            print("A: ", A.D2H(), A.dtype)

            check_array = np.zeros(self.n, dtype = np_c.C[myTypes['SpinType']])
            check_array.fill(self.in_const * 9)
            
            checks = []
            checks += [AllTrue(A.D2H() == check_array)]
            
            tenet.End()
            self.assertTrue(AllTrue(checks))        
        
        def test_IdpyKernelLoopCU(self):
            cu = CUDA()
            cu.SetDevice()
            tenet = cu.GetTenet()
            
            grid, block = ((self.n + self.block_size - 1)//self.block_size, 1, 1), (self.block_size, 1, 1)
            
            myTypes = CustomTypes({'SpinType': 'unsigned int'})
            np_c = NpTypes()
            SumOne = self.K_SumOne(custom_types = myTypes.Push(),
                                   constants = {'DATA_N': self.n})

            A = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 0, tenet = tenet)
            mem_dict = {'A': A}
            SumOne_Loop = IdpyLoop([mem_dict],
                                   [
                                       [
                                           (SumOne(tenet = tenet,
                                                   grid = grid,
                                                   block = block), ['A'])
                                       ]
                                   ])

            print()
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Loop.Run(range(1))")
            SumOne_Loop.Run(range(1))            
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Loop.Run(range(8))")
            SumOne_Loop.Run(range(8))            
            print("A: ", A.D2H(), A.dtype)

            check_array = np.zeros(self.n, dtype = np_c.C[myTypes['SpinType']])
            check_array.fill(9)
            
            checks = []
            checks += [AllTrue(A.D2H() == check_array)]
            
            tenet.End()
            self.assertTrue(AllTrue(checks))        


        def test_IdpyKernelCU(self):
            cu = CUDA()
            cu.SetDevice()
            tenet = cu.GetTenet()
            
            grid, block = ((self.n + self.block_size - 1)//self.block_size, 1, 1), (self.block_size, 1, 1)
            
            myTypes = CustomTypes({'SpinType': 'unsigned int'})
            np_c = NpTypes()
            SumOne = self.K_SumOne(custom_types = myTypes.Push(),
                                   constants = {'DATA_N': self.n})
            SumOne_Idea = SumOne(tenet = tenet, grid = grid, block = block)

            A = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 0, tenet = tenet)
            print()
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Idea.Deploy([A])")
            SumOne_Idea.Deploy([A])            
            print("A: ", A.D2H(), A.dtype)

            check_array = np.zeros(self.n, dtype = np_c.C[myTypes['SpinType']])
            check_array.fill(1)

            checks = [AllTrue(A.D2H() == check_array)]
            
            tenet.End()
            self.assertTrue(AllTrue(checks))        

        
        def test_IdpyMethodCU(self):
            cu = CUDA()
            cu.SetDevice()
            tenet = cu.GetTenet()
            SwapArrays = self.M_SwapArrays(tenet)
            zeros = IdpyMemory.Const(self.n, dtype = np.int32, const = 0, tenet = tenet)
            ones = IdpyMemory.Const(self.n, dtype = np.int32, const = 1, tenet = tenet)
            mem_list = [zeros, ones]
            print()
            print("[0]: ", mem_list[0].D2H(), mem_list[0].dtype)
            print("[1]: ", mem_list[1].D2H(), mem_list[1].dtype)
            print("Swapping")
            SwapArrays.Deploy(mem_list)
            print("[0]: ", mem_list[0].D2H(), mem_list[0].dtype)
            print("[1]: ", mem_list[1].D2H(), mem_list[1].dtype)

            check_1 = np.zeros(self.n, dtype = np.int32)
            check_0 = np.zeros(self.n, dtype = np.int32)
            check_0.fill(1)

            checks = []
            checks += [AllTrue(mem_list[0].D2H() == check_0)]
            checks += [AllTrue(mem_list[1].D2H() == check_1)]
            
            tenet.End()
            self.assertTrue(AllTrue(checks))        

        def test_IdpyMethodLoopCU(self):
            cu = CUDA()
            cu.SetDevice()
            tenet = cu.GetTenet()
            
            zeros = IdpyMemory.Const(self.n, dtype = np.int32, const = 0, tenet = tenet)
            ones = IdpyMemory.Const(self.n, dtype = np.int32, const = 1, tenet = tenet)
            mem_dict = {'zeros': zeros, 'ones': ones}
            SwapArraysLoop = IdpyLoop([mem_dict],
                                          [
                                              [
                                                  (self.M_SwapArrays(tenet), ['zeros', 'ones'])
                                              ]
                                          ])
            print()
            print("['zeros']: ", mem_dict['zeros'].D2H(), mem_dict['zeros'].dtype)
            print("['ones']: ", mem_dict['ones'].D2H(), mem_dict['ones'].dtype)
            print("SwapArraysLoop(range(1))")
            SwapArraysLoop.Run(range(1))
            print("['zeros']: ", mem_dict['zeros'].D2H(), mem_dict['zeros'].dtype)
            print("['ones']: ", mem_dict['ones'].D2H(), mem_dict['ones'].dtype)
            print("SwapArraysLoop(range(4))")
            SwapArraysLoop.Run(range(4))
            print("['zeros']: ", mem_dict['zeros'].D2H(), mem_dict['zeros'].dtype)
            print("['ones']: ", mem_dict['ones'].D2H(), mem_dict['ones'].dtype)
            print("SwapArraysLoop(range(7))")
            SwapArraysLoop.Run(range(7))
            print("['zeros']: ", mem_dict['zeros'].D2H(), mem_dict['zeros'].dtype)
            print("['ones']: ", mem_dict['ones'].D2H(), mem_dict['ones'].dtype)
            checks = []
            checks += [AllTrue(list(mem_dict['zeros'].D2H() == np.zeros(self.n, dtype = np.int32)))]
            check_ones = np.zeros(self.n, dtype = np.int32)
            check_ones.fill(1)
            checks += [AllTrue(list(mem_dict['ones'].D2H() == check_ones))]

            tenet.End()
            self.assertTrue(AllTrue(checks))        
            
    if IsModuleThere('pyopencl'):
        from idpy.OpenCL.OpenCL import OpenCL
        from idpy.OpenCL.OpenCL import Tenet as CLTenet

        def test_IdpyKernelFuncLoopMultStreamCL(self):
            ocl = OpenCL()
            ocl.SetDevice()
            tenet = ocl.GetTenet()
            
            grid, block = ((self.n + self.block_size - 1)//self.block_size, 1, 1), (self.block_size, 1, 1)
            
            myTypes = CustomTypes({'SpinType': 'unsigned int'})
            np_c = NpTypes()
            SumTwoArrConst = self.K_SumTwoArrays(custom_types = myTypes.Push(),
                                                 constants = {'DATA_N': self.n},
                                                 f_classes = [self.F_SumTwoArraysPtr,
                                                              self.F_SumTwoArraysRet,
                                                              self.F_SumTwoArraysVal])

            SumConst = self.K_SumConst(custom_types = myTypes.Push(), constants = {'DATA_N': self.n})

            A = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 0, tenet = tenet)

            B = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 1, tenet = tenet)

            C = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 2, tenet = tenet)

            D = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 3, tenet = tenet)

            print()
            print("A: ", A.D2H(), A.dtype)
            print("B: ", B.D2H(), B.dtype)
            print("C: ", C.D2H(), C.dtype)
            print("D: ", D.D2H(), D.dtype)
            '''
            Checking result
            '''
            a, b, c, cc = A.D2H()[0], B.D2H()[0], C.D2H()[0], self.in_const
            d = D.D2H()[0]
            
            for i in range(2):
                '''first stream'''
                c += cc
                c += a + b
                c += a + b
                c += a + b + cc
                a, c = c, a
                '''second stream'''
                d += cc

            # https://stackoverflow.com/questions/5710690/pycuda-passing-variable-by-value-to-kernel
            ##
            mem_dict_0 = {'A': A, 'B': B, 'C': C, 'const': np_c.C[myTypes['SpinType']](self.in_const)}
            mem_dict_1 = {'D': D, 'const': np_c.C[myTypes['SpinType']](self.in_const)}
            SumTwoArrConst_Loop = IdpyLoop(
                [mem_dict_0, mem_dict_1],
                [
                    [
                        (SumTwoArrConst(tenet = tenet, grid = grid, block = block),
                         ['A', 'B', 'C', 'const']),
                        (self.M_SwapArrays(tenet), ['A', 'C'])
                    ],
                    [
                        (SumConst(tenet = tenet, grid = grid, block = block), ['D', 'const']),
                    ]
                ])
            
            print()
            print("SumTwoArrConst_Loop.Run(range(2))")
            SumTwoArrConst_Loop.Run(range(2))

            print("A: ", A.D2H(), A.dtype, a)
            print("B: ", B.D2H(), B.dtype, b)
            print("C: ", C.D2H(), C.dtype, c)
            print("D: ", D.D2H(), D.dtype, d)
            
            checks = []
            check_array = np.full(self.n, a, dtype = np_c.C[myTypes['SpinType']])
            checks += [AllTrue(A.D2H() == check_array)]
            check_array = np.full(self.n, b, dtype = np_c.C[myTypes['SpinType']])
            checks += [AllTrue(B.D2H() == check_array)]
            check_array = np.full(self.n, c, dtype = np_c.C[myTypes['SpinType']])
            checks += [AllTrue(C.D2H() == check_array)]
            check_array = np.full(self.n, d, dtype = np_c.C[myTypes['SpinType']])
            checks += [AllTrue(D.D2H() == check_array)]
            
            tenet.End()
            self.assertTrue(AllTrue(checks))        

        
        def test_IdpyKernelLoopConstCL(self):
            ocl = OpenCL()
            ocl.SetDevice()
            tenet = ocl.GetTenet()
            
            grid, block = ((self.n + self.block_size - 1)//self.block_size, 1, 1), (self.block_size, 1, 1)
            
            myTypes = CustomTypes({'SpinType': 'unsigned int'})
            np_c = NpTypes()
            SumConst = self.K_SumConst(custom_types = myTypes.Push(),
                                       constants = {'DATA_N': self.n})

            A = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 0, tenet = tenet)
            
            # https://stackoverflow.com/questions/5710690/pycuda-passing-variable-by-value-to-kernel
            mem_dict = {'A': A, 'const': np.int32(self.in_const)}
            SumOne_Loop = IdpyLoop([mem_dict],
                                       [
                                           [
                                               (SumConst(tenet = tenet,
                                                         grid = grid,
                                                         block = block), ['A', 'const'])
                                           ]
                                       ])

            print()
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Loop.Run(range(1))")
            SumOne_Loop.Run(range(1))            
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Loop.Run(range(8))")
            SumOne_Loop.Run(range(8))            
            print("A: ", A.D2H(), A.dtype)

            check_array = np.zeros(self.n, dtype = np_c.C[myTypes['SpinType']])
            check_array.fill(self.in_const * 9)
            
            checks = []
            checks += [AllTrue(A.D2H() == check_array)]
            
            tenet.End()
            self.assertTrue(AllTrue(checks))        
        
        def test_IdpyKernelLoopCL(self):
            ocl = OpenCL()
            ocl.SetDevice()
            tenet = ocl.GetTenet()
            
            grid, block = ((self.n + self.block_size - 1)//self.block_size, 1, 1), (self.block_size, 1, 1)
            
            myTypes = CustomTypes({'SpinType': 'unsigned int'})
            np_c = NpTypes()
            SumOne = self.K_SumOne(custom_types = myTypes.Push(),
                                   constants = {'DATA_N': self.n})

            A = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']],
                                 const = 0, tenet = tenet)
            mem_dict = {'A': A}
            SumOne_Loop = IdpyLoop([mem_dict],
                                       [
                                           [
                                               (SumOne(tenet = tenet,
                                                       grid = grid,
                                                       block = block), ['A'])
                                           ]
                                       ])

            print()
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Loop.Run(range(1))")
            SumOne_Loop.Run(range(1))            
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Loop.Run(range(8))")
            SumOne_Loop.Run(range(8))            
            print("A: ", A.D2H(), A.dtype)

            check_array = np.zeros(self.n, dtype = np_c.C[myTypes['SpinType']])
            check_array.fill(9)
            
            checks = []
            checks += [AllTrue(A.D2H() == check_array)]

            tenet.End()
            self.assertTrue(AllTrue(checks))        

        def test_IdpyKernelCL(self):
            ocl = OpenCL()
            ocl.SetDevice()
            tenet = ocl.GetTenet()
            
            grid, block = ((self.n + self.block_size - 1)//self.block_size, 1, 1), (self.block_size, 1, 1)
            
            myTypes = CustomTypes({"SpinType": "unsigned int"})
            np_c = NpTypes()
            SumOne = self.K_SumOne(custom_types = myTypes.Push(), constants = {'DATA_N': self.n})
            SumOne_Idea = SumOne(tenet = tenet, grid = grid, block = block)

            A = IdpyMemory.Const(self.n, dtype = np_c.C[myTypes['SpinType']], const = 0, tenet = tenet)
            print()
            print("A: ", A.D2H(), A.dtype)
            print("SumOne_Idea.Deploy([A])")
            SumOne_Idea.Deploy([A])            
            print("A: ", A.D2H(), A.dtype)

            check_array = np.zeros(self.n, dtype = np_c.C[myTypes['SpinType']])
            check_array.fill(1)

            checks = [AllTrue(A.D2H() == check_array)]
            
            tenet.End()
            self.assertTrue(AllTrue(checks))        


        def test_IdpyMethodCL(self):
            ocl = OpenCL()
            ocl.SetDevice()
            tenet = ocl.GetTenet()
            SwapArrays = self.M_SwapArrays(tenet)
            zeros = IdpyMemory.Const(self.n, dtype = np.int32, const = 0, tenet = tenet)
            ones = IdpyMemory.Const(self.n, dtype = np.int32, const = 1, tenet = tenet)
            mem_list = [zeros, ones]
            print()
            print("[0]: ", mem_list[0].D2H(), mem_list[0].dtype)
            print("[1]: ", mem_list[1].D2H(), mem_list[1].dtype)
            print("Swapping")
            SwapArrays.Deploy(mem_list)
            print("[0]: ", mem_list[0].D2H(), mem_list[0].dtype)
            print("[1]: ", mem_list[1].D2H(), mem_list[1].dtype)

            check_1 = np.zeros(self.n, dtype = np.int32)
            check_0 = np.zeros(self.n, dtype = np.int32)
            check_0.fill(1)

            checks = []
            checks += [AllTrue(mem_list[0].D2H() == check_0)]
            checks += [AllTrue(mem_list[1].D2H() == check_1)]
            
            tenet.End()
            self.assertTrue(AllTrue(checks))
            
        def test_IdpyMethodLoopCL(self):
            cl = OpenCL()
            cl.SetDevice()
            tenet = cl.GetTenet()
            
            zeros = IdpyMemory.Const(self.n, dtype = np.int32, const = 0, tenet = tenet)
            ones = IdpyMemory.Const(self.n, dtype = np.int32, const = 1, tenet = tenet)
            mem_dict = {'zeros': zeros, 'ones': ones}
            SwapArraysLoop = IdpyLoop([mem_dict],
                                          [
                                              [
                                                  (self.M_SwapArrays(tenet), ['zeros', 'ones'])
                                              ]
                                          ])
            print()
            print("['zeros']: ", mem_dict['zeros'].D2H(), mem_dict['zeros'].dtype)
            print("['ones']: ", mem_dict['ones'].D2H(), mem_dict['ones'].dtype)
            print("SwapArraysLoop(range(1))")
            SwapArraysLoop.Run(range(1))
            print("['zeros']: ", mem_dict['zeros'].D2H(), mem_dict['zeros'].dtype)
            print("['ones']: ", mem_dict['ones'].D2H(), mem_dict['ones'].dtype)  
            print("SwapArraysLoop(range(4))")
            SwapArraysLoop.Run(range(4))
            print("['zeros']: ", mem_dict['zeros'].D2H(), mem_dict['zeros'].dtype)
            print("['ones']: ", mem_dict['ones'].D2H(), mem_dict['ones'].dtype)              
            print("SwapArraysLoop(range(7))")
            SwapArraysLoop.Run(range(7))
            print("['zeros']: ", mem_dict['zeros'].D2H(), mem_dict['zeros'].dtype)
            print("['ones']: ", mem_dict['ones'].D2H(), mem_dict['ones'].dtype)              
            checks = []
            checks += [AllTrue(list(mem_dict['zeros'].D2H() == np.zeros(self.n, dtype = np.int32)))]
            check_ones = np.zeros(self.n, dtype = np.int32)
            check_ones.fill(1)
            checks += [AllTrue(list(mem_dict['ones'].D2H() == check_ones))]

            tenet.End()
            self.assertTrue(AllTrue(checks))        
        
if __name__ == '__main__':
    unittest.main()
