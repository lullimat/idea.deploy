#!/usr/bin/env python3
"""Smoke tests for Metal async Deploy / IdpyLoop encode batching."""

import unittest

import numpy as np

from idpy.IdpyCode import idpy_langs_sys, METAL_T, IDPY_T, GetTenet, IdpyMemory
from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyMethod, IdpyLoop
from idpy.Utils.CustomTypes import CustomTypes


class K_Inc(IdpyKernel):
    def __init__(self, custom_types=None, constants={}, f_classes=[],
                 optimizer_flag=None):
        super().__init__(custom_types=custom_types, constants=constants,
                         f_classes=f_classes, optimizer_flag=optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'NType * A': ['global', 'restrict']}
        self.kernels[IDPY_T] = """
        if(g_tid < DATA_N){
            A[g_tid] += 1;
        }
        """


class M_Swap(IdpyMethod):
    def Deploy(self, args_list=None, idpy_stream=None):
        args_list[0], args_list[1] = args_list[1], args_list[0]
        return IdpyMethod.PassIdpyStream(self, idpy_stream=idpy_stream)


# pymetallic get_status: 1=completed, 0=in-flight, -1=error/notEnqueued
_MTL_STATUS_COMPLETED = 1


@unittest.skipUnless(idpy_langs_sys.get(METAL_T, False), "Metal not available")
class TestMetalAsyncDispatch(unittest.TestCase):
    def setUp(self):
        try:
            self.tenet = GetTenet({"lang": METAL_T, "device": 0, "cl_kind": "gpu"})
        except Exception as exc:
            self.skipTest(f"Metal tenet unavailable: {exc}")
        self.n = 1 << 16
        self.block = (256, 1, 1)
        self.grid = ((self.n + 255) // 256, 1, 1)
        ct = CustomTypes({'NType': 'unsigned int'}).Push()
        self.idea = K_Inc(
            custom_types=ct, constants={'DATA_N': self.n},
        )(tenet=self.tenet, grid=self.grid, block=self.block)

    def tearDown(self):
        try:
            self.tenet.End()
        except Exception:
            pass

    def test_deploy_returns_cb_async_then_wait(self):
        A = IdpyMemory.Const(self.n, dtype=np.uint32, const=0, tenet=self.tenet)
        cb = self.idea.Deploy([A])
        self.assertTrue(hasattr(cb, 'wait_until_completed'))
        self.assertTrue(hasattr(cb, 'get_status'))
        # May already be completed for tiny work; wait must make status completed.
        cb.wait_until_completed()
        self.assertEqual(cb.get_status(), _MTL_STATUS_COMPLETED)
        self.assertTrue(np.all(A.D2H() == 1))

    def test_two_deploys_one_wait_then_d2h(self):
        A = IdpyMemory.Const(self.n, dtype=np.uint32, const=0, tenet=self.tenet)
        cb0 = self.idea.Deploy([A])
        cb1 = self.idea.Deploy([A])
        # No host wait between Deploys; sync once on the last CB.
        self.assertIsNot(cb0, cb1)
        cb1.wait_until_completed()
        self.assertEqual(cb1.get_status(), _MTL_STATUS_COMPLETED)
        self.assertTrue(np.all(A.D2H() == 2))

    def test_deploy_profiling_waits(self):
        A = IdpyMemory.Const(self.n, dtype=np.uint32, const=0, tenet=self.tenet)
        cb, t = self.idea.DeployProfiling([A])
        self.assertEqual(cb.get_status(), _MTL_STATUS_COMPLETED)
        self.assertGreaterEqual(t, 0.0)
        self.assertTrue(np.all(A.D2H() == 1))

    def test_tenet_finish_drains(self):
        A = IdpyMemory.Const(self.n, dtype=np.uint32, const=0, tenet=self.tenet)
        self.idea.Deploy([A])
        self.assertIsNotNone(self.tenet.last_command_buffer)
        self.tenet.Finish()
        self.assertIsNone(self.tenet.last_command_buffer)
        self.assertTrue(np.all(A.D2H() == 1))


@unittest.skipUnless(idpy_langs_sys.get(METAL_T, False), "Metal not available")
class TestMetalIdpyLoopBatch(unittest.TestCase):
    def setUp(self):
        try:
            self.tenet = GetTenet({"lang": METAL_T, "device": 0, "cl_kind": "gpu"})
        except Exception as exc:
            self.skipTest(f"Metal tenet unavailable: {exc}")
        self.n = 4096
        self.block = (256, 1, 1)
        self.grid = ((self.n + 255) // 256, 1, 1)
        self.ct = CustomTypes({'NType': 'unsigned int'}).Push()
        self.k_inc = K_Inc(
            custom_types=self.ct, constants={'DATA_N': self.n},
        )

    def tearDown(self):
        try:
            self.tenet.End()
        except Exception:
            pass

    def _count_cbs(self, run_fn):
        queue = self.tenet.queue
        orig = queue.make_command_buffer
        n = {'c': 0}

        def counted():
            n['c'] += 1
            return orig()

        queue.make_command_buffer = counted
        try:
            run_fn()
        finally:
            queue.make_command_buffer = orig
        return n['c']

    def test_three_kernels_one_cb(self):
        A = IdpyMemory.Const(self.n, dtype=np.uint32, const=0, tenet=self.tenet)
        ideas = [
            self.k_inc(tenet=self.tenet, grid=self.grid, block=self.block)
            for _ in range(3)
        ]
        loop = IdpyLoop(
            [{'A': A}],
            [[(ideas[0], ['A']), (ideas[1], ['A']), (ideas[2], ['A'])]],
        )

        def run():
            loop.Run(range(1))

        self.assertEqual(self._count_cbs(run), 1)
        self.assertTrue(np.all(A.D2H() == 3))

    def test_kernels_method_kernel_two_cbs(self):
        A = IdpyMemory.Const(self.n, dtype=np.uint32, const=0, tenet=self.tenet)
        B = IdpyMemory.Const(self.n, dtype=np.uint32, const=0, tenet=self.tenet)
        i0 = self.k_inc(tenet=self.tenet, grid=self.grid, block=self.block)
        i1 = self.k_inc(tenet=self.tenet, grid=self.grid, block=self.block)
        i2 = self.k_inc(tenet=self.tenet, grid=self.grid, block=self.block)
        swap = M_Swap(tenet=self.tenet)
        # Inc A×2, swap names, Inc dict['A'] (orig B): orig A=2, orig B=1
        mem = {'A': A, 'B': B}
        loop = IdpyLoop(
            [mem],
            [[(i0, ['A']), (i1, ['A']), (swap, ['A', 'B']), (i2, ['A'])]],
        )

        def run():
            loop.Run(range(1))

        self.assertEqual(self._count_cbs(run), 2)
        self.assertTrue(np.all(A.D2H() == 2))
        self.assertTrue(np.all(B.D2H() == 1))
        self.assertIs(mem['A'], B)
        self.assertIs(mem['B'], A)

    def test_loop_ends_with_method_still_syncs(self):
        A = IdpyMemory.Const(self.n, dtype=np.uint32, const=0, tenet=self.tenet)
        B = IdpyMemory.Const(self.n, dtype=np.uint32, const=0, tenet=self.tenet)
        i0 = self.k_inc(tenet=self.tenet, grid=self.grid, block=self.block)
        swap = M_Swap(tenet=self.tenet)
        mem = {'A': A, 'B': B}
        loop = IdpyLoop(
            [mem],
            [[(i0, ['A']), (swap, ['A', 'B'])]],
        )
        loop.Run(range(1))
        # GPU Inc completed before host swap; locals keep original buffers
        self.assertTrue(np.all(A.D2H() == 1))
        self.assertTrue(np.all(B.D2H() == 0))
        self.assertIs(mem['A'], B)
        self.assertIs(mem['B'], A)


if __name__ == '__main__':
    unittest.main(verbosity=2)
