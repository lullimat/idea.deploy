#!/usr/bin/env python3
"""Smoke tests for MetalMemoryPool buffer reuse."""

import gc
import unittest

import numpy as np

from idpy.IdpyCode import idpy_langs_sys, METAL_T, GetTenet, IdpyMemory


@unittest.skipUnless(idpy_langs_sys.get(METAL_T, False), "Metal not available")
class TestMetalMemoryPool(unittest.TestCase):
    def setUp(self):
        try:
            self.tenet = GetTenet({"lang": METAL_T, "device": 0, "cl_kind": "gpu"})
        except Exception as exc:
            self.skipTest(f"Metal tenet unavailable: {exc}")
        self.assertIsNotNone(self.tenet.mem_pool)

    def tearDown(self):
        try:
            self.tenet.End()
        except Exception:
            pass

    def test_zeros_reuse(self):
        pool = self.tenet.mem_pool
        V = 1 << 16
        a = IdpyMemory.Zeros(V, dtype=np.float32, tenet=self.tenet)
        self.assertEqual(a.shape, (V,))
        self.assertTrue(np.all(a.D2H() == 0))
        self.assertGreater(pool.active_bytes(), 0)
        a.release_to_pool()
        self.assertGreater(pool.held_bytes(), 0)
        held = pool.held_bytes()
        b = IdpyMemory.Zeros(V, dtype=np.float32, tenet=self.tenet)
        # Recycled: held should drop (buffer moved active)
        self.assertLess(pool.held_bytes(), held)
        self.assertTrue(np.all(b.D2H() == 0))
        b.release_to_pool()

    def test_many_alloc_free(self):
        pool = self.tenet.mem_pool
        V = 4096
        for _ in range(32):
            a = IdpyMemory.Zeros(V, dtype=np.float32, tenet=self.tenet)
            a.release_to_pool()
        gc.collect()
        # Free-list should hold at least one recycled buffer
        self.assertGreaterEqual(pool.held_bytes(), V * 4)

    def test_on_device_roundtrip(self):
        host = np.arange(1024, dtype=np.float32)
        dev = IdpyMemory.OnDevice(host, tenet=self.tenet)
        self.assertTrue(np.allclose(dev.D2H(), host))
        dev.release_to_pool()


if __name__ == '__main__':
    unittest.main(verbosity=2)
