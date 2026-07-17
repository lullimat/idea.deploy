#!/usr/bin/env python3
"""Milestone 1: IdpyKernel periodic convolution vs numpy JSymmetricTensor path."""

import sys
import unittest
import numpy as np

from idpy.Utils.IdpySymbolic import SymmetricTensor, JSymmetricTensor
from idpy.IdpyCode import idpy_langs_sys
from idpy.IdpyCode import CTYPES_T, OCL_T, METAL_T, CUDA_T
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyStencils.IdpyConvolution import (
    K_ConvolvePeriodic,
    convolve_periodic,
    pack_lattice_flat,
    unpack_lattice_flat,
    clear_active_tenet,
    clear_idea_cache,
    set_convolution_shape,
    clear_convolution_shape,
    _idea_cache,
    _tenet_of,
)


def _centered_dx_kernel_1d(dtype=np.float64):
    c = np.zeros(5, dtype=dtype)
    c[:] = np.array([1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0], dtype=dtype)
    return c


def _centered_dx_kernel_2d(dtype=np.float64):
    """5-point 1D derivative along axis 0, padded to 5x5 center column."""
    c = np.zeros((5, 5), dtype=dtype)
    c[:, 2] = np.array(
        [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0], dtype=dtype,
    )
    return c


def _laplacian_kernel_3d(dtype=np.float64):
    c = np.zeros((3, 3, 3), dtype=dtype)
    c[1, 1, 1] = -6.0
    c[0, 1, 1] = c[2, 1, 1] = 1.0
    c[1, 0, 1] = c[1, 2, 1] = 1.0
    c[1, 1, 0] = c[1, 1, 2] = 1.0
    return c


def _numpy_convolve(kernel, field):
    d = field.ndim
    st = JSymmetricTensor(
        d=d, rank=0, ranks=[0, 0],
        c_dict={0: np.array(kernel, copy=True)},
        dtype=kernel.dtype,
    )
    fld = SymmetricTensor(
        d=d, rank=0,
        c_dict={0: np.array(field, copy=True)},
        dtype=field.dtype,
    )
    return np.asarray((st @ fld)[0])


def _taps(kernel):
    center = tuple(s // 2 for s in kernel.shape)
    nz = np.nonzero(kernel)
    offsets = np.array(list(zip(*nz)), dtype=np.int64) - np.array(center)
    coeffs = kernel[nz]
    return offsets, coeffs


class TestConvolutionNumpy(unittest.TestCase):
    def test_numpy_path_smoke_2d(self):
        rng = np.random.default_rng(0)
        shape = (32, 32)
        field = np.ascontiguousarray(rng.standard_normal(shape))
        kernel = _centered_dx_kernel_2d()
        out = _numpy_convolve(kernel, field)
        self.assertEqual(out.shape, shape)
        self.assertTrue(np.isfinite(out).all())

    def test_numpy_path_smoke_1d_3d(self):
        rng = np.random.default_rng(1)
        f1 = np.ascontiguousarray(rng.standard_normal(64))
        out1 = _numpy_convolve(_centered_dx_kernel_1d(), f1)
        self.assertEqual(out1.shape, (64,))
        f3 = np.ascontiguousarray(rng.standard_normal((16, 16, 16)))
        out3 = _numpy_convolve(_laplacian_kernel_3d(), f3)
        self.assertEqual(out3.shape, (16, 16, 16))


class _DeviceConvolutionMixin:
    """Shared flat LBM-path checks for ctypes / OpenCL / Metal."""

    tol = 1e-12

    def _compare(self, kernel, field, tol=None):
        if tol is None:
            tol = self.tol
        ref = _numpy_convolve(kernel, field)
        shape = field.shape
        set_convolution_shape(shape, tenet=self.tenet)
        clear_active_tenet()

        k_dev = IdpyMemory.OnDevice(np.ascontiguousarray(kernel), tenet=self.tenet)
        f_flat = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.tenet)
        self.assertIs(_tenet_of(f_flat), self.tenet)

        # Ownership from src — no set_active_tenet
        out_flat = convolve_periodic(f_flat, *_taps(kernel), shape=shape)
        out = unpack_lattice_flat(out_flat.D2H(), shape)
        err = np.max(np.abs(out - ref))
        if err > tol:
            kern = K_ConvolvePeriodic(*_taps(kernel), shape)
            print(kern.dump_code(), file=sys.stderr)
        self.assertLessEqual(err, tol, msg=f"convolve_periodic max|Δ|={err}")

        st = JSymmetricTensor(
            d=field.ndim, rank=0, ranks=[0, 0],
            c_dict={0: k_dev},
            dtype=kernel.dtype,
        )
        fld = SymmetricTensor(
            d=field.ndim, rank=0, c_dict={0: f_flat}, dtype=field.dtype,
        )
        out_t = st @ fld
        out2 = unpack_lattice_flat(
            out_t[0].D2H() if hasattr(out_t[0], 'D2H') else out_t[0],
            shape,
        )
        err2 = np.max(np.abs(out2 - ref))
        if err2 > tol:
            kern = K_ConvolvePeriodic(*_taps(kernel), shape)
            print(kern.dump_code(), file=sys.stderr)
        self.assertLessEqual(err2, tol, msg=f"matmul device max|Δ|={err2}")

    def test_random_field_2d(self):
        rng = np.random.default_rng(1)
        field = np.ascontiguousarray(rng.standard_normal((32, 32)))
        self._compare(_centered_dx_kernel_2d(), field)

    def test_periodic_wrap_2d(self):
        field = np.zeros((32, 32), dtype=np.float64)
        field[0, 0] = 1.0
        field[-1, -1] = -1.0
        self._compare(_centered_dx_kernel_2d(), field)

    def test_1d_and_3d(self):
        rng = np.random.default_rng(3)
        self._compare(
            _centered_dx_kernel_1d(),
            np.ascontiguousarray(rng.standard_normal(64)),
        )
        self._compare(
            _laplacian_kernel_3d(),
            np.ascontiguousarray(rng.standard_normal((12, 12, 12))),
        )


@unittest.skipUnless(idpy_langs_sys.get(CTYPES_T, False), "ctypes not available")
class TestConvolutionCTypes(_DeviceConvolutionMixin, unittest.TestCase):
    def setUp(self):
        clear_idea_cache()
        clear_convolution_shape()
        clear_active_tenet()
        from idpy.CTypes.CTypes import CTypes
        self.tenet = CTypes().GetTenet()

    def tearDown(self):
        clear_active_tenet()
        clear_convolution_shape()
        clear_idea_cache()
        try:
            self.tenet.End()
        except Exception:
            pass


@unittest.skipUnless(idpy_langs_sys.get(OCL_T, False), "OpenCL not available")
class TestConvolutionOpenCL(_DeviceConvolutionMixin, unittest.TestCase):
    """fp32 only — Apple OpenCL often reports FP64=0."""
    tol = 5e-5

    def setUp(self):
        clear_idea_cache()
        clear_convolution_shape()
        clear_active_tenet()
        from idpy.OpenCL.OpenCL import OpenCL
        ocl = OpenCL()
        if 'gpu' in ocl.devices:
            ocl.SetDevice(kind='gpu', device=0)
        elif 'cpu' in ocl.devices:
            ocl.SetDevice(kind='cpu', device=0)
        else:
            self.skipTest("no OpenCL devices")
        try:
            self.tenet = ocl.GetTenet()
        except Exception as exc:
            self.skipTest(f"OpenCL tenet unavailable: {exc}")

    def tearDown(self):
        clear_active_tenet()
        clear_convolution_shape()
        clear_idea_cache()
        try:
            self.tenet.End()
        except Exception:
            pass

    def _compare(self, kernel, field, tol=None):
        field32 = np.ascontiguousarray(field, dtype=np.float32)
        kernel32 = np.ascontiguousarray(kernel, dtype=np.float32)
        ref = _numpy_convolve(
            kernel.astype(np.float64), field.astype(np.float64),
        ).astype(np.float32)
        shape = field32.shape
        set_convolution_shape(shape, tenet=self.tenet)
        clear_active_tenet()
        k_dev = IdpyMemory.OnDevice(kernel32, tenet=self.tenet)
        f_flat = IdpyMemory.OnDevice(pack_lattice_flat(field32), tenet=self.tenet)
        self.assertIs(_tenet_of(f_flat), self.tenet)
        out_flat = convolve_periodic(f_flat, *_taps(kernel32), shape=shape)
        out = unpack_lattice_flat(out_flat.D2H(), shape)
        err = np.max(np.abs(out - ref))
        self.assertLessEqual(err, self.tol, msg=f"convolve_periodic max|Δ|={err}")
        st = JSymmetricTensor(
            d=field32.ndim, rank=0, ranks=[0, 0],
            c_dict={0: k_dev}, dtype=kernel32.dtype,
        )
        fld = SymmetricTensor(
            d=field32.ndim, rank=0, c_dict={0: f_flat}, dtype=field32.dtype,
        )
        out_t = st @ fld
        out2 = unpack_lattice_flat(
            out_t[0].D2H() if hasattr(out_t[0], 'D2H') else out_t[0], shape,
        )
        err2 = np.max(np.abs(out2 - ref))
        self.assertLessEqual(err2, self.tol, msg=f"matmul device max|Δ|={err2}")


@unittest.skipUnless(idpy_langs_sys.get(METAL_T, False), "Metal not available")
class TestConvolutionMetal(_DeviceConvolutionMixin, unittest.TestCase):
    tol = 5e-5

    def setUp(self):
        clear_idea_cache()
        clear_convolution_shape()
        clear_active_tenet()
        from idpy.IdpyCode import GetTenet
        try:
            self.tenet = GetTenet({"lang": METAL_T, "device": 0, "cl_kind": "gpu"})
        except Exception as exc:
            self.skipTest(f"Metal tenet unavailable: {exc}")

    def tearDown(self):
        clear_active_tenet()
        clear_convolution_shape()
        clear_idea_cache()
        try:
            self.tenet.End()
        except Exception:
            pass

    def _compare(self, kernel, field, tol=None):
        field32 = np.ascontiguousarray(field, dtype=np.float32)
        kernel32 = np.ascontiguousarray(kernel, dtype=np.float32)
        ref = _numpy_convolve(
            kernel.astype(np.float64), field.astype(np.float64),
        ).astype(np.float32)
        shape = field32.shape
        set_convolution_shape(shape, tenet=self.tenet)
        clear_active_tenet()
        k_dev = IdpyMemory.OnDevice(kernel32, tenet=self.tenet)
        f_flat = IdpyMemory.OnDevice(pack_lattice_flat(field32), tenet=self.tenet)
        self.assertIs(_tenet_of(f_flat), self.tenet)
        out_flat = convolve_periodic(f_flat, *_taps(kernel32), shape=shape)
        out = unpack_lattice_flat(out_flat.D2H(), shape)
        err = np.max(np.abs(out - ref))
        self.assertLessEqual(err, self.tol, msg=f"convolve_periodic max|Δ|={err}")
        st = JSymmetricTensor(
            d=field32.ndim, rank=0, ranks=[0, 0],
            c_dict={0: k_dev}, dtype=kernel32.dtype,
        )
        fld = SymmetricTensor(
            d=field32.ndim, rank=0, c_dict={0: f_flat}, dtype=field32.dtype,
        )
        out_t = st @ fld
        out2 = unpack_lattice_flat(
            out_t[0].D2H() if hasattr(out_t[0], 'D2H') else out_t[0], shape,
        )
        err2 = np.max(np.abs(out2 - ref))
        self.assertLessEqual(err2, self.tol, msg=f"matmul device max|Δ|={err2}")


@unittest.skipUnless(idpy_langs_sys.get(CUDA_T, False), "CUDA not available")
class TestConvolutionCUDA(_DeviceConvolutionMixin, unittest.TestCase):
    def setUp(self):
        clear_idea_cache()
        clear_convolution_shape()
        clear_active_tenet()
        from idpy.CUDA.CUDA import CUDA
        try:
            cu = CUDA()
            if not cu.devices:
                self.skipTest("no CUDA devices")
            cu.SetDevice(0)
            self.tenet = cu.GetTenet()
        except Exception as exc:
            self.skipTest(f"CUDA tenet unavailable: {exc}")

    def tearDown(self):
        clear_active_tenet()
        clear_convolution_shape()
        clear_idea_cache()
        try:
            self.tenet.End()
        except Exception:
            pass


@unittest.skipUnless(idpy_langs_sys.get(CTYPES_T, False), "ctypes not available")
class TestConvolutionOwnership(unittest.TestCase):
    """LBM-style tenet ownership / cache isolation (ctypes)."""

    def setUp(self):
        clear_idea_cache()
        clear_convolution_shape()
        clear_active_tenet()
        from idpy.CTypes.CTypes import CTypes
        self.t0 = CTypes().GetTenet()
        self.t1 = CTypes().GetTenet()
        self.assertIsNot(self.t0, self.t1)

    def tearDown(self):
        clear_active_tenet()
        clear_convolution_shape()
        clear_idea_cache()
        for t in (self.t0, self.t1):
            try:
                t.End()
            except Exception:
                pass

    def test_mismatch_tenet_raises(self):
        shape = (16,)
        field = np.ones(shape, dtype=np.float64)
        f0 = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.t0)
        with self.assertRaises(ValueError):
            convolve_periodic(
                f0, *_taps(_centered_dx_kernel_1d()), shape=shape, tenet=self.t1,
            )

    def test_cache_keyed_by_tenet(self):
        shape = (16,)
        kernel = _centered_dx_kernel_1d()
        field = np.ones(shape, dtype=np.float64)
        taps = _taps(kernel)
        f0 = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.t0)
        f1 = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.t1)
        clear_idea_cache()
        convolve_periodic(f0, *taps, shape=shape)
        n_after_0 = len(_idea_cache)
        convolve_periodic(f1, *taps, shape=shape)
        self.assertEqual(len(_idea_cache), n_after_0 + 1)
        keys = list(_idea_cache.keys())
        self.assertNotEqual(keys[0][0], keys[1][0])

    def test_symbolic_without_active_tenet(self):
        shape = (32, 32)
        kernel = _centered_dx_kernel_2d()
        rng = np.random.default_rng(4)
        field = np.ascontiguousarray(rng.standard_normal(shape))
        set_convolution_shape(shape, tenet=self.t0)
        clear_active_tenet()
        k_dev = IdpyMemory.OnDevice(np.ascontiguousarray(kernel), tenet=self.t0)
        f_flat = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.t0)
        st = JSymmetricTensor(
            d=2, rank=0, ranks=[0, 0], c_dict={0: k_dev}, dtype=kernel.dtype,
        )
        fld = SymmetricTensor(
            d=2, rank=0, c_dict={0: f_flat}, dtype=field.dtype,
        )
        out = unpack_lattice_flat(np.asarray((st @ fld)[0].D2H()), shape)
        ref = _numpy_convolve(kernel, field)
        self.assertLessEqual(np.max(np.abs(out - ref)), 1e-12)


@unittest.skipUnless(idpy_langs_sys.get(CUDA_T, False), "CUDA not available")
class TestConvolutionOwnershipCUDA(unittest.TestCase):
    """LBM-style tenet ownership / cache isolation (CUDA)."""

    def setUp(self):
        clear_idea_cache()
        clear_convolution_shape()
        clear_active_tenet()
        from idpy.CUDA.CUDA import CUDA
        try:
            cu0, cu1 = CUDA(), CUDA()
            if not cu0.devices:
                self.skipTest("no CUDA devices")
            cu0.SetDevice(0)
            cu1.SetDevice(0)
            self.t0 = cu0.GetTenet()
            self.t1 = cu1.GetTenet()
        except Exception as exc:
            self.skipTest(f"CUDA tenet unavailable: {exc}")
        self.assertIsNot(self.t0, self.t1)

    def tearDown(self):
        clear_active_tenet()
        clear_convolution_shape()
        clear_idea_cache()
        for t in (getattr(self, 't0', None), getattr(self, 't1', None)):
            if t is None:
                continue
            try:
                t.End()
            except Exception:
                pass

    def test_ownership_stamp(self):
        field = np.ones(16, dtype=np.float64)
        arr = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.t0)
        self.assertIs(arr.tenet, self.t0)
        self.assertIs(_tenet_of(arr), self.t0)

    def test_mismatch_tenet_raises(self):
        shape = (16,)
        field = np.ones(shape, dtype=np.float64)
        f0 = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.t0)
        with self.assertRaises(ValueError):
            convolve_periodic(
                f0, *_taps(_centered_dx_kernel_1d()), shape=shape, tenet=self.t1,
            )

    def test_cache_keyed_by_tenet(self):
        shape = (16,)
        kernel = _centered_dx_kernel_1d()
        field = np.ones(shape, dtype=np.float64)
        taps = _taps(kernel)
        f0 = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.t0)
        f1 = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.t1)
        clear_idea_cache()
        convolve_periodic(f0, *taps, shape=shape)
        n_after_0 = len(_idea_cache)
        convolve_periodic(f1, *taps, shape=shape)
        self.assertEqual(len(_idea_cache), n_after_0 + 1)
        keys = list(_idea_cache.keys())
        self.assertNotEqual(keys[0][0], keys[1][0])

    def test_symbolic_without_active_tenet(self):
        shape = (32, 32)
        kernel = _centered_dx_kernel_2d()
        rng = np.random.default_rng(4)
        field = np.ascontiguousarray(rng.standard_normal(shape))
        set_convolution_shape(shape, tenet=self.t0)
        clear_active_tenet()
        k_dev = IdpyMemory.OnDevice(np.ascontiguousarray(kernel), tenet=self.t0)
        f_flat = IdpyMemory.OnDevice(pack_lattice_flat(field), tenet=self.t0)
        st = JSymmetricTensor(
            d=2, rank=0, ranks=[0, 0], c_dict={0: k_dev}, dtype=kernel.dtype,
        )
        fld = SymmetricTensor(
            d=2, rank=0, c_dict={0: f_flat}, dtype=field.dtype,
        )
        out = unpack_lattice_flat(np.asarray((st @ fld)[0].D2H()), shape)
        ref = _numpy_convolve(kernel, field)
        self.assertLessEqual(np.max(np.abs(out - ref)), 1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
