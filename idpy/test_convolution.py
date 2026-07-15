#!/usr/bin/env python3
"""Milestone 1: IdpyKernel periodic convolution vs numpy JSymmetricTensor path."""

import sys
import unittest
import numpy as np

from idpy.Utils.IdpySymbolic import SymmetricTensor, JSymmetricTensor
from idpy.IdpyCode import idpy_langs_sys
from idpy.IdpyCode import CTYPES_T, OCL_T
from idpy.IdpyCode import IdpyMemory
from idpy.IdpyStencils.IdpyConvolution import (
    K_ConvolvePeriodic,
    convolve_periodic,
    set_active_tenet,
    clear_active_tenet,
    clear_idea_cache,
    set_convolution_shape,
    clear_convolution_shape,
)


def _centered_dx_kernel_2d():
    """5-point 1D derivative [1/12, -2/3, 0, 2/3, -1/12] along axis 0, as 5x1 pad to 5x5 center."""
    c = np.zeros((5, 5), dtype=np.float64)
    # taps along first axis at center column j=2
    vals = np.array([1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0])
    c[:, 2] = vals
    return c


def _numpy_convolve(kernel, field):
    st = JSymmetricTensor(
        d=2, rank=0, ranks=[0, 0],
        c_dict={0: np.array(kernel, copy=True)},
        dtype=kernel.dtype,
    )
    fld = SymmetricTensor(
        d=2, rank=0,
        c_dict={0: np.array(field, copy=True)},
        dtype=field.dtype,
    )
    return np.asarray((st @ fld)[0])


class TestConvolutionNumpy(unittest.TestCase):
    def test_numpy_path_smoke(self):
        rng = np.random.default_rng(0)
        shape = (32, 32)
        field = rng.standard_normal(shape)
        kernel = _centered_dx_kernel_2d()
        out = _numpy_convolve(kernel, field)
        self.assertEqual(out.shape, shape)
        self.assertTrue(np.isfinite(out).all())


@unittest.skipUnless(idpy_langs_sys.get(CTYPES_T, False), "ctypes not available")
class TestConvolutionCTypes(unittest.TestCase):
    def setUp(self):
        clear_idea_cache()
        clear_convolution_shape()
        from idpy.CTypes.CTypes import CTypes
        self.tenet = CTypes().GetTenet()
        set_active_tenet(self.tenet)

    def tearDown(self):
        clear_active_tenet()
        clear_convolution_shape()
        clear_idea_cache()
        try:
            self.tenet.End()
        except Exception:
            pass

    def _compare(self, kernel, field, tol=1e-12):
        ref = _numpy_convolve(kernel, field)
        shape = field.shape
        set_convolution_shape(shape)

        k_dev = IdpyMemory.OnDevice(np.asfortranarray(kernel), tenet=self.tenet)
        f_flat = IdpyMemory.OnDevice(
            np.asfortranarray(field).ravel(order='F'), tenet=self.tenet,
        )
        # Direct helper
        out_flat = convolve_periodic(
            f_flat,
            *self._taps(kernel),
            shape=shape,
            tenet=self.tenet,
        )
        out = np.asarray(out_flat.D2H()).reshape(shape, order='F')
        err = np.max(np.abs(out - ref))
        if err > tol:
            kern = K_ConvolvePeriodic(*self._taps(kernel), shape)
            print(kern.dump_code(), file=sys.stderr)
        self.assertLessEqual(err, tol, msg=f"convolve_periodic max|Δ|={err}")

        # Through JSymmetricTensor @
        st = JSymmetricTensor(
            d=2, rank=0, ranks=[0, 0],
            c_dict={0: k_dev},
            dtype=kernel.dtype,
        )
        # field as multi-d device array
        f_dev = IdpyMemory.OnDevice(np.asfortranarray(field), tenet=self.tenet)
        fld = SymmetricTensor(d=2, rank=0, c_dict={0: f_dev}, dtype=field.dtype)
        out_t = st @ fld
        out2 = np.asarray(out_t[0].D2H() if hasattr(out_t[0], 'D2H') else out_t[0])
        err2 = np.max(np.abs(out2 - ref))
        if err2 > tol:
            kern = K_ConvolvePeriodic(*self._taps(kernel), shape)
            print(kern.dump_code(), file=sys.stderr)
        self.assertLessEqual(err2, tol, msg=f"matmul device max|Δ|={err2}")

    @staticmethod
    def _taps(kernel):
        center = tuple(s // 2 for s in kernel.shape)
        nz = np.nonzero(kernel)
        offsets = np.array(list(zip(*nz)), dtype=np.int64) - np.array(center)
        coeffs = kernel[nz]
        return offsets, coeffs

    def test_random_field(self):
        rng = np.random.default_rng(1)
        field = rng.standard_normal((32, 32))
        self._compare(_centered_dx_kernel_2d(), field)

    def test_periodic_wrap(self):
        field = np.zeros((32, 32), dtype=np.float64)
        field[0, 0] = 1.0
        field[-1, -1] = -1.0
        self._compare(_centered_dx_kernel_2d(), field)


@unittest.skipUnless(idpy_langs_sys.get(OCL_T, False), "OpenCL not available")
class TestConvolutionOpenCL(unittest.TestCase):
    def setUp(self):
        clear_idea_cache()
        clear_convolution_shape()
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
        set_active_tenet(self.tenet)

    def tearDown(self):
        clear_active_tenet()
        clear_convolution_shape()
        clear_idea_cache()
        try:
            self.tenet.End()
        except Exception:
            pass

    def test_fp32_vs_numpy(self):
        rng = np.random.default_rng(2)
        shape = (32, 32)
        field64 = rng.standard_normal(shape)
        kernel64 = _centered_dx_kernel_2d()
        ref = _numpy_convolve(kernel64, field64).astype(np.float32)

        field = field64.astype(np.float32)
        kernel = kernel64.astype(np.float32)
        set_convolution_shape(shape)

        center = tuple(s // 2 for s in kernel.shape)
        nz = np.nonzero(kernel)
        offsets = np.array(list(zip(*nz)), dtype=np.int64) - np.array(center)
        coeffs = kernel[nz]

        f_flat = IdpyMemory.OnDevice(
            np.asfortranarray(field).ravel(order='F'), tenet=self.tenet,
        )
        try:
            out_flat = convolve_periodic(
                f_flat, offsets, coeffs, shape=shape, tenet=self.tenet,
            )
        except Exception as exc:
            kern = K_ConvolvePeriodic(offsets, coeffs, shape)
            print(kern.dump_code(), file=sys.stderr)
            raise
        out = np.asarray(out_flat.D2H()).reshape(shape, order='F')
        err = np.max(np.abs(out - ref))
        # fp32 on GPU vs fp64 reference cast to fp32
        self.assertLessEqual(err, 5e-5, msg=f"OpenCL max|Δ|={err}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
