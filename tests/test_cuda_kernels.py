import unittest

import numpy as np


class CudaKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            from sim import cuda_kernels

            if cp.cuda.runtime.getDeviceCount() == 0 or not cuda_kernels.load():
                raise unittest.SkipTest("CUDA kernel is unavailable")
            cls.cp = cp
            cls.kernels = cuda_kernels
        except Exception as exc:
            raise unittest.SkipTest(f"CUDA kernel is unavailable: {exc}") from exc

    def test_polar_filter_matches_numpy(self):
        rng = np.random.default_rng(42)
        weights = np.linspace(0, 1, 12, dtype=np.float32)
        passes = 6

        for shape in ((12, 24), (3, 12, 24)):
            with self.subTest(shape=shape):
                field = rng.normal(size=shape).astype(np.float32)
                smooth = field.copy()
                for _ in range(passes):
                    smooth = (0.25 * np.roll(smooth, 1, axis=-1)
                              + 0.5 * smooth
                              + 0.25 * np.roll(smooth, -1, axis=-1))
                weight_shape = (1,) * (len(shape) - 2) + (12, 1)
                expected = field + weights.reshape(weight_shape) * (smooth - field)

                actual = self.kernels.polar_filter(
                    self.cp.asarray(field), self.cp.asarray(weights), passes)
                np.testing.assert_allclose(self.cp.asnumpy(actual), expected,
                                           rtol=2e-6, atol=2e-6)

    def test_adv_diff_matches_numpy_for_2d_and_batched_fields(self):
        rng = np.random.default_rng(7)
        nlat, nlon = 19, 37
        invdx = np.linspace(1.0e-5, 3.0e-5, nlat, dtype=np.float32)
        invdy = np.float32(1.7e-5)
        diffusivity = np.float32(120.0)
        dt = np.float32(30.0)

        for shape in ((nlat, nlon), (3, nlat, nlon)):
            with self.subTest(shape=shape):
                field = rng.normal(280.0, 5.0, size=shape).astype(np.float32)
                u = rng.normal(0.0, 8.0, size=shape).astype(np.float32)
                v = rng.normal(0.0, 8.0, size=shape).astype(np.float32)
                scale_shape = (1,) * (len(shape) - 2) + (nlat, 1)
                idx = invdx.reshape(scale_shape)
                west = np.roll(field, 1, axis=-1)
                east = np.roll(field, -1, axis=-1)
                south = np.concatenate(
                    [field[..., :1, :], field[..., :-1, :]], axis=-2)
                north = np.concatenate(
                    [field[..., 1:, :], field[..., -1:, :]], axis=-2)
                dfdx = np.where(u > 0, (field - west) * idx,
                                (east - field) * idx)
                dfdy = np.where(v > 0, (field - south) * invdy,
                                (north - field) * invdy)
                lap = ((west + east - 2 * field) * idx * idx
                       + (north + south - 2 * field) * invdy * invdy)
                expected = field + dt * (
                    -u * dfdx - v * dfdy + diffusivity * lap)

                actual = self.kernels.adv_diff(
                    self.cp.asarray(field), self.cp.asarray(u),
                    self.cp.asarray(v), self.cp.asarray(invdx), invdy,
                    diffusivity, dt)
                np.testing.assert_allclose(self.cp.asnumpy(actual), expected,
                                           rtol=2e-6, atol=2e-5)

    def test_gradient_matches_numpy_for_2d_and_batched_fields(self):
        rng = np.random.default_rng(11)
        nlat, nlon = 19, 37
        invdx = np.linspace(1.0e-5, 3.0e-5, nlat, dtype=np.float32)
        invdy = np.float32(1.7e-5)

        for shape in ((nlat, nlon), (4, nlat, nlon)):
            with self.subTest(shape=shape):
                field = rng.normal(size=shape).astype(np.float32)
                scale_shape = (1,) * (len(shape) - 2) + (nlat, 1)
                idx = invdx.reshape(scale_shape)
                west = np.roll(field, 1, axis=-1)
                east = np.roll(field, -1, axis=-1)
                south = np.concatenate(
                    [field[..., :1, :], field[..., :-1, :]], axis=-2)
                north = np.concatenate(
                    [field[..., 1:, :], field[..., -1:, :]], axis=-2)
                expected_x = (east - west) * (np.float32(0.5) * idx)
                expected_y = (north - south) * (np.float32(0.5) * invdy)

                actual_x, actual_y = self.kernels.gradient(
                    self.cp.asarray(field), self.cp.asarray(invdx), invdy)
                np.testing.assert_allclose(self.cp.asnumpy(actual_x), expected_x,
                                           rtol=3e-6, atol=3e-7)
                np.testing.assert_allclose(self.cp.asnumpy(actual_y), expected_y,
                                           rtol=3e-6, atol=3e-7)

    def test_divergence_matches_numpy_for_2d_and_batched_fields(self):
        rng = np.random.default_rng(13)
        nlat, nlon = 19, 37
        invdx = np.linspace(1.0e-5, 3.0e-5, nlat, dtype=np.float32)
        invdy = np.float32(1.7e-5)
        lats = np.linspace(-89.0, 89.0, nlat, dtype=np.float32)
        coslat = np.cos(np.radians(lats)).astype(np.float32)
        invcoslat = (1.0 / np.maximum(coslat, np.float32(0.2))).astype(np.float32)

        for shape in ((nlat, nlon), (4, nlat, nlon)):
            with self.subTest(shape=shape):
                u = rng.normal(size=shape).astype(np.float32)
                v = rng.normal(size=shape).astype(np.float32)
                scale_shape = (1,) * (len(shape) - 2) + (nlat, 1)
                idx = invdx.reshape(scale_shape)
                cos = coslat.reshape(scale_shape)
                invcos = invcoslat.reshape(scale_shape)
                dudx = (np.roll(u, -1, axis=-1)
                        - np.roll(u, 1, axis=-1)) * (np.float32(0.5) * idx)
                vc = v * cos
                south = np.concatenate(
                    [vc[..., :1, :], vc[..., :-1, :]], axis=-2)
                north = np.concatenate(
                    [vc[..., 1:, :], vc[..., -1:, :]], axis=-2)
                expected = dudx + (north - south) * (
                    np.float32(0.5) * invdy) * invcos

                actual = self.kernels.divergence(
                    self.cp.asarray(u), self.cp.asarray(v),
                    self.cp.asarray(invdx), invdy,
                    self.cp.asarray(coslat), self.cp.asarray(invcoslat))
                np.testing.assert_allclose(self.cp.asnumpy(actual), expected,
                                           rtol=3e-6, atol=3e-7)


if __name__ == "__main__":
    unittest.main()
