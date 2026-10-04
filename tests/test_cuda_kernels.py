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

    def test_mass_transport_matches_numpy(self):
        from sim.physics import Ops
        from sim.primitive import mass_consistent_transport

        rng = np.random.default_rng(17)
        nz, nlat, nlon = 4, 19, 37
        lats = np.linspace(-89.0, 89.0, nlat, dtype=np.float32)
        ops = Ops(np, lats, nlon, cos_clamp=0.2)
        shape = (nz, nlat, nlon)
        field = rng.normal(280.0, 5.0, size=shape).astype(np.float32)
        u = rng.normal(0.0, 8.0, size=shape).astype(np.float32)
        v = rng.normal(0.0, 8.0, size=shape).astype(np.float32)
        mu_old = rng.uniform(75000.0, 95000.0, size=(nlat, nlon)).astype(
            np.float32)
        mu_new = (mu_old + rng.normal(0.0, 20.0, size=(nlat, nlon))).astype(
            np.float32)
        sigma_thickness = np.array([0.15, 0.20, 0.25, 0.40], np.float32)
        dp_old = (sigma_thickness[:, None, None] * mu_old[None]).astype(
            np.float32)
        dp_new = (sigma_thickness[:, None, None] * mu_new[None]).astype(
            np.float32)
        div_mass = rng.normal(0.0, 2.0e-3, size=shape).astype(np.float32)
        interface_flux = np.zeros((nz + 1, nlat, nlon), np.float32)
        interface_flux[1:-1] = rng.normal(
            0.0, 0.02, size=(nz - 1, nlat, nlon)).astype(np.float32)
        diffusivity = 120.0
        dt = 20.0

        expected = mass_consistent_transport(
            np, ops, field, u, v, dp_old, dp_new,
            div_mass, interface_flux, diffusivity, dt)
        actual = self.kernels.mass_transport(
            self.cp.asarray(field), self.cp.asarray(u), self.cp.asarray(v),
            self.cp.asarray(dp_old), self.cp.asarray(dp_new),
            self.cp.asarray(div_mass), self.cp.asarray(interface_flux),
            self.cp.asarray(ops.invdx[:, 0]), ops.invdy,
            diffusivity, dt)
        np.testing.assert_allclose(
            self.cp.asnumpy(actual), expected, rtol=4e-6, atol=4e-5)

    def test_muscl_transport_matches_numpy(self):
        from sim.physics import Ops
        from sim.primitive import mass_consistent_transport

        rng = np.random.default_rng(23)
        nz, nlat, nlon = 3, 17, 35
        lats = np.linspace(-85.0, 85.0, nlat, dtype=np.float32)
        ops = Ops(
            np, lats, nlon, cos_clamp=0.2,
            advection_scheme="muscl_tvd", advection_limiter="mc")
        shape = (nz, nlat, nlon)
        field = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        u = rng.normal(0.0, 8.0, size=shape).astype(np.float32)
        v = rng.normal(0.0, 8.0, size=shape).astype(np.float32)
        # Hybrid layers: a pure-pressure top layer above two sigma layers.
        ps = rng.uniform(75000.0, 95000.0, size=(nlat, nlon)).astype(
            np.float32)
        ps_new = (ps + rng.normal(0.0, 10.0, size=(nlat, nlon))).astype(
            np.float32)
        d_hyai = np.array([0.0, 6000.0, 14000.0], np.float32)[:, None, None]
        d_hybi = np.array([0.3, 0.5, 0.0], np.float32)[:, None, None]
        dp_old = (d_hyai + d_hybi * ps[None]).astype(np.float32)
        dp_new = (d_hyai + d_hybi * ps_new[None]).astype(np.float32)
        mass_u = dp_old * u
        mass_v = dp_old * v
        div_mass = ops.finite_volume_divergence(mass_u, mass_v)
        interface_flux = np.zeros((nz + 1, nlat, nlon), np.float32)
        interface_flux[1:-1] = rng.normal(
            0.0, 0.01, size=(nz - 1, nlat, nlon)).astype(np.float32)

        expected = mass_consistent_transport(
            np, ops, field, u, v, dp_old, dp_new, div_mass,
            interface_flux, 0.0, 10.0)
        actual = self.kernels.mass_transport(
            self.cp.asarray(field), self.cp.asarray(u), self.cp.asarray(v),
            self.cp.asarray(dp_old), self.cp.asarray(dp_new),
            self.cp.asarray(div_mass), self.cp.asarray(interface_flux),
            self.cp.asarray(ops.invdx[:, 0]), ops.invdy, 0.0, 10.0,
            scheme="muscl_tvd", limiter="mc",
            coslat=self.cp.asarray(ops.coslat[:, 0]),
            invcoslat=self.cp.asarray(ops.invcoslat[:, 0]))
        np.testing.assert_allclose(
            self.cp.asnumpy(actual), expected, rtol=8e-6, atol=8e-6)

    def test_muscl_adv_diff_matches_numpy(self):
        from sim.physics import Ops

        rng = np.random.default_rng(29)
        nlat, nlon = 17, 35
        lats = np.linspace(-85.0, 85.0, nlat, dtype=np.float32)
        ops = Ops(
            np, lats, nlon, cos_clamp=0.2,
            advection_scheme="muscl_tvd", advection_limiter="vanleer")
        field = rng.normal(size=(2, nlat, nlon)).astype(np.float32)
        u = rng.normal(0.0, 7.0, size=field.shape).astype(np.float32)
        v = rng.normal(0.0, 7.0, size=field.shape).astype(np.float32)
        expected = ops.adv_diff_step(field, u, v, 50.0, 12.0)
        actual = self.kernels.adv_diff(
            self.cp.asarray(field), self.cp.asarray(u), self.cp.asarray(v),
            self.cp.asarray(ops.invdx[:, 0]), ops.invdy, 50.0, 12.0,
            scheme="muscl_tvd", limiter="vanleer",
            coslat=self.cp.asarray(ops.coslat[:, 0]),
            invcoslat=self.cp.asarray(ops.invcoslat[:, 0]))
        np.testing.assert_allclose(
            self.cp.asnumpy(actual), expected, rtol=8e-6, atol=8e-6)

    def test_hydrostatic_column_matches_numpy(self):
        from sim.physics import RD
        from sim.primitive import hydrostatic_state

        rng = np.random.default_rng(19)
        nz, nlat, nlon = 4, 9, 17
        shape = (nz, nlat, nlon)
        temperature = rng.uniform(220.0, 300.0, size=shape).astype(np.float32)
        humidity = rng.uniform(0.0, 0.02, size=shape).astype(np.float32)
        surface_pressure = rng.uniform(
            75000.0, 103000.0, size=(nlat, nlon)).astype(np.float32)
        surface_geopotential = rng.uniform(
            0.0, 30000.0, size=(nlat, nlon)).astype(np.float32)
        hyai = np.array([0.0, 2800.0, 6500.0, 9000.0, 10000.0], np.float32)
        hybi = np.array([1.0, 0.72, 0.42, 0.11, 0.0], np.float32)

        expected = hydrostatic_state(
            np, temperature, humidity, surface_pressure, hyai, hybi,
            surface_geopotential)
        actual = self.kernels.hydrostatic_state(
            self.cp.asarray(temperature), self.cp.asarray(humidity),
            self.cp.asarray(surface_pressure), self.cp.asarray(hyai),
            self.cp.asarray(hybi), self.cp.asarray(surface_geopotential), RD)
        for actual_field, expected_field in zip(actual, expected):
            np.testing.assert_allclose(
                self.cp.asnumpy(actual_field), expected_field,
                rtol=8e-6, atol=8e-2)


if __name__ == "__main__":
    unittest.main()
