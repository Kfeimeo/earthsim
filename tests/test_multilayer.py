import unittest

import numpy as np

from sim.config import load_config
from sim.model import EarthModel


def small_config():
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["grid"].update(nlat=12, nlon=24, topo_file="")
    cfg["grid"]["topo_files"] = []
    cfg["time"]["dt"] = 60.0
    cfg["data"]["init_mode"] = "ideal"
    return cfg


class MultilayerModelTests(unittest.TestCase):
    def test_layer_initialization_and_step(self):
        model = EarthModel(small_config())
        self.assertEqual(model.T_layers.shape, (5, 12, 24))
        self.assertTrue(np.all(np.diff(model.levels_m) > 0))
        self.assertLess(float(model.T_layers[-1].mean()),
                        float(model.T_layers[0].mean()))

        model.step(10)
        model.check_health()
        self.assertTrue(np.isfinite(model.w_layers).all())
        self.assertLessEqual(float(np.abs(model.w_layers).max()),
                             float(model.cfg.physics.vertical.w_max))
        self.assertTrue(np.shares_memory(model.u, model.u_layers))

    def test_ideal_initial_wind_prior(self):
        cfg = small_config()
        model = EarthModel(cfg)
        self.assertEqual(model.u_layers.shape, (5, 12, 24))
        self.assertGreater(float(np.abs(model.u_layers).max()), 1.0)
        self.assertGreater(float(np.abs(model.v_layers).max()), 0.1)

        cfg["physics"]["ideal_wind_enabled"] = False
        calm = EarthModel(cfg)
        self.assertEqual(float(np.abs(calm.u_layers).max()), 0.0)
        self.assertEqual(float(np.abs(calm.v_layers).max()), 0.0)

    def test_terrain_adds_low_level_drag(self):
        model = EarthModel(small_config())
        wind = np.full((model.nlat, model.nlon), 10.0, np.float32)
        calm = np.zeros_like(wind)
        au, _ = model._terrain_acceleration(wind, calm, 0)
        land = np.asarray(model.land) > 0.5
        ocean = ~land
        self.assertGreater(float(np.abs(au[land]).mean()),
                           float(np.abs(au[ocean]).mean()))

        # A purely cross-slope wind is both blocked and turned along contours.
        model.land[...] = 1
        model.elev[...] = 2000
        model.terrain_slope_x[...] = 0.02
        model.terrain_slope_y[...] = 0
        model.terrain_slope[...] = 0.02
        au, av = model._terrain_acceleration(wind, calm, 0)
        self.assertLess(float(au.mean()), -float(model.cfg.physics.drag) * 10)
        self.assertGreater(float(np.abs(av).mean()), 0)

    def test_air_temperature_edit_updates_lowest_layer(self):
        model = EarthModel(small_config())
        before = model.T_layers[0].copy()
        model.apply_temp_edit(float(model.lats[6]), 0, radius_km=500,
                              delta=3, target="air")
        self.assertGreater(float(np.max(model.T_layers[0] - before)), 2.5)
        self.assertTrue(np.shares_memory(model.Ta, model.T_layers))

    def test_wind_region_edits_update_selected_layer(self):
        model = EarthModel(small_config())
        lat = float(model.lats[6])
        layer = 1
        model.u_layers[layer][6, 0] = 20.0
        model.v_layers[layer][6, 0] = 10.0

        model.apply_wind_zero_edit(lat, 0, radius_km=700, layer=layer)

        self.assertLess(float(np.hypot(model.u_layers[layer][6, 0],
                                       model.v_layers[layer][6, 0])), 5.0)
        surface_before = model.u_layers[0].copy()

        model.apply_cyclone_edit(lat, 0, radius_km=900,
                                 strength_ms=35, layer=layer)

        self.assertGreater(float(np.hypot(model.u_layers[layer],
                                          model.v_layers[layer]).max()), 5.0)
        np.testing.assert_allclose(model.u_layers[0], surface_before)

    def test_vertical_transport_conserves_layer_integral(self):
        model = EarthModel(small_config())
        field = np.linspace(260, 300, model.nz, dtype=np.float32)[:, None, None]
        field = field + np.zeros_like(model.T_layers)
        w = np.zeros_like(field)
        w[1:-1] = 0.02
        before = np.sum(field * model.layer_dz_m[:, None, None], axis=0)

        after = model._vertical_transport(field, w, diffusivity=4.0, dt=30.0)
        after_sum = np.sum(after * model.layer_dz_m[:, None, None], axis=0)

        np.testing.assert_allclose(after_sum, before, rtol=2e-6, atol=1e-3)

    def test_cloud_cover_dissipates_in_dry_air(self):
        model = EarthModel(small_config())
        model.cloud[...] = 1.0
        model.precip[...] = 0.0
        zero = np.zeros_like(model.cloud)
        dry_rh = np.full_like(model.cloud, 0.20)
        moist_rh = np.full_like(model.cloud, 0.90)

        dry_cloud = model._update_cloud_cover(zero, dry_rh, zero)
        moist_cloud = model._update_cloud_cover(zero, moist_rh, zero)

        self.assertLess(float(dry_cloud.mean()), float(moist_cloud.mean()))
        self.assertLess(float(moist_cloud.mean()), 1.0)

    def test_ground_water_exchanges_rain_and_evaporation_conservatively(self):
        cfg = small_config()
        cfg["physics"].update(
            ground_water_capacity_mm=100.0,
            initial_ground_water_mm=10.0,
            ground_runoff_tau=1.0e30,
        )
        model = EarthModel(cfg)
        model.land[...] = 1.0
        model.ocean[...] = 0.0
        model.ground_water[...] = 10.0
        land_evap = np.full_like(model.ground_water, 0.001)
        rain = np.full_like(model.ground_water, 2.0)

        model._update_ground_water(land_evap, rain)

        np.testing.assert_allclose(model.ground_water, 11.94, atol=1e-6)

    def test_land_evaporation_cannot_exceed_ground_water(self):
        cfg = small_config()
        cfg["physics"].update(initial_ground_water_mm=0.01, land_evap=1.0)
        model = EarthModel(cfg)
        model.land[...] = 1.0
        model.ocean[...] = 0.0
        model.ground_water[...] = 0.01
        hot = np.full_like(model.Ts, 330.0)
        dry = np.zeros_like(model.q)
        wind = np.full_like(model.Ts, 100.0)

        total_evap, land_evap = model._surface_evaporation(hot, dry, wind)

        np.testing.assert_allclose(total_evap, land_evap)
        self.assertLessEqual(float(land_evap.max()), 0.01 / model.dt)

    def test_wetter_ground_has_faster_relative_runoff(self):
        cfg = small_config()
        cfg["physics"].update(
            ground_water_capacity_mm=100.0,
            ground_runoff_tau=600.0,
            ground_runoff_exponent=2.0,
        )
        model = EarthModel(cfg)
        model.land[...] = 1.0
        model.ocean[...] = 0.0
        zero = np.zeros_like(model.ground_water)

        model.ground_water[...] = 25.0
        model._update_ground_water(zero, zero)
        dry_fractional_loss = (25.0 - float(model.ground_water.mean())) / 25.0
        model.ground_water[...] = 100.0
        model._update_ground_water(zero, zero)
        wet_fractional_loss = (100.0 - float(model.ground_water.mean())) / 100.0

        self.assertGreater(wet_fractional_loss, dry_fractional_loss)

    def test_two_layer_ocean_exchanges_heat_conservatively(self):
        cfg = small_config()
        cfg["physics"]["ocean_layers"]["enabled"] = True
        cfg["physics"]["ocean_layers"].update(
            lower_depth_m=200.0, interlayer_heat_exchange=1.0e-5,
            interlayer_drag=0.0, deep_drag=0.0, deep_visc=0.0)
        cfg["physics"].update(tau_ocean=0.0, drag_ocean=0.0, visc_ocean=0.0)
        model = EarthModel(cfg)
        model.land[...] = 0.0
        model.ocean[...] = 1.0
        model.ice[...] = 0.0
        model.Ts[...] = 290.0
        model.To_deep[...] = 280.0
        zero = np.zeros_like(model.Ts)
        before = (model.ocean_upper_depth_m * model.Ts
                  + model.ocean_lower_depth_m * model.To_deep).mean()

        model.Ts = model._advance_ocean(model.Ts, zero, zero)

        after = (model.ocean_upper_depth_m * model.Ts
                 + model.ocean_lower_depth_m * model.To_deep).mean()
        self.assertLess(float(model.Ts.mean()), 290.0)
        self.assertGreater(float(model.To_deep.mean()), 280.0)
        np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-3)

    def test_rejects_unsorted_levels(self):
        cfg = small_config()
        cfg["physics"]["vertical"]["levels_m"] = [1000, 500]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            EarthModel(cfg)


if __name__ == "__main__":
    unittest.main()
