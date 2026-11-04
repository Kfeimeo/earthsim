import unittest
import numpy as np
from sim.config import load_config
from sim.model import EarthModel
from sim.physics import RHO_W, CW


class SeaIceTests(unittest.TestCase):
    def model(self, core="primitive_equations"):
        cfg = load_config()
        cfg["backend"] = "cpu"
        cfg["physics"]["dynamics"]["core"] = core
        cfg["grid"].update(nlat=12, nlon=24, topo_file="", topo_files=[])
        cfg["data"]["init_mode"] = "ideal"
        cfg["time"]["dt"] = 60.0
        return EarthModel(cfg)

    def test_freeze_melt_energy_and_diagnostic_persistence(self):
        m = self.model()
        ocean = m.ocean > .5
        land = ~ocean
        freezing = float(np.float32(m.cfg.physics.sea_ice.freezing_temp_k))
        capacity = RHO_W * CW * m.cfg.physics.mld
        latent = m.cfg.physics.sea_ice.latent_heat_j_kg
        m.sea_ice_mass[:] = 0
        m.Ts[:] = freezing - 1
        before = capacity * m.Ts.astype(np.float64)
        m._diag_surface()
        after = capacity * m.Ts.astype(np.float64) - latent * m.sea_ice_mass
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-4)
        np.testing.assert_allclose(m.Ts[ocean], freezing)
        np.testing.assert_allclose(m.Ts[land], freezing - 1)
        np.testing.assert_allclose(m.sea_ice_mass[ocean], capacity / latent)
        self.assertTrue(np.all(m.ice[ocean] == 1))
        mass = m.sea_ice_mass.copy()
        m._diag_surface()
        np.testing.assert_array_equal(m.sea_ice_mass, mass)
        m.Ts[ocean] += .25
        m._diag_surface()
        np.testing.assert_allclose(m.sea_ice_mass[ocean], mass[ocean] * .75)
        np.testing.assert_allclose(m.Ts[ocean], freezing)
        m.Ts[ocean] += 1
        m._diag_surface()
        np.testing.assert_allclose(m.sea_ice_mass[ocean], 0, atol=1e-8)
        np.testing.assert_allclose(m.Ts[ocean], freezing + .25)

    def test_frame_roundtrip_preserves_frozen_mass(self):
        m = self.model()
        m.Ts[m.ocean > .5] = 270.2
        m._diag_surface()
        fields = m.fields_cpu(include_layers=True)
        mass = m.sea_ice_mass.copy()
        m.sea_ice_mass = np.zeros_like(m.sea_ice_mass)
        m.load_frame_state(fields, m.t)
        np.testing.assert_allclose(m.sea_ice_mass, mass, rtol=1e-6)
        m.step(2)
        m.check_health()

    def test_both_dynamical_cores_accumulate_ice(self):
        for core in ("primitive_equations", "shallow_water"):
            m = self.model(core)
            m.Ts[m.ocean > .5] = 270.2
            m.step()
            self.assertGreater(float(m.sea_ice_mass.max()), 0)
            self.assertGreater(float(m.ice.max()), .1)
            self.assertTrue(np.isfinite(m.Ts).all())

    def test_invalid_mass_scale_rejected(self):
        m = self.model()
        m.cfg["physics"]["sea_ice"]["full_cover_mass_kg_m2"] = 0
        with self.assertRaises(ValueError):
            m._diag_surface()

    def test_sustained_cooling_accumulates_and_legacy_mode_is_available(self):
        m = self.model()
        ocean = m.ocean > .5
        freezing = np.float32(m.cfg.physics.sea_ice.freezing_temp_k)
        m.Ts[ocean] = freezing
        m.sea_ice_mass[:] = 0
        capacity = RHO_W * CW * m.cfg.physics.mld
        latent = m.cfg.physics.sea_ice.latent_heat_j_kg
        for _ in range(8):
            m.Ts[ocean] -= .03125
            m._diag_surface()
        np.testing.assert_allclose(m.sea_ice_mass[ocean], .25 * capacity / latent)
        self.assertTrue(np.all(m.ice[ocean] == 1))
        m.cfg["physics"]["sea_ice"]["enabled"] = False
        m.Ts[ocean] = 271.2
        m._diag_surface()
        np.testing.assert_allclose(m.ice[ocean], .1, atol=2e-5)
