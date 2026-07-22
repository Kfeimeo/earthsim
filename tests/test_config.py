from sim.config import load_config


def test_blank_values_keep_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
time:
  dt:
  start: "2026-05-01T00:00:00"
""",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.time.dt == 240.0
    assert cfg.time.start == "2026-05-01T00:00:00"


def test_numeric_strings_are_coerced_to_default_type(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
physics:
  diff_T: "4.0e4"
""",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.physics.diff_T == 40000.0
    assert isinstance(cfg.physics.diff_T, float)


def test_legacy_physics_under_data_is_not_silently_ignored(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  init_mode: ideal
  H0: 4321.0
  visc: 12345.0
  drag: 7.0e-6
""",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.data.init_mode == "ideal"
    assert "H0" not in cfg.data
    assert cfg.physics.H0 == 4321.0
    assert cfg.physics.visc == 12345.0
    assert cfg.physics.drag_ocean_atmosphere == 7.0e-6
    assert cfg.physics.drag_land_atmosphere == 7.0e-6


def test_explicit_physics_values_win_over_legacy_data_layout(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  visc: 12345.0
physics:
  visc: 67890.0
""",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.physics.visc == 67890.0


def test_unknown_configuration_key_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
physics:
  viscc: 12345.0
""",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert "physics.viscc" in str(exc)
    else:
        raise AssertionError("unknown configuration key was silently accepted")
