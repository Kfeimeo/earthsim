from earthsim.config import load_config


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
