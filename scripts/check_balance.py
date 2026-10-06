"""Inspect startup balance, optionally step, and export the full profiles."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.config import load_config
from sim.model import EarthModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backend", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--nlat", type=int)
    parser.add_argument("--nlon", type=int)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--output", default="output/initial_balance.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backend:
        cfg["backend"] = args.backend
    for key in ("nlat", "nlon"):
        if getattr(args, key) is not None:
            cfg["grid"][key] = getattr(args, key)
    if cfg.physics.dynamics.core != "primitive_equations":
        parser.error("balance checks require primitive_equations")
    m = EarthModel(cfg)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"initialization_source": m.initialization_source,
              "latitude_deg": m.lats.tolist(), "initial_balance": m.initial_balance}
    # Save failed diagnostics too, before any formal step is attempted.
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not m.initial_balance["passed"]:
        raise SystemExit("initial balance failed; see " + str(path))
    # Persist every diagnostic sample so long runs do not lose their history.
    history_path = path.with_suffix(".jsonl")
    with history_path.open("w", encoding="utf-8") as history:
        for _ in range(args.steps):
            try:
                m.step()
            finally:
                sample = m.stability_diagnostics
                if sample and sample["step"] == m.step_count - 1:
                    history.write(json.dumps({"stability": sample,
                        "stabilization": m.stabilization_budget}) + "\n")
                    history.flush()
                elif sample and not sample["finite"]:
                    history.write(json.dumps({"stability": sample}) + "\n")
                    history.flush()
    m.check_health()
    report["steps"] = m.step_count
    report["stabilization_budget_total"] = m.stabilization_budget_total
    report["last_stability_diagnostics"] = m.stability_diagnostics
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Balance diagnostics written to {path}")


if __name__ == "__main__":
    main()
