"""Sanity check the existing v2 base JSONs before kicking off the RunPod slice runs.
Confirms each base file is exactly what the merger will assume:
  - n_games == 200
  - config_idx exactly {0..9}
  - 20 reps per cell
  - 3 archetypes (single-{distributive,compatible,integrative})
"""
import json, sys
from collections import Counter
from pathlib import Path

EXPECTED_BASE_CFGS = set(range(10))
EXPECTED_BASE_ARCHETYPES = {"single-distributive", "single-compatible", "single-integrative"}
EXPECTED_BASE_NAMES = {
    "base_model", "grpo_self_only_560", "grpo_fair_only_560",
    "grpo_all_equal_560", "grpo_self_fair_equal_560",
    "lagrpo_self_only_760", "lagrpo_all_equal_760", "lagrpo_fair_only_760",
}

def check(p: Path):
    d = json.load(open(p, encoding="utf-8"))
    games = d.get("games", [])
    cfgs = Counter(g["config_idx"] for g in games)
    archs = set(g["archetype"] for g in games)
    n = len(games)

    issues = []
    if n != 200:
        issues.append(f"n_games={n} (expected 200)")
    if set(cfgs.keys()) != EXPECTED_BASE_CFGS:
        issues.append(f"config_idx={sorted(cfgs.keys())} (expected 0..9)")
    if archs != EXPECTED_BASE_ARCHETYPES:
        issues.append(f"archetypes={sorted(archs)} (expected single-{{distributive,compatible,integrative}})")
    if set(cfgs.values()) != {20}:
        issues.append(f"reps per cell = {sorted(set(cfgs.values()))} (expected [20])")

    status = "OK" if not issues else "FAIL"
    print(f"  [{status}] {p.name}: n={n} cells={len(cfgs)} archs={len(archs)}")
    for i in issues: print(f"        -> {i}")
    return not issues


def main():
    if len(sys.argv) != 2:
        print("usage: python verify_v2_bases.py <v2_dir>"); sys.exit(2)
    d = Path(sys.argv[1])
    bases = sorted(d.glob("*.json"))
    bases = [p for p in bases if "_samples" not in p.stem and "_cfg" not in p.stem
             and not p.stem.endswith("_full14")]

    found_names = {p.stem for p in bases}
    missing_names = EXPECTED_BASE_NAMES - found_names
    extra_names   = found_names - EXPECTED_BASE_NAMES

    print(f"Checking {len(bases)} base JSONs in {d} ...")
    if missing_names:
        print(f"  [WARN] missing base files: {sorted(missing_names)}")
    if extra_names:
        print(f"  [INFO] extra (non-panel) JSONs ignored: {sorted(extra_names)}")

    all_ok = True
    for p in bases:
        if p.stem in EXPECTED_BASE_NAMES:
            all_ok &= check(p)
    print()
    print("ALL OK" if (all_ok and not missing_names) else "BASES NOT READY")
    sys.exit(0 if (all_ok and not missing_names) else 1)


if __name__ == "__main__":
    main()
