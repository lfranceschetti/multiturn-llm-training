"""Verify that every <name>_full14.json in a directory covers all 28 sample cells
(14 configs × 2 roles), all 7 archetypes, and the expected number of games."""
import argparse, json, sys
from collections import Counter
from pathlib import Path

EXPECTED_SAMPLE_IDX = set(range(28))
EXPECTED_ARCHETYPES = {
    "single-distributive", "single-compatible", "single-integrative",
    "non-integrative distributive", "non-integrative compatible",
    "integrative distributive", "integrative compatible",
}


def check(p: Path):
    d = json.load(open(p, encoding="utf-8"))
    games = d["games"]
    cfgs = Counter(g["config_idx"] for g in games)
    archs = set(g["archetype"] for g in games)
    n = len(games)
    reps = d["args"].get("repetitions") or 20
    expected_n = len(EXPECTED_SAMPLE_IDX) * reps  # 28 * 20 = 560

    missing_cells = EXPECTED_SAMPLE_IDX - set(cfgs.keys())
    extra_cells   = set(cfgs.keys()) - EXPECTED_SAMPLE_IDX
    missing_arch  = EXPECTED_ARCHETYPES - archs
    extra_arch    = archs - EXPECTED_ARCHETYPES
    rep_counts    = sorted(set(cfgs.values()))

    ok = (n == expected_n
          and not missing_cells
          and not extra_cells
          and not missing_arch
          and rep_counts == [reps])

    print(f"\n=== {p.name} ===")
    print(f"  games={n}  expected={expected_n}  reps_field={reps}")
    print(f"  sample cells covered: {len(cfgs)}/28  reps per cell: {rep_counts}")
    print(f"  archetypes covered: {len(archs)}/7")
    if missing_cells: print(f"  [MISSING CELLS] {sorted(missing_cells)}")
    if extra_cells:   print(f"  [EXTRA CELLS]   {sorted(extra_cells)}")
    if missing_arch:  print(f"  [MISSING ARCH]  {sorted(missing_arch)}")
    if extra_arch:    print(f"  [EXTRA ARCH]    {sorted(extra_arch)}")
    print(f"  agreement_rate = {d['metrics']['agreement_rate']:.3f}")
    print(f"  status: {'OK' if ok else 'INCOMPLETE'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Directory containing *_full14.json files to check")
    ap.add_argument("--suffix", default="_full14", help="Suffix marker for merged files")
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob(f"*{args.suffix}.json"))
    if not files:
        print(f"No *{args.suffix}.json files in {d}", file=sys.stderr)
        sys.exit(2)
    all_ok = True
    for f in files:
        all_ok &= check(f)
    print()
    print("ALL OK" if all_ok else "SOME FILES INCOMPLETE")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
