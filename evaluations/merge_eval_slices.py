"""
Merge a base run_negotiation_eval.py JSON (the original 10-sample eval, configs 0..9)
with one or more slice files (e.g. _cfg10-15, _cfg16-21, _cfg22-27) into a single
combined JSON whose `metrics` block is recomputed over the union of `games`.

Usage:
    # Single base + arbitrary number of explicit extras:
    python evaluations/merge_eval_slices.py \\
        --base  evaluations/results/negotiation/v2/lagrpo_fair_only_760.json \\
        --extras evaluations/results/negotiation/v2/lagrpo_fair_only_760_cfg10-15.json \\
                 evaluations/results/negotiation/v2/lagrpo_fair_only_760_cfg16-21.json \\
                 evaluations/results/negotiation/v2/lagrpo_fair_only_760_cfg22-27.json \\
        --out   evaluations/results/negotiation/v2/lagrpo_fair_only_760_full14.json

    # Batch mode: for every <name>.json in DIR, auto-discover every
    # <name>_samples<a>-<b>.json (or legacy <name>_cfg<a>-<b>.json) slice file
    # and merge them all:
    python evaluations/merge_eval_slices.py --batch evaluations/results/negotiation/v2

The batch mode is the standard workflow after running run_v2_extra_gpu1.sh,
run_v2_extra_gpu2.sh, run_v2_extra_gpu3.sh -- it will merge all three slices
(samples 10-15, 16-21, 22-27) plus the original base file (samples 0-9) into
one *_full14.json per model, covering all 14 scenarios in both roles
(28 sample cells, 560 games).
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

SLICE_RE = re.compile(r"_(?:samples|cfg)\d+-\d+$")


def _agreed(g):
    a = g.get("agreed")
    return (a is True) or (str(a).lower() == "true")


def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return 0.0


def recompute_metrics(games, args):
    n = len(games)
    if n == 0:
        return {}
    U_A = [_f(g["U_A"]) for g in games]
    U_B = [_f(g["U_B"]) for g in games]
    rs  = [_f(g["ratio_self"])    for g in games]
    rw  = [_f(g["ratio_welfare"]) for g in games]
    rn  = [_f(g["ratio_nash"])    for g in games]
    rc  = [_f(g["ratio_rcoop"])   for g in games]
    agr = [int(_agreed(g)) for g in games]

    def m(xs): return sum(xs) / len(xs)
    def s(xs):
        mu = m(xs); return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5

    metrics = {
        "n_games": n,
        "n_samples": len({g.get("config_idx") for g in games}),
        "n_underlying_configs": len({g.get("config_idx", -1) // 2 for g in games}),
        "repetitions": args.get("repetitions"),
        "agreement_rate": m(agr),
        "U_A_mean": m(U_A), "U_A_std": s(U_A),
        "U_B_mean": m(U_B), "U_B_std": s(U_B),
        "social_welfare_mean": m([a + b for a, b in zip(U_A, U_B)]),
        "ratio_self_mean": m(rs),
        "ratio_welfare_mean": m(rw),
        "ratio_nash_mean": m(rn),
        "ratio_rcoop_mean": m(rc),
    }

    a_idx = [i for i, x in enumerate(agr) if x]
    if a_idx:
        metrics.update({
            "agreed_ratio_self_mean":    m([rs[i] for i in a_idx]),
            "agreed_ratio_welfare_mean": m([rw[i] for i in a_idx]),
            "agreed_ratio_nash_mean":    m([rn[i] for i in a_idx]),
            "agreed_ratio_rcoop_mean":   m([rc[i] for i in a_idx]),
        })

    by_arch = defaultdict(list)
    for g in games:
        by_arch[g.get("archetype", "unknown")].append(g)
    per_arch = {}
    for arch, gs in by_arch.items():
        u_a = [_f(g["U_A"]) for g in gs]
        u_b = [_f(g["U_B"]) for g in gs]
        per_arch[arch] = {
            "count": len(gs),
            "U_A_mean": m(u_a), "U_B_mean": m(u_b),
            "agreement_rate": m([int(_agreed(g)) for g in gs]),
            "ratio_self_mean":    m([_f(g["ratio_self"])    for g in gs]),
            "ratio_welfare_mean": m([_f(g["ratio_welfare"]) for g in gs]),
            "ratio_nash_mean":    m([_f(g["ratio_nash"])    for g in gs]),
            "ratio_rcoop_mean":   m([_f(g["ratio_rcoop"])   for g in gs]),
        }
    metrics["per_archetype"] = per_arch
    return metrics


def merge(base_path, extra_paths, out_path):
    base = json.load(open(base_path, encoding="utf-8"))
    games = list(base["games"])
    seen_cfgs = {g.get("config_idx") for g in games}

    used_extras = []
    for ep in extra_paths:
        ej = json.load(open(ep, encoding="utf-8"))
        new_cfgs = {g.get("config_idx") for g in ej["games"]}
        overlap = new_cfgs & seen_cfgs
        if overlap:
            print(f"  [WARN] {Path(ep).name}: overlapping config_idx {sorted(overlap)} -- "
                  f"games appended anyway, will double-count if intentional.")
        games.extend(ej["games"])
        seen_cfgs |= new_cfgs
        used_extras.append(str(ep))

    args = dict(base.get("args", {}))
    args["merged_from"] = [str(base_path)] + used_extras
    args["merged_config_indices"] = sorted({g.get("config_idx") for g in games})
    metrics = recompute_metrics(games, args)
    out = {"args": args, "metrics": metrics, "games": games}
    Path(out_path).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    n_arch = len(metrics.get("per_archetype", {}))
    print(f"  [OK]  -> {out_path}")
    print(f"        n_games={metrics['n_games']}  n_sample_cells={metrics['n_samples']}  "
          f"archetypes={n_arch}  agreement={metrics['agreement_rate']:.3f}")


def is_slice_file(p: Path) -> bool:
    return SLICE_RE.search(p.stem) is not None


def base_stem_for(p: Path) -> str:
    return SLICE_RE.sub("", p.stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",   help="Base JSON (e.g. configs 0..9)")
    ap.add_argument("--extras", nargs="*", default=[],
                    help="One or more slice JSONs (e.g. _cfg10-15, _cfg16-21, _cfg22-27)")
    ap.add_argument("--out",    help="Output JSON path")
    ap.add_argument("--batch",  help="Directory: auto-merge every <name>.json with all "
                                     "matching <name>_cfg<a>-<b>.json slice files.")
    ap.add_argument("--out-suffix", default="_full14",
                    help="Suffix for merged output filename (default _full14).")
    args = ap.parse_args()

    if args.batch:
        d = Path(args.batch)
        all_jsons = sorted(d.glob("*.json"))
        # Group slice files by their base stem
        slices_by_base = defaultdict(list)
        for p in all_jsons:
            if is_slice_file(p):
                slices_by_base[base_stem_for(p)].append(p)
        # For each potential base file, merge with its slices
        for p in all_jsons:
            stem = p.stem
            if is_slice_file(p):
                continue
            if stem.endswith(args.out_suffix):
                continue
            slices = sorted(slices_by_base.get(stem, []))
            if not slices:
                print(f"  [skip] no slices for {stem}")
                continue
            print(f"\n[merge] {stem}: base + {len(slices)} slice(s)")
            for s in slices:
                print(f"        + {s.name}")
            out = d / f"{stem}{args.out_suffix}.json"
            merge(p, slices, out)
    else:
        if not (args.base and args.extras and args.out):
            ap.error("Pass --base, --extras (one or more), --out (or --batch DIR)")
        merge(args.base, args.extras, args.out)


if __name__ == "__main__":
    main()
