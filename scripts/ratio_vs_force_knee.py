#!/usr/bin/env python3
"""
ratio_vs_force_knee.py

Reads nn_force_distance_v2.csv (columns: mol_id, calc, atom_i, atom_j,
elem_i, elem_j, r_ij_A, f_rel, f_common, pair_key, is_mutual_nn) and studies
the relationship between the dimensionless covalent-radius-ratio

    x = r_ij_A / r0(elem_i, elem_j)

(r0 = sum of covalent radii for the pair, same Cordero-style table and
0.75 A fallback as crowding_detector.py) and the two-body radial force
component f_rel.

Since a mutual nearest-neighbor pair (is_mutual_nn == True) is written as
TWO rows (once from each atom's perspective, same r_ij_A/f_rel, opposite
f_common), each *physical contact* is deduplicated to a single row via
pair_key (mol_id, calc, pair_key) before any of the statistics/fit/plot
below are computed -- otherwise mutual pairs would be double-counted
relative to one-sided (non-mutual) nearest-neighbor contacts.

Outputs:
  1. Summary statistics of x and |f_rel|, binned by x, printed to stdout.
  2. An estimated "knee" point in the log(x) vs log(|f_rel|) relationship,
     found via a plain numpy grid search over candidate breakpoints (no
     scipy dependency). The candidate grid is now a FIXED grid over a wide
     x-range (default 0.05 to 1.3), NOT a narrow 5-95% quantile clip of the
     observed x values -- the compressed/strained tail is only ~2.4% of all
     rows (867/35804 at r_ij<0.8A in the earlier CSV histogram), so a
     quantile-based grid would very likely exclude exactly the region the
     knee is expected to be in. The single-line (no-knee) baseline fit's
     SSE is also computed and reported alongside the two-segment fit's SSE,
     with the resulting SSE reduction fraction, so the "knee" result is
     falsifiable: if the two-segment fit barely beats the single line, that
     is reported plainly rather than asserting a knee exists regardless.
  3. A log-log scatter plot (x vs |f_rel|, colored/marked by is_mutual_nn)
     with a vertical line at the estimated knee x, saved as a PNG in the
     same directory as this script. If matplotlib is unavailable or the
     plot fails for any reason, this is caught and reported as a warning --
     the numeric stats/knee output above still print regardless.
  4. The estimated knee x value and a suggested midpoint score in [0, 1]
     (NOT rescaled into any other range), following the logarithmic/
     geometric framing discussed: s = (log(x_stable) - log(knee_x)) /
     (log(x_stable) - log(x_unstable)), clipped to [0, 1]. s=0 means the
     knee sits at (or above) the "fully stable" reference ratio; s=1 means
     it sits at (or below) the "fully unstable/fusion-like" reference
     ratio.

Usage:
    python ratio_vs_force_knee.py [--csv PATH] [--grid-x-min 0.05]
                                   [--grid-x-max 1.3] [--n-candidates 80]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = Path("analysis_output/2node_new/nn_force_distance_v2.csv")
DEFAULT_PNG = Path("ratio_vs_force_knee.png")

REQUIRED_COLUMNS = ["mol_id", "calc", "elem_i", "elem_j", "r_ij_A", "f_rel"]

# Cordero et al. (2008)-style single-bond covalent radii, Angstrom.
# Same table as crowding_detector.py; unknown elements fall back to 0.75 A.
COVALENT_RADII_A = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
}
DEFAULT_COVALENT_RADIUS_A = 0.75


def covalent_radius(elem: str) -> float:
    return COVALENT_RADII_A.get(elem, DEFAULT_COVALENT_RADIUS_A)


def r0_for_pair(elem_i: str, elem_j: str) -> float:
    return covalent_radius(elem_i) + covalent_radius(elem_j)


def load_and_dedup(csv_path: Path):
    """Load nn_force_distance_v2.csv, compute x = r_ij_A / r0, and dedup
    mutual-NN pairs via (mol_id, calc, pair_key) so each physical contact
    is counted once. Returns (data, dedup, dedup_idx) dicts of numpy arrays,
    or raises RuntimeError with a clear message if the schema doesn't match
    or no usable rows are found."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise RuntimeError(
                f"SCHEMA_MISMATCH: {csv_path} is missing expected column(s) {missing}. "
                f"Actual header: {fieldnames}. This usually means the CSV was produced "
                f"by a different/older version of nn_force_distance*.py, or --csv points "
                f"at the wrong file."
            )

        mol_id, calc, elem_i, elem_j = [], [], [], []
        r_ij_A, f_rel, is_mutual_nn = [], [], []
        pair_key_full = []
        n_rows_seen = 0
        n_rows_skipped = 0

        for row in reader:
            n_rows_seen += 1
            try:
                r_ij = float(row["r_ij_A"])
                frel = float(row["f_rel"])
            except (ValueError, KeyError):
                n_rows_skipped += 1
                continue
            mol_id.append(row["mol_id"])
            calc.append(row["calc"])
            elem_i.append(row["elem_i"])
            elem_j.append(row["elem_j"])
            r_ij_A.append(r_ij)
            f_rel.append(frel)
            is_mutual_nn.append(row.get("is_mutual_nn", "False") == "True")
            pair_key_full.append((row["mol_id"], row["calc"], row.get("pair_key", "")))

    if n_rows_seen == 0:
        raise RuntimeError(f"EMPTY_CSV: {csv_path} has a header but no data rows.")
    if len(r_ij_A) == 0:
        raise RuntimeError(
            f"NO_USABLE_ROWS: {csv_path} had {n_rows_seen} rows, but all {n_rows_skipped} "
            f"were skipped due to unparseable r_ij_A/f_rel values -- check the CSV contents."
        )
    if n_rows_skipped > 0:
        print(f"WARNING: skipped {n_rows_skipped}/{n_rows_seen} rows with unparseable "
              f"r_ij_A/f_rel values.")

    n = len(r_ij_A)
    r0 = np.array([r0_for_pair(elem_i[k], elem_j[k]) for k in range(n)])
    x = np.array(r_ij_A) / r0

    data = {
        "mol_id": np.array(mol_id),
        "calc": np.array(calc),
        "elem_i": np.array(elem_i),
        "elem_j": np.array(elem_j),
        "r_ij_A": np.array(r_ij_A),
        "f_rel": np.array(f_rel),
        "x": x,
        "is_mutual_nn": np.array(is_mutual_nn),
        "pair_key_full": pair_key_full,
    }

    # Deduplicate physical contacts via (mol_id, calc, pair_key): keep the
    # first occurrence of each key. Non-mutual rows have a pair_key too
    # (typically unique to that row), so this only collapses true mutual
    # duplicates without affecting one-sided nearest-neighbor rows.
    seen = set()
    dedup_idx = []
    for idx, key in enumerate(pair_key_full):
        if key not in seen:
            seen.add(key)
            dedup_idx.append(idx)
    dedup_idx = np.array(dedup_idx, dtype=int)

    dedup = {k: v[dedup_idx] if isinstance(v, np.ndarray) else v
             for k, v in data.items() if k != "pair_key_full"}
    return data, dedup, dedup_idx


def print_binned_stats(x, f_rel, label):
    print(f"[{label}] n={len(x)}")
    print(f"  x         : min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f} median={np.median(x):.4f}")
    absf = np.abs(f_rel)
    print(f"  |f_rel|   : min={absf.min():.6f} max={absf.max():.4f} mean={absf.mean():.4f} median={np.median(absf):.6f}")
    print(f"  FRACTION_REPULSIVE (f_rel > 0): {np.mean(f_rel > 0):.4f}")
    print()
    edges = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, np.inf]
    print("  Binned by x:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            print(f"    x in [{lo}, {hi}): n=0")
            continue
        med_absf = np.median(absf[mask])
        frac_rep = np.mean(f_rel[mask] > 0)
        print(f"    x in [{lo:.1f}, {hi:.1f}): n={cnt:6d}  median|f_rel|={med_absf:10.5f}  frac_repulsive={frac_rep:.3f}")
    print()


def fit_line(lx, ly):
    """Least-squares fit ly = a*lx + b. Returns (a, b, sse)."""
    if len(lx) < 2:
        return 0.0, float(ly.mean()) if len(ly) else 0.0, 0.0
    A = np.vstack([lx, np.ones_like(lx)]).T
    (a, b), residuals, rank, sv = np.linalg.lstsq(A, ly, rcond=None)
    pred = a * lx + b
    sse = float(np.sum((ly - pred) ** 2))
    return float(a), float(b), sse


def find_knee(x, f_rel, grid_x_min=0.05, grid_x_max=1.3, n_candidates=80,
              min_force_for_log=1e-8, repulsive_only=False):
    """Grid-search a two-segment piecewise-linear fit in log(x) vs
    log(|f_rel|) space to find the breakpoint minimizing total SSE.

    The candidate grid is a FIXED linspace in log(x) between log(grid_x_min)
    and log(grid_x_max) -- NOT a quantile-based grid of the observed x
    values. This matters because the compressed/strained tail is a small
    minority of rows (~2.4% below r_ij<0.8A in the full dataset); a
    quantile-clipped grid (e.g. 5th-95th percentile of observed log(x))
    could easily exclude that tail entirely, making it structurally
    impossible to ever find a knee there.

    Also fits a SINGLE line over the whole (masked) range as a baseline,
    so the two-segment fit can be judged against a null hypothesis of "no
    knee" rather than accepted at face value (a two-segment fit will always
    return *some* minimizing breakpoint, knee or not).

    Returns dict with breakpoint_x, breakpoint_log_x, total_sse (two-segment),
    baseline_sse (single line), sse_reduction_fraction, and the two fitted
    line segments (slope/intercept for x <= breakpoint and x > breakpoint).
    """
    absf = np.abs(f_rel)
    mask = (x > 0) & (absf > min_force_for_log) & np.isfinite(x) & np.isfinite(absf)
    if repulsive_only:
        # Exclude the attractive/zero-crossing region near x~1 entirely,
        # so the zero-crossing of f_rel (which produces a spurious V-shape
        # in log|f_rel| space near equilibrium bond length) cannot be
        # mistaken for a real knee in the repulsive wall.
        mask = mask & (f_rel > 0)
    lx = np.log(x[mask])
    ly = np.log(absf[mask])

    if len(lx) < 10:
        return {
            "breakpoint_x": float("nan"), "breakpoint_log_x": float("nan"),
            "total_sse": float("nan"), "baseline_sse": float("nan"),
            "sse_reduction_fraction": float("nan"),
            "left_slope": float("nan"), "left_intercept": float("nan"), "n_left": 0,
            "right_slope": float("nan"), "right_intercept": float("nan"), "n_right": 0,
            "n_used": len(lx),
            "warning": "fewer than 10 usable (x, |f_rel|) points -- knee estimate not meaningful",
        }

    order = np.argsort(lx)
    lx_sorted = lx[order]
    ly_sorted = ly[order]

    # Baseline: single line over the whole used range.
    baseline_a, baseline_b, baseline_sse = fit_line(lx_sorted, ly_sorted)

    # Fixed grid over a wide x-range, clipped to what's actually observed
    # (can't search outside the data), but NOT clipped to a narrow quantile.
    log_grid_lo = max(np.log(grid_x_min), lx_sorted.min())
    log_grid_hi = min(np.log(grid_x_max), lx_sorted.max())
    if log_grid_hi <= log_grid_lo:
        log_grid_lo, log_grid_hi = lx_sorted.min(), lx_sorted.max()
    candidates = np.linspace(log_grid_lo, log_grid_hi, n_candidates)

    best = None
    for bp in candidates:
        left = lx_sorted <= bp
        right = ~left
        n_left = int(np.sum(left))
        n_right = int(np.sum(right))
        if n_left < 5 or n_right < 5:
            continue
        a1, b1, sse1 = fit_line(lx_sorted[left], ly_sorted[left])
        a2, b2, sse2 = fit_line(lx_sorted[right], ly_sorted[right])
        total_sse = sse1 + sse2
        if best is None or total_sse < best["total_sse"]:
            best = {
                "breakpoint_log_x": float(bp),
                "breakpoint_x": float(np.exp(bp)),
                "total_sse": total_sse,
                "left_slope": a1, "left_intercept": b1, "n_left": n_left,
                "right_slope": a2, "right_intercept": b2, "n_right": n_right,
            }

    if best is None:
        best = {
            "breakpoint_log_x": float(np.median(lx_sorted)),
            "breakpoint_x": float(np.exp(np.median(lx_sorted))),
            "total_sse": float("nan"),
            "left_slope": float("nan"), "left_intercept": float("nan"), "n_left": 0,
            "right_slope": float("nan"), "right_intercept": float("nan"), "n_right": 0,
            "warning": "grid search found no valid candidate split (too few points per side) "
                       "-- falling back to median x as a placeholder, NOT a real knee estimate",
        }

    best["baseline_slope"] = baseline_a
    best["baseline_intercept"] = baseline_b
    best["baseline_sse"] = baseline_sse
    if np.isfinite(best["total_sse"]) and baseline_sse > 0:
        best["sse_reduction_fraction"] = float(1.0 - best["total_sse"] / baseline_sse)
    else:
        best["sse_reduction_fraction"] = float("nan")
    best["n_used"] = len(lx)
    best["grid_log_x_range"] = (float(log_grid_lo), float(log_grid_hi))
    best.setdefault("warning", None)

    best["lx_used"] = lx
    best["ly_used"] = ly
    return best


def suggested_midpoint_score(knee_x, x_stable=0.9, x_unstable=0.1):
    """Given the estimated knee x, report a score s in [0, 1] on the
    logarithmic/geometric stability scale discussed: x_stable (default 0.9,
    "fully stable") maps to s=0, x_unstable (default 0.1, "fully unstable /
    fusion-like") maps to s=1. s = (log(x_stable) - log(knee_x)) /
    (log(x_stable) - log(x_unstable)), clipped to [0, 1]. This is returned
    as a PLAIN score in [0, 1] -- it is NOT rescaled into any other range;
    what the caller does with that score (e.g. use it directly as a
    per-contact "compression score") is a separate decision.
    """
    if not (np.isfinite(knee_x) and knee_x > 0):
        return {
            "knee_x": knee_x, "s": float("nan"),
            "geometric_mean_midpoint_x_for_reference": float(np.sqrt(x_stable * x_unstable)),
            "warning": "knee_x is not a finite positive number -- score not computed",
        }
    log_stable = np.log(x_stable)
    log_unstable = np.log(x_unstable)
    s_raw = (log_stable - np.log(knee_x)) / (log_stable - log_unstable)
    s = float(np.clip(s_raw, 0.0, 1.0))
    geometric_midpoint_x = float(np.sqrt(x_stable * x_unstable))
    return {
        "knee_x": knee_x,
        "s": s,
        "geometric_mean_midpoint_x_for_reference": geometric_midpoint_x,
    }


def make_plot(dedup, knee, out_path: Path):
    """Best-effort plot. Returns (success: bool, message: str)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return False, f"matplotlib unavailable or failed to import ({e}); skipping plot."

    try:
        x = dedup["x"]
        f_rel = dedup["f_rel"]
        absf = np.abs(f_rel)
        is_mutual = dedup["is_mutual_nn"]

        mask_plot = (x > 0) & (absf > 1e-8) & np.isfinite(x) & np.isfinite(absf)

        fig, ax = plt.subplots(figsize=(9, 7))
        for flag, label, color, marker in [(True, "mutual NN", "tab:blue", "o"),
                                            (False, "one-sided NN", "tab:orange", "^")]:
            m = mask_plot & (is_mutual == flag)
            if np.any(m):
                ax.scatter(x[m], absf[m], s=8, alpha=0.35, label=f"{label} (n={int(np.sum(m))})",
                           color=color, marker=marker)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("x = r_ij / r0 (covalent-radius-ratio, log scale)")
        ax.set_ylabel("|f_rel| (log scale)")
        ax.set_title("Compression ratio vs. two-body radial force (dedup pairs)")

        knee_x = knee.get("breakpoint_x", float("nan"))
        if np.isfinite(knee_x):
            ax.axvline(knee_x, color="red", linestyle="--", linewidth=1.5,
                       label=f"estimated knee x={knee_x:.4f}")

        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return True, f"Saved plot to {out_path}"
    except Exception as e:
        return False, f"Plotting failed ({e}); numeric results above are unaffected."


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help="Path to nn_force_distance_v2.csv")
    p.add_argument("--out-png", type=Path, default=DEFAULT_PNG,
                    help="Output PNG path")
    p.add_argument("--grid-x-min", type=float, default=0.05,
                    help="Lower bound (in x, not log) of the FIXED knee-search grid")
    p.add_argument("--grid-x-max", type=float, default=1.3,
                    help="Upper bound (in x, not log) of the FIXED knee-search grid")
    p.add_argument("--n-candidates", type=int, default=80,
                    help="Number of candidate breakpoints in the grid search")
    p.add_argument("--repulsive-only", action="store_true",
                    help="Restrict knee search to f_rel > 0 rows only, excluding the "
                         "attractive/zero-crossing region near x~1 (avoids a spurious "
                         "knee at the equilibrium zero-crossing of f_rel)")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"CSV_NOT_FOUND: {args.csv}")
        print("Pass the correct path explicitly with --csv PATH.")
        sys.exit(1)

    print(f"Loading {args.csv} ...")
    try:
        data, dedup, dedup_idx = load_and_dedup(args.csv)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    n_total = len(data["r_ij_A"])
    n_dedup = len(dedup["r_ij_A"])
    print(f"TOTAL_ROWS={n_total}  DEDUP_ROWS(unique physical contacts)={n_dedup}")
    print(f"MUTUAL_NN_FRACTION (dedup): {np.mean(dedup['is_mutual_nn']):.4f}")
    print()

    print("=== Summary on ALL rows (atom-centric, includes mutual-NN duplicates) ===")
    print_binned_stats(data["x"], data["f_rel"], "ALL rows")

    print("=== Summary on DEDUP rows (one row per physical contact via pair_key) ===")
    print_binned_stats(dedup["x"], dedup["f_rel"], "DEDUP rows")

    print("Estimating knee via grid-search two-segment piecewise-linear fit "
          "in log(x) vs log(|f_rel|) space (dedup rows only, FIXED wide x-range grid) ...")
    print(f"REPULSIVE_ONLY: {args.repulsive_only}")
    knee = find_knee(dedup["x"], dedup["f_rel"],
                      grid_x_min=args.grid_x_min,
                      grid_x_max=args.grid_x_max,
                      n_candidates=args.n_candidates,
                      repulsive_only=args.repulsive_only)

    print()
    print("KNEE_RESULT:")
    if knee.get("warning"):
        print(f"  WARNING: {knee['warning']}")
    print(f"  n_used (finite x>0, |f_rel|>1e-8)  : {knee.get('n_used')}")
    lo_hi = knee.get("grid_log_x_range")
    if lo_hi:
        print(f"  grid searched over x in [{np.exp(lo_hi[0]):.4f}, {np.exp(lo_hi[1]):.4f}] "
              f"(log-space linspace, {args.n_candidates} candidates)")
    print(f"  breakpoint_x (r_ij/r0)      : {knee['breakpoint_x']:.4f}")
    print(f"  breakpoint_log_x            : {knee['breakpoint_log_x']:.4f}")
    print(f"  two_segment_total_sse       : {knee['total_sse']:.4f}")
    print(f"  baseline_single_line_sse    : {knee.get('baseline_sse', float('nan')):.4f}")
    sse_red = knee.get("sse_reduction_fraction", float("nan"))
    print(f"  sse_reduction_fraction      : {sse_red:.4f}  "
          f"(fraction of baseline SSE removed by allowing a knee; near 0 means "
          f"the two-segment fit barely beats a single line -- i.e. no real knee)")
    print(f"  left  segment (x<=knee): slope={knee['left_slope']:.4f}  "
          f"intercept={knee['left_intercept']:.4f}  n={knee['n_left']}")
    print(f"  right segment (x> knee): slope={knee['right_slope']:.4f}  "
          f"intercept={knee['right_intercept']:.4f}  n={knee['n_right']}")
    print()

    score_info = suggested_midpoint_score(knee["breakpoint_x"])
    print("STABILITY_SCORE_SUGGESTION (s in [0,1]; s=0 at x=0.9 'fully stable', "
          "s=1 at x=0.1 'fully unstable/fusion-like'):")
    for k, v in score_info.items():
        print(f"  {k}: {v}")
    print()

    ok, msg = make_plot(dedup, knee, args.out_png)
    print(("PLOT_OK: " if ok else "PLOT_SKIPPED: ") + msg)


if __name__ == "__main__":
    main()
