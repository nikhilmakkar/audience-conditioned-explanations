"""Analyze the frozen fixed-direction free-generation dose response."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return sum(values) / len(values)


def sem(values):
    if len(values) < 2:
        return float("nan")
    center = mean(values)
    return math.sqrt(sum((x - center) ** 2 for x in values) / (len(values) * (len(values) - 1)))


def signflip_p(values):
    """Exact one-sided randomization p, including equality in the tail."""
    observed = mean(values)
    tail = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        statistic = mean([sign * value for sign, value in zip(signs, values)])
        if statistic >= observed - 1e-12:
            tail += 1
    return tail / (2 ** len(values))


def slope(xs, ys):
    xbar, ybar = mean(xs), mean(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum(
        (x - xbar) ** 2 for x in xs
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.input)
    for row in rows:
        for key in ("alpha", "term_coverage", "word_count"):
            row[key] = float(row[key])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    domain_rows = []
    summary_rows = []
    for metric in ("term_coverage", "word_count"):
        for condition in ("pooled", "domain_expert", "matched_other_expert"):
            subset = rows if condition == "pooled" else [r for r in rows if r["condition"] == condition]
            effects, slopes = [], []
            for source in sorted({r["source_id"] for r in subset}):
                source_rows = [r for r in subset if r["source_id"] == source]
                by_alpha = {
                    alpha: mean([r[metric] for r in source_rows if r["alpha"] == alpha])
                    for alpha in (-2.0, -1.0, 0.0, 1.0, 2.0)
                }
                effect = by_alpha[2.0] - by_alpha[-2.0]
                source_slope = slope(list(by_alpha), list(by_alpha.values()))
                effects.append(effect)
                slopes.append(source_slope)
                domain_rows.append({
                    "metric": metric,
                    "condition": condition,
                    "source_id": source,
                    **{f"alpha_{alpha:g}": by_alpha[alpha] for alpha in by_alpha},
                    "plus2_minus_minus2": effect,
                    "linear_slope": source_slope,
                })
            summary_rows.append({
                "metric": metric,
                "condition": condition,
                "contrast_mean": mean(effects),
                "contrast_se": sem(effects),
                "contrast_positive_domains": sum(value > 0 for value in effects),
                "contrast_exact_one_sided_p": signflip_p(effects),
                "slope_mean": mean(slopes),
                "slope_se": sem(slopes),
                "slope_positive_domains": sum(value > 0 for value in slopes),
                "slope_exact_one_sided_p": signflip_p(slopes),
            })

    for filename, values in (("generation_by_domain.csv", domain_rows), ("generation_summary.csv", summary_rows)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
