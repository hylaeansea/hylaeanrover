"""Check Stage 0 locomotion gate reports.

Parses the text files produced by `examples/evaluate.py` in
`runs/reports/` and prints a compact pass/fail table for the Stage 0
promotion gate.

Example:

    python examples/check_stage0_gate.py
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCENARIOS = (
    "terrain_fixed_1_0",
    "terrain_fixed_1_5",
    "terrain_fixed_2_0",
    "terrain_mixed_1_2",
)

MEAN_RE = re.compile(r"^\s+(?P<key>[a-z_]+)\s+mean=\s*(?P<value>-?\d+(?:\.\d+)?)")
PERCENT_RE = re.compile(
    r"^\s+(?P<key>flip_rate|out_of_power)\s+(?P<value>-?\d+(?:\.\d+)?)%"
)
REASONS_RE = re.compile(r"^\s+terminal reasons:\s+(?P<value>.+)$")


@dataclass(frozen=True)
class Report:
    path: Path
    metrics: dict[str, float]
    terminal_reasons: str = ""
    parse_error: str | None = None

    def metric(self, name: str, default: float = 0.0) -> float:
        return self.metrics.get(name, default)


def parse_report(path: Path) -> Report:
    metrics: dict[str, float] = {}
    terminal_reasons = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        mean_match = MEAN_RE.match(line)
        if mean_match:
            metrics[mean_match.group("key")] = float(mean_match.group("value"))
            continue
        percent_match = PERCENT_RE.match(line)
        if percent_match:
            metrics[percent_match.group("key")] = (
                float(percent_match.group("value")) / 100.0
            )
            continue
        reasons_match = REASONS_RE.match(line)
        if reasons_match:
            terminal_reasons = reasons_match.group("value")
    if "distance" not in metrics:
        raise ValueError(f"{path} does not look like an evaluate.py report")
    return Report(path=path, metrics=metrics, terminal_reasons=terminal_reasons)


def load_report(path: Path) -> Report | None:
    if not path.exists():
        return None
    try:
        return parse_report(path)
    except (OSError, ValueError) as exc:
        return Report(path=path, metrics={}, parse_error=str(exc))


def status_label(failed: bool, warned: bool) -> str:
    if failed:
        return "FAIL"
    if warned:
        return "WARN"
    return "PASS"


def pct(value: float) -> str:
    return f"{value * 100.0:5.1f}%"


def ratio(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.2f}x"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("runs/reports"),
        help="directory containing evaluate.py report text files",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="comma-separated terrain scenarios to check",
    )
    parser.add_argument(
        "--min-distance-ratio",
        type=float,
        default=2.0,
        help="hard minimum trained/random short and medium distance ratio",
    )
    parser.add_argument(
        "--target-distance-ratio",
        type=float,
        default=3.0,
        help="soft target trained/random short and medium distance ratio; below this warns",
    )
    parser.add_argument(
        "--max-flip-rate",
        type=float,
        default=0.10,
        help="maximum acceptable flip rate for trained short/medium/long evals",
    )
    parser.add_argument(
        "--warn-out-of-power-rate",
        type=float,
        default=0.80,
        help="warn when trained medium out-of-power rate exceeds this",
    )
    parser.add_argument(
        "--long-report",
        default="stage0_best_mixed_long.txt",
        help="optional sampled long-horizon report filename",
    )
    args = parser.parse_args()

    reports_dir = args.reports_dir
    scenarios = tuple(s.strip() for s in args.scenarios.split(",") if s.strip())
    any_failed = False
    any_warned = False

    print("Stage 0 locomotion gate")
    print(f"reports: {reports_dir}")
    print(
        "criteria: "
        f"short+medium ratio >= {args.min_distance_ratio:.2f}x "
        f"(target {args.target_distance_ratio:.2f}x), "
        f"flip <= {pct(args.max_flip_rate)}, "
        f"out-of-power warn > {pct(args.warn_out_of_power_rate)}"
    )
    print()
    print(
        "scenario                 "
        "random_d  short_d  short_r  med_d    med_r   flip    oop    status"
    )
    print("-" * 88)

    for scenario in scenarios:
        random_report = load_report(reports_dir / f"stage0_random_{scenario}.txt")
        short_report = load_report(reports_dir / f"stage0_best_{scenario}_short.txt")
        medium_report = load_report(reports_dir / f"stage0_best_{scenario}_medium.txt")

        failed = False
        warned = False
        notes: list[str] = []
        random_distance: float | None = None
        short_distance: float | None = None
        short_ratio: float | None = None
        medium_distance: float | None = None
        medium_ratio: float | None = None
        medium_flip = 0.0
        medium_oop = 0.0

        if random_report is None:
            failed = True
            notes.append("missing random report")
        elif random_report.parse_error:
            failed = True
            notes.append(f"invalid random report: {random_report.parse_error}")
        else:
            random_distance = random_report.metric("distance")

        if short_report is None:
            warned = True
            notes.append("missing short trained report")
        elif short_report.parse_error:
            failed = True
            notes.append(f"invalid short report: {short_report.parse_error}")
        else:
            short_distance = short_report.metric("distance")
            if random_distance and random_distance > 0.0:
                short_ratio = short_distance / random_distance
                if short_ratio < args.min_distance_ratio:
                    failed = True
                    notes.append(f"short ratio {short_ratio:.2f}x")
                elif short_ratio < args.target_distance_ratio:
                    warned = True
                    notes.append(f"short below target {short_ratio:.2f}x")
            short_flip = short_report.metric("flip_rate")
            if short_flip > args.max_flip_rate:
                failed = True
                notes.append(f"short flip {pct(short_flip)}")

        if medium_report is None:
            failed = True
            notes.append("missing medium trained report")
        elif medium_report.parse_error:
            failed = True
            notes.append(f"invalid medium report: {medium_report.parse_error}")
        else:
            medium_distance = medium_report.metric("distance")
            medium_flip = medium_report.metric("flip_rate")
            medium_oop = medium_report.metric("out_of_power")
            if random_distance and random_distance > 0.0:
                medium_ratio = medium_distance / random_distance
                if medium_ratio < args.min_distance_ratio:
                    failed = True
                    notes.append(f"medium ratio {medium_ratio:.2f}x")
                elif medium_ratio < args.target_distance_ratio:
                    warned = True
                    notes.append(f"medium below target {medium_ratio:.2f}x")
            if medium_flip > args.max_flip_rate:
                failed = True
                notes.append(f"medium flip {pct(medium_flip)}")
            if medium_oop > args.warn_out_of_power_rate:
                warned = True
                notes.append(f"out_of_power {pct(medium_oop)}")

        any_failed = any_failed or failed
        any_warned = any_warned or warned
        status = status_label(failed, warned)
        print(
            f"{scenario:24s} "
            f"{(random_distance or 0.0):8.2f} "
            f"{(short_distance or 0.0):8.2f} "
            f"{ratio(short_ratio):>8s} "
            f"{(medium_distance or 0.0):8.2f} "
            f"{ratio(medium_ratio):>7s} "
            f"{pct(medium_flip):>7s} "
            f"{pct(medium_oop):>7s} "
            f"{status:>7s}"
        )
        if notes:
            print(f"  notes: {', '.join(notes)}")

    long_path = reports_dir / args.long_report
    long_report = load_report(long_path)
    if long_report is None:
        any_warned = True
        print()
        print(f"long sample: WARN missing {long_path}")
    elif long_report.parse_error:
        any_warned = True
        print()
        print(f"long sample: WARN invalid {long_path}")
        print(f"  notes: {long_report.parse_error}")
    else:
        long_flip = long_report.metric("flip_rate")
        long_oop = long_report.metric("out_of_power")
        long_failed = long_flip > args.max_flip_rate
        any_failed = any_failed or long_failed
        print()
        print(
            "long sample: "
            f"distance={long_report.metric('distance'):.2f} "
            f"flip={pct(long_flip)} "
            f"out_of_power={pct(long_oop)} "
            f"status={status_label(long_failed, False)}"
        )
        print(f"  terminal reasons: {long_report.terminal_reasons}")

    print()
    if any_failed:
        print("OVERALL: FAIL - do not promote Stage 0 yet.")
        raise SystemExit(1)
    if any_warned:
        print(
            "OVERALL: WARN - gate is close or incomplete; review notes before promotion."
        )
        raise SystemExit(2)
    print("OVERALL: PASS - Stage 0 meets the configured report gates.")


if __name__ == "__main__":
    main()
