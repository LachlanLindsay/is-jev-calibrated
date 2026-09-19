"""Write the audit up: one machine-readable JSON file and one readable Markdown."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .analysis import AuditResult, calibrated_bins_table
from .selective import GateRecommendation

SIMULATED_BANNER = (
    "> **These numbers are synthetic.** They come from the built-in simulator, "
    "not from a model. They exist to exercise the pipeline and to check that the "
    "audit recovers a miscalibration it was given on purpose. Do not cite them.\n"
)


def verdict(result: AuditResult) -> str:
    """One sentence a reader can quote, derived only from the numbers."""
    ece = result.ece
    if ece < 0.02:
        grade = "well calibrated"
    elif ece < 0.05:
        grade = "usably calibrated"
    elif ece < 0.10:
        grade = "noticeably miscalibrated"
    else:
        grade = "badly miscalibrated"

    gap = result.overconfidence
    if abs(gap) < 0.01:
        direction = "with no systematic lean in either direction"
    elif gap > 0:
        direction = f"and overconfident by {gap:.1%} on average"
    else:
        direction = f"and underconfident by {abs(gap):.1%} on average"

    slope = result.fit["slope"]
    if slope < 0.85:
        shape = " Its probabilities are too extreme: a logistic refit pulls them toward the middle"
    elif slope > 1.15:
        shape = " Its probabilities are too timid: a logistic refit pushes them outward"
    else:
        shape = " The shape of the curve is close to right"
    shape += f" (fitted slope {slope:.2f}, where 1.00 is perfect)."

    ranking = ""
    if result.auroc == result.auroc:  # not NaN
        ranking = (
            f" Separately from calibration, the confidence ranks its own errors at "
            f"AUROC {result.auroc:.3f}, which is what a confidence *gate* actually relies on."
        )

    return (
        f"On {result.n:,} labelled examples from `{result.task}`, this model is **{grade}** "
        f"(ECE {ece:.3f}, 95% CI {result.ece_ci[0]:.3f}-{result.ece_ci[1]:.3f}) {direction}."
        f"{shape}{ranking}"
    )


def _gate_rows(gates: Sequence[GateRecommendation]) -> list[str]:
    rows = []
    for g in gates:
        if g.threshold is None:
            rows.append(f"| {g.target_error:.0%} | not reachable | - | - | - | {g.note} |")
            continue
        flag = "yes" if g.guaranteed else "provisional"
        rows.append(
            f"| {g.target_error:.0%} | >= {g.threshold:.4f} | {g.coverage:.1%} | "
            f"{g.observed_error:.2%} | {g.error_upper:.2%} | {flag} |"
        )
    return rows


#: The risk-coverage curve has one point per distinct confidence value, which is
#: tens of thousands of rows on a real run. The chart uses all of them; the JSON
#: carries an evenly spaced sample, which is plenty to redraw or to cite.
MAX_COVERAGE_POINTS = 200


def to_json(result: AuditResult, max_coverage_points: int = MAX_COVERAGE_POINTS) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    payload = convert(asdict(result))
    # ``extra`` is a scratch space for the plotting layer (it holds the raw
    # confidence vector); the raw data already lives in the results JSONL.
    payload.pop("extra", None)
    payload["coverage"] = convert(thin(result.coverage, max_coverage_points))
    payload["coverage_points_total"] = len(result.coverage)
    payload["verdict"] = verdict(result)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    return payload


def thin(points: Sequence[Any], limit: int) -> list[Any]:
    """Evenly sample a long sequence down to ``limit`` items, keeping both ends."""
    if limit < 2 or len(points) <= limit:
        return list(points)
    step = (len(points) - 1) / (limit - 1)
    idx = sorted({int(round(i * step)) for i in range(limit)} | {0, len(points) - 1})
    return [points[i] for i in idx]


def write_report(
    result: AuditResult,
    out_dir: str | Path,
    chart_paths: Sequence[Path] = (),
    run_meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``report.md`` and ``results.json`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(json.dumps(to_json(result), indent=2) + "\n")

    charts = {p.name: p for p in chart_paths}
    lines: list[str] = []
    add = lines.append

    add(f"# Calibration audit: `{result.task}`")
    add("")
    if result.simulated:
        add(SIMULATED_BANNER)
    add(f"*Model:* `{result.model or 'unknown'}` &nbsp;&nbsp; *Task type:* `{result.kind}` "
        f"&nbsp;&nbsp; *Examples:* {result.n:,} &nbsp;&nbsp; "
        f"*Generated:* {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    add("")
    add("## Verdict")
    add("")
    add(verdict(result))
    add("")

    add("## Headline numbers")
    add("")
    add("| metric | value | what it means |")
    add("| --- | --- | --- |")
    add(f"| accuracy | {result.accuracy:.2%} | share of decisions that matched the label |")
    add(f"| majority-class baseline | {result.base_rate:.2%} | accuracy of always answering the commonest label |")
    add(f"| mean confidence | {result.mean_confidence:.2%} | what the model claimed, on average |")
    add(f"| overconfidence | {result.overconfidence:+.2%} | mean confidence minus accuracy; positive = charges more than it delivers |")
    add(f"| ECE (15 equal-width bins) | {result.ece:.4f} | average distance between claimed and delivered |")
    add(f"| ECE 95% CI | {result.ece_ci[0]:.4f} - {result.ece_ci[1]:.4f} | 1,000-resample bootstrap |")
    add(f"| ECE (15 equal-mass bins) | {result.ece_adaptive:.4f} | same idea, bins holding equal counts |")
    add(f"| MCE (bins with n >= 30) | {result.mce:.4f} | worst single band -- what a gate trips over |")
    add(f"| Brier score | {result.brier:.4f} | proper score over the top label; lower is better |")
    add(f"| - reliability | {result.brier_parts['reliability']:.4f} | the miscalibration part; 0 is perfect |")
    add(f"| - resolution | {result.brier_parts['resolution']:.4f} | how far it separates cases; higher is better |")
    add(f"| - uncertainty | {result.brier_parts['uncertainty']:.4f} | irreducible difficulty of the task |")
    add(f"| AUROC of confidence vs. correctness | {result.auroc:.4f} | can confidence rank its own errors? 0.5 = no signal |")
    if result.nll is not None:
        add(f"| NLL of the true label | {result.nll:.4f} | punishes confident mistakes hardest |")
    if result.multiclass_brier is not None:
        add(f"| multiclass Brier | {result.multiclass_brier:.4f} | over the whole distribution, not just the winner |")
    add(f"| logistic refit slope | {result.fit['slope']:.4f} | 1.00 = right shape; < 1 = too extreme |")
    add(f"| logistic refit intercept | {result.fit['intercept']:+.4f} | 0.00 = no constant bias |")
    if result.ordinal:
        add(f"| mean absolute error (rungs) | {result.ordinal['mae_rungs']:.3f} | ordered rubrics: how far off, not just whether |")
        add(f"| within one rung | {result.ordinal['within_one_rung']:.2%} | |")
    if result.latency_ms:
        add(f"| latency p50 / p95 | {result.latency_ms['p50']:.0f} ms / {result.latency_ms['p95']:.0f} ms | end to end, including network |")
    if result.cost_usd is not None:
        add(f"| total cost | ${result.cost_usd:.4f} | ${result.cost_usd / max(result.n, 1) * 1000:.4f} per 1,000 decisions |")
    add("")

    if "reliability.png" in charts:
        add("## Is the number honest?")
        add("")
        add("![Reliability diagram](reliability.png)")
        add("")
    add("Every bar above is also a row here, with its 95% Wilson interval:")
    add("")
    add(_bins_table(result))
    add("")

    if "risk-coverage.png" in charts:
        add("## Where is it safe to automate?")
        add("")
        add("![Risk-coverage curve](risk-coverage.png)")
        add("")
    add("| error budget | gate at confidence | coverage | observed error | 95% upper bound | meets budget |")
    add("| --- | --- | --- | --- | --- | --- |")
    lines.extend(_gate_rows(result.gates))
    add("")
    add(
        "A gate is only recommended when the *upper* bound on its error clears the "
        "budget, not the point estimate -- picking the threshold that happened to "
        "look best on this sample is how a gate that tested at 1% ships at 4%."
    )
    add("")

    if "confidence.png" in charts:
        add("## What does it claim?")
        add("")
        add("![Confidence histogram](confidence.png)")
        add("")

    add("## How this was measured")
    add("")
    add(f"- Task type `{result.kind}` over {len(result.labels)} labels: `{'`, `'.join(result.labels[:12])}`"
        + (" ..." if len(result.labels) > 12 else ""))
    add("- Confidence is the probability the model assigned to the label it picked. "
        "Nothing here asks a model to write a confidence number into its own output; "
        "a self-reported number is a different object from a calibrated distribution.")
    add("- ECE is reported with both equal-width bins (the readable version) and "
        "equal-mass bins (the version that does not let a near-empty bin swing the result).")
    add("- Intervals on bins are Wilson score intervals; the interval on ECE is a "
        "1,000-resample percentile bootstrap.")
    if run_meta:
        add(f"- Run: provider `{run_meta.get('provider', '?')}`, "
            f"{run_meta.get('n_failed', 0)} failed request(s) excluded, "
            f"{run_meta.get('wall_seconds', 0):.0f}s wall clock.")
    add("")
    add("## What this does not show")
    add("")
    add("- Calibration measured on one dataset is calibration on that dataset. "
        "A model calibrated on product reviews can be badly calibrated on medical text.")
    add("- Bounded context matters: every example here fits comfortably in context. "
        "Long or noisy states are a separate experiment.")
    add("- Type-safe output is not correct output. Everything counted as an error "
        "below was a perfectly valid value that happened to be wrong.")
    add("- Confidence gating trades coverage for accuracy. The escalated tail still "
        "has to go somewhere, and that somewhere has its own cost.")
    add("")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _bins_table(result: AuditResult) -> str:
    rows = ["| confidence bin | n | mean confidence | observed accuracy | 95% CI | gap |", "| --- | --- | --- | --- | --- | --- |"]
    for row in calibrated_bins_table(result.bins):
        if not row["n"]:
            continue
        rows.append(
            f"| {row['bin']} | {row['n']:,} | {row['mean_confidence']:.3f} | "
            f"{row['observed_accuracy']:.3f} | {row['ci_low']:.3f} - {row['ci_high']:.3f} | "
            f"{row['gap']:+.3f} |"
        )
    return "\n".join(rows)
