"""The three charts the audit publishes.

One chart, one question:

* the reliability diagram -- *is the number honest?*
* the risk-coverage curve -- *where can I stop looking at these by hand?*
* the confidence histogram -- *what does it claim, and how often?*

Every plotted value also appears in the report's tables, so nothing is readable
only as a picture. Both themes are rendered; neither is an automatic flip of the
other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from . import theme as theme_mod
from .analysis import AuditResult
from .metrics import Bin
from .selective import CoveragePoint, GateRecommendation

FONT_STACK = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]


def _style(t: theme_mod.Theme) -> dict:
    return {
        "figure.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "text.color": t.ink,
        "axes.labelcolor": t.ink_secondary,
        "axes.edgecolor": t.axis,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t.grid,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": t.ink_muted,
        "ytick.color": t.ink_muted,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 160,
    }


def _finish(fig: Figure, ax, t: theme_mod.Theme, result: AuditResult | None) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t.axis)
    if result is not None and result.simulated:
        _watermark(fig, t)


def _watermark(fig: Figure, t: theme_mod.Theme) -> None:
    """Stamp synthetic output so it can never be mistaken for a measurement."""
    fig.text(
        0.5,
        0.5,
        "SIMULATED DATA",
        fontsize=34,
        color=t.ink_muted,
        alpha=0.18,
        ha="center",
        va="center",
        rotation=24,
        zorder=10,
        weight="bold",
    )


def _titles(ax, t: theme_mod.Theme, title: str, subtitle: str) -> None:
    """Title above subtitle, both left-aligned, neither on top of the other."""
    ax.set_title(title, color=t.ink, loc="left", pad=30)
    ax.text(
        0.0,
        1.012,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color=t.ink_muted,
        ha="left",
        va="bottom",
    )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


# ------------------------------------------------------------ reliability


def reliability_diagram(
    result: AuditResult,
    out_path: str | Path,
    theme: str = "light",
    bins: Sequence[Bin] | None = None,
    title: str | None = None,
) -> Path:
    """Predicted confidence against observed accuracy.

    The whole claim under audit in one picture: a calibrated model's bars sit on
    the diagonal. Bars below it are overconfidence -- the model charging more
    than it delivers. Error bars are 95% Wilson intervals, so a bar that misses
    the diagonal by less than its own interval is not evidence of anything.
    The panel underneath carries the bin counts, because a dramatic gap in a bin
    holding nine predictions is noise.
    """
    t = theme_mod.get(theme)
    bins = list(bins if bins is not None else result.bins)
    filled = [b for b in bins if b.count > 0]
    if not filled:
        raise ValueError("no populated bins to plot")
    total = sum(b.count for b in bins)

    with plt.rc_context(_style(t)):
        fig, (ax, ax_count) = plt.subplots(
            2, 1, figsize=(7.2, 6.4), gridspec_kw={"height_ratios": [3.1, 1.0], "hspace": 0.16}, sharex=True
        )

        centres = np.array([(b.lower + b.upper) / 2 for b in filled])
        widths = np.array([b.upper - b.lower for b in filled])
        accs = np.array([b.accuracy for b in filled])
        lows = np.array([b.accuracy - b.ci_low for b in filled])
        highs = np.array([b.ci_high - b.accuracy for b in filled])

        ax.plot(
            [0, 1],
            [0, 1],
            color=t.ink_muted,
            linewidth=1.4,
            linestyle=(0, (5, 4)),
            zorder=2,
            label="perfectly calibrated",
        )
        ax.bar(
            centres,
            accs,
            width=widths * 0.86,
            color=t.series_1,
            edgecolor=t.surface,
            linewidth=1.2,
            zorder=3,
            label="observed accuracy",
        )
        ax.errorbar(
            centres,
            accs,
            yerr=[lows, highs],
            fmt="none",
            ecolor=t.ink_secondary,
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            zorder=4,
        )

        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 1)
        ax.set_ylabel("observed accuracy")
        _titles(
            ax,
            t,
            title or f"Is it calibrated?  {result.task}",
            f"n = {result.n:,}   ECE {result.ece:.3f}   accuracy {result.accuracy:.1%}   "
            f"mean confidence {result.mean_confidence:.1%}",
        )
        ax.legend(loc="upper left", labelcolor=t.ink_secondary)

        # One direct label, on the worst-calibrated bin that carries enough of
        # the traffic to matter -- a dramatic gap over 2% of the data is not the
        # headline, and the reader should not have to hunt for the one that is.
        believable = [b for b in filled if b.count >= max(50, 0.05 * total)]
        if believable:
            worst = max(believable, key=lambda b: abs(b.gap))
            if abs(worst.gap) > 0.01:
                cx = (worst.lower + worst.upper) / 2
                direction = "over" if worst.gap > 0 else "under"
                # Draw the gap itself: bar top up to the diagonal at this bin.
                ax.plot(
                    [cx, cx],
                    [worst.accuracy, cx],
                    color=t.series_2,
                    linewidth=1.8,
                    solid_capstyle="butt",
                    zorder=5,
                )
                on_left = cx > 0.5
                ax.annotate(
                    f"{abs(worst.gap):.0%} {direction}confident here\n"
                    f"(n={worst.count:,}, {worst.count / total:.0%} of decisions)",
                    xy=(cx, (worst.accuracy + cx) / 2),
                    xytext=(-10 if on_left else 10, 0),
                    textcoords="offset points",
                    ha="right" if on_left else "left",
                    va="center",
                    fontsize=9,
                    color=t.ink_secondary,
                    zorder=6,
                )

        shares = np.array([b.count / total for b in filled])
        ax_count.bar(
            centres,
            shares,
            width=widths * 0.86,
            color=t.ink_muted,
            edgecolor=t.surface,
            linewidth=1.2,
        )
        ax_count.set_ylabel("share of\ndecisions")
        ax_count.set_xlabel("confidence reported by the model")
        ax_count.set_ylim(0, max(float(shares.max()) * 1.18, 0.05))
        ax_count.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax_count.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

        for axis in (ax, ax_count):
            axis.grid(axis="y")
            axis.grid(axis="x", visible=False)
            _finish(fig, axis, t, None)

        if result.simulated:
            _watermark(fig, t)
        return _save(fig, Path(out_path))


# ---------------------------------------------------------- risk / coverage


def risk_coverage_plot(
    result: AuditResult,
    out_path: str | Path,
    theme: str = "light",
    title: str | None = None,
) -> Path:
    """Error rate against the share of traffic you let through.

    The deployment question. Read right to left: as you raise the confidence
    threshold you hand back coverage and buy down error. The marked point is the
    recommended gate -- the cheapest threshold whose 95% upper error bound still
    clears the target.
    """
    t = theme_mod.get(theme)
    points: list[CoveragePoint] = sorted(result.coverage, key=lambda p: p.coverage)
    if not points:
        raise ValueError("no coverage points to plot")

    cov = np.array([p.coverage for p in points])
    err = np.array([p.error for p in points])
    err_hi = np.array([p.error_upper for p in points])

    with plt.rc_context(_style(t)):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))

        ax.fill_between(cov, err, err_hi, color=t.series_1, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(cov, err, color=t.series_1, linewidth=2.0, zorder=3, label="error rate on accepted decisions")

        gate = _headline_gate(result.gates)
        if gate is not None and gate.threshold is not None:
            ax.axhline(gate.target_error, color=t.ink_muted, linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)
            ax.annotate(
                f"{gate.target_error:.0%} error budget",
                xy=(0.99, gate.target_error),
                xytext=(0, 6),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                color=t.ink_muted,
            )
            ax.plot(
                [gate.coverage],
                [gate.observed_error],
                marker="o",
                markersize=9,
                linestyle="none",
                color=t.series_2,
                markeredgecolor=t.surface,
                markeredgewidth=2,
                zorder=5,
                label="recommended gate",
            )
            # Keep the callout inside the axes whichever side the gate lands on.
            on_left = gate.coverage < 0.5
            caveat = "" if gate.guaranteed else "\n(provisional: sample too thin to guarantee)"
            ax.annotate(
                f"gate at confidence >= {gate.threshold:.3f}\n"
                f"{gate.coverage:.0%} of traffic automated, {gate.observed_error:.1%} error"
                f"{caveat}",
                xy=(gate.coverage, gate.observed_error),
                xytext=(16 if on_left else -16, 52),
                textcoords="offset points",
                ha="left" if on_left else "right",
                fontsize=9,
                color=t.ink_secondary,
                zorder=6,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(float(err_hi.max()) * 1.18, 0.02))
        ax.set_xlabel("coverage - share of decisions handled automatically")
        ax.set_ylabel("error rate")
        _titles(
            ax,
            t,
            title or f"Where is it safe to automate?  {result.task}",
            f"n = {result.n:,}   AURC {result.aurc:.4f}   shaded band = 95% upper bound on error",
        )
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", labelcolor=t.ink_secondary)
        _finish(fig, ax, t, result)
        return _save(fig, Path(out_path))


def _headline_gate(gates: Sequence[GateRecommendation]) -> GateRecommendation | None:
    """The gate worth drawing: the tightest budget the model actually meets.

    Guaranteed gates win over provisional ones even at a looser budget -- a 1%
    gate that rests on forty predictions is not the recommendation to put on a
    chart. Among equals, the tightest budget is the interesting one; a 10%
    budget the model clears trivially says nothing.
    """
    usable = [g for g in gates if g.threshold is not None and g.coverage > 0]
    if not usable:
        return None
    guaranteed = [g for g in usable if g.guaranteed]
    return min(guaranteed or usable, key=lambda g: g.target_error)


# ------------------------------------------------------- confidence profile


def confidence_histogram(
    result: AuditResult,
    out_path: str | Path,
    theme: str = "light",
    n_bins: int = 25,
    title: str | None = None,
) -> Path:
    """How often the model claims what.

    The gap between the two reference lines is the headline overconfidence: mean
    confidence on the right of accuracy means the model is charging more than it
    delivers, averaged over everything.
    """
    t = theme_mod.get(theme)
    counts = np.zeros(n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for b in result.bins_adaptive:
        if b.count:
            idx = min(int(b.mean_confidence * n_bins), n_bins - 1)
            counts[idx] += b.count
    # Prefer the exact distribution when the caller kept it around.
    raw = result.extra.get("confidences")
    if raw is not None:
        counts, _ = np.histogram(np.asarray(raw, dtype=float), bins=edges)
    shares = counts / max(counts.sum(), 1)

    with plt.rc_context(_style(t)):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        centres = (edges[:-1] + edges[1:]) / 2
        ax.bar(
            centres,
            shares,
            width=(edges[1] - edges[0]) * 0.86,
            color=t.series_1,
            edgecolor=t.surface,
            linewidth=1.0,
        )

        top = max(float(shares.max()) * 1.34, 0.05)

        # The gap between what it claims and what it delivers, shown as an area
        # rather than left for the reader to subtract two numbers.
        lo, hi = sorted((result.accuracy, result.mean_confidence))
        if hi - lo > 0.005:
            ax.axvspan(lo, hi, color=t.series_2, alpha=0.10, linewidth=0, zorder=1)

        markers = [
            (result.accuracy, "accuracy", t.series_2, 0.95),
            (result.mean_confidence, "mean confidence", t.ink_secondary, 0.80),
        ]
        for value, label, colour, height in markers:
            ax.axvline(value, color=colour, linewidth=1.6, linestyle=(0, (5, 4)), zorder=4)
            # Labels are staggered vertically so the two never collide, and are
            # pushed inward near the right edge so neither runs off the axes.
            inward = value > 0.80
            ax.annotate(
                f"{label} {value:.1%}",
                xy=(value, top * height),
                xytext=(-6 if inward else 6, 0),
                textcoords="offset points",
                ha="right" if inward else "left",
                va="center",
                fontsize=9,
                color=t.ink_secondary,
                zorder=5,
            )

        if hi - lo > 0.005:
            sign = "+" if result.overconfidence > 0 else ""
            ax.annotate(
                f"{sign}{result.overconfidence * 100:.1f}pp",
                xy=((lo + hi) / 2, top * 0.64),
                ha="center",
                va="center",
                fontsize=9,
                color=t.ink_secondary,
                zorder=5,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, top)
        ax.set_xlabel("confidence reported by the model")
        ax.set_ylabel("share of decisions")
        _titles(
            ax,
            t,
            title or f"What does it claim?  {result.task}",
            f"n = {result.n:,}   shaded band = the gap between claimed and delivered",
        )
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        _finish(fig, ax, t, result)
        return _save(fig, Path(out_path))


def render_all(
    result: AuditResult,
    out_dir: str | Path,
    themes: Sequence[str] = ("light", "dark"),
    confidences: Sequence[float] | None = None,
) -> list[Path]:
    """Render every chart in every theme."""
    out_dir = Path(out_dir)
    if confidences is not None:
        result.extra["confidences"] = list(confidences)
    written: list[Path] = []
    for th in themes:
        suffix = "" if th == "light" else f".{th}"
        written.append(reliability_diagram(result, out_dir / f"reliability{suffix}.png", theme=th))
        written.append(risk_coverage_plot(result, out_dir / f"risk-coverage{suffix}.png", theme=th))
        written.append(confidence_histogram(result, out_dir / f"confidence{suffix}.png", theme=th))
    return written
