"""Command line: ``run``, ``analyze``, ``audit``, ``probe``.

    python -m jevcal audit examples/tasks/sentiment.json --provider gateway
    python -m jevcal audit examples/tasks/sentiment.json --provider simulated
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import report as report_mod
from .analysis import DEFAULT_TARGETS, analyse
from .providers import build_provider
from .providers.base import ProviderError
from .runner import align, read_predictions, run_task
from .types import load_task


def _add_task_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("task", help="path to a task spec JSON file")
    p.add_argument("--limit", type=int, default=None, help="use at most N examples")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="shuffle before --limit, for a random subset instead of the first N")


def _add_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="gateway", choices=("gateway", "simulated"),
                   help="gateway = the real model via its evaluation API; "
                        "simulated = offline synthetic data, for testing the pipeline")
    p.add_argument("--surface", default="gateway", choices=("gateway", "typesafe"),
                   help="which endpoint serves Jev: 'gateway' = Vercel AI Gateway "
                        "/v1/evaluate (generally available); 'typesafe' = TypeSafe's own "
                        "/v1/systemone (waitlisted; check your agreement before publishing)")
    p.add_argument("--model", default="", help="model id; defaults to the surface's own default")
    p.add_argument("--probability-source", default="auto",
                   choices=("auto", "probabilities", "probability"),
                   help="which response field carries the distribution; 'auto' picks the one "
                        "matching the task kind, which is almost always right. Run `probe` first")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--no-resume", action="store_true", help="start the results file from scratch")
    p.add_argument("--keep-raw", action="store_true", help="store full response bodies (large)")
    p.add_argument("--sim-temperature", type=float, default=0.55,
                   help="simulator only: < 1 makes the fake model overconfident, 1.0 perfect")
    p.add_argument("--sim-bias", type=float, default=0.0, help="simulator only: constant logit shift")
    p.add_argument("--sim-seed", type=int, default=0)


def _add_analysis_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--targets", type=float, nargs="*", default=list(DEFAULT_TARGETS),
                   help="error budgets to recommend confidence gates for")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--themes", nargs="*", default=["light", "dark"], choices=["light", "dark"])
    p.add_argument("--no-charts", action="store_true")


def _load(args: argparse.Namespace):
    task = load_task(args.task)
    if args.limit:
        task = task.subset(args.limit, seed=args.sample_seed)
    return task


def _make_provider(args: argparse.Namespace):
    return build_provider(
        args.provider,
        model=args.model,
        surface=args.surface,
        probability_source=args.probability_source,
        keep_raw=args.keep_raw,
        temperature=args.sim_temperature,
        bias=args.sim_bias,
        seed=args.sim_seed,
    )


def cmd_run(args: argparse.Namespace) -> int:
    task = _load(args)
    provider = _make_provider(args)
    out = Path(args.out or f"runs/{task.name}.{args.provider}.jsonl")
    print(f"running {len(task.examples):,} {task.kind} decisions -> {out}", file=sys.stderr)
    meta = run_task(
        task,
        provider,
        out,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        keep_raw=args.keep_raw,
    )
    meta.save(out.with_suffix(".meta.json"))
    print(f"done: {meta.n_ok:,} ok, {meta.n_failed:,} failed, {meta.wall_seconds:.1f}s", file=sys.stderr)
    if meta.n_failed:
        print(f"note: {meta.n_failed} failed rows are excluded from the analysis", file=sys.stderr)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    results_path = Path(args.results)
    examples, predictions = align(task, list(read_predictions(results_path)))
    if not examples:
        print(f"no usable predictions in {results_path}", file=sys.stderr)
        return 1

    meta_path = results_path.with_suffix(".meta.json")
    run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    result = analyse(
        task,
        examples,
        predictions,
        n_bins=args.bins,
        targets=args.targets,
        bootstrap_resamples=args.bootstrap,
        simulated=bool(run_meta.get("simulated")),
    )

    out_dir = Path(args.out or "report")
    charts: list[Path] = []
    if not args.no_charts:
        from .plots import render_all

        charts = render_all(
            result,
            out_dir,
            themes=args.themes,
            confidences=[p.confidence for p in predictions],
        )
    path = report_mod.write_report(result, out_dir, charts, run_meta)

    print(report_mod.verdict(result))
    print(f"\nwrote {path} and {out_dir / 'results.json'}"
          + (f" plus {len(charts)} charts" if charts else ""), file=sys.stderr)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """run + analyze, which is what you want almost every time."""
    task = _load(args)
    provider = _make_provider(args)
    out = Path(args.out or f"runs/{task.name}.{args.provider}.jsonl")
    print(f"running {len(task.examples):,} {task.kind} decisions -> {out}", file=sys.stderr)
    meta = run_task(
        task, provider, out,
        concurrency=args.concurrency, resume=not args.no_resume, keep_raw=args.keep_raw,
    )
    meta.save(out.with_suffix(".meta.json"))

    examples, predictions = align(task, list(read_predictions(out)))
    if not examples:
        print("every request failed; nothing to analyse", file=sys.stderr)
        return 1

    result = analyse(
        task, examples, predictions,
        n_bins=args.bins, targets=args.targets,
        bootstrap_resamples=args.bootstrap, simulated=meta.simulated,
    )
    report_dir = Path(args.report_dir)
    charts: list[Path] = []
    if not args.no_charts:
        from .plots import render_all

        charts = render_all(result, report_dir, themes=args.themes,
                            confidences=[p.confidence for p in predictions])
    path = report_mod.write_report(result, report_dir, charts, asdict(meta))

    print()
    print(report_mod.verdict(result))
    print(f"\nwrote {path}", file=sys.stderr)
    return 0


def cmd_make_task(args: argparse.Namespace) -> int:
    """Turn a CSV or JSONL of labelled rows into a task spec."""
    from .datasets import from_csv, from_jsonl, save_task

    src = Path(args.data)
    loader = from_csv if src.suffix.lower() == ".csv" else from_jsonl
    label_map = json.loads(args.label_map) if args.label_map else None
    task = loader(
        src,
        name=args.name or src.stem,
        question=args.question,
        kind=args.kind,
        labels=args.labels or None,
        text_field=args.text_field,
        label_field=args.label_field,
        id_field=args.id_field,
        instructions=args.instructions,
        label_map=label_map,
    )
    spec = Path(args.out)
    save_task(task, spec, spec.with_suffix(".jsonl"))
    print(
        f"wrote {spec}: {len(task.examples):,} {task.kind} examples over "
        f"{len(task.labels)} labels ({', '.join(task.labels[:8])}"
        + (" ..." if len(task.labels) > 8 else "")
        + ")",
        file=sys.stderr,
    )
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Send one request and print the raw response.

    Do this once before a long run. It shows exactly where your gateway puts the
    probability distribution, so you can pin ``--probability-source`` instead of
    discovering halfway through 10,000 requests that the extraction silently
    picked a path you did not intend.
    """
    task = _load(args)
    if args.provider != "gateway":
        print("probe is only meaningful for the gateway provider", file=sys.stderr)
        return 2
    from .providers.gateway import GatewayProvider, extract_decision

    provider: GatewayProvider = build_provider(
        "gateway",
        model=args.model,
        surface=args.surface,
        probability_source=args.probability_source,
        keep_raw=True,
    )  # type: ignore[assignment]
    example = task.examples[0]
    print("--- request ---")
    print(json.dumps(provider.build_payload(task, example), indent=2))
    body = provider.raw_call(task, example)
    print("--- response ---")
    print(json.dumps(body, indent=2))
    print(f"--- endpoint: {provider.endpoint} ---")
    print("--- extraction ---")
    for source in ("probabilities", "probability"):
        try:
            predicted, dist = extract_decision(body, task, source)  # type: ignore[arg-type]
            print(f"{source}: {predicted!r} at {dist[predicted]:.4f}  {dist}")
        except ProviderError as exc:
            print(f"{source}: unavailable ({exc})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevcal",
        description="Independent calibration audit for typed-decision models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="send decisions and record them")
    _add_task_args(p_run)
    _add_provider_args(p_run)
    p_run.add_argument("--out", default=None, help="results JSONL path")
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="score a results file and write the report")
    p_an.add_argument("task")
    p_an.add_argument("results", help="results JSONL written by `run`")
    _add_analysis_args(p_an)
    p_an.add_argument("--out", default="report", help="report directory")
    p_an.set_defaults(func=cmd_analyze)

    p_audit = sub.add_parser("audit", help="run and analyze in one go")
    _add_task_args(p_audit)
    _add_provider_args(p_audit)
    _add_analysis_args(p_audit)
    p_audit.add_argument("--out", default=None, help="results JSONL path")
    p_audit.add_argument("--report-dir", default="report")
    p_audit.set_defaults(func=cmd_audit)

    p_make = sub.add_parser("make-task", help="build a task spec from a CSV or JSONL file")
    p_make.add_argument("data", help="CSV or JSONL with one labelled row per line")
    p_make.add_argument("--out", required=True, help="task spec path to write")
    p_make.add_argument("--question", required=True, help="the typed question to ask about every row")
    p_make.add_argument("--kind", default="choice", choices=("choice", "score", "noul"))
    p_make.add_argument("--labels", nargs="*", default=None,
                        help="permitted answers; required for `score`, where the order is the rubric")
    p_make.add_argument("--name", default=None)
    p_make.add_argument("--text-field", default="text")
    p_make.add_argument("--label-field", default="label")
    p_make.add_argument("--id-field", default="id")
    p_make.add_argument("--instructions", default="", help="extra framing shown to the model")
    p_make.add_argument("--label-map", default=None,
                        help='JSON object renaming raw labels, e.g. \'{"0":"negative","1":"positive"}\'')
    p_make.set_defaults(func=cmd_make_task)

    p_probe = sub.add_parser("probe", help="send one request and show the raw response")
    _add_task_args(p_probe)
    _add_provider_args(p_probe)
    p_probe.set_defaults(func=cmd_probe)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
