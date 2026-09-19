# Sample output (synthetic)

What `python -m jevcal audit` produces, so you can see the shape of the result
without running anything.

**These numbers are not a measurement of any model.** They come from the
built-in simulator, configured to be overconfident on purpose (temperature 0.55,
which distorts the reported probability in logit space). The charts carry a
`SIMULATED DATA` watermark for the same reason.

Reproduce it exactly:

```bash
python -m jevcal audit examples/tasks/demo-choice.json --provider simulated
```

The one thing worth reading here is that the audit recovers the distortion it
was given: a fitted slope of 0.51 against an injected 0.55. That is the check
that makes the machinery trustworthy enough to point at a real model.

See [`report.md`](report.md).
