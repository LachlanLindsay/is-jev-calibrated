# Calibration audit: `clinc150-inscope`

*Model:* `jev-1.13.0` &nbsp;&nbsp; *Task type:* `choice` &nbsp;&nbsp; *Examples:* 22,500 &nbsp;&nbsp; *Generated:* 2026-09-20 00:24 UTC

## Verdict

On 22,500 labelled examples from `clinc150-inscope`, this model is **usably calibrated** (ECE 0.021, 95% CI 0.019-0.024) and overconfident by 2.1% on average. Its probabilities are too extreme: a logistic refit pulls them toward the middle (fitted slope 0.28, where 1.00 is perfect). Separately from calibration, the confidence ranks its own errors at AUROC 0.851, which is what a confidence *gate* actually relies on.

## Headline numbers

| metric | value | what it means |
| --- | --- | --- |
| accuracy | 92.63% | share of decisions that matched the label |
| majority-class baseline | 0.67% | accuracy of always answering the commonest label |
| mean confidence | 94.71% | what the model claimed, on average |
| overconfidence | +2.08% | mean confidence minus accuracy; positive = charges more than it delivers |
| ECE (15 equal-width bins) | 0.0209 | average distance between claimed and delivered |
| ECE 95% CI | 0.0188 - 0.0245 | 1,000-resample bootstrap |
| ECE (15 equal-mass bins) | 0.0208 | same idea, bins holding equal counts |
| MCE (bins with n >= 30) | 0.0714 | worst single band -- what a gate trips over |
| Brier score | 0.0529 | proper score over the top label; lower is better |
| - reliability | 0.0005 | the miscalibration part; 0 is perfect |
| - resolution | 0.0146 | how far it separates cases; higher is better |
| - uncertainty | 0.0683 | irreducible difficulty of the task |
| AUROC of confidence vs. correctness | 0.8509 | can confidence rank its own errors? 0.5 = no signal |
| NLL of the true label | 0.6713 | punishes confident mistakes hardest |
| multiclass Brier | 0.1174 | over the whole distribution, not just the winner |
| logistic refit slope | 0.2833 | 1.00 = right shape; < 1 = too extreme |
| logistic refit intercept | +0.8427 | 0.00 = no constant bias |
| latency p50 / p95 | 160 ms / 284 ms | end to end, including network |
| total cost | $2.3532 | $0.1046 per 1,000 decisions |

## Is the number honest?

![Reliability diagram](reliability.png)

Every bar above is also a row here, with its 95% Wilson interval:

| confidence bin | n | mean confidence | observed accuracy | 95% CI | gap |
| --- | --- | --- | --- | --- | --- |
| 0.13-0.20 | 4 | 0.181 | 0.000 | 0.000 - 0.490 | +0.181 |
| 0.20-0.27 | 18 | 0.239 | 0.111 | 0.031 - 0.328 | +0.128 |
| 0.27-0.33 | 34 | 0.302 | 0.324 | 0.191 - 0.492 | -0.021 |
| 0.33-0.40 | 80 | 0.371 | 0.300 | 0.211 - 0.408 | +0.071 |
| 0.40-0.47 | 124 | 0.436 | 0.371 | 0.291 - 0.459 | +0.065 |
| 0.47-0.53 | 281 | 0.504 | 0.452 | 0.395 - 0.510 | +0.052 |
| 0.53-0.60 | 394 | 0.572 | 0.574 | 0.524 - 0.622 | -0.002 |
| 0.60-0.67 | 347 | 0.636 | 0.634 | 0.582 - 0.683 | +0.002 |
| 0.67-0.73 | 484 | 0.701 | 0.653 | 0.609 - 0.694 | +0.048 |
| 0.73-0.80 | 509 | 0.771 | 0.719 | 0.678 - 0.756 | +0.052 |
| 0.80-0.87 | 632 | 0.838 | 0.807 | 0.774 - 0.836 | +0.031 |
| 0.87-0.93 | 1,283 | 0.904 | 0.891 | 0.873 - 0.907 | +0.013 |
| 0.93-1.00 | 18,310 | 0.994 | 0.975 | 0.973 - 0.977 | +0.019 |

## Where is it safe to automate?

![Risk-coverage curve](risk-coverage.png)

| error budget | gate at confidence | coverage | observed error | 95% upper bound | meets budget |
| --- | --- | --- | --- | --- | --- |
| 1% | not reachable | - | - | - | no threshold reaches 1.0% error; the most selective usable slice still errs at 1.6% |
| 5% | >= 0.6900 | 93.6% | 4.66% | 4.96% | yes |
| 10% | >= 0.1414 | 100.0% | 7.37% | 7.72% | yes |

A gate is only recommended when the *upper* bound on its error clears the budget, not the point estimate -- picking the threshold that happened to look best on this sample is how a gate that tested at 1% ships at 4%.

## What does it claim?

![Confidence histogram](confidence.png)

## How this was measured

- Task type `choice` over 150 labels: `accept_reservations`, `account_blocked`, `alarm`, `application_status`, `apr`, `are_you_a_bot`, `balance`, `bill_balance`, `bill_due`, `book_flight`, `book_hotel`, `calculator` ...
- Confidence is the probability the model assigned to the label it picked. Nothing here asks a model to write a confidence number into its own output; a self-reported number is a different object from a calibrated distribution.
- ECE is reported with both equal-width bins (the readable version) and equal-mass bins (the version that does not let a near-empty bin swing the result).
- Intervals on bins are Wilson score intervals; the interval on ECE is a 1,000-resample percentile bootstrap.
- Run: provider `gateway`, 0 failed request(s) excluded, 0s wall clock.

## What this does not show

- Calibration measured on one dataset is calibration on that dataset. A model calibrated on product reviews can be badly calibrated on medical text.
- Bounded context matters: every example here fits comfortably in context. Long or noisy states are a separate experiment.
- Type-safe output is not correct output. Everything counted as an error below was a perfectly valid value that happened to be wrong.
- Confidence gating trades coverage for accuracy. The escalated tail still has to go somewhere, and that somewhere has its own cost.

