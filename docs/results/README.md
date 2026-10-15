# Results from your completed run

Run folder: `20261015T054141_911671Z`. The timestamp is UTC: this corresponds to 15 October 2026, 13:41:41 in China.

**Status: PASS, 15/15 jobs completed.** GERTAD and four GoalAdapter baselines were evaluated with seeds 17, 29 and 43. Each job used 2,981 physical observations and 38,390 goal queries across 11 conditions.

The program rebuilt observable features, ran fresh inference using the supplied trained weights, and recalculated TEST metrics. Every recorded canonical prediction array matched its frozen reference exactly. Primary and common aggregate metric differences were zero. PASS means reproduction checks passed; it does not mean that every trajectory was classified correctly.

This run used fixed validation thresholds and calibrations. It performed no training, calibration refitting or latency benchmark. The four external endpoints are the study's GoalAdapter adaptations.

## Main comparison

Primary-score metrics; mean +/- sample standard deviation over three seeds. These four metrics use a 0 to 1 scale, with higher values preferred.

| Method | F1 | BA | AUROC | AP |
|---|---:|---:|---:|---:|
| GERTAD | 0.9623 +/- 0.0007 | 0.9653 +/- 0.0018 | 0.9769 +/- 0.0013 | 0.9616 +/- 0.0024 |
| LM-TAD GoalAdapter | 0.9004 +/- 0.0363 | 0.9056 +/- 0.0323 | 0.9715 +/- 0.0108 | 0.9620 +/- 0.0115 |
| CETrajAD GoalAdapter | 0.9459 +/- 0.0035 | 0.9415 +/- 0.0037 | 0.9840 +/- 0.0011 | 0.9811 +/- 0.0006 |
| GM-VSAE GoalAdapter | 0.9274 +/- 0.0108 | 0.9316 +/- 0.0078 | 0.9816 +/- 0.0009 | 0.9744 +/- 0.0031 |
| IDK GoalAdapter | 0.9162 +/- 0.0142 | 0.9190 +/- 0.0168 | 0.9761 +/- 0.0020 | 0.9674 +/- 0.0020 |

GERTAD has the highest mean F1 and BA among these five evaluated endpoints. CETrajAD GoalAdapter has the highest mean AUROC and AP. These are metric-specific comparisons; the sample standard deviations are not confidence intervals.

| Metric | Meaning |
|---|---|
| F1 | Combines precision and recall for classification. |
| BA | Balanced accuracy: averages recall across classes. |
| AUROC | Area under the receiver operating characteristic curve; summarizes score ranking. |
| AP | Average precision; summarizes precision-recall performance. |

## Which files to open

| File | Contents |
|---|---|
| [comparison_mean_sd.csv](comparison_mean_sd.csv) | Five methods, three-seed mean and sample standard deviation for every aggregate metric. Start here. |
| [comparison_by_seed.csv](comparison_by_seed.csv) | One row for each of the 15 method/seed jobs. |
| [classification_by_condition.csv](classification_by_condition.csv) | Classification metrics for each method, seed and condition. |
| [SUMMARY.json](SUMMARY.json) | Machine-readable completion status and aggregate means. |
| [RUN_PLAN.json](RUN_PLAN.json) | Method/seed plan and Python/library versions. |
| `tad_17/`, `tad_29/`, `tad_43/` | GERTAD outputs. The internal directory identifier `tad` is preserved. |
| `lm_*`, `ce_*`, `gm_*`, `idk_*` | Outputs from the four adapted comparison methods. |

Every method/seed directory contains `predictions.npz` (fresh numerical predictions), `primary_metrics.json`, `common_metrics.json`, `trace_audit.json`, and `COMPLETED.json`. The three comparison CSVs use primary-score metrics; common-score metrics remain separate in each method/seed directory.

These files contain predictions and numerical evaluations. This entry point does not export the manuscript figures, train models, or reproduce historical hardware timings.
