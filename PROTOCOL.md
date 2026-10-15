# Frozen evaluation protocol

The default experiment evaluates five methods and three fixed seeds on the
same controlled TEST population. It preserves the final shared-latent GERTAD
configuration and raw equal-weight executable scorer. The external methods
retain their trained common GoalAdapters and original native representation
dependencies.

All thresholds and logistic probability calibrations are fixed from the
recorded validation SMOKE conditions: `clean`, `gps_30`, `sampling_4`, and
`outage_30`. The program performs no fitting, threshold search, model selection
or checkpoint replacement.

The current selection was frozen before the final quality/runtime evaluation.
Earlier development involved historical TEST reuse, as recorded in the study.
This is a reproduction of that frozen benchmark, with its stated development
history. It does not constitute a new independent validation population.

Classification uses equal-user macro metrics within each condition and equal
conditions for the aggregate. Average precision uses continuous scores and
eligible two-class users. Calibration errors use pooled task rows within a
condition. Opposite-label goal pairs are averaged within observation, within
user, then across eligible users. Recovery and ordering retain their original
scope and unique observation/category-pair definitions.

Primary external endpoints use learned GoalAdapter logits. The separately
calibrated common-score diagnostic applies the executable scorer to recovered
events. Those two endpoints remain separate in the output files.

The program saves fresh predictions, compares them with frozen output arrays,
recomputes primary and common metrics, and checks every aggregate against the
saved numerical reference. Integer IDs, thresholds and operating decisions
must match exactly. Floating prediction arrays use `atol=1e-6, rtol=1e-6`;
aggregate metrics use an absolute tolerance of `1e-12`. Per-array differences
are recorded in each `COMPLETED.json`.

Historical timing and TRAIN-only component studies retain their original
experimental roles. Their model training and hardware timing are not executed
by this frozen TEST evaluator. No new confidence intervals are generated.
