# Recorded verification

The author's run `20261015T054141_911671Z` completed all 15 jobs with Python
3.12.4 and the pinned CPU dependencies. Every canonical prediction array
matched the frozen reference exactly. The maximum absolute difference in
primary and common aggregate metrics was zero. The actual terminal screenshot,
unaltered CSV tables and per-job checks are in `docs/`.

An earlier complete local-bundle validation used an independent directory and
a newly installed dependency environment, with network and original-project
access blocked during evaluation. Neither block was triggered. Those records
remain in the author's private audit materials.

The public snapshot excludes GeoLife-derived trajectory coordinates and two
original external encoder files. The single entry reconstructs the frozen
trajectory input after the official GeoLife ZIP is placed at the documented
local path, and retrieves the pinned encoder sources. Reconstruction from
the original 190 trajectory files reproduces all 271 TEST contexts and the
complete compressed input SHA-256 exactly. The frozen row-selection recipe
contains source hashes and indices rather than coordinate arrays.

A separate independent directory started with the distributed files and the
official Microsoft ZIP at the README path. The single command
`python run_experiments.py` reconstructed the input, retrieved both encoders
and completed all 15 jobs. Every canonical prediction array matched exactly;
the maximum primary/common aggregate metric difference was zero. The compact
record is [docs/input-preparation.json](docs/input-preparation.json).

The recovery model, goal scorer, experiment settings, thresholds, calibration
values and supplied checkpoint tensors retain the accepted frozen versions.
No training, calibration refit, timing benchmark, new intervals or manuscript
figure generation is performed by publication preparation.
