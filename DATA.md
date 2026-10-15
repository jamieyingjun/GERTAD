# Data sources and required inputs

## GeoLife trajectory source

GeoLife is available from the [official Microsoft download page](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
and [dataset user guide](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/).
The original GPS archive and the controlled trajectory coordinates are not
redistributed in this repository. The GeoLife 1.3 user guide includes the
Microsoft Research License Agreement for non-commercial use and restricts
redistribution, including derivatives.

Download **`Geolife Trajectories 1.3.zip`** from that official page and place
the unchanged archive at `data/geolife/Geolife Trajectories 1.3.zip` inside
the project. Double-click `run_experiments.cmd`, or run
`python run_experiments.py` in the project folder. The entry prepares data
and then runs the frozen TEST comparison. Full download instructions and
the directory layout are in [README.md](README.md).

The unchanged evaluation code uses the locally reconstructed `data/worlds.json.gz`:

```text
SHA-256: 01c196ae0e75d039c21d3e2082887d89633c895661d8f7d4a9a8d6f8c085e1c9
```

`prepare_external.py` rebuilds this file from 190 original `.plt` files,
containing 271 frozen windows of 50 points each. The recipe in
`config/geolife_windows.json` contains only source-file identities and
zero-based point-row indices after the six PLT header lines. It contains no
GPS coordinate or timestamp arrays. Indices preserve the original selected
rows, including the window whose retained rows are not contiguous.

Each original trajectory file is checked before use. Latitude
and longitude are converted to the original binary32 precision; date and
time fields are interpreted as UTC Unix seconds. The supplied POI entity
arrays and context order are preserved. The final compressed input must
match the complete frozen SHA-256 above before it is written or evaluated.
This reconstruction was verified against the original local input with an
exact byte match. Preparation does not choose new windows, generate new
labels or call the map service.

An already extracted GeoLife folder is also supported at
`data/geolife/Geolife Trajectories 1.3/Data/`. Advanced users can select a
different ZIP, extracted dataset folder or `Data` folder with
`python prepare_external.py --geolife PATH`. An existing prepared file can
be checked and copied with `--worlds PATH`. Raw data and prepared coordinates
remain local and are ignored by Git.

## POI candidates

`map_candidates.json.gz` contains the unchanged `context_id` and `entities`
fields from those controlled contexts, separated from trajectory coordinates
and timestamps. POI candidates were obtained through the **AMap POI Search API**.
The provider retains its rights and applicable terms. API attribution is not
a new redistribution or reuse license. See the [AMap API documentation](https://lbs.amap.com/api/webservice/guide/api/search)
and [service agreement](https://lbs.amap.com/pages/terms/).

## Controlled evaluation metadata

`observations.json.gz`, `tasks.json.gz`, `goals.json`, `schema.json` and
`order_truth.npz` contain the controlled benchmark definitions, labels and
separate diagnostic supervision. Original category IDs and entity ordering
are preserved, with English display labels stored separately.
The goals and event/violation labels describe controlled simulations;
they are not historical records of participants' goal completion.

Reference predictions and metrics are opened only after fresh predictions
are saved; file-integrity hashing can read their bytes earlier. Goals and
evaluation truth do not enter physical recovery as labels or target events.

## Availability scope

This public repository provides source, frozen weights, provider-derived POI
candidates, controlled metadata and recorded results. Full frozen TEST
evaluation requires the user's official GeoLife download. The single entry
then reconstructs the frozen trajectory file and retrieves the pinned
external encoder sources automatically. The original full local reproduction
package remains separate from this public snapshot.
