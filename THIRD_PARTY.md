# Third-party source identities

| Component | Source | Identity and distribution |
|---|---|---|
| LM-TAD and GM-VSAE supporting code | [jonathankabala/LMTAD](https://github.com/jonathankabala/LMTAD) | Commit `80bb89a8ea108db8f13cb9959826424e9c45f41c`; retained notice in `third_party/LMTAD/LICENSE.txt`. The original trajectory encoder file is obtained locally by `prepare_external.py`. |
| CETrajAD LSTM encoder | [ShuruiCao/comp-ensemble-ad](https://github.com/ShuruiCao/comp-ensemble-ad) | Commit `c10b5e86ab94d16f2878527e7b63d0489ea5777a`; no repository redistribution license was located. The original file is obtained locally from upstream. |
| IDK isolation kernel | [IsolationKernel/Codes](https://github.com/IsolationKernel/Codes) | Commit `0ccee89d8e35c9d370fa6cc6dc5785fc4583a074`; original non-commercial notice retained in `third_party/IDK/LICENSE.txt`. |

`config/external_sources.json` records the pinned URLs and SHA-256 values.
Setup preserves the original frozen source bytes, including their original
notices and references. Line-ending normalization is accepted only when it
reproduces the exact expected hash. These locally obtained files are ignored
by Git; no original source attribution is rewritten.

The GM-VSAE representation is the pinned research port used in the study.
All external comparison endpoints retain the study's trained GoalAdapters;
they are not presented as the original papers' unmodified evaluation results.
No new blanket license is applied to third-party code, data or model states.
