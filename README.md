# fasim — prototype tunnel-seismic simulator (2-D plan view)

Hand-tunable prototype for generating synthetic first-arrival training data
for the tunnel-seismic picking project. It is deliberately small: paint a
vp/vs/rho model with a tunnel cavity, shoot Deepwave's elastic propagator,
compute two labels per trace (eikonal traveltime and observable onset), save
everything to HDF5 and make figures. Everything is a knob in one YAML file.

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e .
```

Runs on the CPU (Apple Silicon wheel of Deepwave 0.0.27). Roughly 1 s/shot at
dx = 0.2 m and 8 s/shot at dx = 0.1 m on an M4; the RTX server will be ~10-50x faster.

## Run

```bash
.venv/bin/python run_experiment.py --config configs/homogeneous.yaml
.venv/bin/python run_experiment.py --config configs/homogeneous_hires.yaml --no-tunnel --name sanity_notunnel_hires  # the kept sanity run
.venv/bin/python run_experiment.py --config configs/homogeneous.yaml --source pressure           # explosive
.venv/bin/python run_experiment.py --config configs/homogeneous.yaml --shots 8 --dx 0.4 --freq 300  # quick look
.venv/bin/python plot_configs.py --field vp                                                      # vp maps of all configs
```

Output goes to `runs/<name>/`: `config.yaml` (resolved config), `run.h5`,
`model_vp.png`, `gather_<RCV>.png` (receiver gathers, all three components,
blue = eikonal + wavelet onset, red = picked onset), `delay_vs_offset.png`,
`snapshots_vx.png`.

CLI overrides: `--shots N --dx --freq --source force|pressure --no-tunnel
--picker --threshold-ref trace|gather|window --rel-threshold --component y|x|p|vec --record-ms`.

Re-picking without re-simulating: `python repick.py runs/homogeneous --threshold-ref gather --suffix _gatherref`
(add `--write-h5` to overwrite the stored labels) or interactively in `notebooks/00_explore_run.ipynb`.

## Dataset generator V1 (`fasim/gen`, `generate.py`)

Implements `docs/dataset_design_v1.md`. Metaparameters live in `fasim/gen/defaults.yaml`
(fixed values or distributions); override with `--meta my.yaml` or `--set key.path=value`.

```bash
.venv/bin/python generate.py sample --seed 0 1 2            # drawn parameters, no simulation
.venv/bin/python generate.py paint  --seed 2 --dims 2       # build and plot both wall variants, no simulation
.venv/bin/python generate.py paint  --seed 2 --dims 3 --set grid.dx_3d=0.4
.venv/bin/python generate.py run    --seeds 2 --dims 2      # simulate wall x source, write runs/gen/2d/draw_000002.h5
.venv/bin/python generate.py compare runs/gen/2d/draw_000002.h5 --rcv 0 1 --comp n
```

Quick, cheap test runs: `--set grid.dx_2d=0.2 --set acquisition.n_shots=6 --set record.length_ms=20`.

Per draw, one HDF5 file: `params_json` (every drawn value and field seed, enough to rebuild the
models), `meta_json`, `geometry/` (receiver and shot positions, impact directions, offsets),
and `wall_<concrete|bare>/` with the models (float16), the eikonal traveltimes and one group per
source (`data[shot, rcv, comp, t]` with comps p, s, n[, v], `t`, `wavelet`). `index.jsonl` in the
output directory lists the draws with their main parameters for slicing.
