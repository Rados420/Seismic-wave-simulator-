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

Metaparameters live in `fasim/gen/defaults.yaml`
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

## Experiments

`notebooks/01_factor_influence.ipynb`: one-factor-at-a-time influence of wall, curvature, roughness,
EDZ, texture, hammer direction, domain size, grid density and 2-D vs 3-D on the simulated gathers,
with conclusions at the end. Helpers in `fasim/gen/experiments.py` (`run_variant` caches each
simulation in `runs/experiments/`, `compare` and `fa_visibility` compute the measures).
Running it from scratch takes about 45 min on an M4 CPU; with the cache it only re-plots.

## GPU

Wave propagation runs on a CUDA GPU through Deepwave; model building, eikonal labels and file
writing stay on the CPU (a few seconds per draw). `--device auto` (default, `sim.device`) picks the
GPU when torch sees one. Shots are batched into as few Deepwave calls as fit in GPU memory, and
explosive and impact shots share the same calls (`sim.combine_sources`). On out-of-memory the batch
is halved and retried.

Server set-up (Linux, NVIDIA driver recent enough for the card; Blackwell needs CUDA 12.8 or newer):

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e .
.venv/bin/python generate.py gpu-info                               # device, memory, tiny GPU-vs-CPU check
.venv/bin/python generate.py bench --seed 0 --dims 2 --compare-cpu 2  # s/shot, peak memory, CPU agreement
.venv/bin/python generate.py bench --seed 0 --dims 3 --compare-cpu 1
```

Use the `cu128` index or a newer one that matches the installed driver. If `gpu-info` reports
"no kernel image", Deepwave's prebuilt CUDA kernels do not cover the card and Deepwave has to be
built from source on the server.

Tuning (`sim:` in `fasim/gen/defaults.yaml`): `bench` prints the measured memory per shot; if it is
far from the estimate, adjust `memory_safety`. `chunk` fixes the shots per call. Small 2-D grids
do not fill a GPU on their own: running two or three `generate.py run` processes with disjoint
`--seed-range` values on the same GPU raises throughput; they can share one output directory.
