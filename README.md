# fasim

Simulates seismic recordings in a tunnel. The output is training data for a model that picks
**first arrivals**: the time at which the first wave from a shot reaches a sensor.

## Concepts

- **Draw**: one simulated example. It has a random tunnel, rock, sensor line and shot line.
  Each draw has an integer **seed**, and the same seed always gives the same draw.
- Each draw is simulated **4 times**: for 2 wall types (concrete-lined or bare rock) and
  2 sources (an explosive charge in a borehole, or a hammer blow on the wall).
- **2-D or 3-D**: 2-D is a horizontal slice through the rock at sensor height and is the default.
  3-D is much slower.
- **Label**: the true first-arrival time for every shot–sensor pair. It is computed from the rock
  model, not picked from the waveforms.

## Install

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e .
```

On an NVIDIA GPU server, install the CUDA build of torch first. Blackwell cards need CUDA 12.8 or newer:

```bash
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
```

## Try it

```bash
.venv/bin/python generate.py sample --seed 0                                   # print the random parameters of draw 0
.venv/bin/python generate.py paint --seed 0                                    # plot its rock models (PNG)
.venv/bin/python generate.py run --seeds 0 --set grid.dx_2d=0.2 --set acquisition.n_shots=4   # quick small simulation
.venv/bin/python generate.py compare runs/gen/2d/draw_000000.h5 --rcv 0        # plot the 4 versions at sensor 0
```

Output goes to `runs/gen/2d/` (`runs/gen/3d/` with `--dims 3`).

## Generate a dataset

```bash
.venv/bin/python generate.py run --seed-range 0 1000 --no-figs --skip-existing   # draws 0 to 999
```

- `--skip-existing` skips draws that are already done, so an interrupted run can be restarted
  with the same command.
- `--device cpu|cuda` sets the device. By default the GPU is used if one is found.
- `--workers N` sets how many CPU processes prepare the next draws while the current one is simulated.
  If the log shows "propagation busy" well below 100 %, increase N.
- Failed draws are listed in `failures.jsonl` and the run continues.

To check a GPU server:

```bash
.venv/bin/python generate.py gpu-info                                  # is the GPU visible and working
.venv/bin/python generate.py bench --seed 0 --compare-cpu 2            # seconds per shot, GPU vs CPU result
```

## Changing parameters

All parameters are in [fasim/gen/defaults.yaml](fasim/gen/defaults.yaml). Each one is either a fixed
value or a random range. You can override them for one run:

```bash
--set record.length_ms=20               # one value
--meta my_settings.yaml                 # a YAML file with only the values you change
```

## Output file

One HDF5 file per draw, `draw_<seed>.h5`. Times are in seconds and positions in metres.

| Path | Content |
|---|---|
| `geometry/rcv_pos`, `geometry/shot_pos` | sensor and shot positions |
| `wall_<concrete\|bare>/<explosive\|impact>/data` | recordings, `[shot, sensor, component, time]` |
| `wall_…/<source>/t` | time axis |
| `wall_…/eikonal` | first-arrival label, `[shot, sensor]` |
| `wall_…/vp`, `vs`, `rho` | rock model: P and S velocity, density |

Components: `p` pressure; `s`, `n` (and `v` in 3-D) particle velocity along the tunnel,
into the rock and vertically. `index.jsonl` in the output folder lists all draws with their
main parameters.

## Code

| Path | Content |
|---|---|
| `generate.py` | command line |
| `fasim/gen/sampler.py` | draws the random parameters |
| `fasim/gen/build.py` | builds the tunnel and rock models |
| `fasim/gen/simulate.py` | wave simulation (Deepwave) and labels |
| `fasim/gen/pipeline.py`, `device.py` | parallel runs and GPU handling |
| `notebooks/` | experiments on which parameters change the data |
| `run_experiment.py`, `configs/`, other `fasim/*.py` | older single-scenario prototype |
