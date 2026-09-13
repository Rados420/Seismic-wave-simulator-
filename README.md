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

## Coordinate frame

* dim 0 = `z_t` along the tunnel (face direction positive), dim 1 = `x_t`
  across the tunnel (0 = centreline). Deepwave's `y` is our dim 0, its `x` our dim 1.
* Tunnel = slot `|x_t| < half_width`, running out through the back boundary
  up to `z_face`. Beyond the face is rock. The EDZ is an annulus of
  `edz_thickness` around the cavity with vp, vs × `edz_factor`, rho × `edz_rho_factor`.
* Receivers sit `embed` metres into the wall at `x_t = ±(half_width + embed)`;
  shots sit `shot_embed` into the `shot_side` wall. Defaults mirror the Amberg
  TSP layout (2 receivers per wall at z = 2 and 7 m, 24 shots from z = 22 m every 1.5 m).
* Cavity cells have vp = vs = rho = 0 → Deepwave's improved vacuum method
  makes the wall a free surface.
* Components recorded per receiver: `y` (velocity along tunnel), `x` (velocity
  across tunnel), `p` (pressure, i.e. isotropic stress; P-waves only).
* Force direction: 0° = wall normal pointing into the rock (hammer blow), 90° = along the tunnel.

## Time axis and labels

* Output `dt` = 20.833 µs (48 kHz) like the field data. Deepwave sub-steps
  internally to satisfy CFL.
* `t = 0` is the **peak of the source Ricker**. Deepwave's half-step stagger
  (forces and velocities at (k-0.5)dt, pressure at k dt) is handled in
  `propagate.time_axis`; the velocity axis is stored, pressure samples are
  dt/2 later than the axis says when the source is a force.
* `eikonal[shot, rcv, comp]`: fast-marching traveltime on the same vp grid
  with the cavity masked, sampled at the actual staggered node of each
  component (`rcv_actual`). Verified equal to offset/vp within 2 µs without a cavity.
* `onset[shot, rcv, 0:3]` = picks on y, x, p; `onset[..., 3]` = pick on the
  configured `labels.component`. Picker is pluggable (`fasim.onset.PICKERS`):
  `amplitude_threshold` (default, causal), `envelope_threshold` (Hilbert
  envelope — **non-causal**: the tails of the strong later S / guided phases leak
  backwards to the P time, do not use it as a label on clean synthetics),
  `sta_lta`, `aic`.
* `wavelet_onset` = the same picker applied to the source wavelet, relative to
  its peak (−1.34 ms for a 600 Hz Ricker at 2 %, −0.67 ms at 1.2 kHz).
* The search window starts `search_before_ms` before the *expected first
  energy* `eikonal + wavelet_onset`. A warning is printed when picks sit at the
  window start (means energy before the expected arrival — widen the window or
  suspect the picker).
* Observability floor: a reference amplitude below −80 dB re. the strongest
  trace of the gather (same component) gives a NaN label. In the 2-D plan view
  this makes all opposite-wall picks NaN (see findings).
* `delay = onset - (eikonal + wavelet_onset)` — the **emergence delay**. It is
  ~0 for a clean straight-ray arrival and grows when the first-arriving P is
  weak (grazing incidence along the cavity wall, radiation-pattern null of the
  hammer force, EDZ refraction).
* `threshold_ref` decides what "relative threshold" means:
  `window` = fraction of the max in `ref_window_ms` after the search start
  (measures P-wave *shape* emergence); `gather` = fraction of the gather max
  (mimics a fixed noise floor: weak P is simply not seen and the pick jumps to
  the next phase); `trace` = fraction of the trace max (dominated by S /
  guided waves — usually not what you want).
* Known artifact: the first ~5 samples of every trace carry a small decaying
  transient (−54 dB re. the P amplitude) from Deepwave's source resampling.
  Harmless for the labels (floor guard) but visible if you normalise per sample.

## Amplitude post-processing (`post:` section, `--q`, `--spreading`)

The elastic solver is lossless and 2-D, so raw traces do not die out the way
field records do. Three proxies, all optional and applied after propagation
(`fasim/postprocess.py`); `repick.py --q 100 --spreading 0.5 --suffix _q100_s05`
applies them to a saved raw run and writes a processed copy next to it:

* `post.q`: constant-Q intrinsic attenuation, exact time-variant filter
  exp(-pi f t / Q) with t = propagation time. Hard rock Q ~ 100-300 at kHz,
  fractured / EDZ rock lower. Verified against the analytic damping to 4 digits.
* `post.spreading_exponent` p: extra (t / 1 ms)^-p decay. 2-D body waves spread
  as 1/sqrt(r) and 2-D surface/guided waves not at all; 3-D is one half-order
  more for each class, so p = 0.5 is the usual 2-D -> 3-D amplitude proxy.
* `rock.texture_std` / `rock.texture_corr_len`: correlated velocity texture
  scatters energy into a coda and out of the direct phases (a real physical
  loss mechanism, simulated rather than post-processed). `configs/textured_hires.yaml`.

`decay.png` (moving-RMS in dB vs time for six offsets of RCV1) is the figure
to look at when tuning these; real hard-rock hammer records typically fall
40-60 dB within 30-50 ms. Measured Q_P for intact Grimsel granodiorite at
0.6-7.5 kHz is only 12-35 (Barbosa et al. 2025, GJI), so Q = 100 is optimistic. Note that the strongest late phase in plan-view 2-D
is the slab-waveguide wall phase, which has no 3-D counterpart of that
strength - part of the "missing decay" is that geometry, not attenuation.

## Heterogeneous media

`structure.family`: `homogeneous`, `fault_zone` (low-velocity band crossing
the axis at `fault_z`, thickness `fault_width`, angle `fault_strike_deg`),
`layered_dip` (different rock beyond a straight interface at `layer_z`).
Plus texture on top of any family (`rock.texture_std`, `rock.texture_corr_len`).
Only the homogeneous configs are kept in `configs/`; the families are exercised
by editing `structure:` in a copy (e.g. `family: fault_zone, fault_z: 40, fault_width: 4,
fault_strike_deg: 70, fault_vp: 3500`). Scenario seeding and batch generation are
deliberately left out of the prototype: they need a planned design (see project notes
on Grimsel / Bedretto data and geological generators).

## Findings from the first runs (2026-09-10)

Reference runs are in `runs/` (`./run_batch.sh` regenerates them, ~10 min on the M4).

* **Time conventions verified.** Without a cavity the pressure-channel picks
  give an apparent velocity of 5804 m/s for vp = 5800; the eikonal equals
  offset/vp within 2 µs; the P-lobe peak sits a constant 0.145 ms after the
  eikonal time (2-D Green's-function shape, not a stagger bug).
* **Cavity is exactly dead.** Receivers placed inside the cavity record 0.0;
  the vacuum method does not leak.
* **Same-wall P onset is kinematically on time.** With the window reference the
  emergence delay of RCV1/RCV2 is 0.014-0.035 ms with cavity + EDZ (dx 0.1,
  1.2 kHz) versus 0.023-0.026 ms without — i.e. the first motion appears at the
  eikonal time; the cavity does not delay the *shape*.
* **The cavity effect is in amplitude.** The same-wall P on the force-parallel
  component is only ~3 % of the S/guided-wave maximum even without a cavity
  (hammer force normal to the wall puts along-wall receivers near the P
  radiation null), and the cavity adds a strong guided phase along the wall.
  With a fixed detection floor (`threshold_ref: gather`, 2 % of gather max) the
  pick jumps to later phases and the "delay" grows from ~2 ms at 15 m to ~8 ms
  at 50 m offset (600 Hz; 2-12 ms at 1.2 kHz). This reproduces the
  experts' "systematically late, offset-dependent" picks — and says the
  emergence delay is an SNR phenomenon, so the noise model in the real
  simulator decides the observable label, not the clean physics alone.
* **2-D plan-view artifact:** the tunnel becomes an infinitely tall slab, so
  the only rock path to the opposite wall runs around the face (~15-19 ms
  instead of ~3 ms) and arrives at −50 dB. Opposite-wall receivers
  (RCV3/RCV4 by default) get NaN labels and are not physical in this
  prototype; draw conclusions from same-wall receivers only. Cross-wall
  geometry needs 3-D (Deepwave supports it; ~10 M cells at dx 0.25 m).
* 600 Hz at dx 0.2 m is a compromise for speed; at 15-50 m offsets it is in the
  near field (r/λ ≈ 1.5-5). Use `homogeneous_hires.yaml` (dx 0.1, 1.2 kHz,
  8 s/shot on the M4) for anything quantitative about the hammer-era data
  (peak ~1.6 kHz).
* Throughput on the M4 CPU (4 torch threads): 1.0 s/shot at dx 0.2 m
  (100 k cells, 30 ms), 8 s/shot at dx 0.1 m.

## Findings on amplitude decay (2026-09-11)

* The "no decay" impression in the 30 ms figures came from per-trace
  normalisation plus a record that ended just as the wall-guided phase arrived.
  With a 100 ms record the raw homogeneous coda falls 80 dB by ~20 ms, i.e. it
  dies *faster* than field data, which has a long scattering coda.
* The −33 dB plateau at 28-50 ms in the homogeneous 100 ms run is physical:
  91 m at S speed is exactly the path to the face corner and back. In plan-view
  2-D the face is a full-height wall and a strong back-scatterer. PML width
  (20 vs 50 cells) changes nothing to 0.1 dB, so the absorbing boundaries are fine.
* Realistic-looking records come from *scattering* (`texture_std` ~0.08 with
  1 m correlation length) plus Q ~100 and the spreading proxy 0.5: the coda then
  decays smoothly to −55 dB at 25 ms.
* Grid cost: a 100 ms record at dx 0.1 costs 21 s/shot on the M4 (3x the 30 ms run).
  First-sample artifact: Deepwave's source resampling leaves a −54 dB transient
  in the first ~5 samples of every trace; ignore (before any arrival).

## Not in the prototype (by decision)

Noise harvesting/mixing, wavelet re-shaping, 3-D, GPU, scenario sampler,
gstools/GemPy textures, dataset-scale storage, reciprocity trick (simulate
from the 4 receiver positions instead of 24 shots: 6x cheaper for force /
velocity pairs), the ML model.
