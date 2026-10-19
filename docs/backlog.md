# Backlog

Open questions, hypotheses and to-dos that come up during experiments. Two or three sentences
each; newest first. Mark an item **done** with a one-line outcome instead of deleting it.

## Open

- **2-D training data need the line-to-point correction** (2026-10-19, notebook 02). Uncorrected 2-D
  first arrivals are 0.16-0.39 ms off the 3-D ones; after the half-derivative filter they agree within
  0.04 ms. 2-D also under-states the amplitude decay with offset (exponent 0.5-1.0 smaller), so decide
  whether to correct amplitudes too or treat them as augmentation.
- **EDZ position relative to the sensors dominates the first arrival** (2026-10-10, notebook 01).
  Removing or thickening the EDZ changes the first-arrival waveform more than any other factor,
  and a 2 m EDZ delays it by 0.3 ms once sensors sit inside it. The EDZ model (sharp boundary,
  thickness from rock quality) and the embedment distribution need a closer look before the dataset.
- **Texture is sampled differently on each grid** (2026-10-10, notebook 01). With von Karman
  texture, dx 0.1 and 0.2 m both differ from 0.05 m by ~50 % at the first arrival, against 7-32 %
  without texture. Band-limit the texture to the grid so the medium does not depend on dx.
- **Random fields depend on the box bounds** (2026-10-10). Changing curvature or domain size moves
  the box and gives a different texture and roughness realisation, which confounds one-factor tests.
  Anchor the fields to a fixed frame so a draw keeps its rock when the geometry changes.
- **Weak P from impact along the wall** (2026-10-09, seed 5). A wall-normal hammer blow sends
  almost no P along the wall. Tilting the blows 30° towards the receivers raises the first arrival
  by 10 dB (notebook 01), so the direction distribution matters a lot. Check against real hammer
  data whether the P arrival is clearly visible there.
- **Explosive vs impact amplitudes are not comparable** (2026-10-03). Deepwave uses different
  units for pressure and force sources, so absolute amplitudes differ by orders of magnitude.
  Decide on a normalisation before mixing both in one dataset.
- **Grid frequency cap clips drawn frequencies** (2026-10-03). The peak frequency is capped to what
  the grid resolves: about 2 kHz at 2-D dx 0.1 m, about 0.9 kHz at 3-D dx 0.25 m. Decide whether
  the source distributions should respect the cap or the grid should be finer.
- **Attenuation is post-processed, not simulated** (2026-09-12). Q_P is stored per draw; measured
  Q_P for Grimsel granodiorite is only 12-35 in the hammer band. If post-processed Q proves too
  crude, switch to a viscoelastic solver (Devito, SPECFEM2D).
- **Ask ETH for the raw Grimsel hammer gathers** (2026-09-12). The published record has only the
  9500 picks; raw tunnel-to-tunnel hammer waveforms would be a real validation set. Contact on the
  record: Joseph Doetsch.

## Done

- **How different is 2-D from the in-plane cut of 3-D?** (2026-10-19, notebook 02). Same timing after the
  line-to-point correction (≤ 0.04 ms); waveforms 44-75 % different at the first arrival; 3-D amplitude
  decays faster with offset; 3-D costs ~1300× more per shot.

- **Concrete changes the first-arrival waveform** (2026-10-10). Confirmed on three draws: 100-168 %
  waveform change at the first arrival, eikonal unchanged, visibility −3 to −6 dB (notebook 01).
- **Is the one-wall box big enough?** (2026-10-10). Yes: a box twice as deep and longer gives
  identical traces for 25 ms records (notebook 01).

- **Source amplitude depended on grid spacing** (2026-10-10). Deepwave sources are densities, so a
  fixed amplitude got 4× weaker when dx halved in 2-D (−12 dB from 0.2 to 0.1 m). Fixed: the wavelet
  is divided by the cell volume dx^dims; amplitudes now agree within 0.6 dB across grids.
