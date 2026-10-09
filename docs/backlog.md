# Backlog

Open questions, hypotheses and to-dos that come up during experiments. Two or three sentences
each; newest first. Mark an item **done** with a one-line outcome instead of deleting it.

## Open

- **Weak P from impact along the wall** (2026-10-09, seed 5). A wall-normal hammer blow sends
  almost no P along the wall, so the first arrival is barely visible on every component; a 7.5°
  direction jitter does not change that. Check against real hammer data: if real P is clearly
  visible, real blows are less wall-normal than modelled (try `sources.impact.direction_jitter_deg=45`).
- **Concrete changes the first-arrival waveform** (2026-10-09, seed 5). For the explosive source
  the concrete-minus-bare difference starts at the P arrival and is as large as the P wavelet
  (component s). Confirm on more draws before treating it as a general effect.
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
