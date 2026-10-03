# Synthetic FA dataset: design decisions (V1)

Status: **experimental phase**. No dataset is being built yet. This file records what has
been decided so far; open points are listed separately at the end.

## Scope

- The target is **3-D**. 2-D runs are obtained by projection: parameters without a 2-D
  meaning (vertical curvature, receiver height on the wall) are ignored.
- **Opposite-wall receivers are omitted completely.** Receivers and shot holes are always on
  the same side of the tunnel. The simulation domain therefore only needs the rock behind
  that wall (plus the face), not the full tunnel cross-section.
- Noise added to the simulated traces is out of scope here; it is a separate post-processing stage.

## Sampling rule

All parameters are drawn **randomly and independently** for every sample, unless stated
otherwise. Two factors are stated otherwise: **tunnel wall** and **source type**. For each
random draw of everything else, all of their variants are simulated:

| | concrete wall | no concrete wall |
|---|---|---|
| explosive | sim 1 | sim 2 |
| impact | sim 3 | sim 4 |

The four simulations share the same rock, tunnel geometry and acquisition, so the dataset can
later be split to measure how wall and source alone influence propagation and model training.

## Components

**Tunnel geometry.** Curvature is a 2-tuple (vertical, horizontal) drawn from a 2-D normal
distribution. Its variance is a dataset metaparameter with a reasonable default.

**Tunnel wall** (enumerated, see above).
- No concrete: the rock wall has randomly generated imperfections (dents, rough edges, jaggedness).
- Concrete: smooth inner surface; the outer surface fills the jaggedness and cracks of the rock
  wall behind it. Concrete velocity gets small random noise.

**Receiver and shot-line placement.** Randomly generated, same side of the tunnel. Random
variables control at least the receiver to first shot hole separation, deviations from the
line, and the mean shot-hole spacing. Several receiver layouts may be taken from one run via
virtual receivers, within the storage budget.

**Source** (enumerated, see above). Explosive and impact. Each follows its own parameter
distributions, controlled by metaparameters.

**Rock.** Hierarchically generated; the most important and least defined component. Aim
for as much variability as possible. Variability costs no extra compute (each simulation
uses one rock realisation either way). The detailed tree is still to be drafted.

**Fixed per dataset.** Tunnel width, distance to the face, number of shots, embedment depth,
record length and sampling interval. Values not chosen yet; prototype defaults are the
Amberg ones (width ~10.5 m, 24 shots, embedment 1.5 m, dt 20.833 µs, 30 ms).

## Rough budget (reference only)

Per random draw: 4 simulations. One-wall domain, GPU throughput still to be benchmarked.

| | per simulation | 10,000 draws (40,000 sims) |
|---|---|---|
| 2-D, dx 0.1 m | ~10 s | ~2 days, ~200 GB |
| 3-D, dx 0.25 m | ~1 min | ~4 weeks |

## Open points

- Rock tree: lithology classes, rock-mass quality, structures, texture, attenuation; ranges and dependencies.
- Concrete thickness and jaggedness amplitudes vs grid resolution (a lining is only a few cells at dx 0.1 m).
- Values of the fixed parameters.
- Hypotheses to test on a few samples in 2-D and 3-D before building: influence of wall type,
  source type, wall jaggedness, curvature, and the one-wall domain cut.

## Implementation V1 (branch `dataset-generator-v1`)

Code: `fasim/gen/` (sampler, builder, simulation), CLI `generate.py`, defaults in
`fasim/gen/defaults.yaml`. Choices made where the notes leave room; all are adjustable:

- **Seeds.** Each draw has its own random stream per component (tunnel, concrete, acquisition,
  sources, rock), so changing one component's distribution leaves the others unchanged. Random
  fields are stored as seeds; the parameter dict alone rebuilds every model.
- **Domain.** One-wall box: 2 m of tunnel interior, 25 m of rock behind the wall, 8 m behind the
  first receiver, 15 m beyond the face, ±15 m vertically in 3-D.
- **Tunnel.** Circular cross-section, radius 5.25 m; the axis is a parabola with curvature
  (vertical, horizontal), zero at mid-survey. Face 10 m after the last shot hole.
- **Wall roughness.** Correlated random surface (rms amplitude, correlation length, overbreak bias),
  shared by both wall variants. Bare: the rough surface is the free surface. Concrete: a smooth
  circular inner surface clearing the deepest rock protrusion by the drawn thickness; the concrete
  fills everything between it and the rough rock. The EDZ follows the rough rock surface and the face.
- **Acquisition.** Receivers from s = 0 at a drawn spacing; gap; shot holes at a drawn mean spacing;
  per-position deviations along the tunnel, in depth and in height. Optional virtual receiver line.
- **Sources.** Ricker wavelet per source type with a drawn peak frequency, capped to what the grid
  resolves. Impact = force along the wall normal, each shot tilted randomly within a drawn cone.
  Explosive = pressure source. Absolute amplitudes of the two are not comparable (different units).
- **Rock tree.** Lithology (weighted table) -> in-situ vp, vp/vs, Q_P; rock-mass quality q in [0, 1]
  reduces vp and vs and sets EDZ thickness and strength; 0-5 structures (fault zone with damage
  halo, dyke, second-lithology interface, karst voids empty or clay-filled in carbonates, gradient),
  anchored inside the receiver-shot zone with probability 0.6; two-scale von Karman texture,
  anisotropic along a fabric plane for foliated and layered rocks; density from Gardner.
- **2-D projection.** Geology is the exact v = 0 slice of the 3-D geology. Tunnel vertical curvature
  and all heights are ignored. Texture is generated on the 2-D grid with the same statistics.
- **Labels.** Eikonal P traveltime per wall variant. Q_P is stored, not simulated (apply with
  `fasim.postprocess`).
- **Not yet:** horseshoe profiles, anisotropic elasticity (fabric only shapes the texture), wavelet
  families beyond Ricker, noise.

Tested only with two small runs (2-D dx 0.2 m, 6 shots; 3-D dx 0.5 m, 2 shots).
