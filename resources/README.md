# Lunar science references

Reference material for the procedural mineral / regolith models in
`src/minerals.rs` and `src/terrain.rs`. Each entry below notes which
part of the simulation it informs, and whether the file is committed
locally or just linked (large PDFs are linked to keep the repo lean).

The runtime does not load any of these at startup; they're for design
and tuning notes.

---

## Composition / mineralogy

### `lunar_sourcebook_ch6.pdf` — *committed*

Heiken, Vaniman, and French, **Lunar Sourcebook: A User's Guide to the
Moon** (1991), Chapter 6 *Lunar Soil*. The canonical reference for
regolith physical properties, grain size distribution, density, and
maturity.

Lunar and Planetary Institute, free PDF:
<https://www.lpi.usra.edu/publications/books/lunar_sourcebook/pdf/Chapter06.pdf>

Informs:
- bulk density assumption (≈ 1500 kg/m³) used to convert wt% → g/m³ in
  `ELEMENTS` (`src/minerals.rs`)
- typical grain-size and porosity numbers if/when we add wheel-slip or
  bearing-capacity models.

### `lunar_sourcebook_ch07_chemistry.pdf` — *committed*

Heiken, Vaniman, and French, **Lunar Sourcebook**, Chapter 7 *Lunar
Chemistry*. Tables of major-element wt% across Apollo sample sites
(mare basalts vs. highland anorthosites vs. KREEP).

<https://www.lpi.usra.edu/publications/books/lunar_sourcebook/pdf/Chapter07.pdf>

Informs:
- the `base ± range` numbers per element in `ELEMENTS` in
  `src/minerals.rs`. Specifically: Si ~21 wt%, Al swings 7–14 wt%
  between mare and highlands, Fe 4–12 wt%, Ti 0.4–4.5 wt%.

### Moon Mineralogy Mapper (M³) — *linked*

Pieters et al. (2009 / 2013), Chandrayaan-1 M³ instrument:
- Instrument description and global mineral mapping:
  <https://www.usgs.gov/publications/moon-mineralogy-mapper-m3-imaging-spectrometer-lunar-science-instrument-description>
- Unsupervised clustering of M³ data (2024 arxiv preprint):
  <https://arxiv.org/abs/2411.03186> (25 MB PDF, not committed)
- TiO₂ / FeO mapping with M³ (Zhang & Bowles, EPSC 2013):
  <https://meetingorganizer.copernicus.org/EPSC2013/EPSC2013-374.pdf>

Informs:
- spatial autocorrelation length of the mineral noise. M³ resolution is
  ~140 m/pixel; using 257² cells over a ~5 km arena gives ≈ 20 m cell
  size, which is finer than M³ but reasonable for a survey simulation.

---

## Subsurface structure / stratigraphy

### Dielectric properties and stratigraphy, SPA basin — *linked*

Yan et al. (2022), Yutu-2 Lunar Penetrating Radar in South Pole–Aitken:
<https://arxiv.org/abs/2203.02840> (29 MB PDF, not committed)

Probably the most directly relevant publication to the
"surface → subsurface" coupling we want to model: it shows three layered
regolith strata at the CE-4 site, distinguishes a paleoregolith from
overlying ejecta, and reports cm-scale dielectric heterogeneity.

Informs (or *should* inform) the subsurface generation in
`src/minerals.rs`. The current `subsurface = surface + 70%-range noise`
is a heuristic; a more faithful model would have at least two
horizontally-correlated layers with vertical mixing depth.

### Gardening of lunar regolith

The "vertical mixing" process that *partially* couples surface to
subsurface composition. Two useful refs:
- Costello et al. (2020), **The mixing of lunar regolith: Vital updates
  to a canonical model**, Icarus:
  <https://www.sciencedirect.com/science/article/abs/pii/S0019103517307066>
- Wang et al. (2022), **The gardening process of lunar regolith by
  small impact craters: a case study in Chang'E-4 landing area**:
  <https://www.sciencedirect.com/science/article/abs/pii/S0019103522000306>
- Hellmann et al. (2024), **Gardening on the Moon: An Advection-Diffusion
  Model**, arxiv preprint linked from the search results.

Informs:
- the *correlation length* of the surface-to-subsurface relationship.
  Real gardening overturns the top ≈ 0.5 m on Gyr timescales, so deep
  deposits stay decoupled from the immediate surface composition —
  consistent with the "hidden hot-spots" we want in the gameplay.

---

## Polar water / volatiles

### `arxiv_2305.20007_lunar_psr_soil.pdf` — *committed*

Kreslavsky et al. (2023), **The Physical State of Lunar Soil in the
Permanently Shadowed Regions**.
<https://arxiv.org/abs/2305.20007>

### `arxiv_2502.06056_lunar_ice_theories.pdf` — *committed*

Schorghofer et al. (2025), **Current Theories of Lunar Ice**.
<https://arxiv.org/abs/2502.06056>

### `arxiv_1801.05754_polar_regolith_looser.pdf` — *committed*

Hayne et al. (2018), **Evidence for exposed water ice in the Moon's
south polar regions**. (Polar regolith density anomalies.)
<https://arxiv.org/abs/1801.05754>

### LCROSS detection — *linked*

Colaprete et al. (2010), **Detection of water in the LCROSS ejecta
plume**. ≈ 5.6 ± 2.9 wt% water at the Cabeus impact site —
the upper end of the H₂O range in `ELEMENTS`.
<https://www.science.org/doi/10.1126/science.1186986>

LEND / LRO follow-up (Sanin et al. 2012):
<https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JE003971>

Informs:
- H₂O `base = 8 000 g/m³, range = 25 000` in `ELEMENTS`. The range is
  intentionally large enough to hit ≈ 50 000 g/m³ in localised hotspots
  (≈ 3 wt%), matching the LCROSS detection.

---

## What the gameplay still ignores

- **Mare vs. highland duality.** Real lunar composition is bimodal —
  basaltic mare in flat lowlands, anorthositic highlands in elevated
  terrain. The current noise model is unimodal Gaussian-ish around each
  element's `base`. A faithful model would correlate the element field
  with the heightmap (low ↔ mare ↔ Fe/Ti-rich, high ↔ highlands ↔ Al-rich).
- **Spatial correlation between elements.** Ti and Fe should co-vary
  (ilmenite); Al and Ca should co-vary (plagioclase). Right now each
  element has an independent noise field.
- **Vertical structure.** Subsurface is sampled as a single layer; real
  regolith has the gardened top layer over megaregolith and basalt /
  anorthosite bedrock.
