# cubetas-morphometry

Analysis code and derived data for the morphometric characterisation and spatial
organisation of cubeta wetlands in the upper Matanza-Riachuelo and Reconquista
basins, Buenos Aires Province, Argentina.

> Grimson, R., Migone, L., González Trilla, G., Schivo, F. (2026). Morphometric
> characterisation and spatial organisation of cubeta wetlands in the upper
> Matanza-Riachuelo and Reconquista basins (Buenos Aires Province, Argentina).
> *Cuadernos de Investigación Geográfica*. [DOI to be assigned]

Archived release: [Zenodo DOI to be assigned]

---

## What is here

Everything needed to reproduce every number, table and figure of the paper from
the data distributed with it, with the exceptions listed under *What is not
distributed*.

```
data/        inputs: layers and tables produced outside this repository
results/     outputs: the CSV data products the paper reports
figures/     outputs: the published figures, PNG and PDF
src/         the analysis pipeline, numbered in execution order
src/lib/     imported modules: configuration and the von Mises fitting code
```

Scripts are numbered in the order they must run. The two modules in `src/lib/`
are imported, never executed, which is why they carry no number. The numbering
has gaps where a figure is not produced by the pipeline.

---

## Running the pipeline

Python 3.11:

```bash
pip install -r requirements.txt
cd src
python 01_morphometry.py
python 02_orientation.py
...
```

`01` and `02` must run before anything else. The rest are independent of one
another.

| Script | Produces | Where it is used |
|---|---|---|
| `01_morphometry.py` | `morphometry.csv`, `morphometry_summary.csv`, `data/cubetas_morphometry.gpkg` | Table 1, and every later script |
| `02_orientation.py` | `orientation_R.csv` | Table 2, Section 4.2 |
| `03_modality.py` | `TableA1_modality.csv` | Table A1, Appendix A.1 and A.2 |
| `04_ripley.py` | `ripley_L.csv`, **Figure 8** | Section 4.5 |
| `05_ripley_inhom.py` | `ripley_inhom.csv` | Section 4.5 |
| `06_slope_position.py` | `topographic_position.csv` | Sections 4.3 and 4.5 |
| `07_drainage.py` | `drainage.csv`, **Figure 4** | Section 4.3 |
| `08_size_dependence.py` | `size_dependence.csv`, **Figure A1** | Appendix A.5 |
| `09_coupling.py` | `valley_coupling.csv`, **Figure 7** | Section 4.4 |
| `12_fig3_area_distribution.py` | `area_distribution.csv`, **Figure 3** | Section 4.1 |
| `13_fig5_rosarios.py` | **Figure 5** | Section 4.4 |
| `14_fig6_roses.py` | **Figure 6** | Section 4.4 |
| `15_estimator_control.py` | `coupling_estimator_control.csv`, `coupling_elongation.csv`, `grid_depletion.csv` | Appendices A.3 and A.4 |

Figures 1 and 2 are not produced by the pipeline; both are included in
`figures/` and both are explained under *What is not distributed*.

### Verification

Two scripts check the repository rather than produce results:

```bash
python verify_numbers.py      # every figure quoted in the paper, against its file
python check_references.py    # broken references, without running anything
```

`verify_numbers.py` names the section of the paper where each number appears, so
a mismatch points at the sentence to revise. It checks 74 quantities and should
report none failed.

---

## Data dictionary

One rule throughout: a name means the same thing everywhere. Three groups deserve
attention, because the underlying quantities are easy to confuse.

### Orientation estimators (`morphometry.csv`, `cubetas_morphometry.gpkg`)

| Column | What it is |
|---|---|
| `theta_mom_geo` | Azimuth of the elongation axis from second-order area moments. **The primary estimator**: the modality decision is taken on it |
| `theta_feret_geo` | Azimuth of the maximum Feret diameter, the longest chord between boundary points. Used where comparability with published values is required |
| `theta_mbr_geo` | Azimuth of the long side of the minimum-area rotated rectangle. A control, not reported in the paper |
| `theta_mom_grid`, `theta_feret_grid`, `theta_mbr_grid` | The same three before the meridian-convergence correction. Kept because Appendix A.3 diagnoses estimator behaviour in grid coordinates |

Every `_geo` azimuth is axial (0–180°) and corrected for meridian convergence, so
no directional result depends on the projection. The correction is about 0.6° in
this area: small, but it is the difference between the three estimators sharing a
frame and not sharing one.

### Valley orientation field (`tensor_cubetas.gpkg`)

Estimated from the structure tensor of the smoothed relief with the cubetas masked
out of the elevation model, and sampled on a ring of 24 points at fixed radii of
200 and 300 m outside each body.

| Column | What it is |
|---|---|
| `az_canada` | Orientation of the field. **This is the field the paper uses** |
| `coh` | Tensor coherence averaged over the ring. The 0.3 threshold that retains 95 % of the census applies to this, and to the θV rose of Figure 6 |
| `ring_consistency` | Axial resultant of the 24 ring azimuths. **A different quantity from `coh`**: this is the local consistency that stratifies Figure 7 |
| `n_valid_fixed` | How many of the 24 ring points carried valid data |
| `az_scaled_ring`, `coh_scaled_ring`, `scaled_consistency`, `n_valid_scaled` | The same quantities on a ring scaled to body size. A control: Appendix A.4 shows it confounds sampling distance with size (Spearman ρ = −0.556 against −0.021 for the fixed ring) and discards it |
| `az_centroid_unmasked`, `coh_centroid_unmasked` | Reading at the cubeta centroid, without masking. A control: it inflates the agreement with cubeta orientation by roughly half in the largest size class |

There is no masked-centroid column: masking a cubeta removes the data under its
own centroid, which is precisely why the field is sampled on a ring.

### Effect sizes

`RBC` is the rank-biserial correlation. Its sign convention differs between
analyses, so it is stated at each use and repeated in the figure captions. No
p-values are reported: the inventory is a census, so statistical significance is
uninformative, and the interpretable quantity is the magnitude of an effect
against an explicit null.

---

## Figures

The nine figures of the paper are `figures/Figure_1` to `Figure_8` and
`Figure_A1`, as PNG and PDF.

Three further figures are written by the pipeline and are **not** figures of the
paper. They are diagnostics, kept because they make the corresponding CSV
interpretable at a glance:

| File | From | What it shows |
|---|---|---|
| `orientation_rose` | `02_orientation.py` | Axial rose of cubeta orientation per basin, the visual counterpart of `orientation_R.csv` |
| `ripley_inhom` | `05_ripley_inhom.py` | Inhomogeneous L against its envelope, which is how Section 4.5 separates first-order intensity from interaction |
| `topographic_position` | `06_slope_position.py` | RBC of slope against smoothing scale, and of distance to the divide |

---

## What is not distributed

**The LiDAR digital elevation models** (10 m, and the original 1 m model of the
RRB) are the property of ACUMAR (Matanza-Riachuelo basin authority) and COMIREC
(Reconquista basin committee). Only point-wise information derived from them is
shared: slope and topographic position sampled at cubetas and at control points,
the channel points with their contributing area, the valley orientation field and
its coherence at each cubeta, and two crops of the topographic position index for
the panels of Figure 5. Requests for the original DEMs should be addressed to the
corresponding agency.

**The intermediate rasters** from which those terrain covariates were sampled,
slope and topographic position at several smoothing scales, are not kept either.
What is distributed is the value of each covariate at every cubeta and at every
control point, which is what the analysis uses; regenerating the rasters
themselves would require the original elevation models.

**Figure 1 is a location map.** It is drawn from third-party context cartography
in three projections: the land outline from Natural Earth, the provincial
boundaries and the watercourse network from the Instituto Geográfico Nacional of
Argentina, and the built-up mass from the Global Human Settlement Layer
(GHS-BUILT-S R2023A, tile `R14_C13`, 100 m, openly available from the JRC Data
Catalogue). None of that is research data, and reproducing the map requires all of
it, so the script that renders the figure lives outside this repository. The
rendered figure is included in `figures/`.

**Figure 2 is a schematic.** Nothing in it is measured from data: it illustrates
the definitions of the three directional signals on three idealised cubetas, and
it was drawn by hand rather than generated from the repository. It is included in
`figures/` like the rest.

The scripts that turn external inputs into the light layers of `data/` live in a
`prep/` directory that is not published, because it cannot run without the
undistributed sources. Their role is recorded below, so that the provenance of
every file in `data/` remains traceable.

---

## Provenance of `data/`

| File | Produced from |
|---|---|
| `cubetas.gpkg` | Multi-temporal wetland detection (Migone et al. 2025), visually inspected, restricted to the non-urban window |
| `area_de_estudio.gpkg` | Basin outlines, non-urban analysis windows, and the drainage divide with the Salado basin |
| `cubetas_morphometry.gpkg` | Regenerated by `01_morphometry.py` |
| `tensor_cubetas.gpkg` | Structure tensor of the smoothed LiDAR relief, cubetas masked out, plus the ring and centroid controls of Appendix A.4 |
| `control_points.gpkg` | 10,000 random points per basin inside the non-urban window, carrying the same terrain covariates as the cubetas |
| `covariates_cubetas.csv` | Slope and topographic position sampled at each cubeta from the LiDAR DEMs |
| `channel_points.csv.gz` | Flow accumulation on the LiDAR DEM (Migone et al. 2026a), pruned to a contributing area of at least 0.2 km², the smallest threshold the analysis evaluates |
| `tpi_fig5.tif` | Topographic position index, 500 m radius, cropped to the detail window of Figure 5 |
| `tpi_rrb_inset.tif` | The same index over the whole RRB, resampled to 200 m, for the locator inset of Figure 5 |

---

## Requirements

See `requirements.txt`: GeoPandas, Shapely 2.x, rasterio, NumPy, SciPy,
scikit-learn, pandas and matplotlib. Vector layers are in POSGAR 2007 / Argentina
Faja 5 (EPSG:5347); the rasters keep their native UTM 21S (EPSG:32721), and every
azimuth is reported relative to geographic north.

## Licence

Code under the terms of `LICENSE`; data under the terms of `LICENSE-data`.

## Citation

See `CITATION.cff`.
