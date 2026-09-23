"""
config.py - central configuration for the cubetas analysis pipeline.

Single source of truth for paths, CRS, layer names, frozen parameters and shared
plotting settings. Every analysis script imports this module so that constants
never diverge between scripts. This module is imported, never executed.

Project: cubetas-morphometry | Python 3.11 | EPSG:5347 (POSGAR 2007 / Argentina Faja 5)
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Repository layout. BASE is the repository root, resolved from this file
# (src/lib/config.py -> up three levels), so the pipeline runs from any clone.
# --------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]
assert (BASE / "src" / "lib").is_dir(), (
    f"config.py is misplaced: BASE resolved to {BASE}. "
    "It must live at <repo>/src/lib/config.py")

DATA    = BASE / "data"         # inputs: layers and tables produced outside the repo
RESULTS = BASE / "results"      # outputs: CSV data products feeding the paper
FIGURES = BASE / "figures"      # outputs: final figures (PNG + PDF)

# --------------------------------------------------------------------------
# Input layers
# --------------------------------------------------------------------------
CUBETAS_GPKG  = DATA / "cubetas.gpkg"
CUBETAS_LAYER = "cubetas"

MORPH_GPKG    = DATA / "cubetas_morphometry.gpkg"
MORPH_LAYER   = "cubetas_morphometry"

AREA_GPKG     = DATA / "area_de_estudio.gpkg"
CUENCAS_LAYER = "cuencas"
SECTOR_LAYER  = "sector_no_urbano"      # non-urban analysis window
DIVIDE_LAYER  = "divisoria_salado"      # drainage divide with the Salado basin

TENSOR_GPKG   = DATA / "tensor_cubetas.gpkg"
TENSOR_LAYER  = "tensor_cubetas"

CONTROL_GPKG  = DATA / "control_points.gpkg"
CONTROL_LAYER = "control_points"

CHANNELS_CSV  = DATA / "channel_points.csv.gz"   # channel points with their
                                                 # contributing area; gzip, read
                                                 # transparently by pandas
TPI_FIG5_TIF  = DATA / "tpi_fig5.tif"              # TPI crop, background of Figure 5
TPI_RRB_INSET = DATA / "tpi_rrb_inset.tif"         # coarse TPI for the Figure 5 inset
COVARIATES_CSV = DATA / "covariates_cubetas.csv"   # slope and topographic
                                 # position sampled at each cubeta, baked
                                 # from the LiDAR DEMs (see prep/)

ID_FIELD    = "cubeta_id"
BASIN_FIELD = "cuenca"
BASINS      = ["CMR", "CRR"]

# --------------------------------------------------------------------------
# CRS
# --------------------------------------------------------------------------
EPSG             = 5347
CENTRAL_MERIDIAN = -60.0        # faja 5, for the grid -> geographic azimuth correction

# --------------------------------------------------------------------------
# Morphometry (script 01)
# --------------------------------------------------------------------------
# theta_mom_geo        second-moment axis, the primary orientation estimator
# theta_feret_geo  maximum Feret diameter, used where comparability is required
# theta_mbr_geo    minimum bounding rectangle, control estimator (not used in the paper)
# *_grid           the same azimuths before the meridian-convergence correction,
#                  kept because Appendix A.3 diagnoses behaviour in grid coordinates
MORPH_VARS = ["S_ha", "P_m", "a_m", "b_m", "feret_max_m", "Dm_m", "IC", "E",
              "theta_mom_geo", "theta_feret_geo", "theta_mbr_geo",
              "theta_mom_grid", "theta_feret_grid", "theta_mbr_grid"]

# Estimators reported in the manuscript. 'mbr' is computed but not published.
ESTIMATORS = ["moments", "feret"]

AREA_CLASSES   = [(None, 1.0, "<1 ha"), (1.0, 2.0, "1-2 ha"), (2.0, None, ">=2 ha")]
SHAPE_FILTER_E = 1.3            # sensitivity check only, never a selection criterion

# --------------------------------------------------------------------------
# Valley orientation field and coupling (scripts 09, 10)
# --------------------------------------------------------------------------
RING_RADII_M   = (200, 300)     # fixed radii, never scaled to cubeta size (App. A.4)
RING_NPOINTS   = 24
CONSISTENCY_TH = 0.3            # applies to the descriptive summary of the field only;
                                # the coupling analysis keeps the full range, because
                                # consistency is the stratifying variable
LENGTH_BINS    = [(0, 50), (50, 100), (100, 200), (200, None)]

# --------------------------------------------------------------------------
# Nulls and resampling
# --------------------------------------------------------------------------
N_PERM    = 999                 # permutation null for agreement between two signals
PERM_SEED = 20260922
N_ISO     = 2000                # draws for the isotropy band of R
ISO_SEED  = 20260101
VM_SEED   = 4242                # restarts of the two-component von Mises fit

# --------------------------------------------------------------------------
# Ripley L (scripts 04, 05)
# --------------------------------------------------------------------------
N_SIM     = 999
RMAX_FRAC = 0.25
N_RADII   = 25
SEED      = 12345

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
# Data keep CMR/CRR in the 'cuenca' attribute; only displayed labels use the
# basin names of the manuscript.
BASIN_LABELS  = {"CMR": "MRB", "CRR": "RRB", "ALL": "Both basins"}
BASIN_COLORS  = {"CMR": "#4477AA", "CRR": "#CC6677"}    # colourblind-safe
SIGNAL_COLORS = {"theta_C": "#117733", "theta_N": "#4477AA", "theta_V": "#882255"}


def basin_label(b):
    return BASIN_LABELS.get(b, b)


def setup_matplotlib():
    """Consistent figure style across all scripts. Returns pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })
    return plt


def ensure_dirs():
    for d in (RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name):
    """Save as PNG + PDF into FIGURES/. Captions belong in the manuscript,
    never burned into the image."""
    ensure_dirs()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    return [FIGURES / f"{name}.{ext}" for ext in ("png", "pdf")]
