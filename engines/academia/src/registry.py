"""Sample registry for an amorphous-solid-dispersion measurement set.

Reads samples.csv + measurements.csv and exposes a small query API.
Everything downstream (parsers, figure functions, the UI) should go through
here instead of hardcoding file URLs.

MEASURED DATA IS NOT PART OF THIS REPOSITORY. Point `SPECTRA_DATA_URL` at your
own store — a raw-file host, or a `file:///` URL for a local mirror — and drop
the matching `samples.csv` / `measurements.csv` index beside this module. The
simulated figure scripts need none of this; only the overlay demos do.
"""
import os
from pathlib import Path
import pandas as pd

# --- config -----------------------------------------------------------------
# Change the store in ONE place. Empty means "no measured data configured",
# which every caller treats as "skip the experimental half".
DATA_BASE_URL = os.environ.get("SPECTRA_DATA_URL", "")

TECHNIQUES = ["FTIR", "NMR", "XRD", "DSC", "UV", "RAMAN"]

REGISTRY_DIR = Path(__file__).resolve().parent

# --- loading ----------------------------------------------------------------
def load_samples(path=None) -> pd.DataFrame:
    return pd.read_csv(path or REGISTRY_DIR / "samples.csv", dtype=str).fillna("")


def load_measurements(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or REGISTRY_DIR / "measurements.csv", dtype=str).fillna("")
    df["include"] = df["include"].str.strip().str.upper().eq("TRUE")
    # Guard against typos in the technique column.
    bad = set(df["technique"].str.upper()) - set(TECHNIQUES)
    if bad:
        raise ValueError(f"Unknown technique(s) in measurements.csv: {sorted(bad)}")
    return df


# --- queries ----------------------------------------------------------------
def get_sample(sample_id, samples=None) -> pd.Series:
    s = load_samples() if samples is None else samples
    hit = s[s["sample_id"] == sample_id]
    if hit.empty:
        raise KeyError(f"No sample with id {sample_id!r}")
    return hit.iloc[0]


def get_files(sample_id, technique=None, include_only=True, measurements=None) -> pd.DataFrame:
    m = load_measurements() if measurements is None else measurements
    m = m[m["sample_id"] == sample_id]
    if technique is not None:
        m = m[m["technique"].str.upper() == technique.upper()]
    if include_only:
        m = m[m["include"]]
    return m


def file_url(path: str) -> str:
    """Turn a repo-relative path into a full fetchable URL."""
    return DATA_BASE_URL + path.lstrip("/")


def urls_for(sample_id, technique, **kw) -> list[str]:
    """The list a plotting function actually wants."""
    return [file_url(p) for p in get_files(sample_id, technique, **kw)["path"]]


def samples_with(technique, measurements=None, samples=None) -> pd.DataFrame:
    """All samples that have at least one (included) file for a technique."""
    m = load_measurements() if measurements is None else measurements
    ids = m[(m["technique"].str.upper() == technique.upper()) & m["include"]]["sample_id"].unique()
    s = load_samples() if samples is None else samples
    return s[s["sample_id"].isin(ids)]


if __name__ == "__main__":
    print(f"{len(load_samples())} samples, {len(load_measurements())} measurements")
    print("DSC samples:", list(samples_with("DSC")["sample_id"]))
    print("CBN-PVP29-1_2 DSC files:", urls_for("CBN-PVP29-1_2", "DSC"))
