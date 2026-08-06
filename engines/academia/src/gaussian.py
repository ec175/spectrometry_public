"""gaussian.py — turn Gaussian 16 output into line lists for spectra.simulate().

Written against Gaussian 16 output (`DFT/*.log`, plus GaussView ASCII exports
`*_ir.txt` / `*_raman.txt`). This module reads either:

  parse_ir / parse_raman   -> (wavenumber cm-1, intensity)     [drop-in like orca.py]
  parse_log_freq           -> raw .log freq job -> dataclass with all three
  parse_g16_ascii          -> GaussView "Peak information"/"Spectra" export

so a computed spectrum overlays a measured one the same way for ORCA or Gaussian:

    import gaussian as g16, spectra as sp
    c, a = g16.parse_ir(NIF_ir_url)                 # GaussView export
    gx, gy = sp.simulate(c, a, "FTIR", freq_scale=0.967)   # B3LYP scaling
    sp.overlay(sp.experimental("nif_xtal", "FTIR"), (gx, gy), "FTIR")

DFT harmonic frequencies are systematically high — apply a scaling factor in
simulate() (B3LYP/6-311++G** ~0.967; wB97XD/6-31G* ~0.95; see context/methods).
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
import re
import numpy as np


# --- shared fetch (URL or local), mirrors spectra.load_xy's source handling --
def _read(source, *, timeout=30):
    s = str(source)
    if s.startswith(("http://", "https://")):
        import requests
        return requests.get(s, timeout=timeout).text
    return Path(source).read_text(errors="ignore")


# === GaussView ASCII export (#Peak information / #Spectra) ===================
@dataclass
class G16Spectrum:
    """Parsed GaussView IR/Raman export."""
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    peaks: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))    # (n, 2): x, y
    spectra: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # (n, 3): x, y, dy/dx
    metadata: dict = field(default_factory=dict)

    def line_list(self):
        """The stick spectrum (centers, intensities) for spectra.simulate()."""
        if self.peaks.size == 0:
            return np.array([]), np.array([])
        return self.peaks[:, 0], self.peaks[:, 1]

    def curve(self):
        """The pre-broadened (x, y) GaussView curve, if present."""
        if self.spectra.size == 0:
            return np.array([]), np.array([])
        return self.spectra[:, 0], self.spectra[:, 1]


def parse_g16_ascii(content: str) -> G16Spectrum:
    """Parse a GaussView IR/Raman ASCII export into a G16Spectrum.

    Format: '#'-prefixed header with 'X-Axis:'/'Y-Axis:' and a title, a
    '# Peak information' section of (x, y) pairs, and a '# Spectra' section of
    (x, y, dy/dx) rows. Robust to the pairs living in comments or as bare rows.
    """
    title = x_label = y_label = ""
    peaks_data, spectra_data = [], []
    in_peaks = in_spectra = False

    for raw in content.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comment = line[1:].strip()
            low = comment.lower()
            if "peak information" in low:
                in_peaks, in_spectra = True, False
                continue
            if low == "spectra":
                in_spectra, in_peaks = True, False
                continue
            if in_peaks:                           # some exports put pairs in comments
                try:
                    vals = [float(v) for v in comment.split()]
                    if len(vals) == 2:
                        peaks_data.append(vals)
                        continue
                except ValueError:
                    pass
            if "X-Axis:" in comment:
                x_label = comment.split("X-Axis:")[1].strip()
            elif "Y-Axis:" in comment:
                y_label = comment.split("Y-Axis:")[1].strip()
            elif not title and ":" not in comment and comment not in ("X", "Y", "DY/DX") \
                    and "X" not in comment:
                title = comment
            continue
        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            continue
        if len(vals) == 2:
            (peaks_data if (in_peaks or not in_spectra) else spectra_data).append(
                vals if in_peaks or not in_spectra else vals + [0.0])
        elif len(vals) >= 3:
            spectra_data.append(vals[:3])

    peaks = np.array(peaks_data, float) if peaks_data else np.empty((0, 2))
    spectra = np.array(spectra_data, float) if spectra_data else np.empty((0, 3))
    return G16Spectrum(title=title, x_label=x_label, y_label=y_label,
                       peaks=peaks, spectra=spectra)


def scale_frequencies(data: G16Spectrum, factor: float) -> G16Spectrum:
    """Return a copy with the frequency (x) axis of peaks and spectra scaled.

    (Equivalent to passing freq_scale to spectra.simulate(); use whichever fits.)
    """
    peaks = data.peaks.copy()
    if peaks.size:
        peaks[:, 0] *= factor
    spectra = data.spectra.copy()
    if spectra.size:
        spectra[:, 0] *= factor
    return replace(data, peaks=peaks, spectra=spectra)


# === raw .log frequency job =================================================
@dataclass
class G16FreqJob:
    """Modes parsed from a raw Gaussian 16 freq .log."""
    freqs: np.ndarray                  # cm-1
    ir: np.ndarray                     # IR intensities (km/mol)
    raman: np.ndarray                  # Raman activities (A^4/amu), empty if not computed

    def ir_lines(self):
        return self.freqs, self.ir

    def raman_lines(self):
        if self.raman.size == 0:
            raise ValueError("no Raman activities in this log (run freq=raman)")
        return self.freqs, self.raman


_NUM = r"[-+]?\d+\.\d+"


def parse_log_freq(source) -> G16FreqJob:
    """Parse 'Frequencies --', 'IR Inten --', 'Raman Activ --' lines of a G16 log.

    Gaussian prints harmonic analysis three modes per row; this collects every
    row of each kind in order. Raman rows appear only if the job computed them.
    Accepts a URL/path or the raw log text directly.
    """
    return _parse_log_freq_text(_read(source))


def _parse_log_freq_text(text) -> G16FreqJob:
    freqs, ir, raman = [], [], []
    for line in text.splitlines():
        if "Frequencies --" in line:
            freqs += [float(x) for x in re.findall(_NUM, line.split("--", 1)[1])]
        elif "IR Inten" in line:
            ir += [float(x) for x in re.findall(_NUM, line.split("--", 1)[1])]
        elif "Raman Activ" in line:
            raman += [float(x) for x in re.findall(_NUM, line.split("--", 1)[1])]
    n = len(freqs)
    ir = (ir + [0.0] * n)[:n]
    return G16FreqJob(freqs=np.array(freqs), ir=np.array(ir),
                      raman=np.array(raman[:n]))


# === NMR GIAO shieldings (for DFT NMR simulation) ===========================
# Gaussian's NMR job prints one block per atom; the isotropic shielding line is
# format-stable across versions, so this is robust (unlike the column-indexed
# vibrational tables). Convert to chemical shifts with nmr.shieldings_to_shifts.
_GIAO = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+Isotropic\s*=\s*([-\d.]+)", re.M)


def parse_nmr_shieldings(source, element=None):
    """Parse 'n  El  Isotropic = sigma' lines -> list of (index, element, sigma).

    element : keep only this element (e.g. "C" or "H") if given.
    Feed the sigmas to nmr.shieldings_to_shifts(sigmas, reference_sigma).
    """
    out = [(int(i), el, float(s)) for i, el, s in _GIAO.findall(_read(source))]
    return [r for r in out if element is None or r[1] == element]


# === drop-in API (matches orca.parse_ir / parse_raman) ======================
# A raw Gaussian .log is identified by "Frequencies --" (GaussView ASCII exports
# never contain it); everything else is treated as an exported peak/spectra file.
def parse_ir(source):
    """(wavenumber cm-1, IR intensity) from a GaussView export or a raw .log."""
    text = _read(source)
    if "Frequencies --" in text:
        return _parse_log_freq_text(text).ir_lines()
    return parse_g16_ascii(text).line_list()


def parse_raman(source):
    """(wavenumber cm-1, Raman activity) from a GaussView export or a raw .log."""
    text = _read(source)
    if "Frequencies --" in text:
        return _parse_log_freq_text(text).raman_lines()
    return parse_g16_ascii(text).line_list()


if __name__ == "__main__":
    # offline smoke test: round-trip a tiny GaussView-style export
    sample = (
        "# Felodipine IR\n# X-Axis: Frequency (cm-1)\n# Y-Axis: Intensity\n"
        "# Peak information\n# 1714.75 27.18\n# 1725.71 547.16\n# 3642.99 55.55\n"
        "# Spectra\n1700 0.1 0\n1720 0.9 0\n1740 0.2 0\n"
    )
    s = parse_g16_ascii(sample)
    c, a = s.line_list()
    print(f"title={s.title!r} peaks={len(c)} strongest={c[int(np.argmax(a))]:.1f} cm-1")
    print("scaled 0.967:", scale_frequencies(s, 0.967).peaks[:, 0].round(1).tolist())
