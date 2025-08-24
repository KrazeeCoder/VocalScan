from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# Columns expected by ml_files/datasets/voice_recordings/parkinsons.data (excluding name/status)
REQUIRED_UCI_PD_COLUMNS: Tuple[str, ...] = (
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
)


def _read_csv_rows(path: str) -> List[List[str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _nan_to_num(x: float, default: float = 0.0) -> float:
    if x is None:
        return default
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return default
    return float(x)


def extract_ucipd_from_wav(
    wav_path: str,
    f0_min_hz: float = 75.0,
    f0_max_hz: float = 500.0,
    time_step: float = 0.0,
) -> Dict[str, float]:
    """Extracts a dict of UCI PD-style voice features from a WAV file.

    This uses Praat algorithms via parselmouth for MDVP jitter/shimmer measures and HNR.
    Nonlinear measures (RPDE, DFA, D2, PPE, spread1, spread2) are provided via
    light-weight approximations so values may not exactly match the original dataset
    but preserve scale and qualitative trends for inference.

    Returns a dict keyed by REQUIRED_UCI_PD_COLUMNS.
    """

    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception as e:  # pragma: no cover - dependency guidance
        raise ImportError(
            "parselmouth is required for voice feature extraction.\n"
            "Install with: pip install praat-parselmouth"
        ) from e

    # --- Robust audio preprocessing (trim silence, pre-emphasis, normalization) ---
    # Improves stability of jitter/shimmer/HNR by focusing on voiced content.
    snd = None
    try:
        import librosa  # type: ignore
        import numpy as _np
        # Load mono waveform at a consistent rate
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        # Trim leading/trailing silence
        y, _ = librosa.effects.trim(y, top_db=25)
        # If too short after trim, fall back to raw loading
        if y.shape[0] < int(0.25 * sr):
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
        # Pre-emphasis to boost higher freqs, approximate high-pass
        try:
            y = librosa.effects.preemphasis(y, coef=0.97)
        except Exception:
            y = _np.concatenate([[y[0]], y[1:] - 0.97 * y[:-1]]) if y.size > 1 else y
        # Normalize peak to ~ -1 dBFS
        peak = float(_np.max(_np.abs(y)) + 1e-9)
        y = 0.89 * (y / peak)
        # Construct Praat Sound from numpy array
        snd = parselmouth.Sound(y, sampling_frequency=sr)
    except Exception:
        # Fallback to direct file-based loading if preprocessing not available
        snd = parselmouth.Sound(wav_path)
    # Pitch extraction (Hertz)
    try:
        pitch = call(snd, "To Pitch", time_step, f0_min_hz, f0_max_hz)
    except Exception:
        # Fallback: auto time step
        pitch = call(snd, "To Pitch", 0.0, f0_min_hz, f0_max_hz)
    try:
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    except Exception:
        mean_f0 = 0.0
    try:
        min_f0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    except Exception:
        min_f0 = mean_f0
    try:
        max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    except Exception:
        max_f0 = mean_f0

    # Point process for perturbation measures
    # Create point process for period-based measures; try robust sequence
    try:
        point_process = call([snd, pitch], "To PointProcess (cc)")
    except Exception:
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", f0_min_hz, f0_max_hz)
        except Exception:
            point_process = None

    # Recommended defaults per Praat docs
    tmin, tmax = 0, 0
    period_floor, period_ceiling = 1.0 / f0_max_hz, 1.0 / f0_min_hz
    max_period_factor = 1.3
    max_amplitude_factor = 1.6

    def _pp_call(cmd: str, *params, default: float = 0.0) -> float:
        try:
            if point_process is None:
                return float(default)
            return float(call([snd, point_process], cmd, *params))
        except Exception:
            return float(default)

    #jitter_local = _pp_call("Get jitter (local)", tmin, tmax, period_floor, period_ceiling, max_period_factor)
    #jitter_local_abs = _pp_call("Get jitter (local, absolute)", tmin, tmax, period_floor, period_ceiling, max_period_factor)
    # Jitter metrics operate on PointProcess only
    jitter_local = call(
        point_process, "Get jitter (local)", tmin, tmax, period_floor, period_ceiling, max_period_factor
    )
    jitter_local_abs = call(
        point_process, "Get jitter (local, absolute)", tmin, tmax, period_floor, period_ceiling, max_period_factor
    )
    jitter_rap = _pp_call("Get jitter (rap)", tmin, tmax, period_floor, period_ceiling, max_period_factor)
    jitter_ppq5 = _pp_call("Get jitter (ppq5)", tmin, tmax, period_floor, period_ceiling, max_period_factor)
    jitter_rap = call(
        point_process, "Get jitter (rap)", tmin, tmax, period_floor, period_ceiling, max_period_factor
    )
    jitter_ppq5 = call(
        point_process, "Get jitter (ppq5)", tmin, tmax, period_floor, period_ceiling, max_period_factor
    )

    shimmer_local = _pp_call(
        "Get shimmer (local)", tmin, tmax, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    )
    shimmer_local_db = _pp_call(
        "Get shimmer (local_dB)", tmin, tmax, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    )
    shimmer_apq3 = _pp_call(
        "Get shimmer (apq3)", tmin, tmax, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    )
    shimmer_apq5 = _pp_call(
        "Get shimmer (apq5)", tmin, tmax, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    )
    shimmer_apq11 = _pp_call(
        "Get shimmer (apq11)", tmin, tmax, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    )
    shimmer_dda = _pp_call(
        "Get shimmer (dda)", tmin, tmax, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    )

    # Harmonicity
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, f0_min_hz, 0.1, 1.0)
        hnr_db = call(harmonicity, "Get mean", 0, 0)
    except Exception:
        hnr_db = 0.0
    # Convert HNR[dB] to NHR (noise-to-harmonic ratio). HNR = 10*log10(H/N) -> NHR = 1/(10^(HNR/10))
    nhr = float(1.0 / (10.0 ** (hnr_db / 10.0))) if np.isfinite(hnr_db) else 0.0

    # ---------- Nonlinear/complex measures (approximations) ----------
    # Extract F0 contour for downstream measures
    f0_values = np.array(pitch.selected_array["frequency"], dtype=np.float64)
    f0_values = f0_values[np.isfinite(f0_values) & (f0_values > 0)]
    if f0_values.size < 5:
        f0_values = np.array([_nan_to_num(mean_f0)], dtype=np.float64)

    # DFA on F0 contour (scale-less exponent)
    def _dfa(x: np.ndarray) -> float:
        # Simple DFA implementation
        x = np.asarray(x, dtype=np.float64)
        x = x - np.mean(x)
        y = np.cumsum(x)
        # box sizes (log-spaced)
        n_vals = np.unique((10 ** np.linspace(np.log10(4), np.log10(max(8, len(y) // 4)), num=10)).astype(int))
        flucts: List[float] = []
        scales: List[float] = []
        for n in n_vals:
            if n < 4 or n > len(y):
                continue
            num_boxes = len(y) // n
            if num_boxes < 2:
                continue
            reshaped = y[: num_boxes * n].reshape(num_boxes, n)
            rms_list = []
            t = np.arange(n)
            for seg in reshaped:
                # linear detrend
                coeffs = np.polyfit(t, seg, 1)
                trend = coeffs[0] * t + coeffs[1]
                resid = seg - trend
                rms_list.append(np.sqrt(np.mean(resid ** 2)))
            flucts.append(np.sqrt(np.mean(np.asarray(rms_list) ** 2)))
            scales.append(n)
        if len(scales) < 2:
            return 0.5
        scales = np.log(np.asarray(scales))
        flucts = np.log(np.asarray(flucts) + 1e-12)
        slope = np.polyfit(scales, flucts, 1)[0]
        return float(slope)

    dfa = _dfa(f0_values)

    # RPDE: crude approximation using entropy of autocorrelation peak distances
    def _rpde(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        x = x - np.mean(x)
        if x.size < 16:
            return 0.4
        # Autocorrelation
        ac = np.correlate(x, x, mode="full")[x.size - 1 :]
        ac = ac / (ac[0] + 1e-12)
        # Detect peaks as candidate periods
        from math import isfinite

        dif = np.diff(ac)
        peaks = np.where((np.hstack(([0.0], dif)) > 0) & (np.hstack((dif, [0.0])) <= 0))[0]
        peaks = peaks[peaks > 1]
        if peaks.size < 3:
            return 0.5
        # Differences between successive peak lags
        periods = np.diff(peaks)
        if periods.size < 1:
            return 0.6
        # Histogram-based normalized Shannon entropy
        bins = max(5, min(20, int(np.sqrt(periods.size))))
        hist, _ = np.histogram(periods, bins=bins, density=True)
        p = hist / (np.sum(hist) + 1e-12)
        p = p[p > 1e-12]
        H = -np.sum(p * np.log(p))
        H_max = np.log(bins)
        val = float(H / (H_max + 1e-12))
        # Clamp to typical RPDE range
        return float(np.clip(val, 0.1, 0.95))

    rpde = _rpde(f0_values)

    # D2 (correlation dimension) rough proxy from F0 variability
    def _corr_dim_proxy(x: np.ndarray) -> float:
        x = (x - np.mean(x)) / (np.std(x) + 1e-9)
        # Use correlation sum slope across radii
        radii = np.logspace(-2, 0, 10)
        counts = []
        for r in radii:
            dists = np.abs(x[:, None] - x[None, :])
            C = np.mean((dists < r).astype(np.float64))
            counts.append(C + 1e-12)
        slope = np.polyfit(np.log(radii), np.log(counts), 1)[0]
        # Typical D2 in dataset ~ 1.8 - 3.2
        return float(np.clip(slope, 1.0, 3.5))

    d2 = _corr_dim_proxy(f0_values)

    # PPE: entropy of normalized log pitch period deviation
    T = 1.0 / np.maximum(f0_values, 1e-6)
    T_med = np.median(T)
    d = np.log(T / (T_med + 1e-12))
    d = np.clip(d, -0.5, 0.5)
    hist, _ = np.histogram(d, bins=20, range=(-0.5, 0.5), density=True)
    p = hist / (np.sum(hist) + 1e-12)
    p = p[p > 1e-12]
    ppe = float(-np.sum(p * np.log(p)))

    # spread1/spread2 proxies from skewness-like moments of normalized F0
    z = (f0_values - np.mean(f0_values)) / (np.std(f0_values) + 1e-9)
    spread1 = float(-np.mean(np.abs(z)) * 2.0)  # negative scale like dataset
    spread2 = float(-np.std(z) * 2.5)

    # Assemble feature dict
    # Convert jitter local to percent to match dataset scale
    mdvp_jitter_percent = float(jitter_local * 100.0)

    feats: Dict[str, float] = {
        "MDVP:Fo(Hz)": _nan_to_num(mean_f0),
        "MDVP:Fhi(Hz)": _nan_to_num(max_f0),
        "MDVP:Flo(Hz)": _nan_to_num(min_f0),
        "MDVP:Jitter(%)": _nan_to_num(mdvp_jitter_percent),
        "MDVP:Jitter(Abs)": _nan_to_num(jitter_local_abs),
        "MDVP:RAP": _nan_to_num(jitter_rap),
        "MDVP:PPQ": _nan_to_num(jitter_ppq5),
        "Jitter:DDP": _nan_to_num(3.0 * jitter_rap),
        "MDVP:Shimmer": _nan_to_num(shimmer_local),
        "MDVP:Shimmer(dB)": _nan_to_num(shimmer_local_db),
        "Shimmer:APQ3": _nan_to_num(shimmer_apq3),
        "Shimmer:APQ5": _nan_to_num(shimmer_apq5),
        "MDVP:APQ": _nan_to_num(shimmer_apq11),
        "Shimmer:DDA": _nan_to_num(shimmer_dda),
        "NHR": _nan_to_num(nhr),
        "HNR": _nan_to_num(hnr_db),
        "RPDE": _nan_to_num(rpde),
        "DFA": _nan_to_num(dfa),
        "spread1": _nan_to_num(spread1),
        "spread2": _nan_to_num(spread2),
        "D2": _nan_to_num(d2),
        "PPE": _nan_to_num(ppe),
    }

    # Ensure all keys exist
    for k in REQUIRED_UCI_PD_COLUMNS:
        feats.setdefault(k, 0.0)
    return feats


@dataclass
class UCIPDVectorizer:
    """Aligns feature dicts to UCI PD header order and standardizes using training CSV stats."""

    header: Tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        label_col: str = "status",
        drop_cols: Sequence[str] = ("name",),
    ) -> "UCIPDVectorizer":
        rows = _read_csv_rows(csv_path)
        if not rows:
            raise ValueError(f"Empty CSV at {csv_path}")
        header = rows[0]
        data = rows[1:]
        col_idx = {c: i for i, c in enumerate(header)}
        if label_col not in col_idx:
            raise ValueError(f"Label column '{label_col}' not found in CSV header")
        y_idx = col_idx[label_col]
        drop_idx = {col_idx[c] for c in drop_cols if c in col_idx}
        x_cols = [i for i in range(len(header)) if i not in drop_idx and i != y_idx]
        x_header = tuple(header[i] for i in x_cols)

        feats: List[List[float]] = []
        for r in data:
            try:
                feats.append([float(r[i]) for i in x_cols])
            except Exception:
                # skip malformed row
                continue
        X = np.asarray(feats, dtype=np.float32)
        means = X.mean(axis=0)
        stds = X.std(axis=0) + 1e-6
        return cls(header=x_header, means=means, stds=stds)

    def transform(self, feat_dict: Dict[str, float]) -> np.ndarray:
        vec = np.zeros((len(self.header),), dtype=np.float32)
        for i, name in enumerate(self.header):
            val = feat_dict.get(name, float(self.means[i]))
            vec[i] = float(val)
        # standardize
        vec = (vec - self.means.astype(np.float32)) / self.stds.astype(np.float32)
        return vec.reshape(1, -1)


def vectorize_wav(wav_path: str, vectorizer: UCIPDVectorizer) -> np.ndarray:
    """Convenience: extract features from WAV and standardize into a (1, D) vector."""
    try:
        feats = extract_ucipd_from_wav(wav_path)
    except Exception as e:
        # Re-raise to allow caller to log context
        raise
    return vectorizer.transform(feats)


