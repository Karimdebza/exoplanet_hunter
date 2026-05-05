"""
validation/tests/depth_consistency.py v2 — phase folding par moitiés temporelles.

POURQUOI ON A CHANGÉ :
La v1 mesurait transit par transit → MAD=264ppm pour signal de 340ppm → inutilisable.
La v2 phase-fold chaque moitié indépendamment → moyenne sur N/2 transits → robuste.

SEUIL :
score = |D_first - D_second| / mean(D_first, D_second)
< 0.20 → PASS
> 0.40 → FAIL
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.types import CleanedLightCurve, TransitCandidate, PhysicalTestResult

TEST_NAME      = "depth_consistency"
PASS_THRESHOLD = 0.20
FAIL_THRESHOLD = 0.40
N_BINS         = 51


def _fold_and_measure(time, flux, period, t0, duration, n_bins=N_BINS, phase_window=2.0):
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    relative_time = phase * period / duration
    in_window = np.abs(relative_time) < phase_window
    if in_window.sum() < n_bins // 2:
        return np.nan
    rel_t = relative_time[in_window]
    flx   = flux[in_window]
    bin_edges   = np.linspace(-phase_window, phase_window, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    folded      = np.full(n_bins, np.nan)
    for i in range(n_bins):
        in_bin = (rel_t >= bin_edges[i]) & (rel_t < bin_edges[i + 1])
        if in_bin.sum() > 0:
            folded[i] = np.median(flx[in_bin])
    nan_mask = np.isnan(folded)
    if nan_mask.all():
        return np.nan
    if nan_mask.any():
        folded[nan_mask] = np.interp(np.flatnonzero(nan_mask),
                                      np.flatnonzero(~nan_mask),
                                      folded[~nan_mask])
    edge_mask = np.abs(bin_centers) > 1.5
    baseline  = np.median(folded[edge_mask]) if edge_mask.sum() >= 3 else np.median(folded)
    if baseline <= 0:
        return np.nan
    depth = float(1.0 - np.min(folded / baseline))
    return depth if depth > 0 else np.nan


def run(lc: CleanedLightCurve, candidate: TransitCandidate) -> PhysicalTestResult:
    period, t0, duration = candidate.period, candidate.t0, candidate.duration
    n_transits_est = int((lc.time.max() - lc.time.min()) / period)
    if n_transits_est < 4:
        return PhysicalTestResult(test_name=TEST_NAME, passed=True, score=0.0,
            details={"n_transits_est": n_transits_est, "warning": "not_enough_transits"})
    t_mid  = (lc.time.min() + lc.time.max()) / 2.0
    first  = lc.time < t_mid
    second = lc.time >= t_mid
    if first.sum() < 100 or second.sum() < 100:
        return PhysicalTestResult(test_name=TEST_NAME, passed=True, score=0.0,
            details={"warning": "not_enough_points_in_halves"})
    d_first  = _fold_and_measure(lc.time[first],  lc.flux[first],  period, t0, duration)
    d_second = _fold_and_measure(lc.time[second], lc.flux[second], period, t0, duration)
    if np.isnan(d_first) or np.isnan(d_second):
        return PhysicalTestResult(test_name=TEST_NAME, passed=True, score=0.0,
            details={"warning": "fold_failed_on_one_half"})
    d_mean = (d_first + d_second) / 2.0
    if d_mean <= 0:
        return PhysicalTestResult(test_name=TEST_NAME, passed=True, score=0.0,
            details={"warning": "mean_depth_zero"})
    score = abs(d_first - d_second) / d_mean
    inconclusive = PASS_THRESHOLD <= score < FAIL_THRESHOLD
    return PhysicalTestResult(
        test_name=TEST_NAME,
        passed=score < FAIL_THRESHOLD,
        score=score,
        details={
            "n_transits_est" : n_transits_est,
            "d_first_ppm"    : round(d_first  * 1e6, 1),
            "d_second_ppm"   : round(d_second * 1e6, 1),
            "d_mean_ppm"     : round(d_mean   * 1e6, 1),
            "drift_relative" : round(score, 4),
            "inconclusive"   : inconclusive,
            "verdict"        : ("STABLE"      if score < PASS_THRESHOLD else
                                "INCONCLUSIVE" if score < FAIL_THRESHOLD else
                                "DRIFTING"),
        }
    )