"""
validation/tests/odd_even.py — Test de profondeur pair/impair.

PRINCIPE PHYSIQUE :
Une vraie planète produit TOUJOURS le même transit — même profondeur,
même durée, même forme. L'objet est le même à chaque passage.

Une binaire à éclipses produit DEUX transits différents :
- Transit primaire   : étoile A passe devant étoile B (profonde)
- Transit secondaire : étoile B passe devant étoile A (moins profonde)

Si on numérote les transits 1, 2, 3, 4... et qu'on compare :
- Transits impairs (1, 3, 5...) : profondeur D1
- Transits pairs   (2, 4, 6...) : profondeur D2

Planète    → D1 ≈ D2
Binaire EB → D1 ≠ D2

SEUIL :
score = |D_odd - D_even| / mean(D_odd, D_even)
< 0.10 → PASS
> 0.20 → FAIL (binaire probable)
entre les deux → INCONCLUSIVE
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.types import CleanedLightCurve, TransitCandidate, PhysicalTestResult

TEST_NAME       = "odd_even_depth"
PASS_THRESHOLD  = 0.10
FAIL_THRESHOLD  = 0.20


def _measure_transit_depth(flux, time, t_center, duration, baseline_window=3.0):
    """Profondeur d'un transit individuel = médiane(baseline) - médiane(in-transit)."""
    half_dur  = duration / 2.0
    half_base = half_dur * baseline_window

    in_transit = (time >= t_center - half_dur)  & (time <= t_center + half_dur)
    baseline   = ((time >= t_center - half_base) & (time < t_center - half_dur)) | \
                 ((time >  t_center + half_dur)  & (time <= t_center + half_base))

    if in_transit.sum() < 3 or baseline.sum() < 6:
        return np.nan

    return float(np.median(flux[baseline]) - np.median(flux[in_transit]))


def run(lc: CleanedLightCurve, candidate: TransitCandidate) -> PhysicalTestResult:
    period, t0, duration = candidate.period, candidate.t0, candidate.duration

    n_min = int(np.ceil( (lc.time.min() - t0) / period))
    n_max = int(np.floor((lc.time.max() - t0) / period))
    transit_numbers = np.arange(n_min, n_max + 1)
    transit_centers = t0 + transit_numbers * period

    mask = (transit_centers >= lc.time.min()) & (transit_centers <= lc.time.max())
    transit_centers = transit_centers[mask]
    transit_numbers = transit_numbers[mask]
    n_transits = len(transit_centers)

    if n_transits < 4:
        return PhysicalTestResult(
            test_name=TEST_NAME, passed=True, score=0.0,
            details={"n_transits": n_transits, "warning": "not_enough_transits"}
        )

    depths   = np.array([_measure_transit_depth(lc.flux, lc.time, tc, duration) for tc in transit_centers])
    parities = transit_numbers % 2
    odd_d    = depths[parities == 1]; odd_d  = odd_d[~np.isnan(odd_d)]
    even_d   = depths[parities == 0]; even_d = even_d[~np.isnan(even_d)]

    if len(odd_d) < 2 or len(even_d) < 2:
        return PhysicalTestResult(
            test_name=TEST_NAME, passed=True, score=0.0,
            details={"warning": "not_enough_valid_transits"}
        )

    d_odd, d_even = float(np.median(odd_d)), float(np.median(even_d))
    d_mean = (d_odd + d_even) / 2.0
    if d_mean <= 0:
        return PhysicalTestResult(test_name=TEST_NAME, passed=True, score=0.0,
                                  details={"warning": "depth_mean_zero"})

    score = abs(d_odd - d_even) / d_mean
    inconclusive = PASS_THRESHOLD <= score < FAIL_THRESHOLD

    return PhysicalTestResult(
        test_name=TEST_NAME,
        passed=score < FAIL_THRESHOLD,  # reject only if > 20%
        score=score,
        details={
            "n_transits"    : n_transits,
            "n_odd"         : len(odd_d),
            "n_even"        : len(even_d),
            "odd_depth_ppm" : round(d_odd  * 1e6, 1),
            "even_depth_ppm": round(d_even * 1e6, 1),
            "diff_relative" : round(score, 4),
            "inconclusive"  : inconclusive,
            "verdict"       : ("PLANET_LIKE" if score < PASS_THRESHOLD else
                               "INCONCLUSIVE" if score < FAIL_THRESHOLD else
                               "BINARY_SUSPECT"),
        }
    )