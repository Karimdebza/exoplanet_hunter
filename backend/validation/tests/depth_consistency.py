"""
validation/tests/depth_consistency.py — Stabilité de la profondeur transit par transit.

PRINCIPE PHYSIQUE :
Une vraie planète a une taille fixe. Chaque transit doit avoir la même
profondeur (à bruit près). Si la profondeur varie significativement
d'un transit à l'autre → le signal n'est pas physiquement cohérent.

Causes de profondeur variable :
- Contamination par une étoile de fond (background EB)
- Activité stellaire (taches) qui modifie la profondeur apparente
- Artefacts instrumentaux Kepler (rolling band, cosmic rays)
- Signal BLS fortuit (pas de vrai transit)

MÉTHODE :
1. Mesurer la profondeur de chaque transit individuel
2. Calculer le coefficient de variation (CV = std / mean)
3. CV < 0.3 → signal stable → PASS
   CV > 0.5 → signal instable → FAIL

DIFFÉRENCE AVEC ODD/EVEN :
Odd/even teste une ASYMÉTRIE SYSTÉMATIQUE pair vs impair (binaire EB).
Depth consistency teste la VARIANCE GLOBALE sur tous les transits (artefacts).
Les deux tests sont complémentaires, pas redondants.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.types import CleanedLightCurve, TransitCandidate, PhysicalTestResult

TEST_NAME      = "depth_consistency"
PASS_THRESHOLD = 0.30   # CV < 30% → stable
FAIL_THRESHOLD = 0.50   # CV > 50% → instable


def _measure_transit_depth(flux, time, t_center, duration, baseline_window=3.0):
    """Profondeur d'un transit individuel (médiane baseline - médiane in-transit)."""
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
    centers = t0 + np.arange(n_min, n_max + 1) * period
    centers = centers[(centers >= lc.time.min()) & (centers <= lc.time.max())]

    if len(centers) < 3:
        return PhysicalTestResult(
            test_name=TEST_NAME, passed=True, score=0.0,
            details={"n_transits": len(centers), "warning": "not_enough_transits"}
        )

    depths = np.array([_measure_transit_depth(lc.flux, lc.time, tc, duration) for tc in centers])
    depths = depths[~np.isnan(depths)]

    if len(depths) < 3:
        return PhysicalTestResult(
            test_name=TEST_NAME, passed=True, score=0.0,
            details={"warning": "not_enough_valid_transits"}
        )

    mean_depth = float(np.mean(depths))
    std_depth  = float(np.std(depths))

    if mean_depth <= 0:
        return PhysicalTestResult(
            test_name=TEST_NAME, passed=False, score=1.0,
            details={"warning": "mean_depth_zero_or_negative"}
        )

    cv = std_depth / mean_depth   # Coefficient de variation
    inconclusive = PASS_THRESHOLD <= cv < FAIL_THRESHOLD
    print(f"DEBUG depths_ppm: {[round(d*1e6,1) for d in depths]}")

    return PhysicalTestResult(
        test_name=TEST_NAME,
        passed=cv < FAIL_THRESHOLD,
        score=cv,
        details={
            "n_transits"      : len(centers),
            "n_valid"         : len(depths),
            "mean_depth_ppm"  : round(mean_depth * 1e6, 1),
            "std_depth_ppm"   : round(std_depth  * 1e6, 1),
            "cv"              : round(cv, 4),
            "inconclusive"    : inconclusive,
            "verdict"         : ("STABLE"        if cv < PASS_THRESHOLD else
                                 "INCONCLUSIVE"   if cv < FAIL_THRESHOLD else
                                 "UNSTABLE"),
            "depths_ppm"      : [round(d * 1e6, 1) for d in depths],
        }
    )