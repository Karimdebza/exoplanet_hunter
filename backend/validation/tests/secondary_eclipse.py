"""
validation/tests/secondary_eclipse.py — Recherche d'éclipse secondaire.

PRINCIPE PHYSIQUE :
Une planète n'émet pas de lumière visible. Quand elle passe DERRIÈRE
l'étoile (phase 0.5), le flux total ne change pas de manière détectable.

Une binaire à éclipses, par contre : quand l'étoile secondaire passe
derrière la primaire (phase 0.5), on perd sa contribution lumineuse.
→ Chute de flux à phase 0.5 = l'objet émet de la lumière = étoile.

MÉTHODE :
1. Phase-folder la courbe sur la période du candidat
2. Mesurer la profondeur moyenne autour de phase 0.5 (±durée/2)
3. Comparer à la profondeur du transit primaire (phase 0.0)

SEUIL :
ratio = depth_secondary / depth_primary
< 0.05 (5%)  → pas d'éclipse secondaire → PASS
> 0.10 (10%) → éclipse secondaire détectée → FAIL (binaire)

LIMITE :
Pour des orbites excentriques, l'éclipse secondaire peut ne pas être
exactement à phase 0.5. Ce test assume une orbite circulaire — valide
pour les Hot Jupiters (circularisation rapide par effets de marée).
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.types import CleanedLightCurve, TransitCandidate, PhysicalTestResult

TEST_NAME      = "secondary_eclipse"
PASS_THRESHOLD = 0.05
FAIL_THRESHOLD = 0.10


def run(lc: CleanedLightCurve, candidate: TransitCandidate) -> PhysicalTestResult:
    period, t0, duration = candidate.period, candidate.t0, candidate.duration

    # Phase folding
    phase = ((lc.time - t0) % period) / period
    phase[phase > 0.5] -= 1.0   # centre sur 0, range [-0.5, 0.5]

    half_dur_phase = (duration / period) / 2.0

    # ── Transit primaire (phase ≈ 0.0) ───────────────────────────────────────
    in_primary  = np.abs(phase) < half_dur_phase
    out_primary = np.abs(phase) > 4 * half_dur_phase   # loin du transit

    if in_primary.sum() < 5 or out_primary.sum() < 20:
        return PhysicalTestResult(
            test_name=TEST_NAME, passed=True, score=0.0,
            details={"warning": "not_enough_points_primary"}
        )

    baseline      = float(np.median(lc.flux[out_primary]))
    depth_primary = float(baseline - np.median(lc.flux[in_primary]))

    if depth_primary <= 0:
        return PhysicalTestResult(
            test_name=TEST_NAME, passed=True, score=0.0,
            details={"warning": "primary_depth_zero_or_negative"}
        )

    # ── Transit secondaire (phase ≈ ±0.5) ────────────────────────────────────
    # On cherche dans une fenêtre de ±2× durée autour de phase 0.5
    secondary_phase_center = 0.5
    in_secondary = np.abs(np.abs(phase) - secondary_phase_center) < half_dur_phase * 2

    if in_secondary.sum() < 5:
        return PhysicalTestResult(
            test_name=TEST_NAME, passed=True, score=0.0,
            details={
                "warning"        : "not_enough_points_secondary",
                "depth_primary_ppm": round(depth_primary * 1e6, 1),
            }
        )

    depth_secondary = float(baseline - np.median(lc.flux[in_secondary]))
    # depth_secondary peut être négatif (bosse au lieu d'un creux) → on prend abs
    # Une bosse à phase 0.5 = réflexion de la planète (pas une binaire, c'est OK)
    ratio = max(0.0, depth_secondary) / depth_primary

    inconclusive = PASS_THRESHOLD <= ratio < FAIL_THRESHOLD

    return PhysicalTestResult(
        test_name=TEST_NAME,
        passed=ratio < FAIL_THRESHOLD,
        score=ratio,
        details={
            "depth_primary_ppm"  : round(depth_primary   * 1e6, 1),
            "depth_secondary_ppm": round(depth_secondary * 1e6, 1),
            "ratio"              : round(ratio, 4),
            "inconclusive"       : inconclusive,
            "verdict"            : ("NO_SECONDARY"       if ratio < PASS_THRESHOLD else
                                    "INCONCLUSIVE"        if ratio < FAIL_THRESHOLD else
                                    "SECONDARY_DETECTED"),
        }
    )