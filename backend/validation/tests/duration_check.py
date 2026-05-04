"""
validation/tests/duration_check.py — Cohérence durée / loi de Kepler.

PRINCIPE PHYSIQUE :
La durée d'un transit dépend de la période orbitale et de la taille
de l'étoile. Pour une orbite circulaire, la 3ème loi de Kepler donne :

    T_transit = (P / π) × arcsin(R★ / a)

où :
- P  : période orbitale
- R★ : rayon stellaire
- a  : demi-grand axe (calculé depuis P et M★ via la 3ème loi)

Si la durée observée est incompatible avec cette formule (trop longue
ou trop courte d'un facteur > 3), le signal n'est pas un transit planétaire.

VALEURS PAR DÉFAUT :
On suppose une étoile de type solaire (R★ = 1 R☉, M★ = 1 M☉).
C'est une approximation — les étoiles Kepler varient de 0.5 à 2 R☉.
En l'absence de données stellaires précises, c'est le meilleur qu'on peut faire.

TOLÉRANCE LARGE (facteur 3) :
On tolère un écart de facteur 3 car :
1. L'incertitude sur R★ et M★ peut être de 50%
2. L'impact du paramètre b (impact parameter) peut réduire/allonger la durée
3. Notre lissage peut avoir légèrement modifié la durée mesurée par BLS

LIMITE CONNUE :
Ce test ne remplace pas une caractérisation stellaire précise.
Il filtre les cas aberrants (durée 10× trop longue = clairement pas un transit).
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.types import CleanedLightCurve, TransitCandidate, PhysicalTestResult

TEST_NAME = "duration_check"

# Constantes physiques
R_SUN  = 6.957e8    # m
M_SUN  = 1.989e30   # kg
G      = 6.674e-11  # m³ kg⁻¹ s⁻²
DAY_S  = 86400.0    # secondes par jour

# Tolérance : on accepte un écart de facteur TOLERANCE_FACTOR
TOLERANCE_FACTOR = 3.0


def _expected_duration_days(
    period_days: float,
    r_star_solar: float = 1.0,
    m_star_solar: float = 1.0,
) -> float:
    """
    Durée théorique du transit pour une orbite circulaire (impact parameter b=0).

    C'est la durée MAXIMALE possible — b > 0 donne des transits plus courts.

    Formule : T = (P/π) × arcsin(R★ / a)
    avec a calculé depuis la 3ème loi : a³ = G M★ P² / (4π²)
    """
    period_s  = period_days * DAY_S
    r_star    = r_star_solar * R_SUN
    m_star    = m_star_solar * M_SUN

    # Demi-grand axe via 3ème loi de Kepler
    a = (G * m_star * period_s**2 / (4 * np.pi**2)) ** (1/3)

    if a <= r_star:
        # Physiquement impossible (planète dans l'étoile)
        return np.nan

    # Durée du transit (b=0 = transit central = durée max)
    sin_arg = min(r_star / a, 1.0)   # clamp pour éviter arcsin > 1
    duration_s = (period_s / np.pi) * np.arcsin(sin_arg)

    return duration_s / DAY_S


def run(
    lc: CleanedLightCurve,
    candidate: TransitCandidate,
    r_star_solar: float = 1.0,
    m_star_solar: float = 1.0,
) -> PhysicalTestResult:
    """
    Compare la durée observée à la durée théorique de Kepler.

    Args:
        lc            : courbe de lumière (pour contexte)
        candidate     : candidat à tester
        r_star_solar  : rayon stellaire en R☉ (défaut = 1.0 = solaire)
        m_star_solar  : masse stellaire en M☉ (défaut = 1.0 = solaire)
    """
    duration_obs = candidate.duration      # jours
    period       = candidate.period        # jours

    duration_expected = _expected_duration_days(period, r_star_solar, m_star_solar)

    if np.isnan(duration_expected) or duration_expected <= 0:
        return PhysicalTestResult(
            test_name=TEST_NAME, passed=True, score=1.0,
            details={"warning": "expected_duration_uncomputable"}
        )

    # Ratio observé / attendu
    ratio = duration_obs / duration_expected

    # PASS si ratio ∈ [1/FACTOR, FACTOR]
    # On tolère les transits plus courts (b > 0) mais pas les plus longs
    # qu'un facteur TOLERANCE_FACTOR
    lower = 1.0 / TOLERANCE_FACTOR
    upper = TOLERANCE_FACTOR
    passed = lower <= ratio <= upper

    return PhysicalTestResult(
        test_name=TEST_NAME,
        passed=passed,
        score=ratio,
        details={
            "period_days"           : round(period, 4),
            "duration_obs_hours"    : round(duration_obs * 24, 2),
            "duration_expected_hours": round(duration_expected * 24, 2),
            "ratio_obs_expected"    : round(ratio, 3),
            "tolerance_factor"      : TOLERANCE_FACTOR,
            "r_star_solar"          : r_star_solar,
            "m_star_solar"          : m_star_solar,
            "verdict"               : "CONSISTENT" if passed else "INCONSISTENT",
        }
    )