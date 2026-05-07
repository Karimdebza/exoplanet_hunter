"""
validation/physical_validator.py — Couche 4 : validation physique.

RÔLE :
Appliquer tous les tests physiques sur un candidat et produire un verdict
final classifié. C'est la dernière ligne de défense avant le reporting.

PIPELINE DES TESTS (tous exécutés, rapport complet) :
1. odd_even_depth      : profondeurs pairs/impairs identiques ?
2. secondary_eclipse   : pas de signal à phase 0.5 ?
3. depth_consistency   : profondeur stable sur tous les transits ?
4. duration_check      : durée cohérente avec la 3ème loi de Kepler ?

LOGIQUE DE CLASSIFICATION :
On ne fait pas un simple "tous passent = planète". On lit les patterns :
- secondary_eclipse FAIL → ECLIPSING_BINARY  (signal fort, conclusion directe)
- odd_even FAIL         → ECLIPSING_BINARY
- depth_consistency FAIL → NOISE ou INCONCLUSIVE
- duration_check FAIL   → INCONCLUSIVE (peut être erreur sur R★)
- Tout PASS             → PLANET_CANDIDATE

CONFIDENCE :
Score [0, 1] calculé comme moyenne pondérée des scores individuels.
Les tests à forte valeur discriminante (secondary, odd_even) ont plus de poids.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import CleanedLightCurve, TransitCandidate, CNNValidation, PhysicalValidation
from validation.tests import odd_even, secondary_eclipse, depth_consistency, duration_check


# Poids des tests pour le calcul de confidence
# secondary_eclipse et odd_even sont les plus discriminants
TEST_WEIGHTS = {
    "odd_even_depth"    : 0.30,
    "secondary_eclipse" : 0.35,
    "depth_consistency" : 0.20,
    "duration_check"    : 0.15,
}


def _classify(tests: list, cnn_score: float) -> tuple[str, float]:
    """
    Détermine la classification finale et le score de confidence.

    Logique :
    - Si secondary_eclipse échoue → ECLIPSING_BINARY (fort signal physique)
    - Si odd_even échoue          → ECLIPSING_BINARY
    - Si depth_consistency échoue → NOISE (signal instable)
    - Si duration_check échoue    → INCONCLUSIVE (incertitude stellaire)
    - Si tout passe               → PLANET_CANDIDATE
    - Si tests contradictoires    → INCONCLUSIVE
    """
    results = {t.test_name: t for t in tests}

    # Règles de classification par priorité
    sec = results.get("secondary_eclipse")
    oe  = results.get("odd_even_depth")
    dc  = results.get("depth_consistency")
    dur = results.get("duration_check")

    # Binaire à éclipses — signal fort
    if (sec and not sec.passed) or (oe and not oe.passed):
        classification = "ECLIPSING_BINARY"

    # Signal instable → bruit
    elif dc and not dc.passed:
        classification = "NOISE"

    # Durée incohérente → on ne peut pas conclure sans données stellaires
    elif dur and not dur.passed:
        classification = "INCONCLUSIVE"

    # Tout passe
    elif all(t.passed for t in tests):
        classification = "PLANET_CANDIDATE"

    else:
        classification = "INCONCLUSIVE"

    

    # Confidence = moyenne pondérée des (1 - score) pour les tests qui passent
    # score = 0 → parfait, score = 1 → limite du seuil
    total_weight = 0.0
    weighted_sum = 0.0
    for t in tests:
        w = TEST_WEIGHTS.get(t.test_name, 0.1)
        # Pour les tests qui passent : contribution positive
        # Pour les tests qui échouent : contribution négative
        contribution = (1.0 - min(t.score, 1.0)) if t.passed else 0.0
        weighted_sum += w * contribution
        total_weight += w

    # Intégrer le score CNN (déjà entre 0 et 1)
    physical_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5
    confidence = 0.6 * physical_confidence + 0.4 * cnn_score


    return classification, round(float(confidence), 3)


def validate(
    lc: CleanedLightCurve,
    cnn_validation: CNNValidation,
    r_star_solar: float = 1.0,
    m_star_solar: float = 1.0,
) -> PhysicalValidation:
    """
    Applique tous les tests physiques et retourne un verdict classifié.

    Args:
        lc             : courbe de lumière nettoyée
        cnn_validation : résultat de la couche 3
        r_star_solar   : rayon stellaire en R☉ (optionnel, améliore duration_check)
        m_star_solar   : masse stellaire en M☉ (optionnel)

    Returns:
        PhysicalValidation avec classification et confidence
    """
    candidate = cnn_validation.candidate

    # Exécuter tous les tests — rapport complet, pas fail-fast
    tests = [
        odd_even.run(lc, candidate),
        secondary_eclipse.run(lc, candidate),
        depth_consistency.run(lc, candidate),
        duration_check.run(lc, candidate, r_star_solar, m_star_solar),
    ]

    classification, confidence = _classify(tests, cnn_validation.cnn_score)

    return PhysicalValidation(
        candidate=candidate,
        cnn_validation=cnn_validation,
        tests=tests,
        classification=classification,
        confidence=confidence,
    )