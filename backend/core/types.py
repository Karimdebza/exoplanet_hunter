"""
core/types.py — Vocabulaire commun du pipeline ExoplanetHunter.

Ce fichier définit les structures de données qui circulent entre les couches.
RÈGLE D'OR : aucune logique métier ici, uniquement du contenu structuré.

Pourquoi des dataclasses et pas des dicts ?
- Typage explicite : ton IDE te corrige avant l'exécution
- Auto-complétion : tu ne tapes pas "candidate['perio']" sans t'en rendre compte
- Documentation : la signature dit tout, pas besoin de lire le code
- Validation possible : on peut ajouter des __post_init__ pour vérifier les invariants

Alternative écartée : Pydantic. Plus puissant (validation runtime, sérialisation JSON
gratuite), mais ajoute une dépendance et de la complexité. À reconsidérer en v3
quand on aura une API HTTP.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# COUCHE 1 — Output de l'ingestion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanedLightCurve:
    """
    Courbe de lumière nettoyée, prête à être analysée.
    
    INVARIANTS (à respecter par tous les producteurs) :
    - time et flux ont la même longueur
    - flux est centré autour de 1.0 (luminosité relative, pas absolue)
    - les NaN ont déjà été supprimés
    - le detrending a déjà été appliqué (variations stellaires longues retirées)
    
    Pourquoi flux_err en optionnel ?
    Parce que tous les pipelines ne l'exposent pas. Mais quand il existe, c'est
    précieux : il pondère le BLS et améliore la détection (chi² weighted).
    """
    time: np.ndarray              # Temps en BKJD (Barycentric Kepler Julian Date)
    flux: np.ndarray              # Flux normalisé autour de 1.0
    star_id: str                  # Ex: "Kepler-5", "KIC 11657614"
    quarters_used: list[int]      # Ex: [3, 4, 5, 6]
    cadence: str = "long"         # "long" (29.4 min) ou "short" (58.85 sec)
    flux_err: Optional[np.ndarray] = None
    
    def __post_init__(self):
        # Validation des invariants — on plante TÔT plutôt que de produire du bruit
        if len(self.time) != len(self.flux):
            raise ValueError(
                f"time ({len(self.time)}) et flux ({len(self.flux)}) "
                f"doivent avoir la même longueur"
            )
        if self.flux_err is not None and len(self.flux_err) != len(self.flux):
            raise ValueError("flux_err doit avoir la même longueur que flux")
        if self.cadence not in ("long", "short"):
            raise ValueError(f"cadence doit être 'long' ou 'short', reçu: {self.cadence}")
    
    @property
    def duration_days(self) -> float:
        """Durée totale d'observation en jours. Utile pour BLS (période max = duration/2)."""
        return float(self.time[-1] - self.time[0])
    
    @property
    def n_points(self) -> int:
        return len(self.flux)


# ─────────────────────────────────────────────────────────────────────────────
# COUCHE 2 — Output de la détection BLS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitCandidate:
    """
    Un signal périodique détecté qui POURRAIT être un transit.
    
    À ce stade, on ne sait PAS si c'est :
    - une vraie planète
    - une binaire à éclipses
    - une variabilité stellaire qui ressemble à un transit
    - un alias d'une autre période (P/2, P*2)
    
    C'est aux couches 3 (CNN) et 4 (validation physique) de trancher.
    
    Pourquoi 'sde' (Signal Detection Efficiency) plutôt que 'snr' ?
    SDE est la métrique standard du BLS : elle mesure combien le pic dépasse
    le bruit du périodogramme. SDE > 7 est le seuil pro de "signal crédible"
    dans la littérature Kepler. SNR au sens classique se calcule sur la
    courbe de lumière, pas sur le périodogramme.
    """
    period: float                 # Période orbitale en jours
    t0: float                     # Temps du premier transit (BKJD)
    duration: float               # Durée du transit en jours
    depth: float                  # Profondeur relative (ex: 0.01 = 1%)
    sde: float                    # Signal Detection Efficiency (BLS)
    
    star_id: str                  # On garde une référence à l'étoile source
    rank: int = 0                 # Rang dans le top-K (0 = signal le plus fort)
    
    def __post_init__(self):
        if self.period <= 0:
            raise ValueError(f"period doit être > 0, reçu: {self.period}")
        if self.duration <= 0 or self.duration >= self.period:
            raise ValueError(
                f"duration ({self.duration}) doit être dans ]0, period[ "
                f"(period={self.period})"
            )
        if self.depth < 0:
            raise ValueError(f"depth doit être >= 0, reçu: {self.depth}")
    
    @property
    def duration_hours(self) -> float:
        """Durée en heures — plus parlant pour comparer aux ordres de grandeur physiques."""
        return self.duration * 24.0
    
    @property
    def depth_ppm(self) -> float:
        """Profondeur en parties par million — unité standard en exoplanétologie."""
        return self.depth * 1_000_000
    
    def __repr__(self) -> str:
        return (
            f"TransitCandidate({self.star_id}, "
            f"P={self.period:.3f}j, "
            f"depth={self.depth_ppm:.0f}ppm, "
            f"dur={self.duration_hours:.2f}h, "
            f"SDE={self.sde:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# COUCHE 3 — Output du CNN (validation de forme)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CNNValidation:
    """
    Résultat du CNN sur un candidat. Le CNN ne décide pas seul, il scoree.
    
    'passed' est calculé par rapport à un threshold configurable. On ne stocke
    PAS le threshold ici parce qu'il peut évoluer (calibration sur dataset).
    """
    candidate: TransitCandidate
    cnn_score: float              # [0, 1]
    passed: bool
    rejection_reason: Optional[str] = None  # Renseigné si passed=False
    
    def __post_init__(self):
        if not 0.0 <= self.cnn_score <= 1.0:
            raise ValueError(f"cnn_score doit être dans [0,1], reçu: {self.cnn_score}")


# ─────────────────────────────────────────────────────────────────────────────
# COUCHE 4 — Output de la validation physique
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicalTestResult:
    """Résultat d'UN test physique (odd/even, secondary, V-shape, etc.)."""
    test_name: str                # Ex: "odd_even_depth"
    passed: bool
    score: float                  # Métrique numérique du test (sens dépend du test)
    details: dict = field(default_factory=dict)  # Données diagnostiques
    
    def __repr__(self) -> str:
        verdict = "✓" if self.passed else "✗"
        return f"{verdict} {self.test_name} (score={self.score:.3f})"


@dataclass
class PhysicalValidation:
    """
    Verdict final après toutes les validations physiques.
    
    'classification' est l'étiquette finale du pipeline. Valeurs possibles :
    - PLANET_CANDIDATE : passe tous les tests, mérite une analyse humaine
    - ECLIPSING_BINARY : forme/profondeur incompatible avec une planète
    - STELLAR_VARIABILITY : variabilité de l'étoile elle-même
    - NOISE : signal pas reproductible
    - INCONCLUSIVE : tests contradictoires, besoin de plus de données
    """
    candidate: TransitCandidate
    cnn_validation: CNNValidation
    tests: list[PhysicalTestResult]
    classification: str
    confidence: float             # [0, 1] — fusion des scores des tests
    
    def __post_init__(self):
        valid_classifications = {
            "PLANET_CANDIDATE", "ECLIPSING_BINARY", 
            "STELLAR_VARIABILITY", "NOISE", "INCONCLUSIVE"
        }
        if self.classification not in valid_classifications:
            raise ValueError(
                f"classification invalide: {self.classification}. "
                f"Valeurs autorisées: {valid_classifications}"
            )
    
    @property
    def n_tests_passed(self) -> int:
        return sum(1 for t in self.tests if t.passed)
    
    @property
    def n_tests_total(self) -> int:
        return len(self.tests)
    
    def get_test(self, name: str) -> Optional[PhysicalTestResult]:
        """Récupère un test par nom. Utile pour le reporting."""
        for t in self.tests:
            if t.test_name == name:
                return t
        return None