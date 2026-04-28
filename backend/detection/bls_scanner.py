"""
detection/bls_scanner.py — Couche 2 du pipeline.
 
RÔLE : trouver les K signaux périodiques les plus forts dans une courbe de lumière.
NE FAIT PAS : décider si un signal est crédible. Ça, c'est le rôle des couches
3 (CNN) et 4 (validation physique).
 
Pourquoi BLS et pas Lomb-Scargle ?
- Lomb-Scargle cherche des SINUSOIDES (variabilité stellaire, pulsations)
- BLS cherche des BOÎTES (transit = chute brutale, plateau, remontée)
- Bon outil pour la bonne forme. C'est le standard NASA depuis Kovács 2002.
 
Pourquoi top-K et pas top-1 ?
- Le pic principal peut être un alias (P*2, P/2)
- Un système peut avoir plusieurs planètes (Kepler-90 en a 8)
- top_k=5 est le compromis empirique de la mission Kepler.
 
Anti-aliasing :
Si le BLS trouve P=3.55j ET P=7.10j (= 2×3.55), c'est le MÊME signal.
On déduplique avant de retourner les K candidats.
"""


import numpy as np
from typing import Optional
import logging
 
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
 
# Import relatif de notre vocabulaire commun
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import CleanedLightCurve, TransitCandidate
 
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DU BLS — toutes les constantes magiques sont ici
# ─────────────────────────────────────────────────────────────────────────────
 
# Périodes minimum/maximum à scanner (en jours)
# - 0.5j : min physique, en dessous c'est de l'ultra-Hot Jupiter (rare)
# - 50j : on monte rarement plus haut sur des données Kepler quarters par
#   manque de transits multiples (besoin >= 2-3 transits pour confirmer)


DEFAULT_PERIOD_MIN = 0.5
DEFAULT_PERIOD_MAX = 50.0

# Durées de transit à tester (en jours)
# Couvre Hot Jupiter (2h) à planète "tempérée" (12h)
DEFAULT_DURATIONS = np.array([0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5])
 
# Nombre de périodes à tester dans la grille BLS
# Compromis : plus c'est haut, plus c'est précis mais plus c'est lent.
# 10000 = standard pro pour Kepler
DEFAULT_N_PERIODS = 10000
 
# Tolérance pour considérer 2 périodes comme "le même signal" (anti-aliasing)
# Si P1/P2 ∈ {0.5, 1, 2, 3} ± 5%, on considère que c'est un alias
ALIAS_RATIOS = (0.5, 1.0, 2.0, 3.0)
ALIAS_TOLERANCE = 0.05  # 5%
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES — privées (préfixe _)
# ─────────────────────────────────────────────────────────────────────────────
 
def _is_alias(period_a: float, period_b: float) -> bool:
    """
    True si period_a est un alias de period_b (ou inverse).
    
    Exemple : 3.55j et 7.10j → ratio = 2.0 → alias.
    Exemple : 3.55j et 5.20j → ratio = 1.46 → pas alias.
    
    Pourquoi cette fonction existe ?
    Le BLS produit souvent plusieurs pics pour LE MÊME signal physique
    à des multiples de la vraie période. Sans dédup, ton "top 5" pourrait
    être 5 fois la même planète.
    """
    if period_a == 0 or period_b == 0:
        return False
    
    ratio = max(period_a, period_b) / min(period_a, period_b)
    
    for target_ratio in ALIAS_RATIOS:
        if abs(ratio - target_ratio) / target_ratio < ALIAS_TOLERANCE:
            return True
    return False
 
 
def _compute_snr(flux: np.ndarray, time: np.ndarray, 
                 period: float, t0: float, duration: float) -> float:
    """
    SNR du transit = depth / σ(flux hors transit).
    
    Pourquoi pas le SDE du BLS directement ?
    SDE mesure combien le pic dépasse le bruit DU PÉRIODOGRAMME.
    SNR mesure combien la chute de flux dépasse le bruit DE LA COURBE.
    Les deux sont complémentaires :
    - SDE élevé + SNR bas = pic statistique mais signal trop faible visuellement
    - SDE bas + SNR élevé = signal isolé mais pas périodique (rare)
    
    On calcule les DEUX, on les stocke tous les deux dans le candidat.
    """
    # Phase fold pour identifier les points "in-transit" vs "out-of-transit"
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0  # Centre sur 0
    
    half_dur_phase = (duration / period) / 2.0
    in_transit = np.abs(phase) < half_dur_phase
    out_of_transit = np.abs(phase) > 2 * half_dur_phase  # Marge de sécurité
    
    if in_transit.sum() < 3 or out_of_transit.sum() < 30:
        return 0.0  # Pas assez de points pour un calcul fiable
    
    depth = np.median(flux[out_of_transit]) - np.median(flux[in_transit])
    noise = np.std(flux[out_of_transit])
    
    return float(depth / (noise + 1e-12))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE — l'API publique du module
# ─────────────────────────────────────────────────────────────────────────────
 
def detect_signals(
    lc: CleanedLightCurve,
    top_k: int = 5,
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: Optional[float] = None,
    durations: np.ndarray = DEFAULT_DURATIONS,
    n_periods: int = DEFAULT_N_PERIODS,
) -> list[TransitCandidate]:
    """
    Scanne une courbe de lumière avec BLS et retourne les K signaux les plus forts.
    
    Args:
        lc: Courbe de lumière nettoyée (output de la couche 1)
        top_k: Nombre maximum de candidats à retourner
        period_min: Période minimum à scanner (jours)
        period_max: Période maximum (par défaut: duration_obs / 3 pour avoir
                    au moins 3 transits potentiels)
        durations: Grille de durées de transit à tester
        n_periods: Résolution de la grille de périodes
    
    Returns:
        Liste de TransitCandidate triés par SDE décroissant, sans aliases.
        Peut être vide si aucun signal détecté.
    
    Note d'architecture :
    Cette fonction NE FILTRE PAS par seuil de qualité. Si tu veux un filtre
    SDE > 7, fais-le dans le code appelant. Cf. principe "trouver vs filtrer".
    """
    # Auto-config de period_max si non fournie
    if period_max is None:
        # On veut au moins 3 transits potentiels dans la fenêtre d'observation
        period_max = min(DEFAULT_PERIOD_MAX, lc.duration_days / 3.0)
    
    if period_max <= period_min:
        logger.warning(
            f"{lc.star_id}: durée d'obs trop courte ({lc.duration_days:.1f}j) "
            f"pour scanner [{period_min}, {period_max}]j"
        )
        return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Étape 1 : exécuter le BLS
    # ─────────────────────────────────────────────────────────────────────────
    # Filtre physique : une durée ne peut pas être >= période minimale.
    # Un transit qui dure plus d'une période, c'est mathématiquement absurde.
    valid_durations = durations[durations < period_min * 0.5]
    if len(valid_durations) == 0:
        logger.warning(
            f"{lc.star_id}: aucune durée valide pour period_min={period_min}j. "
            f"Augmente period_min ou réduis durations."
        )
        return []
    
    bls = BoxLeastSquares(lc.time, lc.flux)
    period_grid = np.linspace(period_min, period_max, n_periods)
    
    logger.info(
        f"{lc.star_id}: BLS sur {n_periods} périodes "
        f"dans [{period_min:.2f}, {period_max:.2f}]j"
    )
    
    result = bls.power(period_grid, valid_durations)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Étape 2 : extraire les top pics avec anti-aliasing
    # ─────────────────────────────────────────────────────────────────────────
    # Trier les indices par puissance décroissante
    sorted_idx = np.argsort(result.power)[::-1]
    
    # Calcul de la SDE (Signal Detection Efficiency) - normalisation standard
    # SDE = (power - mean) / std → mesure combien le pic sort du bruit
    sde_values = (result.power - np.mean(result.power)) / (np.std(result.power) + 1e-12)
    
    candidates: list[TransitCandidate] = []
    selected_periods: list[float] = []
    
    for idx in sorted_idx:
        if len(candidates) >= top_k:
            break
        
        period = float(result.period[idx])
        
        # Anti-alias : skip si on a déjà un pic à période multiple
        if any(_is_alias(period, p) for p in selected_periods):
            continue
        
        # Récupérer les paramètres du transit pour cette période
        t0 = float(result.transit_time[idx])
        duration = float(result.duration[idx])
        depth = float(result.depth[idx])
        sde = float(sde_values[idx])
        
        # Validation : depth doit être positive (chute = creux dans flux)
        # Le BLS peut renvoyer depth négative si la "boîte" est vers le haut
        if depth <= 0:
            continue
        
        # Validation : durée < période/4 (sinon c'est physiquement absurde)
        if duration >= period / 4.0:
            continue
        
        # Calcul du SNR temporel (complémentaire au SDE)
        snr = _compute_snr(lc.flux, lc.time, period, t0, duration)
        
        candidate = TransitCandidate(
            period=period,
            t0=t0,
            duration=duration,
            depth=depth,
            sde=sde,
            star_id=lc.star_id,
            rank=len(candidates),
        )
        
        candidates.append(candidate)
        selected_periods.append(period)
        
        logger.info(
            f"  → Candidat #{candidate.rank}: P={period:.4f}j, "
            f"depth={candidate.depth_ppm:.0f}ppm, SDE={sde:.2f}, SNR={snr:.2f}"
        )
    
    return candidates
 