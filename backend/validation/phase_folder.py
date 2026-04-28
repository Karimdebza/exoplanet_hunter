"""
validation/phase_folder.py — Produit l'input standardisé du CNN.

CONCEPT CENTRAL :
Le CNN reçoit toujours un signal de FORME IDENTIQUE :
- Longueur fixe : 201 points
- Centré sur le transit (point 100 = milieu du transit)
- Normalisé : 1.0 = flux nominal, baisse = transit
- Échelle horizontale : ±2× la durée du transit (pour voir les bords)

POURQUOI 201 POINTS ?
- Impair → un point central exact (index 100)
- Assez pour résoudre la forme (ingress, plateau, egress)
- Pas trop pour rester rapide à entraîner

POURQUOI ±2× DURATION ?
- Le transit lui-même prend ~50% de la fenêtre
- 25% de chaque côté pour voir le baseline (référence du flux nominal)
- Permet au CNN de juger la "remise à 1" après le transit (test de planète)

DIFFÉRENCE AVEC L'ANCIEN PIPELINE :
Avant : segments temporels arbitraires, transit pas centré, taille variable
Maintenant : le CNN voit TOUJOURS le même type d'image — il apprend la FORME pure.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import CleanedLightCurve, TransitCandidate


# Configuration standardisée — TOUS les inputs du CNN passent par ici
CNN_INPUT_LENGTH = 201           # Doit être impair pour avoir un centre exact
CNN_PHASE_WINDOW = 2.0           # ±2× durée du transit


@dataclass
class FoldedTransit:
    """
    Un transit phase-foldé prêt à être donné au CNN.
    
    INVARIANTS :
    - flux a toujours CNN_INPUT_LENGTH points
    - flux est centré sur 1.0 (baseline) avec creux vers le bas
    - Le minimum (transit) est autour de l'index central
    """
    flux: np.ndarray                  # Shape (CNN_INPUT_LENGTH,)
    candidate: TransitCandidate       # Référence au candidat source
    n_transits_folded: int            # Nb de transits moyennés (qualité du signal)
    
    def __post_init__(self):
        if len(self.flux) != CNN_INPUT_LENGTH:
            raise ValueError(
                f"flux doit avoir {CNN_INPUT_LENGTH} points, reçu: {len(self.flux)}"
            )
    
    def to_cnn_input(self) -> np.ndarray:
        """Convertit en shape attendu par Keras : (1, length, 1)."""
        return self.flux.reshape(1, CNN_INPUT_LENGTH, 1)


def fold_transit(
    lc: CleanedLightCurve,
    candidate: TransitCandidate,
    n_bins: int = CNN_INPUT_LENGTH,
    phase_window: float = CNN_PHASE_WINDOW,
) -> Optional[FoldedTransit]:
    """
    Replie tous les transits d'un candidat sur une seule période et bin.
    
    Args:
        lc: Courbe de lumière nettoyée
        candidate: Candidat avec période, t0, durée connus
        n_bins: Résolution de sortie (défaut 201)
        phase_window: Demi-largeur en multiples de durée (défaut 2.0)
    
    Returns:
        FoldedTransit prêt pour le CNN, ou None si pas assez de données
    
    ALGORITHME :
    1. Calcule la phase de chaque point : phase ∈ [-0.5, 0.5]
    2. Garde uniquement les points dans la fenêtre ±phase_window × durée
    3. Bin (moyenne par bucket) sur n_bins points équidistants
    4. Normalise pour avoir baseline ≈ 1.0
    """
    period = candidate.period
    t0 = candidate.t0
    duration = candidate.duration
    
    # ─────────────────────────────────────────────────────────────────────────
    # Étape 1 : calcul de la phase normalisée [-0.5, 0.5], 0 = centre du transit
    # ─────────────────────────────────────────────────────────────────────────
    phase = ((lc.time - t0 + 0.5 * period) % period) / period - 0.5
    
    # Conversion phase → "temps relatif au transit" en unités de durée
    # phase=0 → centre, phase=duration/period/2 → bord du transit
    relative_time = phase * period / duration
    
    # ─────────────────────────────────────────────────────────────────────────
    # Étape 2 : sélection des points dans la fenêtre ±phase_window
    # ─────────────────────────────────────────────────────────────────────────
    in_window = np.abs(relative_time) < phase_window
    
    if in_window.sum() < n_bins // 4:
        # Pas assez de points pour faire un fold propre
        return None
    
    rel_t_window = relative_time[in_window]
    flux_window = lc.flux[in_window]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Étape 3 : binning — moyenne par bucket pour réduire le bruit
    # ─────────────────────────────────────────────────────────────────────────
    bin_edges = np.linspace(-phase_window, phase_window, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    folded_flux = np.full(n_bins, np.nan)
    
    for i in range(n_bins):
        in_bin = (rel_t_window >= bin_edges[i]) & (rel_t_window < bin_edges[i + 1])
        if in_bin.sum() > 0:
            folded_flux[i] = np.median(flux_window[in_bin])  # Médiane > moyenne (robuste outliers)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Étape 4 : remplissage des bins vides par interpolation
    # ─────────────────────────────────────────────────────────────────────────
    nan_mask = np.isnan(folded_flux)
    if nan_mask.all():
        return None
    if nan_mask.any():
        folded_flux[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            folded_flux[~nan_mask]
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Étape 5 : normalisation à baseline = 1.0
    # On utilise les bords de la fenêtre comme référence (loin du transit)
    # ─────────────────────────────────────────────────────────────────────────
    edge_mask = np.abs(bin_centers) > 1.5  # bords externes
    if edge_mask.sum() >= 5:
        baseline = np.median(folded_flux[edge_mask])
    else:
        baseline = np.median(folded_flux)
    
    folded_flux = folded_flux / baseline
    
    # ─────────────────────────────────────────────────────────────────────────
    # Étape 6 : compter les transits effectivement repliés (qualité du signal)
    # ─────────────────────────────────────────────────────────────────────────
    n_transits = int((lc.time[-1] - lc.time[0]) / period)
    
    return FoldedTransit(
        flux=folded_flux.astype(np.float32),
        candidate=candidate,
        n_transits_folded=n_transits,
    )