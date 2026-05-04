"""
validation/dataset_builder.py — Génération du dataset d'entraînement.

LE PROBLÈME FONDAMENTAL :
Pour entraîner un classifieur de FORME, il faut :
- Des POSITIFS : vraies planètes confirmées NASA, phase-foldées proprement
- Des NÉGATIFS : faux positifs réalistes (binaires, variabilité, bruit)

POURQUOI C'EST DIFFICILE :
1. Peu de planètes confirmées (~5000 KOI confirmées) — données rares
2. Les faux positifs réalistes ne sont pas faciles à générer
3. Si les positifs et négatifs sont trop différents, le CNN apprend à
   distinguer DEUX DATASETS plutôt que VRAI vs FAUX TRANSIT
   → biais de dataset = mort silencieuse du modèle

NOTRE STRATÉGIE :
A) Positifs synthétiques : transits "boîte" idéaux + bruit gaussien
B) Positifs réels (optionnel, si lightkurve dispo) : KOI confirmés
C) Négatifs synthétiques diversifiés :
   - Bruit pur (rien de périodique)
   - V-shape (binaire à éclipses qui passe en grazing)
   - Asymétriques (variabilité stellaire)
   - Doubles creux (binaire avec éclipse secondaire)
   - Profondeurs variables entre transits

CETTE STRATÉGIE EST IMPARFAITE :
Le CNN va apprendre à rejeter NOS faux positifs, pas tous les vrais cas pathologiques.
C'est pourquoi la couche 4 (validation physique) est CRUCIALE.
Le CNN n'est qu'un PREMIER FILTRE, pas le verdict final.
"""

import numpy as np
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.phase_folder import CNN_INPUT_LENGTH


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATEURS DE FORMES — chacun produit un signal de CNN_INPUT_LENGTH points
# ─────────────────────────────────────────────────────────────────────────────

def _generate_box_transit(depth: float, duration_frac: float, noise_level: float) -> np.ndarray:
    """
    Transit "boîte" idéalisé : descente brutale, plateau, remontée brutale.
    
    POSITIF — c'est le pattern qu'on veut reconnaître.
    
    Args:
        depth: Profondeur relative (ex: 0.01 = 1%)
        duration_frac: Fraction de la fenêtre occupée par le transit (ex: 0.25 = 25%)
        noise_level: Écart-type du bruit gaussien
    """
    flux = np.ones(CNN_INPUT_LENGTH)
    center = CNN_INPUT_LENGTH // 2
    half_width = int(CNN_INPUT_LENGTH * duration_frac / 2)
    flux[center - half_width : center + half_width] -= depth
    flux += np.random.normal(0, noise_level, CNN_INPUT_LENGTH)
    return flux.astype(np.float32)


def _generate_u_shape_transit(depth: float, duration_frac: float, noise_level: float) -> np.ndarray:
    """
    Transit en U réaliste : descente progressive (limb darkening), plateau, remontée.
    
    POSITIF — plus réaliste qu'une boîte, c'est ce que voit Kepler.
    """
    flux = np.ones(CNN_INPUT_LENGTH)
    center = CNN_INPUT_LENGTH // 2
    half_width = int(CNN_INPUT_LENGTH * duration_frac / 2)
    
    # Profil de transit avec ingress/egress lissés
    x = np.arange(CNN_INPUT_LENGTH)
    # Hanning window centrée → forme en U avec bords doux
    transit_profile = np.zeros(CNN_INPUT_LENGTH)
    in_transit = np.abs(x - center) < half_width
    transit_profile[in_transit] = -depth * np.cos(
        np.pi * (x[in_transit] - center) / (2 * half_width)
    ) ** 2
    
    flux += transit_profile
    flux += np.random.normal(0, noise_level, CNN_INPUT_LENGTH)
    return flux.astype(np.float32)


def _generate_v_shape(depth: float, duration_frac: float, noise_level: float) -> np.ndarray:
    """
    V-shape : descente linéaire jusqu'au minimum puis remontée linéaire.
    
    NÉGATIF — caractéristique d'une binaire à éclipses en grazing
    (les deux étoiles s'effleurent juste, pas de plateau).
    """
    flux = np.ones(CNN_INPUT_LENGTH)
    center = CNN_INPUT_LENGTH // 2
    half_width = int(CNN_INPUT_LENGTH * duration_frac / 2)
    
    x = np.arange(CNN_INPUT_LENGTH)
    in_transit = np.abs(x - center) < half_width
    flux[in_transit] -= depth * (1.0 - np.abs(x[in_transit] - center) / half_width)
    
    flux += np.random.normal(0, noise_level, CNN_INPUT_LENGTH)
    return flux.astype(np.float32)


def _generate_asymmetric_dip(depth: float, duration_frac: float, noise_level: float) -> np.ndarray:
    """
    Creux asymétrique : descente lente, remontée rapide (ou inverse).
    
    NÉGATIF — typique de variabilité stellaire (taches solaires) ou de défauts
    instrumentaux. Une vraie planète est SYMÉTRIQUE.
    """
    flux = np.ones(CNN_INPUT_LENGTH)
    center = CNN_INPUT_LENGTH // 2
    half_width = int(CNN_INPUT_LENGTH * duration_frac / 2)
    
    x = np.arange(CNN_INPUT_LENGTH)
    asymmetry = np.random.uniform(0.3, 0.7)  # ratio de l'asymétrie
    
    left_end = center - int(half_width * asymmetry * 2)
    right_end = center + int(half_width * (1 - asymmetry) * 2)
    
    if left_end < center:
        left_x = np.arange(left_end, center)
        flux[left_end:center] -= depth * (left_x - left_end) / max(1, center - left_end)
    if right_end > center:
        right_x = np.arange(center, right_end)
        flux[center:right_end] -= depth * (1 - (right_x - center) / max(1, right_end - center))
    
    flux += np.random.normal(0, noise_level, CNN_INPUT_LENGTH)
    return flux.astype(np.float32)


def _generate_pure_noise(noise_level: float) -> np.ndarray:
    """
    Bruit pur — pas de signal du tout.
    
    NÉGATIF — un BLS qui retourne un candidat sans transit réel devrait
    produire ce genre de signal phase-foldé.
    """
    flux = np.ones(CNN_INPUT_LENGTH) + np.random.normal(0, noise_level, CNN_INPUT_LENGTH)
    return flux.astype(np.float32)


def _generate_sinusoidal(amplitude: float, n_cycles: float, noise_level: float) -> np.ndarray:
    """
    Variation sinusoïdale — variabilité stellaire pure.
    
    NÉGATIF — une étoile pulsante phase-foldée à sa propre période donne
    une sinusoïde, pas un transit.
    """
    x = np.linspace(0, 2 * np.pi * n_cycles, CNN_INPUT_LENGTH)
    flux = 1.0 - amplitude * np.cos(x)  # creux au centre
    flux += np.random.normal(0, noise_level, CNN_INPUT_LENGTH)
    return flux.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DU DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_synthetic_dataset(
    n_positives: int = 5000,
    n_negatives: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dataset v3.3 — fusion meilleur de Gemini + rigueur originale.

    POSITIFS (2 types) :
    - 30% boîtes : Hot Jupiters (ingress/egress quasi-instantanés)
    - 70% U-shapes cos² : transits réalistes avec limb darkening

    Pourquoi garder les boîtes ? Si on entraîne UNIQUEMENT sur U-shapes,
    le CNN apprend "épaules douces = planète" et REJETTE les vrais Hot Jupiters
    qui ont un profil rectangulaire. Les deux formes sont physiquement valides.

    NÉGATIFS (4 types, tous avec bruit ET duration ALÉATOIRES) :
    - V-shapes : binaires à éclipses grazing
    - Sinusoïdes : variabilité stellaire / pulsations
    - Asymétriques : taches stellaires, défauts instrumentaux
    - Bruit pur : faux candidats BLS sans signal réel

    CORRECTION v3.2 (Gemini) :
    - Bug : bruit et duration_frac FIXES pour les négatifs
    → CNN apprenait à distinguer les distributions, pas les formes
    - Fix : bruit et duration_frac aléatoires pour TOUS les types
    """
    np.random.seed(seed)

    print(f"🔨 Génération Dataset v3.3 : {n_positives} positifs + {n_negatives} négatifs...")

    samples = []
    labels = []

    # ─── POSITIFS ────────────────────────────────────────────────────────────
    # 30% boîtes (Hot Jupiters), 70% U-shapes cos² (limb darkening)
    n_box    = int(n_positives * 0.30)
    n_ushape = n_positives - n_box

    for _ in range(n_box):
        depth        = np.random.uniform(0.0001, 0.020)   # 100 ppm → 2%
        duration_frac = np.random.uniform(0.10, 0.35)
        noise        = np.random.uniform(0.00005, 0.0004)
        samples.append(_generate_box_transit(depth, duration_frac, noise))
        labels.append(1)

    for _ in range(n_ushape):
        depth        = np.random.uniform(0.0001, 0.015)   # descend à 100 ppm
        duration_frac = np.random.uniform(0.15, 0.35)
        noise        = np.random.uniform(0.00005, 0.0003)
        samples.append(_generate_u_shape_transit(depth, duration_frac, noise))
        labels.append(1)

    # ─── NÉGATIFS ────────────────────────────────────────────────────────────
    # Règle : bruit ET duration_frac TOUJOURS aléatoires
    # Sinon le CNN apprend la distribution du bruit, pas la forme du signal.
    n_v    = n_negatives // 4
    n_sin  = n_negatives // 4
    n_asym = n_negatives // 4
    n_noise = n_negatives - n_v - n_sin - n_asym

    for _ in range(n_v):
        depth        = np.random.uniform(0.001, 0.05)
        duration_frac = np.random.uniform(0.10, 0.35)    # ← aléatoire
        noise        = np.random.uniform(0.00005, 0.0004) # ← aléatoire
        samples.append(_generate_v_shape(depth, duration_frac, noise))
        labels.append(0)

    for _ in range(n_sin):
        amplitude = np.random.uniform(0.0005, 0.01)
        n_cycles  = np.random.uniform(0.5, 2.0)           # ← aléatoire
        noise     = np.random.uniform(0.00005, 0.0004)    # ← aléatoire
        samples.append(_generate_sinusoidal(amplitude, n_cycles, noise))
        labels.append(0)

    for _ in range(n_asym):
        depth        = np.random.uniform(0.001, 0.02)
        duration_frac = np.random.uniform(0.15, 0.40)    # ← aléatoire
        noise        = np.random.uniform(0.00005, 0.0004) # ← aléatoire
        samples.append(_generate_asymmetric_dip(depth, duration_frac, noise))
        labels.append(0)

    for _ in range(n_noise):
        noise = np.random.uniform(0.0001, 0.001)          # ← aléatoire
        samples.append(_generate_pure_noise(noise))
        labels.append(0)

    X = np.array(samples).reshape(-1, CNN_INPUT_LENGTH, 1)
    y = np.array(labels, dtype=np.int32)

    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    print(f"✅ Dataset v3.3 : X.shape={X.shape}, positifs={int(y.sum())}")
    return X, y