# augmentation.py — ExoplanetHunter v3
# Data augmentation sur les transits + fusion des deux sources de données
#
# POURQUOI CE FICHIER ?
# CNN avec 790 transits → trop peu → Recall = 0
# Après augmentation → ~4000+ transits → CNN peut vraiment apprendre
#
# DEUX SOURCES DE DONNÉES :
# 1. exoTrain.csv (Kaggle) → 5087 étoiles, 37 transits, flux brut déjà propre
# 2. lightkurve (NASA) → nos 7297 segments, 790 transits, éphémérides réelles
#
# DATA AUGMENTATION — 5 techniques physiquement réalistes :
# 1. Bruit gaussien     → simule le bruit instrumental
# 2. Décalage temporel  → le transit n'est pas toujours centré
# 3. Scaling amplitude  → étoiles de brillances différentes
# 4. Flip horizontal    → la courbe est symétrique
# 5. Baseline shift     → décalage de la ligne de base

import numpy as np
import pandas as pd

WINDOW_SIZE = 200
N_AUGMENT   = 30  # Nombre de variantes par transit réel


# ─────────────────────────────────────────────────────────
# SECTION 1 — Chargement exoTrain.csv
# ─────────────────────────────────────────────────────────

def load_exotrain(path: str = 'exoTrain.csv') -> tuple:
    """
    Charge le dataset Kaggle exoTrain.csv.
    
    Format : LABEL (1=normal, 2=transit) + FLUX.1 à FLUX.3197
    On découpe chaque courbe en segments de 200 points.
    """
    print(f"📥 Chargement {path}...")
    df = pd.read_csv(path)
    
    print(f"  Étoiles     : {len(df)}")
    print(f"  Avec transit: {(df['LABEL'] == 2).sum()}")
    print(f"  Sans transit: {(df['LABEL'] == 1).sum()}")
    
    # Conversion labels : 2→1 (transit), 1→0 (normal)
    y_stars = (df['LABEL'] == 2).astype(int).values
    X_stars = df.drop(columns=['LABEL']).values  # (5087, 3197)
    
    segments = []
    labels   = []
    
    for i, (flux, label) in enumerate(zip(X_stars, y_stars)):
        # Normalisation : centré autour de 0
        flux = flux - np.mean(flux)
        flux = flux / (np.std(flux) + 1e-8)
        
        # Découpe en segments de 200 points
        for j in range(0, len(flux) - WINDOW_SIZE, 100):
            seg = flux[j:j + WINDOW_SIZE]
            segments.append(seg)
            # Si l'étoile a un transit, tous ses segments sont labellisés 1
            # (approximation — le transit peut être dans n'importe quel segment)
            # On affine avec un seuil sur le minimum
            has_dip = np.min(seg) < -2.0  # Plus de 2 écarts-types sous la moyenne
            labels.append(1 if (label == 1 and has_dip) else 0)
    
    X = np.array(segments)
    y = np.array(labels)
    
    print(f"  Segments total   : {len(X)}")
    print(f"  Segments transit : {y.sum()}")
    
    return X, y


# ─────────────────────────────────────────────────────────
# SECTION 2 — Data Augmentation
# ─────────────────────────────────────────────────────────

def augment_segment(segment: np.ndarray) -> list:
    """
    Génère N_AUGMENT variantes d'un segment de transit.
    
    Chaque technique simule une réalité physique différente.
    """
    variants = [segment.copy()]  # On garde l'original
    
    for _ in range(N_AUGMENT - 1):
        aug = segment.copy()
        
        # Technique 1 — Bruit gaussien (bruit instrumental)
        # Amplitude aléatoire entre 0.1% et 0.5% du signal
        noise_level = np.random.uniform(0.001, 0.005)
        aug += np.random.normal(0, noise_level * np.std(aug), len(aug))
        
        # Technique 2 — Décalage temporel (transit pas centré)
        # Décale le signal de ±20 points
        shift = np.random.randint(-20, 20)
        aug = np.roll(aug, shift)
        
        # Technique 3 — Scaling amplitude (brillance de l'étoile)
        # Multiplie par un facteur entre 0.8 et 1.2
        scale = np.random.uniform(0.8, 1.2)
        aug = aug * scale
        
        # Technique 4 — Flip horizontal (symétrie de la courbe)
        if np.random.random() > 0.5:
            aug = aug[::-1].copy()
        
        # Technique 5 — Baseline shift (décalage ligne de base)
        baseline = np.random.uniform(-0.001, 0.001)
        aug += baseline
        
        variants.append(aug)
    
    return variants


def augment_dataset(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Applique la data augmentation uniquement sur les transits.
    Les segments normaux ne sont pas augmentés.
    
    Résultat : dataset équilibré avec beaucoup plus de transits.
    """
    print(f"\n🔬 Data Augmentation...")
    print(f"  Avant : {(y==1).sum()} transits / {(y==0).sum()} normaux")
    
    transit_idx = np.where(y == 1)[0]
    normal_idx  = np.where(y == 0)[0]
    
    # Augmente chaque transit
    aug_segments = []
    aug_labels   = []
    
    for idx in transit_idx:
        variants = augment_segment(X[idx])
        aug_segments.extend(variants)
        aug_labels.extend([1] * len(variants))
    
    # Combine : originaux + augmentés
    X_aug = np.concatenate([X, np.array(aug_segments)], axis=0)
    y_aug = np.concatenate([y, np.array(aug_labels)],   axis=0)
    
    # Mélange aléatoire
    shuffle_idx = np.random.permutation(len(X_aug))
    X_aug = X_aug[shuffle_idx]
    y_aug = y_aug[shuffle_idx]
    
    print(f"  Après : {(y_aug==1).sum()} transits / {(y_aug==0).sum()} normaux")
    print(f"  Ratio : {(y_aug==0).sum() / (y_aug==1).sum():.1f}x")
    
    return X_aug, y_aug


# ─────────────────────────────────────────────────────────
# SECTION 3 — Fusion des deux sources
# ─────────────────────────────────────────────────────────

def build_combined_dataset(exotrain_path: str = 'exoTrain.csv',
                           use_lightkurve: bool = True) -> tuple:
    """
    Construit le dataset combiné :
    exoTrain.csv + lightkurve + data augmentation
    """
    all_X = []
    all_y = []
    
    # Source 1 — exoTrain.csv
    try:
        X_kaggle, y_kaggle = load_exotrain(exotrain_path)
        all_X.append(X_kaggle)
        all_y.append(y_kaggle)
        print(f"✅ exoTrain.csv chargé")
    except FileNotFoundError:
        print(f"⚠️  exoTrain.csv non trouvé — utilisation lightkurve uniquement")
    
    # Source 2 — lightkurve (nos données)
    if use_lightkurve:
        from data import build_dataset
        print(f"\n📡 Chargement données lightkurve...")
        X_lk, y_lk = build_dataset()
        # Reshape (N, 200, 1) → (N, 200)
        X_lk = X_lk.reshape(X_lk.shape[0], X_lk.shape[1])
        all_X.append(X_lk)
        all_y.append(y_lk)
        print(f"✅ lightkurve chargé")
    
    # Fusion
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"\n📊 Avant augmentation :")
    print(f"   Total    : {len(X)}")
    print(f"   Transits : {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Data augmentation sur les transits
    X_aug, y_aug = augment_dataset(X, y)
    
    # Reshape pour CNN : (N, 200, 1)
    X_aug = X_aug.reshape(X_aug.shape[0], X_aug.shape[1], 1)
    
    print(f"\n📊 Dataset final :")
    print(f"   Total    : {len(X_aug)}")
    print(f"   Transits : {y_aug.sum()} ({y_aug.mean()*100:.1f}%)")
    print(f"   Shape    : {X_aug.shape}")
    
    return X_aug, y_aug


if __name__ == "__main__":
    # Test — charge tout et affiche les stats
    X, y = build_combined_dataset(
        exotrain_path='exoTrain.csv',
        use_lightkurve=True
    )
    print("\n✅ Dataset prêt pour le CNN !")
    print(f"   Lance maintenant : python model.py")