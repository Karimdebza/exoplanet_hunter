# data.py — ExoplanetHunter v2
# Préparation des données avec labeling correct basé sur éphémérides connues
#
# POURQUOI CE FICHIER EXISTE ?
# Le problème du labeling arbitraire (min flux < -0.001) donnait
# des faux labels → le modèle apprenait du bruit, pas des transits.
# Ici on utilise les périodes et durées CONNUES de chaque planète
# pour savoir exactement quand un transit se produit.

import numpy as np
import lightkurve as lk
import warnings
warnings.filterwarnings('ignore')
import pickle
# ─────────────────────────────────────────────
# ÉPHÉMÉRIDES CONNUES — périodes et durées réelles
# Source : NASA Exoplanet Archive
# period   = période orbitale en jours
# duration = durée du transit en jours
# t0       = temps du premier transit connu (BKJD)
# ─────────────────────────────────────────────
KNOWN_PLANETS = {
    'Kepler-10': {'period': 0.8375,  'duration': 0.019, 't0': 130.308},
    'Kepler-90': {'period': 7.0082,  'duration': 0.064, 't0': 134.082},
    'Kepler-7':  {'period': 4.8855,  'duration': 0.125, 't0': 136.853},
    'Kepler-5':  {'period': 3.5485,  'duration': 0.120, 't0': 122.892},
    # 'Kepler-6':  {'period': 3.2347,  'duration': 0.110, 't0': 121.486},
    'Kepler-8':  {'period': 3.5225,  'duration': 0.107, 't0': 120.960},
    'Kepler-12': {'period': 4.4380,  'duration': 0.155, 't0': 123.955},
    'Kepler-17': {'period': 1.4857,  'duration': 0.094, 't0': 120.623},
}

WINDOW_SIZE = 200   # Nombre de points par segment (environ 4 heures de données Kepler)
STEP        = 100    # Pas de glissement entre segments (overlap pour plus de données)

def download_lightcurve(star_name: str, quarter: int = 4):
    print(f"  📥 Téléchargement {star_name} Q{quarter}...")
    
    result = lk.search_lightcurve(star_name, author='Kepler', quarter=quarter)
    if len(result) == 0:
        raise ValueError(f"Aucune donnée trouvée pour {star_name}")
    
    lc = result.download()
    lc = lc.remove_nans().remove_outliers(sigma=5)
    lc = lc.flatten(window_length=401)
    
    flux = lc.flux.value - 1.0
    time = lc.time.value
    
    return time, flux


def find_t0_with_bls(star_name: str, period: float, quarter: int = 4) -> float:
    """
    Trouve le t0 réel via BLS au lieu de le mettre à la main.
    Plus fiable que les éphémérides hardcodées.
    """
    result = lk.search_lightcurve(star_name, author='Kepler', quarter=quarter)
    lc = result.download().remove_nans().flatten(window_length=401)
    
    # BLS cherche autour de la période connue ± 10%
    period_grid = np.linspace(period * 0.9, period * 1.1, 1000)
    bls = lc.to_periodogram(method='bls', period=period_grid)
    
    t0 = float(bls.transit_time_at_max_power.value)
    print(f"  🔍 {star_name} t0 trouvé : {t0:.3f} BKJD")
    return t0

def is_in_transit(time_point: float, planet_params: dict) -> bool:
    """
    Vérifie si un instant donné correspond à un transit.
    
    COMMENT ÇA MARCHE :
    On calcule la phase du point dans l'orbite (entre 0 et 1).
    Si la phase est proche de 0 (début d'orbite), c'est un transit.
    
    phase = ((t - t0) mod period) / period
    Un transit se produit quand phase ≈ 0 (ou ≈ 1, c'est pareil)
    """
    t0       = planet_params['t0']
    period   = planet_params['period']
    duration = planet_params['duration']
    
    # Phase normalisée entre -0.5 et 0.5
    phase = ((time_point - t0) % period) / period
    if phase > 0.5:
        phase -= 1.0
    
    # Demi-durée en fraction d'orbite
    half_duration = (duration / period) / 2.0
    
    return abs(phase) < half_duration


def make_segments(time: np.ndarray, flux: np.ndarray,
                  planet_params: dict) -> tuple:
    """
    Découpe la courbe en segments et assigne les labels.
    
    Label = 1 si le segment contient un transit réel
    Label = 0 sinon (bruit normal)
    
    POURQUOI DES SEGMENTS ?
    Le CNN a besoin d'inputs de taille fixe.
    On glisse une fenêtre de 200 points sur toute la courbe.
    """
    segments = []
    labels   = []
    
    for i in range(0, len(flux) - WINDOW_SIZE, STEP):
        segment     = flux[i : i + WINDOW_SIZE]
        time_window = time[i : i + WINDOW_SIZE]
        
        # Le segment est labellisé 1 si AU MOINS UN point est en transit
        in_transit = any(is_in_transit(t, planet_params) for t in time_window)
        
        segments.append(segment)
        labels.append(1 if in_transit else 0)
    
    return np.array(segments), np.array(labels)


def build_dataset(stars: dict = None, quarters:  list = [3, 4, 5]) -> tuple:
    """
    Construit le dataset complet pour l'entraînement.
    
    Paramètre stars : dict {nom_étoile: planet_params}
    Par défaut utilise KNOWN_PLANETS.
    """
    if stars is None:
        stars = KNOWN_PLANETS
    
    all_segments = []
    all_labels   = []
    
    print("🔭 Construction du dataset...\n")

    for quarter in quarters : 
        print(f"  📡 Quarter {quarter}...")
        for star_name, planet_params in stars.items():
            try:
                time, flux = download_lightcurve(star_name, quarter)
                segs, labs = make_segments(time, flux, planet_params)
                
                n_transits = labs.sum()
                n_total    = len(labs)
                print(f"  ✅ {star_name} : {n_total} segments, {n_transits} avec transit")
                
                all_segments.append(segs)
                all_labels.append(labs)
                
            except Exception as e:
                print(f"  ❌ {star_name} : {e}")
    
    if not all_segments:
        raise RuntimeError("Aucune donnée récupérée. Vérifie ta connexion.")
    
    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels,   axis=0)
    
    # Reshape pour CNN 1D : (samples, timesteps, features)
    # 1 feature = la valeur du flux à chaque instant
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"\n📊 Dataset final :")
    print(f"   Total segments  : {len(X)}")
    print(f"   Avec transit    : {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Sans transit    : {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"   Shape X         : {X.shape}")
    
    return X, y


# ──────────────────────────────────────────────────────────
# BLOC DE SAUVEGARDE (C'est ça qui crée le .pkl)
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. On lance la construction
    X, y = build_dataset()

    # 2. On prépare le dictionnaire
    data_to_save = {
        'X': X,
        'y': y
    }

    # 3. On enregistre physiquement sur le disque
    print(f"\n💾 Sauvegarde en cours...")
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print("✅ Fichier 'dataset.pkl' créé avec succès !")
    print("🚀 Tu peux maintenant lancer 'python model.py'")