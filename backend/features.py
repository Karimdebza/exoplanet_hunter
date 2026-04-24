# features.py — ExoplanetHunter v2
# Extraction de features physiques sur les segments de courbes de lumière
#
# POURQUOI CETTE APPROCHE ?
# Le CNN reçoit 200 points bruts → trop de bruit, pas assez de données
# XGBoost reçoit 8 features résumant physiquement le segment → plus robuste
#
# UN TRANSIT EN CHIFFRES :
# - min très bas      → baisse soudaine de luminosité
# - skewness négatif  → asymétrie vers le bas
# - min/mean ratio    → creux profond par rapport à la moyenne
# - std élevée        → variation importante dans le segment
 

import numpy as np 

from scipy import stats


def extract_features(segment: np.ndarray) -> dict:
    """
    Extrait 12 features physiques d'un segment de 200 points.
    
    Chaque feature capture un aspect différent d'un transit :
    """

    segment = segment.flatten()

    f_min = np.min(segment)
    f_mean = np.mean(segment)
    f_std = np.std(segment)

    f_min_mean_ratio = f_min / (abs(f_mean) + 1e-8)
    f_skewness = float(stats.skew(segment))
    f_kurtosis = float(stats.kurtosis(segment))
    f_q05 = np.percentile(segment, 5)
    f_q95 = np.percentile(segment, 95)
    f_range = f_q95 - f_q05
    f_min_position = float(np.argmin(segment)) / len(segment)
    threshold = f_mean - 2 * f_std
    f_dip_duration = float(np.sum(segment < threshold)) / len(segment)
    f_depth = (f_mean - f_min) / (f_std + 1e-8)
 
    return {
        'min':            f_min,
        'mean':           f_mean,
        'std':            f_std,
        'min_mean_ratio': f_min_mean_ratio,
        'skewness':       f_skewness,
        'kurtosis':       f_kurtosis,
        'q05':            f_q05,
        'q95':            f_q95,
        'range':          f_range,
        'min_position':   f_min_position,
        'dip_duration':   f_dip_duration,
        'depth':          f_depth,
    }

def build_feature_matrix(X: np.ndarray) -> np.ndarray:
    """
    Transforme un dataset de segments en matrice de features.
    
    Input  : X shape (n_segments, 200, 1)
    Output : X_feat shape (n_segments, 12)
    """
    print(f"Extraction des features sur {len(X)} segments...")
    
    features_list = []
    for segment in X:
        feat = extract_features(segment)
        features_list.append(list(feat.values()))
    
    X_feat = np.array(features_list)
    print(f"Shape features : {X_feat.shape}")
    
    return X_feat
 
 
def get_feature_names() -> list:
    """Retourne les noms des features dans l'ordre."""
    return [
        'min', 'mean', 'std', 'min_mean_ratio',
        'skewness', 'kurtosis', 'q05', 'q95',
        'range', 'min_position', 'dip_duration', 'depth'
    ]
 
 
if __name__ == "__main__":
    # Test rapide sur un segment synthétique
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
 
    # Segment normal (bruit)
    normal = np.random.normal(0, 0.001, 200)
    
    # Segment avec transit (creux en U)
    transit = np.random.normal(0, 0.001, 200)
    transit[80:120] -= 0.01 * np.hanning(40)
 
    feat_normal  = extract_features(normal)
    feat_transit = extract_features(transit)
 
    print("\n── Comparaison Normal vs Transit ──")
    print(f"{'Feature':<20} {'Normal':>10} {'Transit':>10}")
    print("-" * 42)
    for key in feat_normal:
        print(f"{key:<20} {feat_normal[key]:>10.4f} {feat_transit[key]:>10.4f}")