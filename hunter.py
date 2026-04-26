# hunter.py — Le vrai chasseur d'exoplanètes
# Scan les 5050 étoiles inconnues et sort les candidats intéressants

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

MODEL_PATH = 'exoplanet_cnn_v2.h5'
THRESHOLD = 0.95       # Au lieu de 0.7
MIN_DETECTIONS = 5  

model = tf.keras.models.load_model(MODEL_PATH)

def scan_star_from_csv(flux_array: np.ndarray) -> dict:
    """Scanne une étoile depuis le CSV."""
    # Normalisation identique à l'entraînement
    flux = flux_array - np.mean(flux_array)
    flux = flux / (np.std(flux) + 1e-8)
    
    segments = []
    for i in range(0, len(flux) - 200, 100):
        segments.append(flux[i:i+200])
    
    X = np.array(segments).reshape(-1, 200, 1)
    probas = model(X, training=False).numpy().flatten()
    
    detections = [i*100 for i, p in enumerate(probas) if p > THRESHOLD]
    max_proba  = float(np.max(probas))
    
    return {
        'n_detections': len(detections),
        'max_proba':    max_proba,
        'detections':   detections,
        'mean_proba':   float(np.mean(probas))
    }


def hunt():
    print("🔭 Chargement exoTrain.csv...")
    df = pd.read_csv('exoTrain.csv')
    
    # On prend UNIQUEMENT les étoiles sans planète connue (label=1)
    unknown = df[df['LABEL'] == 1].copy()
    print(f"🎯 {len(unknown)} étoiles inconnues à scanner...")
    
    candidates = []
    
    for idx, row in unknown.iterrows():
        flux = row.drop('LABEL').values.astype(float)
        result = scan_star_from_csv(flux)
        
        # Candidat si assez de détections avec haute confiance
        if result['n_detections'] >= MIN_DETECTIONS:
            candidates.append({
                'star_idx':     idx,
                'n_detections': result['n_detections'],
                'max_proba':    result['max_proba'],
                'detections':   result['detections'],
            })
            print(f"  🪐 CANDIDAT #{len(candidates)} — étoile {idx} "
                  f"| {result['n_detections']} détections "
                  f"| max_proba={result['max_proba']:.3f}")
        
        if idx % 100 == 0:
            print(f"  ... {idx}/{len(unknown)} étoiles scannées")
    
    print(f"\n✅ Scan terminé — {len(candidates)} candidats trouvés")
    
    # Sauvegarde
    with open('candidates.pkl', 'wb') as f:
        pickle.dump(candidates, f)
    
    return candidates


if __name__ == "__main__":
    hunt()