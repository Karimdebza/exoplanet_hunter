import lightkurve as lk
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('exoplanet_cnn_v2.h5')

lc = lk.search_lightcurve('Kepler-4', author='Kepler', quarter=4).download()
lc = lc.remove_nans().flatten(window_length=401)
flux = lc.flux.value - 1.0

all_probas = []
for i in range(0, len(flux) - 200, 100):
    segment = flux[i:i+200].reshape(1, 200, 1)
    proba = float(model(segment, training=False).numpy()[0][0])
    all_probas.append((i, proba))

print(f"Proba max : {max(p for _,p in all_probas):.4f}")
detections = [(i,p) for i,p in all_probas if p > 0.5]
print(f"Détections seuil 0.5 : {len(detections)}")

# --- AJOUTE ÇA À LA FIN DE TON SCRIPT ---

# 1. On ne garde que les indices (le temps i)
indices = [p[0] for p in detections]

# 2. Filtrage : On cherche la période
if len(indices) > 5:
    # Calcul des écarts entre chaque détection
    diffs = np.diff(indices)
    
    # La médiane nous donne la période orbitale la plus probable
    probable_period = np.median(diffs)
    
    # On regarde combien de détections respectent ce rythme (marge de 15%)
    valid_hits = sum(1 for d in diffs if abs(d - probable_period) < (probable_period * 0.15))
    score = (valid_hits / len(diffs)) * 100

    print(f"\n--- RÉSULTAT DU SCAN ---")
    print(f"Période détectée : {probable_period} points")
    print(f"Score de régularité : {score:.2f}%")
    
    if score > 80:
        print("💎 VERDICT : Signal périodique parfait. Kepler-4 b est confirmée par l'IA !")
    else:
        print("⚠️ VERDICT : Signal irrégulier. Probablement du bruit.")