import lightkurve as lk
import numpy as np
import tensorflow as tf


# 1. Charger ton modèle (le fichier .h5 que tu as généré)
print("🧠 Chargement de l'IA...")
model = tf.keras.models.load_model('exoplanet_detector_v1.h5')

# 2. Télécharger une nouvelle étoile (Kepler-90)
print("🔭 Observation de Kepler-90...")
search = lk.search_lightcurve('Kepler-90', author='Kepler', quarter=5)
lc = search.download().remove_nans().flatten(window_length=401)

# 3. Préparer un segment de 200 points pour l'IA
# On prend un segment au hasard (par exemple à l'index 1000)
segment = lc.flux.value[1000:1200] - 1.0
X_test = segment.reshape((1, 200, 1)) # Format attendu par le CNN

# 4. LA PRÉDICTION
prediction = model.predict(X_test)
score = prediction[0][0]

print("-" * 30)
print(f"RESULTAT DE L'ANALYSE :")
if score > 0.5:
    print(f"✅ EXOPLANÈTE DÉTECTÉE ! (Confiance : {score*100:.2f}%)")
else:
    print(f"❌ RIEN À SIGNALER. (Confiance : {(1-score)*100:.2f}%)")
print("-" * 30)