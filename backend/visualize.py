import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Charger le modèle et les données
MODEL_PATH = 'exoplanet_cnn_v2.h5'
DATA_PATH = 'dataset.pkl'

print("🔭 Chargement du chasseur d'exoplanètes...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)
    X, y = data['X'], data['y']

# 2. Chercher des segments où le modèle est sûr qu'il y a un transit
print("🧠 Analyse des segments en cours...")
predictions = model.predict(X, verbose=0)

# On cherche les indices où la probabilité est > 0.7 (Haute confiance)
high_confidence_indices = np.where(predictions > 0.7)[0]

if len(high_confidence_indices) == 0:
    print("pff... l'IA n'est pas très confiante aujourd'hui. On baisse le seuil à 0.3.")
    high_confidence_indices = np.where(predictions > 0.3)[0]

# 3. Afficher les 3 meilleures détections
for i in high_confidence_indices[:3]:
    plt.figure(figsize=(10, 4))
    plt.plot(X[i], color='#1f77b4', label='Signal Lumineux')
    
    status = "VRAI TRANSIT" if y[i] == 1 else "FAUX POSITIF"
    proba = predictions[i][0] * 100
    
    plt.title(f"Détection : {proba:.2f}% de certitude | Réalité : {status}")
    plt.xlabel("Temps (Segments de 30 min)")
    plt.ylabel("Luminosité normalisée")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    filename = f"detection_{i}.png"
    plt.savefig(filename)
    print(f"✅ Graphique généré : {filename}")
    plt.close()