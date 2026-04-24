from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import lightkurve as lk

app = Flask(__name__)
CORS(app)

# 1. Chargement du modèle (une seule fois au lancement)
MODEL_PATH = 'exoplanet_cnn_v2.h5'
print(f"🤖 Chargement du modèle {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

def analyze_star(star_name):
    """Logique métier : Téléchargement -> Prediction -> Nettoyage"""
    try:
        # Téléchargement Kepler Quarter 4
        print(f"📡 Téléchargement des données pour {star_name}...")
        search = lk.search_lightcurve(star_name, author='Kepler', quarter=4)
        if len(search) == 0:
            return None, "Étoile introuvable dans les archives Kepler."
        
        lc = search.download().remove_nans().flatten(window_length=401)
        flux = (lc.flux.value - 1.0).astype(float)
        time = lc.time.value.astype(float)

        detections = []
        step = 100
        window = 200

        # Scan par fenêtre glissante
        print(f"🧠 Analyse par le CNN...")
        for i in range(0, len(flux) - window, step):
            segment = flux[i:i+window].reshape(1, window, 1)
            # Utilisation de model.predict pour obtenir la proba
            proba = float(model.predict(segment, verbose=0)[0][0])
            
            if proba > 0.5:
                detections.append({
                    "idx": i,
                    "time": time[i + window//2], # On prend le milieu du transit
                    "proba": round(proba, 4)
                })

        return {
            "star_name": star_name,
            "time": time.tolist(),
            "flux": flux.tolist(),
            "detections": detections,
            "count": len(detections)
        }, None

    except Exception as e:
        return None, str(e)

@app.route('/api/scan', methods=['POST'])
def scan():
    data = request.json
    star_id = data.get('star_name', 'Kepler-4')
    
    result, error = analyze_star(star_id)
    
    if error:
        return jsonify({"status": "error", "message": error}), 400
    
    return jsonify({"status": "success", "result": result})

if __name__ == '__main__':
    # On lance sur le port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)