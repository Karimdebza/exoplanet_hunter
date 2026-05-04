"""
scriptes/train_cnn_v3.py — Entraîne le CNN sur dataset synthétique.

USAGE :
    python scriptes/train_cnn_v3.py

OUTPUT :
    models/exoplanet_cnn_v3.h5 (modèle entraîné, prêt à être chargé)

RAPPEL HONNÊTE :
Ce modèle est entraîné sur du SYNTHÉTIQUE pur. Il sera bon pour distinguer
les FORMES caricaturales (vrai transit vs V-shape vs bruit), mais il aura
des trous sur les cas pathologiques réels (binaires sophistiquées, tâches
solaires complexes). C'est pourquoi il s'inscrit dans un PIPELINE et n'est
pas le verdict final.

Pour aller plus loin (phase 2) : enrichir avec des KOI confirmés réels
téléchargés via lightkurve.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.dataset_builder import build_synthetic_dataset
from validation.cnn_model import train_cnn


if __name__ == "__main__":
    # Génération du dataset synthétique
    X, y = build_synthetic_dataset(
        n_positives=10000,
        n_negatives=10000,
        seed=42,
    )
    
    # Entraînement
    model = train_cnn(
        X, y,
        model_path="models/exoplanet_cnn_v3.h5",
        epochs=60,
        batch_size=64,
    )
    
    print("\n✅ Entraînement terminé. Modèle prêt à explorer des étoiles.")
    print("   Lance maintenant : python scriptes/explore_star.py <KIC>")