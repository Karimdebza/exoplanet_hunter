"""
validation/cnn_model.py — Le classifieur de forme.

RÔLE DU CNN :
Recevoir un transit phase-foldé centré (201 points) et répondre :
"Cette forme ressemble-t-elle à un vrai transit de planète ? Score [0,1]."

CE QU'IL FAIT :
- Apprend les patterns visuels d'un transit : descente, plateau, remontée
- Distingue forme de planète (U-shape) vs forme suspecte (V, asymétrique, plate)

CE QU'IL NE FAIT PAS :
- Décider seul (couche 4 = validation physique pour les tests odd/even, etc.)
- Détecter la périodicité (déjà fait par BLS, couche 2)

ARCHITECTURE — POURQUOI CES CHOIX :

1. Conv1D(16, kernel=11) : 11 points = environ la moitié d'un transit en bins.
   Ce filtre apprend les "morceaux locaux" : ingress, egress, fond du creux.
   16 filtres = 16 motifs locaux différents.

2. Conv1D(32, kernel=7) : combine les motifs locaux en motifs plus larges
   (ex: "ingress + plateau" = début de transit complet).

3. GlobalAveragePooling1D : résume toute la séquence en un vecteur de 32 valeurs.
   Avantage vs Flatten : invariant à la position exacte du transit (le transit
   peut être légèrement décalé, le CNN s'en fiche). C'est un point CRUCIAL.

4. Dense(32) + Dropout(0.3) : combinaison finale + régularisation.
   Dropout 0.3 = 30% des neurones désactivés à chaque batch d'entraînement.
   → empêche le sur-apprentissage sur le bruit du dataset.

5. Dense(1, sigmoid) : sortie en probabilité [0, 1].

ALTERNATIVES ÉCARTÉES :
- LSTM : utile pour des séquences avec dépendances longues. Notre transit est
  court et local, le CNN suffit. LSTM serait plus lent sans gain.
- Transformer : overkill pour 201 points. À reconsidérer si on passe à des
  séquences de 10000 points.
- ResNet-like : trop profond pour notre tâche, risque d'overfit.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import TransitCandidate, CNNValidation
from validation.phase_folder import FoldedTransit, CNN_INPUT_LENGTH


# Configuration globale du modèle
DEFAULT_MODEL_PATH = "models/exoplanet_cnn_v3.h5"
DEFAULT_THRESHOLD = 0.5  # Seuil de décision passed/failed

def _normalize_input(X:np.ndarray) -> np.ndarray:
    """
    Centre les flux autour de 0 pour stabiliser l'apprentissage.
    
    flux nominal = 1.0 → après centrage = 0.0
    transit profond = 0.99 → -0.01
    
    Cette normalisation DOIT être appliquée IDENTIQUEMENT à l'entraînement
    et à l'inférence. Sinon = data leak silencieux.
    """
    return X - 1.0

def build_cnn(input_length: int = CNN_INPUT_LENGTH) -> tf.keras.Model:
    """Architecture CNN v3.1 — sans BatchNorm, avec Dropout distribué."""
    model = models.Sequential([
        layers.Input(shape=(input_length, 1)),
        
        # Bloc 1 : motifs locaux (ingress, egress, fond)
        layers.Conv1D(16, kernel_size=11, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Bloc 2 : motifs globaux (transit complet)
        layers.Conv1D(32, kernel_size=7, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Réduction par moyenne globale
        layers.GlobalAveragePooling1D(),
        
        # Tête de classification
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )
    return model
 

def train_cnn(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
    epochs: int = 30,
    batch_size: int = 32,
    verbose: int = 1,
) -> tf.keras.Model:
    """Entraîne le CNN sur des transits phase-foldés normalisés."""
    print("=" * 60)
    print("ENTRAÎNEMENT CNN v3.1 — Phase-folded transits")
    print("=" * 60)
    print(f"Dataset : {X.shape[0]} échantillons de {X.shape[1]} points")
    print(f"Positifs : {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"Négatifs : {int((1-y).sum())} ({(1-y).mean()*100:.1f}%)")
    
    # NORMALISATION : centrer autour de 0
    X_normalized = _normalize_input(X)
    print(f"Range après normalisation : [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=y
    )
    
    class_weights_array = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight = {0: float(class_weights_array[0]), 1: float(class_weights_array[1])}
    print(f"\nclass_weight: {class_weight}")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, verbose=1
        ),
    ]
    
    model = build_cnn(input_length=X.shape[1])
    if verbose:
        model.summary()
    
    print("\n🚀 Début entraînement...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )
    
    # Évaluation finale
    print("\n" + "=" * 60)
    print("ÉVALUATION SUR TEST")
    print("=" * 60)
    results = model.evaluate(X_test, y_test, verbose=0)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name:12s} : {val:.4f}")
    
    # Diagnostics
    if 'auc' in model.metrics_names:
        test_auc = results[model.metrics_names.index('auc')]
        if test_auc < 0.7:
            print("\n⚠️  AUC de test faible (< 0.7) — le modèle a du mal à généraliser.")
        elif test_auc < 0.9:
            print("\n✓ AUC correct mais améliorable.")
        else:
            print("\n✅ AUC excellent — modèle prêt à l'usage.")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\n💾 Modèle sauvegardé → {model_path}")
    
    return model
 
 
def predict(
    folded: FoldedTransit,
    model: tf.keras.Model,
    threshold: float = DEFAULT_THRESHOLD,
) -> CNNValidation:
    """
    Prédit si un transit phase-foldé ressemble à une vraie planète.
    
    IMPORTANT : applique la MÊME normalisation qu'à l'entraînement.
    """
    X_raw = folded.to_cnn_input()
    X_normalized = _normalize_input(X_raw)  # cohérence train/inference
    
    score = float(model.predict(X_normalized, verbose=0)[0][0])
    passed = score >= threshold
    
    rejection_reason = None
    if not passed:
        if score < 0.2:
            rejection_reason = f"Forme non-conforme à un transit (score={score:.3f})"
        else:
            rejection_reason = f"Score sous le seuil {threshold} (score={score:.3f})"
    
    return CNNValidation(
        candidate=folded.candidate,
        cnn_score=score,
        passed=passed,
        rejection_reason=rejection_reason,
    )

def load_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """Charge un modèle pré-entraîné. Plante proprement si introuvable."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. "
            f"Lance d'abord scriptes/train_cnn_v3.py"
        )
    return tf.keras.models.load_model(model_path)