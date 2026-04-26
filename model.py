# model.py — ExoplanetHunter v2
# Architecture CNN 1D pour détecter les transits d'exoplanètes
#
# POURQUOI CNN 1D ET PAS XGBOOST ?
# XGBoost traite FLUX.1, FLUX.2... comme des colonnes indépendantes.
# Il ne comprend pas que ce sont des points CONSÉCUTIFS dans le temps.
# CNN 1D glisse une fenêtre sur la séquence → il "voit" la forme du transit.
#
# POURQUOI PAS LSTM DIRECTEMENT ?
# CNN 1D est plus rapide à entraîner et suffit pour détecter des patterns
# locaux comme un transit. LSTM est utile quand on a besoin de mémoire
# longue distance (détecter la PÉRIODE de répétition). On l'ajoutera en v3.
 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Nettoie les messages inutiles

print("🚀 Initialisation du modèle...")
import tensorflow as tf
print("✅ TensorFlow chargé avec succès")

# Juste avant la ligne qui charge tes données (ex: np.load ou pd.read_csv)
print("📂 Chargement des données d'exoplanètes...")
import numpy as np
import pickle
# import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
 
WINDOW_SIZE = 200

MODEL_PATH  = 'exoplanet_cnn_v2.h5'

def build_cnn(input_shape:tuple = (200,1))-> tf.keras.Model:
    
    """
    Architecture CNN 1D.
    
    LECTURE COUCHE PAR COUCHE :
    
    Conv1D(32, kernel_size=15) 
    → 32 filtres, chacun regarde 15 points consécutifs
    → apprend 32 patterns différents (début transit, creux, fin transit...)
    
    MaxPooling1D(2)
    → réduit la taille de moitié en gardant les valeurs max
    → rend le modèle robuste aux petits décalages temporels
    
    Conv1D(64, kernel_size=10)
    → 64 filtres sur les patterns déjà détectés
    → apprend des patterns plus complexes (forme globale du transit)
    
    GlobalAveragePooling1D()
    → résume toute la séquence en un seul vecteur
    → peu importe où le transit apparaît dans le segment
    
    Dense(64) → couche fully connected, combine tous les indices
    Dropout(0.3) → désactive 30% des neurones aléatoirement pendant l'entraînement
                 → évite le surapprentissage (overfitting)
    
    Dense(1, sigmoid) → sortie entre 0 et 1
                      → 0 = bruit normal, 1 = transit détecté
    """

    model = models.Sequential([
        layers.Conv1D(16, kernel_size=15, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=10, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.Precision(name='precision')]
    )
    return model



def train(X:np.ndarray, y:np.ndarray) -> tf.keras.Model:
    """
    Entraîne le CNN sur les données préparées.
    
    POINTS IMPORTANTS :
    - class_weight : compense le déséquilibre (peu de transits vs beaucoup de bruit)
    - EarlyStopping : arrête si le modèle ne s'améliore plus → évite l'overfitting
    - ReduceLROnPlateau : réduit le learning rate si stagnation
    """


    print("=" * 50)
    print("ENTRAÎNEMENT CNN 1D")
    print("=" * 50)
    print(f"Données : {X.shape[0]} segments de {WINDOW_SIZE} points")
    print(f"Transits : {y.sum()} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test =  train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    classes = np.unique(y_train)

    weights = compute_class_weight('balanced', classes=classes,y=y_train)

    class_weight = {0: 1.0, 1: 8.0}
    print(f"\nclass_weight : {class_weight}")
    print(f"→ Une erreur sur transit pénalise {weights[1]:.1f}x plus")

    model = build_cnn(input_shape=(WINDOW_SIZE,1))
    model.summary()

    callbacks=[

            tf.keras.callbacks.EarlyStopping(
                monitor='val_recall',
                patience=5,
                mode='max',
                restore_best_weights=True
            ),

         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1)

     ]
    
    print("\n🚀 Début entraînement...")

    history = model.fit(X_train,y_train,
                        epochs=30,
                        batch_size=32,
                        validation_split=0.2,
                        class_weight=class_weight,
                        callbacks=callbacks,
                        verbose=1
                        )
    
    print("\n" + "=" * 50)
    print("ÉVALUATION SUR TEST")
    print("=" * 50)

    results = model.evaluate(X_test,y_test,verbose=0)

    for name, val in zip(model.metrics_names, results):
        print(f"  {name:12s} : {val:.4f}")

    
    _plot_history(history)

    model.save(MODEL_PATH)
    print(f"\n💾 Modèle sauvegardé → {MODEL_PATH}")

    return model


def _plot_history(history):
    """Sauvegarde les courbes d'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
 
    axes[1].plot(history.history['recall'],     label='Train Recall')
    axes[1].plot(history.history['val_recall'], label='Val Recall')
    axes[1].set_title('Recall (détection transits)')
    axes[1].legend()
 
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("📊 Graphique sauvegardé → training_history.png")
 

def predict(flux_segment: np.ndarray,
            threshold: float = 0.3) -> dict:
    """
    Prédit si un segment de courbe de lumière contient un transit.
    
    threshold=0.3 : on abaisse le seuil de 0.5 à 0.3
    → on préfère avoir des faux positifs plutôt que rater un vrai transit
    → même logique que pour la fraude bancaire
    """
    model = tf.keras.models.load_model(MODEL_PATH)
 
    # Preprocessing identique à l'entraînement
    segment = flux_segment - 1.0  # Normalisation
    X = segment.reshape(1, WINDOW_SIZE, 1)
 
    proba = float(model.predict(X, verbose=0)[0][0])
    is_transit = proba >= threshold
 
    return {
        "is_transit":    is_transit,
        "probability":   round(proba, 4),
        "confidence_pct": round(proba * 100 if is_transit else (1 - proba) * 100, 2),
        "risk_level":    "HIGH" if proba > 0.7 else "MEDIUM" if proba > 0.3 else "LOW",
        "threshold_used": threshold
    }
 
 
# if __name__ == "__main__":
#     # Pour tester sans données réelles
#     print("Test avec données synthétiques...")
#     np.random.seed(42)
 
#     # Simule 1000 segments : 900 bruit + 100 transits
#     X_fake = np.random.normal(0, 0.001, (1000, 200, 1)).astype(np.float32)
 
#     # Ajoute une forme de transit dans les 100 premiers segments
#     transit_shape = np.zeros(200)
#     transit_shape[80:120] = -0.01 * np.hanning(40)  # Forme en U
#     for i in range(100):
#         X_fake[i, :, 0] += transit_shape
 
#     y_fake = np.array([1]*100 + [0]*900)
 
#     model = train(X_fake, y_fake)
#     print("\n✅ Test terminé — modèle fonctionnel")

if __name__ == "__main__":
    from augmentation import build_combined_dataset
    
    print("📥 Chargement dataset combiné...")
    X, y = build_combined_dataset(
        exotrain_path='exoTrain.csv',
        use_lightkurve=True
    )
    
    model = train(X, y)


