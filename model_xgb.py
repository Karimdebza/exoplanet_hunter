import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from features import build_feature_matrix, get_feature_names


MODEL_PATH = 'exoplanet_xgb.pk1'

def train(X: np.ndarray, y: np.ndarray)-> XGBClassifier:
    print("=" * 50)
    print("EXTRACTION DES FEATURES")
    print("=" * 50)
    X_feat = build_feature_matrix(X)

    print("\n" + "=" * 50)
    print("ENTRAÎNEMENT XGBOOST")
    print("=" * 50)
    print(f"Segments : {len(X_feat)}")
    print(f"Transits : {y.sum()} ({y.mean()*100:.1f}%)")
 
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )
 
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nscale_pos_weight = {ratio:.1f}")
 
    model = XGBClassifier(
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='aucpr',
        n_jobs=-1
    )
 
    params = {
        'max_depth':     [3, 5, 7],
        'n_estimators':  [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample':     [0.8, 1.0],
    }
 
    print("\n🔍 GridSearchCV en cours...")
    grid = GridSearchCV(model, params, scoring='recall', cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
 
    print(f"\nMeilleurs paramètres : {grid.best_params_}")
 
    print("\n" + "=" * 50)
    print("RÉSULTATS SUR TEST")
    print("=" * 50)
    y_pred  = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
 
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Transit']))
    print(f"Recall  : {recall_score(y_test, y_pred):.4f}")
    print(f"F1      : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-PR  : {roc_auc_score(y_test, y_proba):.4f}")
 
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatrice de confusion :")
    print(f"  Transits trouvés : {cm[1][1]}")
    print(f"  Transits ratés   : {cm[1][0]}  ← minimiser ça")
    print(f"  Faux positifs    : {cm[0][1]}")
 
    _plot_importance(best)
 
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best, f)
    print(f"\n💾 Modèle sauvegardé → {MODEL_PATH}")
 
    return best

def _plot_importance(model):
    names  = get_feature_names()
    scores = model.feature_importances_
    idx    = np.argsort(scores)[::-1]
 
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(names)), scores[idx], color='steelblue')
    plt.xticks(range(len(names)), [names[i] for i in idx], rotation=45, ha='right')
    plt.title("Importance des features — XGBoost ExoplanetHunter")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("📊 feature_importance.png sauvegardé")


def predict_star(star_name: str, model_path: str = MODEL_PATH, threshold: float = 0.3):
    import lightkurve as lk
    from features import extract_features
 
    print(f"\n🔭 Analyse de {star_name}...")
 
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
 
    lc = lk.search_lightcurve(star_name, author='Kepler', quarter=4).download()
    lc = lc.remove_nans().flatten(window_length=401)
    flux = lc.flux.value - 1.0
 
    detections = []
    all_probas  = []
 
    for i in range(0, len(flux) - 200, 100):
        segment = flux[i:i + 200]
        feat    = list(extract_features(segment).values())
        proba   = float(model.predict_proba([feat])[0][1])
        all_probas.append(proba)
        if proba >= threshold:
            detections.append({'segment': i, 'probability': proba})
 
    print(f"Probabilité max  : {max(all_probas):.4f}")
    print(f"Probabilité moy  : {np.mean(all_probas):.4f}")
    print(f"Transits détectés (seuil {threshold}) : {len(detections)}")
    for d in detections[:10]:
        print(f"  Segment {d['segment']:4d} → {d['probability']*100:.1f}%")
 
    return detections
def merge_detections(detections, gap=300):
    """
    Fusionne les détections consécutives en un seul événement.
    gap = distance max entre deux détections pour les considérer comme le même transit
    """
    if not detections:
        return []
    
    merged = []
    current_group = [detections[0]]
    
    for d in detections[1:]:
        if d['segment'] - current_group[-1]['segment'] <= gap:
            current_group.append(d)
        else:
            # Garde la détection avec la plus haute probabilité du groupe
            best = max(current_group, key=lambda x: x['probability'])
            merged.append(best)
            current_group = [d]
    
    best = max(current_group, key=lambda x: x['probability'])
    merged.append(best)
    
    return merged
 
if __name__ == "__main__":
    from data import build_dataset
 
    print("📥 Chargement des données Kepler...")
    X, y = build_dataset()
 
    model = train(X, y)

    print("\n" + "=" * 50)
    print("TEST SUR ÉTOILE INCONNUE — Kepler-4")
    print("=" * 50)
    detections = predict_star('Kepler-4', threshold=0.65)
    merged = merge_detections(detections)
    print(f"\nAprès fusion : {len(merged)} transits uniques")
for d in merged:
    print(f"  Segment {d['segment']:4d} → {d['probability']*100:.1f}%")