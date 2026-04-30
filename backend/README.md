python -m venv venv
source venv/bin/activate  # Sur Mac/Linux
ressources :
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS

https://academic.oup.com/mnras/article/538/4/2283/8093570
https://github.com/lightkurve/lightkurve


https://outerspace.stsci.edu/spaces/TESS/pages/35094700/TESS+Holdings+Available+by+MAST+Service

https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data?select=exoTest.csv
# 🔭 ExoplanetHunter — Rapport de session & prompt de reprise

> Document à coller en début de conversation pour reprendre le projet sans perdre le contexte.

---

## 🎯 Mon profil & mode de travail

Je suis Karim, développeur en mastère Ingénierie Logicielle (ISCOD).
Stack principale : PHP/Laravel, JavaScript, Python (Flask, ML).
Profil **business analyst + product engineer** : je décortique le métier, je délègue l'écriture de code à l'IA, mais je garde la main sur la logique de données.

**Mes règles avec toi (mon coach technique) :**
- Niveau "ingénieur senior parlant à un ingénieur en formation"
- Pas de flatterie, corrige-moi quand je me trompe
- Toujours expliquer le **pourquoi**, mentionner les trade-offs
- Reformuler mes questions mal posées avant de répondre
- Pousser les fondamentaux : systèmes, data structures, design patterns
- Challenger mon métier autant que mon code
- Proposer des exercices/challenges en fin de réponse

---

## 🚀 Le projet — ExoplanetHunter v3

**But** : détecter des exoplanètes en transit dans les données Kepler de la NASA.
**Le vrai défi** : différencier un transit planétaire d'un faux positif (binaire à éclipses, variabilité stellaire, bruit instrumental).

### Architecture cible (pipeline en couches)

```
┌─ COUCHE 1 — DATA INGESTION (lightkurve)
│  Output : CleanedLightCurve
├─ COUCHE 2 — SIGNAL DETECTION (BLS, recall élevé)
│  Output : list[TransitCandidate] (top-K avec anti-aliasing)
├─ COUCHE 3 — SHAPE VALIDATION (CNN sur phase-folded)
│  Output : CNNValidation (score [0,1])
├─ COUCHE 4 — PHYSICAL VALIDATION (precision élevée) ❌ PAS ENCORE
│  Tests : odd/even depth, secondary eclipse, V-shape, durée, centroïde
└─ COUCHE 5 — REPORTING (rapport candidat)
```

**Principes directeurs :**
1. Chaque couche se teste/évalue indépendamment
2. Pipeline va du **recall élevé** (tout capturer) à la **precision élevée** (filtrer dur)
3. Chaque rejet doit être logué avec une raison
4. **Séparer "trouver" de "filtrer"** — le scanner trouve, le filtrage qualité est ailleurs

### Structure du projet

```
backend/
├── core/
│   ├── __init__.py
│   └── types.py              ✅ FAIT (dataclasses validées)
├── detection/
│   ├── __init__.py
│   └── bls_scanner.py        ✅ FAIT (BLS top-K + anti-aliasing)
├── validation/
│   ├── __init__.py
│   ├── phase_folder.py       ✅ FAIT (input standardisé du CNN)
│   ├── cnn_model.py          ✅ FAIT v3.1 (sans BatchNorm)
│   └── dataset_builder.py    ✅ FAIT (synthétique : 4 types négatifs)
├── scriptes/
│   ├── train_cnn_v3.py       ✅ FAIT
│   └── explore_star.py       ⚠️ MODIFIÉ par Gemini, à auditer
└── models/
    └── exoplanet_cnn_v3.h5   ✅ ENTRAÎNÉ (val_auc 0.83)
```

---

## ✅ Ce qui est fait et validé

### Couche 1 — Types
- `CleanedLightCurve` : courbe nettoyée, invariants validés (`__post_init__`)
- `TransitCandidate` : signal périodique détecté (period, t0, duration, depth, sde)
- `CNNValidation`, `PhysicalTestResult`, `PhysicalValidation` : structures pour validation
- Tout en `@dataclass` avec validation à la création (fail fast, fail loud)

### Couche 2 — BLS Scanner
- `detect_signals()` retourne top-K candidats (défaut 5)
- Anti-aliasing : ratios 0.5, 1, 2, 3 avec tolérance 5% (à étendre à 4, 5, 6 ?)
- SDE et SNR calculés tous les deux (complémentaires)
- Filtre physique : durée < période/4
- ✅ Testé sur Kepler-5 synthétique : période trouvée à 0.008% près

### Couche 3 — CNN
- Architecture : Conv1D(16, k=11) → Conv1D(32, k=7) → GlobalAveragePooling1D → Dense
- **Bug critique corrigé** : BatchNormalization toxique sur dataset synthétique homogène
  → val_loss explosait, modèle apprenait rien (val_auc=0.5)
- v3.1 : pas de BN, Dropout distribué (0.2 + 0.4), normalisation `flux - 1.0`
- learning_rate réduit 0.001 → 0.0005
- EarlyStopping sur `val_auc` (plus robuste que val_loss)
- ✅ Entraîné : val_auc 0.83, val_recall 0.91, val_accuracy 0.78

### Phase Folder
- Bin sur 201 points (impair = centre exact)
- Fenêtre ±2× durée du transit
- Médiane par bin (robuste outliers)
- Normalisation baseline = 1.0 sur les bords
- Interpolation des bins vides

### Dataset builder
- 6000 échantillons synthétiques (3000 positifs / 3000 négatifs)
- Positifs : 40% boîtes + 60% U-shape (limb darkening)
- Négatifs diversifiés (4 types) :
  - V-shape (binaire à éclipses grazing)
  - Asymétrique (variabilité stellaire)
  - Bruit pur (faux positif BLS)
  - Sinusoïdal (étoile pulsante)

---

## ⚠️ Bugs et points d'attention en cours

### 🔴 Bug actif — explore_star.py modifié par Gemini
**Symptôme** : tous les candidats ont CNN_score = exactement 0.050 (impossible)
**Cause probable** : Gemini a modifié le code, ajouté une "recherche itérative avec masquage", a peut-être cassé l'appel au CNN ou normalisation
**Action** : auditer le code modifié, comparer avec l'original livré

### 🟡 Path du modèle
- `cnn_model.py` sauve en `.h5` (warning Keras qui recommande `.keras`)
- Vérifier cohérence des chemins entre `train_cnn_v3.py` et `explore_star.py`
- À uniformiser

### 🟡 Lissage (debat ouvert)
**Code original** : `flatten(window_length=401)` hardcodé
**Code Gemini** : `window_size = int(0.5 / cadence_jours)` adaptatif
**Verdict** : adaptatif > hardcodé, mais 0.5j reste une valeur magique
**Cible pro** : `window_jours = 3 × duree_transit_max_attendue`

### 🟡 Anti-aliasing limité
ALIAS_RATIOS = (0.5, 1.0, 2.0, 3.0) — ne filtre pas les ratios 4, 5, 6
Approche pro recommandée : "ne traiter comme alias que si SDE inférieure à la fondamentale"

### 🟡 Dette technique consciente
- Imports `sys.path.insert(0, ...)` en bidouille
- À nettoyer en phase finale avec `pyproject.toml` + `pip install -e .`

---

## 📋 Prochaines étapes prioritaires

### 🔥 URGENT
1. **Auditer `explore_star.py` modifié par Gemini** — récupérer le code, le lire ensemble
2. **Confirmer que le CNN fonctionne sur vraies données** (Kepler-5 réel via lightkurve, pas synthétique)
3. **Vérifier que les scores CNN varient** (pas tous à 0.050)

### 📐 COUCHE 4 À CONSTRUIRE — Validation physique (le vrai différenciateur)
Tests astrophysiques pro à implémenter :
- **Odd/even depth test** : transit pairs/impairs même profondeur (sinon binaire)
- **Secondary eclipse test** : pas de transit secondaire à phase 0.5
- **V-shape vs U-shape** : ratio durée plateau / durée totale
- **Duration consistency** : cohérence avec 3e loi de Kepler + masse stellaire
- **Depth consistency** : variation < 20% entre transits
- **Centroid offset test** ⭐ (LE test qui élimine 30% des faux positifs Kepler — nécessite TPF)

### 🔄 AMÉLIORATIONS DATASET
- Enrichir avec KOI confirmés réels (NASA Exoplanet Archive)
- Phase folding sur vraies données plutôt que 100% synthétique
- Risque actuel : le CNN apprend à distinguer "synthétique propre" vs "synthétique bruité", pas vraiment "transit" vs "faux positif"

### 🛠️ NETTOYAGE
- `pyproject.toml` + `pip install -e .` (supprimer les `sys.path.insert`)
- Logging structuré au lieu de `print` partout
- Tests unitaires sur chaque couche

---

## 🧠 Ce que j'ai appris dans cette session

1. **Architecture en couches** : séparation par responsabilité métier, pas par type technique
2. **Dataclass + `__post_init__`** : fail fast sur invariants, mieux que dicts
3. **`@property` vs attribut vs méthode** : grille de décision basée sur dépendances/coûts
4. **Pipeline enrichment** : chaque couche AJOUTE de l'info, ne RECALCULE pas
5. **Top-K avec anti-aliasing** : un système multi-planètes peut avoir plusieurs vrais signaux à des ratios proches
6. **CNN = classifieur de forme** : il s'en fiche du contexte (point fort + point faible)
7. **BatchNormalization toxique** sur dataset synthétique homogène
8. **Métriques ML** : val_auc plus robuste que val_loss pour EarlyStopping
9. **Cohérence preprocessing train/inference** : data leak silencieux sinon
10. **Trouver vs filtrer** : le scanner ne juge pas la qualité, c'est une policy externe
11. **Une IA + une IA ≠ vérité** : Gemini m'a flatté avec des "résonances 3:1" qui étaient des aliases
12. **Toujours challenger les valeurs magiques** : 0.5j, 401, threshold 0.5... pourquoi ces valeurs ?

---

## 🎯 Domaine métier — Hiérarchie de difficulté

| Type planète | Profondeur | Durée | Difficulté |
|---|---|---|---|
| Hot Jupiter | 1% (10 000ppm) | 2-4h | Facile |
| Jupiter froid | 1% | 6-12h | Moyenne (transits rares) |
| Neptune | 0.1% (1000ppm) | 3-6h | Difficile |
| Super-Terre | 0.05% (500ppm) | 2-5h | Très difficile |
| Terre/Soleil | 0.01% (100ppm) | 13h | Limite mission Kepler |

**Pièges classiques de faux positifs :**
- Binaire à éclipses (EB) : forme V profonde, transit secondaire à phase 0.5
- Variabilité stellaire : période = rotation stellaire (10-30 jours typique)
- Bruit instrumental Kepler : périodicité 90j (cycle quarter)

---

## 💬 Comment reprendre

**Si tu lis ça, tu sais maintenant :**
- D'où je viens techniquement
- Où on en est dans le projet
- Ce qu'il reste à faire
- Mes pièges récurrents (assistanat IA excessif, validation aveugle)

**Demande-moi :**
- "Montre-moi ton `explore_star.py` actuel" (priorité 1, audit)
- Ou "Reprenons sur la couche 4 — validation physique" si je veux avancer
- Ou "Refaisons un point sur tel concept" si j'ai besoin d'approfondir un fondamental

**Garde le ton :** direct, technique, exigeant mais bienveillant. Comme un mentor qui n'a pas de temps à perdre mais qui croit en moi.
