# bls_filter.py — Vérifie la périodicité des candidats
import pickle
import numpy as np
import pandas as pd

with open('candidates.pkl', 'rb') as f:
    candidates = pickle.load(f)

df = pd.read_csv('exoTrain.csv')
unknown = df[df['LABEL'] == 1].reset_index(drop=True)

print(f"🔍 Analyse BLS sur {len(candidates)} candidats...\n")

for cand in candidates:
    idx       = cand['star_idx']
    dets      = cand['detections']  # positions des segments détectés
    
    if len(dets) < 3:
        continue
    
    # Calcule les intervalles entre détections consécutives
    intervals = [dets[i+1] - dets[i] for i in range(len(dets)-1)]
    mean_interval = np.mean(intervals)
    std_interval  = np.std(intervals)
    regularity    = 1 - (std_interval / (mean_interval + 1e-8))
    
    print(f"Étoile {idx:4d} | {len(dets):2d} détections "
          f"| intervalle moyen={mean_interval:.0f} pts "
          f"| régularité={regularity:.2f} "
          f"| max_proba={cand['max_proba']:.3f}")