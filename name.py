import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('exoTrain.csv')
unknown = df[df['LABEL'] == 1].reset_index(drop=True)

# flux = unknown.iloc[265].drop('LABEL').values.astype(float)
# flux_norm = (flux - np.mean(flux)) / np.std(flux)

# plt.figure(figsize=(20, 5))
# plt.plot(flux_norm, color='steelblue', linewidth=0.5, alpha=0.8)
# plt.axhline(-3, color='orange', linestyle='--', label='-3σ')
# plt.axhline(-5, color='red', linestyle='--', label='-5σ')
# plt.title('Étoile 265 — Courbe de lumière complète')
# plt.xlabel('Temps (points Kepler × 30min)')
# plt.ylabel('Flux normalisé (σ)')
# plt.legend()
# plt.tight_layout()
# plt.savefig('star_265.png')
# print("✅ star_265.png sauvegardé")

flux = unknown.iloc[3697].drop('LABEL').values.astype(float)
flux_norm = (flux - np.mean(flux)) / np.std(flux)

plt.figure(figsize=(20, 5))
plt.plot(flux_norm, color='steelblue', linewidth=0.5, alpha=0.8)
plt.axhline(-3, color='orange', linestyle='--', label='-3σ')
plt.axhline(-5, color='red', linestyle='--', label='-5σ')
plt.title('Étoile 3697 — Courbe de lumière complète')
plt.xlabel('Temps (points Kepler × 30min)')
plt.ylabel('Flux normalisé (σ)')
plt.legend()
plt.tight_layout()
plt.savefig('star_3697.png')