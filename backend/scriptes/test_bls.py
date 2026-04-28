"""
Test du BLS scanner avec une courbe synthétique simulant Kepler-5b.

Kepler-5b : Hot Jupiter, P=3.55j, depth ~0.7%, durée ~2.9h.
Si le scanner trouve correctement la période, le test passe.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.types import CleanedLightCurve
from detection.bls_scanner import detect_signals
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# Génération d'une courbe synthétique type Kepler-5b
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)

PERIOD_TRUE = 3.5485    # jours - Kepler-5b
DEPTH_TRUE = 0.0067     # 6700 ppm
DURATION_TRUE = 0.120   # ~2.88h
T0_TRUE = 1.5           # premier transit

# 90 jours d'observation, cadence Kepler longue (~30 min)
time = np.arange(0, 90, 30/60/24)  # 30 min en jours
flux = np.ones_like(time)

# Bruit gaussien réaliste Kepler (~150 ppm RMS pour étoile mag 12)
flux += np.random.normal(0, 150e-6, len(time))

# Injection des transits (boîte simplifiée)
n_transits = 0
t_transit = T0_TRUE
while t_transit < time[-1]:
    in_transit = np.abs(time - t_transit) < DURATION_TRUE / 2
    flux[in_transit] -= DEPTH_TRUE
    n_transits += 1
    t_transit += PERIOD_TRUE

print(f"Courbe synthétique générée : {len(time)} points, {n_transits} transits injectés")
print(f"Période vraie : {PERIOD_TRUE}j, depth : {DEPTH_TRUE*1e6:.0f}ppm\n")

# ─────────────────────────────────────────────────────────────────────────────
# Création de la LightCurve et lancement du scanner
# ─────────────────────────────────────────────────────────────────────────────
lc = CleanedLightCurve(
    time=time,
    flux=flux,
    star_id="Kepler-5-synthetic",
    quarters_used=[3, 4, 5],
)

candidates = detect_signals(lc, top_k=5, period_min=0.5, period_max=15.0)

# ─────────────────────────────────────────────────────────────────────────────
# Vérifications
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("VÉRIFICATIONS")
print("="*60)

print(f"\nNombre de candidats trouvés : {len(candidates)}")

if not candidates:
    print("✗ ÉCHEC : aucun candidat trouvé")
    sys.exit(1)

best = candidates[0]
period_error = abs(best.period - PERIOD_TRUE) / PERIOD_TRUE

print(f"\nMeilleur candidat : {best}")
print(f"\n  Période trouvée : {best.period:.4f}j")
print(f"  Période vraie   : {PERIOD_TRUE:.4f}j")
print(f"  Erreur          : {period_error*100:.3f}%")

if period_error < 0.01:  # Erreur < 1%
    print(f"\n✓ SUCCÈS : période trouvée à {period_error*100:.3f}% près")
else:
    print(f"\n✗ ÉCHEC : erreur trop grande ({period_error*100:.3f}%)")
    sys.exit(1)

# Vérification de l'anti-aliasing : aucun candidat ne doit être un multiple
print("\n--- Vérification anti-aliasing ---")
for i, c in enumerate(candidates[1:], 1):
    ratio = c.period / best.period
    print(f"  #{i} P={c.period:.3f}j, ratio au #0 = {ratio:.3f}")
    if abs(ratio - round(ratio)) < 0.05 and round(ratio) in [1, 2, 3]:
        print(f"  ✗ ALIAS DÉTECTÉ — anti-aliasing a échoué")
        sys.exit(1)

print("\n✓ Anti-aliasing OK")
print("\n" + "="*60)
print("TOUS LES TESTS PASSENT")
print("="*60)