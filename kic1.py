import lightkurve as lk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# lc = lk.search_lightcurve('KIC 8191672', author='Kepler', quarter=[3,4,5]).download_all().stitch()
# lc = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=401)

# flux = lc.flux.value
# flux = (flux - np.mean(flux)) / np.std(flux)
# time = lc.time.value

# plt.figure(figsize=(20, 5))
# plt.plot(time, flux, color='steelblue', linewidth=0.5, alpha=0.8)
# plt.axhline(-3, color='red', linestyle='--', alpha=0.5)
# plt.title('KIC 8191672 — Candidat planétaire (période ~3.66 jours)')
# plt.xlabel('Temps (BKJD)')
# plt.ylabel('Flux normalisé')
# plt.tight_layout()
# plt.savefig('kic_8191672.png')
# print("✅ kic_8191672.png sauvegardé")

lc = lk.search_lightcurve('KIC 9941662', author='Kepler', quarter=[3,4,5]).download_all().stitch()
lc = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=401)
flux = (lc.flux.value - np.mean(lc.flux.value)) / np.std(lc.flux.value)
time = lc.time.value

# plt.figure(figsize=(20, 5))
# plt.plot(time, flux, color='steelblue', linewidth=0.5, alpha=0.8)
# plt.axhline(-3, color='red', linestyle='--', alpha=0.5)
# plt.title('KIC 9941662 — Candidat (période ~1.77 jours)')
# plt.xlabel('Temps (BKJD)')
# plt.ylabel('Flux normalisé')
# plt.tight_layout()
# plt.savefig('kic_9941662.png')

# Zoom sur la zone 350-450 où le signal est propre
mask = (time > 350) & (time < 450)
plt.figure(figsize=(15, 5))
plt.plot(time[mask], flux[mask], color='steelblue', linewidth=0.8)
plt.axhline(-3, color='red', linestyle='--', alpha=0.5)
plt.title('KIC 9941662 — Zoom zone propre (350-450 BKJD)')
plt.xlabel('Temps (BKJD)')
plt.ylabel('Flux normalisé')
plt.tight_layout()
plt.savefig('kic_9941662_zoom.png')