
import lightkurve as lk
import numpy as np
import tensorflow as tf
import pickle

MODEL_PATH = 'exoplanet_cnn_v2.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# KOI non confirmés — statut "CANDIDATE" dans l'archive NASA
# Source : https://exoplanetarchive.ipac.caltech.edu
# Ces étoiles ont un signal suspect mais jamais confirmé
KOI_CANDIDATES = [
    'KIC 757450',   # KOI-256  — signal périodique détecté
    'KIC 4544670',  # KOI-314  — plusieurs candidats
    'KIC 6521045',  # KOI-523  — période ~10 jours
    'KIC 8191672',  # KOI-688  — non confirmé
    'KIC 5358624',  # KOI-817  — candidat chaud
    'KIC 6716021',  # KOI-1161 — signal faible
    'KIC 9941662',  # KOI-1599 — deux candidats
    'KIC 3558849',  # KOI-1726 — non confirmé
]



def merge_detections(detections, gap=300):
    if not detections:
        return []
    merged = []
    current = [detections[0]]
    for d in detections[1:]:
        if d['segment'] - current[-1]['segment'] <= gap:
            current.append(d)
        else:
            merged.append(max(current, key=lambda x: x['proba']))
            current = [d]
    merged.append(max(current, key=lambda x: x['proba']))
    return merged

def scan_koi(star_name):
    print(f"\n🔭 Analyse de {star_name}...")
    try:
        # Télécharge plusieurs quarters pour plus de signal
        lc = lk.search_lightcurve(
            star_name, author='Kepler', quarter=[3,4,5]
        ).download_all().stitch()
        
        lc = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=401)
        # Par ça
        flux = lc.flux.value
        flux = flux - np.mean(flux)
        flux = flux / (np.std(flux) + 1e-8)  # Normalisation Z-score
        time = lc.time.value

        # Scan batch
        segments, times_mid = [], []
        for i in range(0, len(flux) - 200, 50):  # Step=50 pour plus de résolution
            segments.append(flux[i:i+200])
            times_mid.append(time[i + 100])

        X = np.array(segments).reshape(-1, 200, 1)
        probas = model(X, training=False).numpy().flatten()

        detections = [
            {'segment': i*50, 'time': times_mid[i], 'proba': float(p)}
            for i, p in enumerate(probas) if p > 0.3  # 0.5 → 0.3
        ]


        merged = merge_detections(detections, gap=300)

        if len(merged) >= 2: 
            intervals = [merged[i+1]['time'] - merged[i]['time']
                        for i in range(len(merged)-1)]
            period_days = np.median(intervals)
            regularity  = 1 - np.std(intervals)/(np.mean(intervals)+1e-8)

            print(f"  ✅ {len(merged)} transits détectés")
            print(f"  📅 Période estimée : {period_days:.2f} jours")
            print(f"  📊 Régularité      : {regularity:.2f}")
            print(f"  🎯 Max probabilité : {max(d['proba'] for d in merged):.3f}")
            print(f"  ℹ️  Proba max : {float(np.max(probas)):.3f} | "
            f"Proba moy : {float(np.mean(probas)):.3f}")

            return {
                'star':       star_name,
                'n_transits': len(merged),
                'period':     period_days,
                'regularity': regularity,
                'detections': merged
            }
        else:
            print(f"  ❌ Signal insuffisant ({len(merged)} transits)")
            return None

    except Exception as e:
        print(f"  ❌ Erreur : {e}")
        return None


if __name__ == "__main__":
    print("🚀 Chasse aux KOI non confirmés...\n")
    results = []

    for star in KOI_CANDIDATES:
        result = scan_koi(star)
        if result:
            results.append(result)

    print(f"\n{'='*50}")
    print(f"RÉSULTATS FINAUX — {len(results)} candidats confirmés")
    print(f"{'='*50}")

    for r in sorted(results, key=lambda x: x['regularity'], reverse=True):
        print(f"\n🪐 {r['star']}")
        print(f"   Période    : {r['period']:.2f} jours")
        print(f"   Régularité : {r['regularity']:.2f}")
        print(f"   Transits   : {r['n_transits']}")

    with open('koi_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Résultats sauvegardés → koi_results.pkl")