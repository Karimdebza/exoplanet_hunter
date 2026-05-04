"""
scriptes/explore_star.py — TON OUTIL D'EXPLORATION.

USAGE :
    python scriptes/explore_star.py "Kepler-5"
    python scriptes/explore_star.py "KIC 11657614"
    python scriptes/explore_star.py "Kepler-90" --quarters 3,4,5,6,7

PIPELINE COMPLET :
1. Téléchargement courbe de lumière (lightkurve)
2. Nettoyage (NaN removal, outliers, detrending)
3. Détection BLS (top-K candidats)
4. Phase folding sur chaque candidat
5. Validation CNN sur la forme
6. Affichage du verdict + graphiques

OUTPUT :
- Texte console avec verdict pour chaque candidat
- Graphique PNG par candidat dans ./outputs/

⚠️ MODE EXPLORATION :
Le verdict CNN est INDICATIF, pas définitif. La couche 4 (validation physique)
n'est pas encore implémentée. Un score CNN haut peut encore être un faux positif
(binaire à éclipses sophistiquée). Considère ça comme un PRÉ-FILTRE.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend non-interactif (pas besoin de display)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightkurve as lk
from core.types import CleanedLightCurve
from detection.bls_scanner import detect_signals
from validation.phase_folder import fold_transit
from validation.cnn_model import load_model, predict, DEFAULT_MODEL_PATH
from validation.physical_validator import validate


def download_and_clean(star_name: str, quarters: list[int],max_transit_duration_hours: float = 8.0) -> CleanedLightCurve:
    """Couche 1 : ingestion."""
    print(f"\n📡 Téléchargement {star_name} (quarters {quarters})...")
    
    search = lk.search_lightcurve(star_name, author='Kepler', quarter=quarters)
    if len(search) == 0:
        raise ValueError(f"Aucune donnée trouvée pour {star_name}")
    
    lc_collection = search.download_all()
    if lc_collection is None:
        raise ValueError(f"Téléchargement échoué pour {star_name}")
    
    lc = lc_collection.stitch()
    lc = lc.remove_nans().remove_outliers(sigma=5)
    # --- LA RÈGLE DU FACTEUR 3 DE KARIM ---
    max_tansit_days =  max_transit_duration_hours / 24.0
    window_days = 3 * max_tansit_days
    cadence_jours = np.median(np.diff(lc.time.value))
    window_size = int(window_days / cadence_jours)
    
    if window_size % 2 == 0:
        window_size +=1
    print(f"   ⚙️ Durée max attendue : {max_transit_duration_hours}h")
    print(f"   ⚙️ Fenêtre de lissage dynamique : {window_size} points ({window_days:.2f} jours)")

    lc = lc.flatten(window_length=window_size, polyorder=2)
    lc = lc.remove_outliers(sigma=4)
    
    cleaned = CleanedLightCurve(
        time=np.array(lc.time.value),
        flux=np.array(lc.flux.value),
        star_id=star_name,
        quarters_used=list(quarters),
    )
    
    print(f"   ✓ {cleaned.n_points} points, {cleaned.duration_days:.1f}j d'observation")
    return cleaned


def plot_candidate(folded, validation, star_name: str, output_dir: str):
    """Génère un graphique pour un candidat."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.linspace(-2, 2, len(folded.flux))
    ax.plot(x, folded.flux, 'o-', color='steelblue', markersize=3, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Centre du transit')
    ax.axhline(1.0, color='green', linestyle=':', alpha=0.5, label='Baseline')
    
    c = validation.candidate
    verdict_color = 'green' if validation.passed else 'red'
    verdict_text = '✓ PASSED' if validation.passed else '✗ REJECTED'
    
    title = (
        f"{star_name} — Candidat #{c.rank} | {verdict_text}\n"
        f"P={c.period:.3f}j, depth={c.depth_ppm:.0f}ppm, "
        f"dur={c.duration_hours:.2f}h, SDE={c.sde:.2f} | "
        f"CNN score={validation.cnn_score:.3f}"
    )
    ax.set_title(title, color=verdict_color)
    ax.set_xlabel("Phase relative (en multiples de durée du transit)")
    ax.set_ylabel("Flux normalisé")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    safe_name = star_name.replace(' ', '_').replace('/', '_')
    filepath = os.path.join(output_dir, f"{safe_name}_candidate_{c.rank}.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=100)
    plt.close(fig)
    return filepath


def explore(star_name: str, quarters: list[int], output_dir: str = "outputs"):
    """Pipeline complet d'exploration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # ─── Couche 1 : Ingestion ────────────────────────────────────────────────
    lc = download_and_clean(star_name, quarters)
    
    # ─── Couche 2 : Détection BLS ────────────────────────────────────────────
    print(f"\n🔍 Détection BLS sur {star_name}...")
    candidates = detect_signals(lc, top_k=5)
    
    if not candidates:
        print("   ✗ Aucun candidat détecté")
        return
    
    print(f"   ✓ {len(candidates)} candidats trouvés")
    
    # ─── Couche 3 : Validation CNN ───────────────────────────────────────────
    print(f"\n🧠 Chargement du CNN...")
    try:
        model = load_model(DEFAULT_MODEL_PATH)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("   Lance d'abord : python scriptes/train_cnn_v3.py")
        return
    
    print(f"\n📊 Validation des candidats :\n")
    print(f"{'#':<3} {'Période':<12} {'Depth':<10} {'Durée':<8} {'SDE':<8} {'CNN':<8} {'Verdict'}")
    print("-" * 75)
    
    for candidate in candidates:
        folded = fold_transit(lc, candidate)
        if folded is None:
            print(f"{candidate.rank:<3} P={candidate.period:.3f}j  → fold échoué (pas assez de points)")
            continue
        
        validation = predict(folded, model)
        physical = validate(lc, validation)
        print(f"   🔬 {physical.classification} (confidence={physical.confidence:.2f})")
        for test in physical.tests:
            print(f"      {test}")
        
        verdict = "✓ PASS" if validation.passed else "✗ FAIL"
        print(
            f"{candidate.rank:<3} "
            f"{candidate.period:<12.4f} "
            f"{candidate.depth_ppm:<10.0f} "
            f"{candidate.duration_hours:<8.2f} "
            f"{candidate.sde:<8.2f} "
            f"{validation.cnn_score:<8.3f} "
            f"{verdict}"
        )
        
        # Génération du graphique
        filepath = plot_candidate(folded, validation, star_name, output_dir)
        if validation.passed:
            print(f"      → 📸 {filepath}")
    
    print("\n" + "=" * 75)
    print(f"⚠️  Rappel : ces verdicts sont INDICATIFS. La validation physique")
    print(f"   (odd/even, secondary eclipse, V-shape, centroid) reste à implémenter.")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore une étoile pour des transits.")
    parser.add_argument("star_name", help="Ex: 'Kepler-5', 'KIC 11657614'")
    parser.add_argument(
        "--quarters", default="3,4,5,6",
        help="Quarters Kepler à utiliser (défaut: 3,4,5,6)"
    )
    parser.add_argument(
        "--output", default="outputs",
        help="Dossier de sortie pour les graphiques (défaut: outputs)"
    )
    args = parser.parse_args()
    
    quarters = [int(q.strip()) for q in args.quarters.split(",")]
    explore(args.star_name, quarters, args.output)