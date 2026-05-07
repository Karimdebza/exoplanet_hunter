"""
api.py — Backend FastAPI pour ExoplanetHunter.

ARCHITECTURE :
- POST /api/scan         : lance un scan (retourne job_id immédiatement)
- GET  /api/scan/{id}    : statut + résultats quand terminé
- GET  /api/history      : liste des scans passés
- GET  /api/health       : healthcheck pour Render

POURQUOI JOB_ID ET PAS RÉPONSE DIRECTE :
Le pipeline prend 2-5 minutes. Un POST synchrone timeout dans le navigateur.
Avec un job_id, Angular poll toutes les 2 secondes et affiche une progress bar.

DÉMARRAGE :
    uvicorn api:app --reload --port 8000

CORS :
    Configuré pour accepter localhost:4200 (Angular dev) + domaine Vercel en prod.
"""

import uuid
import asyncio
import base64
import os
import sys
import json
import traceback
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Import pipeline ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.types import CleanedLightCurve
from detection.bls_scanner import detect_signals
from validation.phase_folder import fold_transit
from validation.cnn_model import load_model, predict, DEFAULT_MODEL_PATH
from validation.physical_validator import validate

import lightkurve as lk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scriptes", "models", "exoplanet_cnn_v3.h5"
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ExoplanetHunter API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",   # Angular dev
        "http://localhost:3000",
        "https://*.vercel.app",    # Vercel prod
        "*",                       # à restreindre en prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State en mémoire (remplacer par Redis en prod) ────────────────────────────
jobs: dict[str, dict] = {}
history: list[dict]   = []
executor = ThreadPoolExecutor(max_workers=2)

# CNN chargé une seule fois au démarrage
_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_model(DEFAULT_MODEL_PATH)
    return _model


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    star_name: str
    quarters: list[int] = [3, 4, 5, 6]
    max_transit_duration_hours: float = 8.0


class ScanResponse(BaseModel):
    job_id: str
    message: str


# ── Pipeline (tourne dans un thread séparé) ───────────────────────────────────
def _run_pipeline(job_id: str, star_name: str, quarters: list[int],
                  max_transit_hours: float):
    """
    Exécute le pipeline complet et met à jour jobs[job_id] au fur et à mesure.
    """
    def update(step: str, progress: float, **kwargs):
        jobs[job_id].update({"step": step, "progress": progress, **kwargs})

    try:
        update("downloading", 0.05)

        # ── Couche 1 : ingestion ──────────────────────────────────────────────
        search = lk.search_lightcurve(star_name, author='Kepler', quarter=quarters)
        if len(search) == 0:
            raise ValueError(f"Aucune donnée Kepler pour '{star_name}'")

        lc_raw = search.download_all()
        if lc_raw is None:
            raise ValueError("Téléchargement échoué")

        lc_stitch = lc_raw.stitch().remove_nans().remove_outliers(sigma=5)

        cadence = float(np.median(np.diff(lc_stitch.time.value)))
        window_days = 3 * max_transit_hours / 24.0
        window_size = max(11, int(window_days / cadence))
        if window_size % 2 == 0:
            window_size += 1

        lc_flat = lc_stitch.flatten(window_length=window_size, polyorder=2)
        lc_flat = lc_flat.remove_outliers(sigma=4)

        lc = CleanedLightCurve(
            time=np.array(lc_flat.time.value),
            flux=np.array(lc_flat.flux.value),
            star_id=star_name,
            quarters_used=list(quarters),
        )

        update("bls", 0.30,
               n_points=lc.n_points,
               duration_days=round(lc.duration_days, 1))

        # ── Couche 2 : BLS ────────────────────────────────────────────────────
        candidates = detect_signals(lc, top_k=5)
        if not candidates:
            jobs[job_id].update({
                "status": "done",
                "progress": 1.0,
                "results": [],
                "message": "Aucun signal détecté",
            })
            _save_history(star_name, quarters, [])
            return

        update("cnn", 0.50, n_candidates=len(candidates))

        # ── Couches 3 + 4 : CNN + validation physique ─────────────────────────
        model   = get_model()
        results = []

        for i, candidate in enumerate(candidates):
            progress = 0.50 + 0.40 * (i / len(candidates))
            update("validating", progress,
                   current_candidate=i, n_candidates=len(candidates))

            folded = fold_transit(lc, candidate)
            if folded is None:
                continue

            cnn_val  = predict(folded, model)
            phys_val = validate(lc, cnn_val)

            # Graphique phase folding → base64
            plot_b64 = _make_plot_b64(folded, cnn_val, phys_val, star_name)

            results.append({
                "rank"           : candidate.rank,
                "period"         : round(candidate.period, 4),
                "depth_ppm"      : round(candidate.depth_ppm, 1),
                "duration_hours" : round(candidate.duration_hours, 2),
                "sde"            : round(candidate.sde, 2),
                "cnn_score"      : round(cnn_val.cnn_score, 3),
                "cnn_passed"     : cnn_val.passed,
                "classification" : phys_val.classification,
                "confidence"     : phys_val.confidence,
                "tests": [
                    {
                        "name"   : t.test_name,
                        "passed" : bool(t.passed),
                        "score"  : float(round(t.score, 4)),
                        "details": {
                            k: (bool(v) if isinstance(v, np.bool_) else
                                float(v) if isinstance(v, (np.floating, np.integer)) else
                                v)
                            for k, v in t.details.items()
                        },
                    }
                    for t in phys_val.tests
                ],
                "plot_base64"    : plot_b64,
            })

        jobs[job_id].update({
            "status"  : "done",
            "progress": 1.0,
            "results" : results,
            "message" : f"{len(results)} candidats analysés",
        })

        _save_history(star_name, quarters, results)

    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        print(traceback.format_exc())
        jobs[job_id].update({...})


def _make_plot_b64(folded, cnn_val, phys_val, star_name: str) -> str:
    """Génère le graphique phase folding et retourne en base64."""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.linspace(-2, 2, len(folded.flux))
        ax.plot(x, folded.flux, 'o-', color='#4FC3F7', markersize=2, alpha=0.8)
        ax.axvline(0, color='#EF5350', linestyle='--', alpha=0.6)
        ax.axhline(1.0, color='#66BB6A', linestyle=':', alpha=0.5)

        c = cnn_val.candidate
        color = '#66BB6A' if phys_val.classification == 'PLANET_CANDIDATE' else '#EF5350'
        ax.set_title(
            f"{star_name} — P={c.period:.3f}j | {phys_val.classification} "
            f"(conf={phys_val.confidence:.2f})\n"
            f"depth={c.depth_ppm:.0f}ppm, SDE={c.sde:.1f}, CNN={cnn_val.cnn_score:.3f}",
            color=color, fontsize=9
        )
        ax.set_facecolor('#0D1117')
        fig.patch.set_facecolor('#0D1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.spines[:].set_color('#30363D')
        ax.set_xlabel("Phase (× durée transit)")
        ax.set_ylabel("Flux normalisé")

        import io
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight',
                    facecolor='#0D1117')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception:
        return ""


def _save_history(star_name: str, quarters: list[int], results: list[dict]):
    """Sauvegarde dans l'historique en mémoire."""
    best = None
    planet_candidates = [r for r in results if r["classification"] == "PLANET_CANDIDATE"]
    if planet_candidates:
        best = max(planet_candidates, key=lambda r: r["confidence"])

    history.insert(0, {
        "id"          : str(uuid.uuid4()),
        "star_name"   : star_name,
        "quarters"    : quarters,
        "date"        : datetime.now().isoformat(),
        "n_candidates": len(results),
        "n_planet_candidates": len(planet_candidates),
        "best_candidate": {
            "period"        : best["period"],
            "classification": best["classification"],
            "confidence"    : best["confidence"],
        } if best else None,
    })


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/scan", response_model=ScanResponse)
def start_scan(req: ScanRequest, background_tasks: BackgroundTasks):
    """Lance un scan en arrière-plan et retourne un job_id immédiatement."""
    if not req.star_name.strip():
        raise HTTPException(status_code=400, detail="star_name requis")
    if not req.quarters:
        raise HTTPException(status_code=400, detail="quarters requis")
    if len(req.quarters) > 18:
        raise HTTPException(status_code=400, detail="Maximum 18 quarters")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status"    : "running",
        "step"      : "initializing",
        "progress"  : 0.0,
        "star_name" : req.star_name,
        "quarters"  : req.quarters,
        "created_at": datetime.now().isoformat(),
        "results"   : None,
        "error"     : None,
    }

    background_tasks.add_task(
        _run_pipeline,
        job_id,
        req.star_name,
        req.quarters,
        req.max_transit_duration_hours,
    )
    

    return ScanResponse(job_id=job_id, message="Scan lancé")


@app.get("/api/scan/{job_id}")
def get_scan(job_id: str):
    """Retourne le statut et les résultats d'un scan."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job introuvable")
    return jobs[job_id]


@app.get("/api/history")
def get_history():
    """Retourne l'historique des scans (50 derniers)."""
    return history[:50]


@app.delete("/api/history")
def clear_history():
    """Vide l'historique."""
    history.clear()
    return {"message": "Historique vidé"}