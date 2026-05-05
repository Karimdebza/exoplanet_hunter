// src/app/pages/search/search.component.ts

import { Component, OnDestroy } from '@angular/core';
import { Router } from '@angular/router';
import { FormControl, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { ExoplanetService } from '../../core/services/exoplanet.service';
import { ScanJob, STEP_LABELS } from '../../core/models/scan.model';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  template: `
    <div class="search-page">

      <!-- Header -->
      <header class="header">
        <div class="logo">
          <span class="logo-icon">◎</span>
          <span class="logo-text">ExoplanetHunter</span>
        </div>
        <nav>
          <a routerLink="/history" class="nav-link">Historique</a>
        </nav>
      </header>

      <!-- Hero -->
      <main class="hero">
        <div class="hero-content">
          <p class="hero-sub">Pipeline de détection BLS → CNN → Validation physique</p>
          <h1 class="hero-title">Chassez des exoplanètes<br>dans les données Kepler</h1>

          <!-- Formulaire -->
          <form [formGroup]="form" (ngSubmit)="onScan()" class="scan-form">

            <div class="field-group">
              <label class="field-label">Étoile cible</label>
              <input
                formControlName="starName"
                class="field-input"
                placeholder="ex: Kepler-7, Kepler-90, KIC 11657614"
                autocomplete="off"
              />
              <span class="field-hint">Nom Kepler ou identifiant KIC</span>
            </div>

            <div class="field-group">
              <label class="field-label">Quarters Kepler</label>
              <input
                formControlName="quarters"
                class="field-input"
                placeholder="ex: 3,4,5,6"
              />
              <span class="field-hint">Quarters à analyser (1–17, séparés par des virgules)</span>
            </div>

            <button
              type="submit"
              class="btn-scan"
              [disabled]="form.invalid || scanning"
            >
              <span *ngIf="!scanning">▶ Lancer le scan</span>
              <span *ngIf="scanning">Analyse en cours...</span>
            </button>

          </form>

          <!-- Progress -->
          <div class="progress-section" *ngIf="scanning || job">
            <div class="progress-bar-track">
              <div
                class="progress-bar-fill"
                [style.width.%]="(job?.progress ?? 0) * 100"
                [class.done]="job?.status === 'done'"
              ></div>
            </div>
            <p class="progress-label">
              {{ job ? stepLabel(job.step) : 'Initialisation...' }}
              <span class="progress-pct">{{ ((job?.progress ?? 0) * 100).toFixed(0) }}%</span>
            </p>
          </div>

          <!-- Erreur -->
          <div class="error-box" *ngIf="error">
            <span class="error-icon">⚠</span> {{ error }}
          </div>

        </div>

        <!-- Étoiles déco -->
        <div class="stars-bg" aria-hidden="true">
          <span *ngFor="let s of stars" class="star"
            [style.left.%]="s.x"
            [style.top.%]="s.y"
            [style.opacity]="s.o"
            [style.width.px]="s.size"
            [style.height.px]="s.size">
          </span>
        </div>
      </main>

    </div>
  `,
  styles: [`
    :host { display: block; }

    .search-page {
      min-height: 100vh;
      background: #0D1117;
      color: #E6EDF3;
      font-family: 'Inter', system-ui, sans-serif;
      position: relative;
      overflow: hidden;
    }

    /* Header */
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 24px 48px;
      border-bottom: 1px solid #21262D;
      position: relative;
      z-index: 10;
    }
    .logo { display: flex; align-items: center; gap: 10px; }
    .logo-icon { font-size: 20px; color: #58A6FF; }
    .logo-text { font-size: 16px; font-weight: 600; letter-spacing: 0.5px; }
    .nav-link {
      color: #8B949E;
      text-decoration: none;
      font-size: 14px;
      transition: color 0.2s;
      &:hover { color: #E6EDF3; }
    }

    /* Hero */
    .hero {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: calc(100vh - 73px);
      position: relative;
    }
    .hero-content {
      width: 100%;
      max-width: 560px;
      padding: 48px 24px;
      position: relative;
      z-index: 10;
    }
    .hero-sub {
      font-size: 12px;
      color: #58A6FF;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      margin-bottom: 16px;
    }
    .hero-title {
      font-size: 40px;
      font-weight: 700;
      line-height: 1.15;
      margin-bottom: 48px;
      color: #E6EDF3;
    }

    /* Form */
    .scan-form { display: flex; flex-direction: column; gap: 24px; }
    .field-group { display: flex; flex-direction: column; gap: 6px; }
    .field-label { font-size: 13px; color: #8B949E; font-weight: 500; }
    .field-input {
      background: #161B22;
      border: 1px solid #30363D;
      border-radius: 8px;
      padding: 12px 16px;
      color: #E6EDF3;
      font-size: 15px;
      font-family: 'JetBrains Mono', monospace, system-ui;
      outline: none;
      transition: border-color 0.2s;
      &:focus { border-color: #58A6FF; }
      &::placeholder { color: #484F58; }
    }
    .field-hint { font-size: 12px; color: #484F58; }

    .btn-scan {
      margin-top: 8px;
      padding: 14px 24px;
      background: #238636;
      border: 1px solid #2EA043;
      border-radius: 8px;
      color: #fff;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
      &:hover:not(:disabled) { background: #2EA043; transform: translateY(-1px); }
      &:disabled { opacity: 0.5; cursor: not-allowed; }
    }

    /* Progress */
    .progress-section { margin-top: 32px; }
    .progress-bar-track {
      height: 4px;
      background: #21262D;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 10px;
    }
    .progress-bar-fill {
      height: 100%;
      background: #58A6FF;
      border-radius: 4px;
      transition: width 0.5s ease;
      &.done { background: #3FB950; }
    }
    .progress-label {
      font-size: 13px;
      color: #8B949E;
      display: flex;
      justify-content: space-between;
    }
    .progress-pct { color: #58A6FF; font-family: monospace; }

    /* Error */
    .error-box {
      margin-top: 24px;
      padding: 12px 16px;
      background: #1C1010;
      border: 1px solid #F85149;
      border-radius: 8px;
      color: #F85149;
      font-size: 13px;
    }
    .error-icon { margin-right: 8px; }

    /* Stars background */
    .stars-bg { position: absolute; inset: 0; pointer-events: none; }
    .star {
      position: absolute;
      background: #fff;
      border-radius: 50%;
    }
  `]
})
export class SearchComponent implements OnDestroy {

  form = new FormGroup({
    starName: new FormControl('', [Validators.required, Validators.minLength(3)]),
    quarters: new FormControl('3,4,5,6', [Validators.required]),
  });

  scanning = false;
  job: ScanJob | null = null;
  error: string | null = null;

  stars = Array.from({ length: 80 }, () => ({
    x   : Math.random() * 100,
    y   : Math.random() * 100,
    o   : Math.random() * 0.5 + 0.1,
    size: Math.random() * 2 + 1,
  }));

  private sub?: Subscription;

  constructor(
    private svc: ExoplanetService,
    private router: Router,
  ) {}

  onScan(): void {
    if (this.form.invalid || this.scanning) return;

    const starName = this.form.value.starName!.trim();
    const quarters = this.form.value.quarters!
      .split(',')
      .map(q => parseInt(q.trim(), 10))
      .filter(q => !isNaN(q) && q >= 1 && q <= 17);

    if (!quarters.length) {
      this.error = 'Quarters invalides. Exemple : 3,4,5,6';
      return;
    }

    this.scanning = true;
    this.error    = null;
    this.job      = null;

    this.sub = this.svc.scan(starName, quarters).subscribe({
      next: (job) => {
        this.job = job;
        if (job.status === 'done') {
          this.scanning = false;
          this.router.navigate(['/results'], {
            state: { job, starName, quarters }
          });
        }
      },
      error: (err: Error) => {
        this.scanning = false;
        this.error    = err.message;
      },
    });
  }

  stepLabel(step: string): string {
    return STEP_LABELS[step as keyof typeof STEP_LABELS] ?? step;
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }
}