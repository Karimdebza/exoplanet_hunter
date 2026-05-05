// src/app/pages/results/results.component.ts

import { Component, OnInit } from '@angular/core';
import { Router, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import {
  ScanJob, Candidate, Classification,
  CLASSIFICATION_CONFIG, PhysicalTest
} from '../../core/models/scan.model';

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule, RouterModule],
  template: `
    <div class="results-page">

      <!-- Header -->
      <header class="header">
        <div class="logo">
          <span class="logo-icon">◎</span>
          <span class="logo-text">ExoplanetHunter</span>
        </div>
        <nav class="nav">
          <a routerLink="/" class="nav-link">← Nouvelle recherche</a>
          <a routerLink="/history" class="nav-link">Historique</a>
        </nav>
      </header>

      <!-- Meta -->
      <div class="meta-bar">
        <div class="meta-info">
          <span class="meta-star">{{ starName }}</span>
          <span class="meta-sep">·</span>
          <span class="meta-quarters">Quarters {{ quarters.join(', ') }}</span>
          <span class="meta-sep">·</span>
          <span class="meta-count">{{ candidates.length }} candidats analysés</span>
        </div>
        <div class="meta-stats">
          <span class="stat planet" *ngIf="planetCount > 0">
            🪐 {{ planetCount }} candidat{{ planetCount > 1 ? 's' : '' }} planète
          </span>
          <span class="stat none" *ngIf="planetCount === 0">
            Aucun candidat planète détecté
          </span>
        </div>
      </div>

      <!-- Empty state -->
      <div class="empty" *ngIf="!candidates.length">
        <p class="empty-icon">🔭</p>
        <p class="empty-title">Aucun signal détecté</p>
        <p class="empty-sub">Essaie avec plus de quarters ou une autre étoile.</p>
        <a routerLink="/" class="btn-back">← Nouvelle recherche</a>
      </div>

      <!-- Candidats -->
      <div class="candidates" *ngIf="candidates.length">
        <div
          class="candidate-card"
          *ngFor="let c of candidates"
          [class.planet]="c.classification === 'PLANET_CANDIDATE'"
        >
          <!-- Card header -->
          <div class="card-header">
            <div class="card-rank">#{{ c.rank }}</div>
            <div class="card-classification"
              [style.color]="classConfig(c.classification).color"
              [style.border-color]="classConfig(c.classification).color + '40'">
              {{ classConfig(c.classification).icon }}
              {{ classConfig(c.classification).label }}
            </div>
            <div class="card-confidence">
              <span class="conf-label">Confiance</span>
              <span class="conf-value">{{ (c.confidence * 100).toFixed(0) }}%</span>
              <div class="conf-bar">
                <div class="conf-fill"
                  [style.width.%]="c.confidence * 100"
                  [style.background]="classConfig(c.classification).color">
                </div>
              </div>
            </div>
          </div>

          <!-- Métriques -->
          <div class="metrics">
            <div class="metric">
              <span class="metric-label">Période</span>
              <span class="metric-value">{{ c.period.toFixed(4) }}<span class="metric-unit">j</span></span>
            </div>
            <div class="metric">
              <span class="metric-label">Profondeur</span>
              <span class="metric-value">{{ c.depth_ppm.toFixed(0) }}<span class="metric-unit">ppm</span></span>
            </div>
            <div class="metric">
              <span class="metric-label">Durée</span>
              <span class="metric-value">{{ c.duration_hours.toFixed(1) }}<span class="metric-unit">h</span></span>
            </div>
            <div class="metric">
              <span class="metric-label">SDE</span>
              <span class="metric-value" [class.sde-high]="c.sde >= 7">{{ c.sde.toFixed(1) }}</span>
            </div>
            <div class="metric">
              <span class="metric-label">CNN</span>
              <span class="metric-value" [class.cnn-pass]="c.cnn_passed">{{ c.cnn_score.toFixed(3) }}</span>
            </div>
          </div>

          <!-- Contenu principal : graphique + tests -->
          <div class="card-body">

            <!-- Graphique phase folding -->
            <div class="plot-container" *ngIf="c.plot_base64">
              <img
                [src]="'data:image/png;base64,' + c.plot_base64"
                alt="Phase folding candidat {{ c.rank }}"
                class="plot-img"
              />
            </div>

            <!-- Tests physiques -->
            <div class="tests">
              <div class="tests-title">Validation physique</div>
              <div class="test-row" *ngFor="let t of c.tests">
                <span class="test-icon">{{ t.passed ? '✓' : '✗' }}</span>
                <span class="test-name" [class.pass]="t.passed" [class.fail]="!t.passed">
                  {{ testLabel(t.name) }}
                </span>
                <span class="test-score">{{ t.score.toFixed(3) }}</span>
                <span class="test-verdict" *ngIf="t.details['verdict']">
                  {{ t.details['verdict'] }}
                </span>
              </div>
            </div>

          </div>
        </div>
      </div>

    </div>
  `,
  styles: [`
    :host { display: block; }

    .results-page {
      min-height: 100vh;
      background: #0D1117;
      color: #E6EDF3;
      font-family: 'Inter', system-ui, sans-serif;
    }

    /* Header */
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 48px;
      border-bottom: 1px solid #21262D;
      position: sticky;
      top: 0;
      background: #0D1117;
      z-index: 100;
    }
    .logo { display: flex; align-items: center; gap: 10px; }
    .logo-icon { font-size: 20px; color: #58A6FF; }
    .logo-text { font-size: 16px; font-weight: 600; }
    .nav { display: flex; gap: 24px; }
    .nav-link {
      color: #8B949E; text-decoration: none; font-size: 14px;
      transition: color 0.2s;
      &:hover { color: #E6EDF3; }
    }

    /* Meta bar */
    .meta-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 48px;
      background: #161B22;
      border-bottom: 1px solid #21262D;
    }
    .meta-info { display: flex; align-items: center; gap: 12px; font-size: 14px; }
    .meta-star { font-weight: 600; font-family: monospace; font-size: 16px; color: #58A6FF; }
    .meta-sep { color: #30363D; }
    .meta-quarters, .meta-count { color: #8B949E; }
    .stat { font-size: 13px; font-weight: 500; }
    .stat.planet { color: #3FB950; }
    .stat.none { color: #484F58; }

    /* Empty */
    .empty {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; padding: 96px 24px; text-align: center;
    }
    .empty-icon { font-size: 48px; margin-bottom: 16px; }
    .empty-title { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
    .empty-sub { color: #8B949E; font-size: 14px; margin-bottom: 32px; }
    .btn-back {
      padding: 10px 20px; background: #21262D; border: 1px solid #30363D;
      border-radius: 8px; color: #E6EDF3; text-decoration: none;
      font-size: 14px; transition: background 0.2s;
      &:hover { background: #30363D; }
    }

    /* Candidates */
    .candidates {
      display: flex; flex-direction: column; gap: 24px;
      padding: 32px 48px; max-width: 1200px; margin: 0 auto;
    }

    .candidate-card {
      background: #161B22;
      border: 1px solid #21262D;
      border-radius: 12px;
      overflow: hidden;
      transition: border-color 0.2s;
      &:hover { border-color: #30363D; }
      &.planet { border-color: #238636; }
    }

    /* Card header */
    .card-header {
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 16px 24px;
      border-bottom: 1px solid #21262D;
      background: #0D1117;
    }
    .card-rank {
      font-family: monospace; font-size: 13px;
      color: #484F58; min-width: 24px;
    }
    .card-classification {
      font-size: 13px; font-weight: 600;
      padding: 4px 10px;
      border: 1px solid;
      border-radius: 20px;
    }
    .card-confidence {
      margin-left: auto; display: flex; align-items: center; gap: 10px;
    }
    .conf-label { font-size: 12px; color: #484F58; }
    .conf-value { font-family: monospace; font-size: 14px; font-weight: 600; }
    .conf-bar {
      width: 80px; height: 4px; background: #21262D;
      border-radius: 4px; overflow: hidden;
    }
    .conf-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }

    /* Metrics */
    .metrics {
      display: flex; gap: 0;
      border-bottom: 1px solid #21262D;
    }
    .metric {
      flex: 1; padding: 16px 24px;
      border-right: 1px solid #21262D;
      &:last-child { border-right: none; }
    }
    .metric-label { display: block; font-size: 11px; color: #484F58; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 600; }
    .metric-unit { font-size: 11px; color: #8B949E; margin-left: 2px; }
    .sde-high { color: #3FB950; }
    .cnn-pass { color: #58A6FF; }

    /* Card body */
    .card-body {
      display: grid;
      grid-template-columns: 1fr 300px;
      gap: 0;
    }

    /* Plot */
    .plot-container {
      padding: 16px;
      border-right: 1px solid #21262D;
    }
    .plot-img {
      width: 100%; border-radius: 8px;
      display: block;
    }

    /* Tests */
    .tests { padding: 20px 24px; }
    .tests-title {
      font-size: 11px; color: #484F58;
      text-transform: uppercase; letter-spacing: 0.5px;
      margin-bottom: 16px;
    }
    .test-row {
      display: flex; align-items: center; gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid #21262D;
      font-size: 13px;
      &:last-child { border-bottom: none; }
    }
    .test-icon { font-size: 12px; min-width: 16px; }
    .test-name {
      flex: 1; font-family: monospace;
      &.pass { color: #3FB950; }
      &.fail { color: #F85149; }
    }
    .test-score { font-family: monospace; color: #484F58; font-size: 12px; }
    .test-verdict {
      font-size: 11px; color: #8B949E;
      background: #0D1117; padding: 2px 6px; border-radius: 4px;
    }
  `]
})
export class ResultsComponent implements OnInit {

  starName  = '';
  quarters: number[] = [];
  candidates: Candidate[] = [];

  get planetCount(): number {
    return this.candidates.filter(c => c.classification === 'PLANET_CANDIDATE').length;
  }

  constructor(private router: Router) {}

  ngOnInit(): void {
    const state = this.router.getCurrentNavigation()?.extras.state
      ?? history.state;

    if (!state?.job) {
      this.router.navigate(['/']);
      return;
    }

    const job: ScanJob = state.job;
    this.starName  = state.starName ?? job.star_name;
    this.quarters  = state.quarters ?? job.quarters;
    this.candidates = job.results ?? [];

    // Trier : PLANET_CANDIDATE en premier, puis par confidence décroissante
    this.candidates.sort((a, b) => {
      if (a.classification === 'PLANET_CANDIDATE' && b.classification !== 'PLANET_CANDIDATE') return -1;
      if (b.classification === 'PLANET_CANDIDATE' && a.classification !== 'PLANET_CANDIDATE') return 1;
      return b.confidence - a.confidence;
    });
  }

  classConfig(c: Classification) {
    return CLASSIFICATION_CONFIG[c];
  }

  testLabel(name: string): string {
    const labels: Record<string, string> = {
      odd_even_depth    : 'Odd/Even depth',
      secondary_eclipse : 'Secondary eclipse',
      depth_consistency : 'Depth consistency',
      duration_check    : 'Duration (Kepler)',
    };
    return labels[name] ?? name;
  }
}