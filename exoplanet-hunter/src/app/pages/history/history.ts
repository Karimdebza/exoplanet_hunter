// src/app/pages/history/history.component.ts

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ExoplanetService } from '../../core/services/exoplanet.service';
import { HistoryEntry, CLASSIFICATION_CONFIG } from '../../core/models/scan.model';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, RouterModule],
  template: `
    <div class="history-page">

      <header class="header">
        <div class="logo">
          <span class="logo-icon">◎</span>
          <span class="logo-text">ExoplanetHunter</span>
        </div>
        <nav class="nav">
          <a routerLink="/" class="nav-link">← Nouvelle recherche</a>
          <button class="btn-clear" (click)="clearHistory()" *ngIf="entries.length">
            Vider l'historique
          </button>
        </nav>
      </header>

      <div class="content">
        <h2 class="page-title">Historique des scans</h2>

        <div class="empty" *ngIf="!entries.length">
          <p class="empty-icon">🔭</p>
          <p class="empty-title">Aucun scan effectué</p>
          <a routerLink="/" class="btn-back">← Lancer un scan</a>
        </div>

        <div class="entries" *ngIf="entries.length">
          <div class="entry" *ngFor="let e of entries">
            <div class="entry-main">
              <span class="entry-star">{{ e.star_name }}</span>
              <span class="entry-quarters">Q{{ e.quarters.join(', ') }}</span>
              <span class="entry-date">{{ formatDate(e.date) }}</span>
            </div>
            <div class="entry-results">
              <span class="entry-count">{{ e.n_candidates }} candidats</span>
              <span class="entry-planet" *ngIf="e.n_planet_candidates > 0">
                🪐 {{ e.n_planet_candidates }} planète{{ e.n_planet_candidates > 1 ? 's' : '' }}
              </span>
              <div class="entry-best" *ngIf="e.best_candidate">
                <span class="best-label">Meilleur :</span>
                <span class="best-period">P={{ e.best_candidate.period.toFixed(3) }}j</span>
                <span class="best-conf">conf={{ (e.best_candidate.confidence * 100).toFixed(0) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  `,
  styles: [`
    :host { display: block; }
    .history-page {
      min-height: 100vh; background: #0D1117;
      color: #E6EDF3; font-family: 'Inter', system-ui, sans-serif;
    }
    .header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 20px 48px; border-bottom: 1px solid #21262D;
    }
    .logo { display: flex; align-items: center; gap: 10px; }
    .logo-icon { font-size: 20px; color: #58A6FF; }
    .logo-text { font-size: 16px; font-weight: 600; }
    .nav { display: flex; gap: 16px; align-items: center; }
    .nav-link {
      color: #8B949E; text-decoration: none; font-size: 14px;
      &:hover { color: #E6EDF3; }
    }
    .btn-clear {
      background: none; border: 1px solid #F85149; border-radius: 6px;
      color: #F85149; padding: 6px 12px; font-size: 13px; cursor: pointer;
      &:hover { background: #1C1010; }
    }
    .content { max-width: 900px; margin: 0 auto; padding: 48px 24px; }
    .page-title { font-size: 24px; font-weight: 600; margin-bottom: 32px; }
    .empty {
      display: flex; flex-direction: column; align-items: center;
      padding: 64px 24px; text-align: center;
    }
    .empty-icon { font-size: 40px; margin-bottom: 16px; }
    .empty-title { font-size: 18px; color: #8B949E; margin-bottom: 24px; }
    .btn-back {
      padding: 10px 20px; background: #21262D; border: 1px solid #30363D;
      border-radius: 8px; color: #E6EDF3; text-decoration: none; font-size: 14px;
    }
    .entries { display: flex; flex-direction: column; gap: 12px; }
    .entry {
      display: flex; justify-content: space-between; align-items: center;
      padding: 16px 20px; background: #161B22;
      border: 1px solid #21262D; border-radius: 10px;
      transition: border-color 0.2s;
      &:hover { border-color: #30363D; }
    }
    .entry-main { display: flex; align-items: center; gap: 16px; }
    .entry-star { font-family: monospace; font-size: 15px; font-weight: 600; color: #58A6FF; }
    .entry-quarters, .entry-date { font-size: 13px; color: #484F58; }
    .entry-results { display: flex; align-items: center; gap: 16px; font-size: 13px; }
    .entry-count { color: #8B949E; }
    .entry-planet { color: #3FB950; font-weight: 500; }
    .entry-best { display: flex; gap: 8px; color: #8B949E; }
    .best-label { color: #484F58; }
    .best-period, .best-conf { font-family: monospace; }
  `]
})
export class HistoryComponent implements OnInit {
  entries: HistoryEntry[] = [];

  constructor(private svc: ExoplanetService) {}

  ngOnInit(): void {
    this.svc.getHistory().subscribe(h => this.entries = h);
  }

  clearHistory(): void {
    this.svc.clearHistory().subscribe(() => this.entries = []);
  }

  formatDate(iso: string): string {
    return new Date(iso).toLocaleString('fr-FR', {
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// APP ROUTES — src/app/app.routes.ts
// ─────────────────────────────────────────────────────────────────────────────
/*
import { Routes } from '@angular/router';
import { SearchComponent }  from './pages/search/search.component';
import { ResultsComponent } from './pages/results/results.component';
import { HistoryComponent } from './pages/history/history.component';

export const routes: Routes = [
  { path: '',         component: SearchComponent  },
  { path: 'results',  component: ResultsComponent },
  { path: 'history',  component: HistoryComponent },
  { path: '**',       redirectTo: ''              },
];
*/


// ─────────────────────────────────────────────────────────────────────────────
// APP CONFIG — src/app/app.config.ts
// ─────────────────────────────────────────────────────────────────────────────
/*
import { ApplicationConfig } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    provideHttpClient(),
  ],
};
*/


// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL STYLES — src/styles.scss
// ─────────────────────────────────────────────────────────────────────────────
/*

*/