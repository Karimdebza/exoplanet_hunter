// src/app/pages/history/history.component.ts

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Router } from '@angular/router';
import { ExoplanetService } from '../../core/services/exoplanet.service';
import { HistoryEntry, CLASSIFICATION_CONFIG } from '../../core/models/scan.model';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './history.html',
  styleUrls: ['./history.scss']
})
export class HistoryComponent implements OnInit {
  entries: HistoryEntry[] = [];
  loadingId: string | null = null;

  constructor(private svc: ExoplanetService, private router: Router) {}

  ngOnInit(): void {
    this.svc.getHistory().subscribe(h => this.entries = h);
  }

  viewEntry(entry: HistoryEntry): void {
    if (this.loadingId) return;
    this.loadingId = entry.id;
    this.svc.getHistoryDetail(entry.id).subscribe({
      next: (detail) => {
        this.loadingId = null;
        this.router.navigate(['/results'], {
          state: {
            job: { results: detail.results, star_name: detail.star_name, quarters: detail.quarters },
            starName: detail.star_name,
            quarters: detail.quarters,
          }
        });
      },
      error: () => { this.loadingId = null; }
    });
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
