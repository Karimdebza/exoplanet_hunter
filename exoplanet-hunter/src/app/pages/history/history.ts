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
  templateUrl: './history.html',
  styleUrls: ['./history.scss']
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
