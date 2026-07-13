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
  templateUrl: './results.html',
  styleUrls: ['./results.scss']
})
export class ResultsComponent implements OnInit {

  starName  = '';
  quarters: number[] = [];
  candidates: Candidate[] = [];

  get planetCount(): number {
    return this.candidates.filter(c => c.classification === 'PLANET_CANDIDATE').length;
  }

  get bestPlanet() {
    return this.candidates.find(c => c.classification === 'PLANET_CANDIDATE') ?? null;
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