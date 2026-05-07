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
  templateUrl: './search.html',
  styleUrls: ['./search.scss']
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
          console.log('STATUS:', job.status, '| PROGRESS:', job.progress);
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