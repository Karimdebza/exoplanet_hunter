// src/app/core/services/exoplanet.service.ts

import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import {
  Observable, interval, switchMap, takeWhile,
  catchError, throwError, tap, startWith
} from 'rxjs';
import { ScanJob, HistoryEntry, HistoryDetail } from '../models/scan.model';
import { environment } from '../../../environments/environment';



@Injectable({ providedIn: 'root' })
export class ExoplanetService {

  // En dev : localhost:8000 / En prod : ton URL Render
  private readonly API = environment.apiUrl;

  constructor(private http: HttpClient) {}

  /**
   * Lance un scan et poll le statut toutes les 2 secondes.
   * Émet chaque mise à jour du job jusqu'à status=done ou error.
   */
  scan(starName: string, quarters: number[]): Observable<ScanJob> {
    return this.http
      .post<{ job_id: string }>(`${this.API}/scan`, {
        star_name: starName,
        quarters,
      })
      .pipe(
        switchMap(({ job_id }) => this.pollJob(job_id)),
        catchError(this.handleError)
      );
  }

  /**
   * Poll le statut d'un job toutes les 2 secondes.
   * S'arrête quand status = 'done' ou 'error'.
   */
  pollJob(jobId: string): Observable<ScanJob> {
    return interval(2000).pipe(
      startWith(0),
      switchMap(() =>
        this.http.get<ScanJob>(`${this.API}/scan/${jobId}`)
      ),
      takeWhile(
        (job) => job.status === 'running',
        true  // émet aussi le dernier état (done/error)
      ),
      catchError(this.handleError)
    );
  }

  /**
   * Récupère l'historique des scans.
   */
  getHistory(): Observable<HistoryEntry[]> {
    return this.http
      .get<HistoryEntry[]>(`${this.API}/history`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Récupère le détail complet d'une entrée d'historique (avec plots).
   */
  getHistoryDetail(id: string): Observable<HistoryDetail> {
    return this.http
      .get<HistoryDetail>(`${this.API}/history/${id}`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Vide l'historique.
   */
  clearHistory(): Observable<void> {
    return this.http
      .delete<void>(`${this.API}/history`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Healthcheck — vérifie que le backend est up.
   */
  checkHealth(): Observable<{ status: string }> {
    return this.http
      .get<{ status: string }>(`${this.API}/health`)
      .pipe(catchError(this.handleError));
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    const message =
      error.status === 0
        ? 'Backend inaccessible — vérifie que uvicorn tourne sur le port 8000'
        : error.error?.detail ?? error.message;
    return throwError(() => new Error(message));
  }
}