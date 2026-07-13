// src/app/core/models/scan.model.ts

export type Classification =
  | 'PLANET_CANDIDATE'
  | 'ECLIPSING_BINARY'
  | 'STELLAR_VARIABILITY'
  | 'NOISE'
  | 'INCONCLUSIVE';

export interface PhysicalTest {
  name: string;
  passed: boolean;
  score: number;
  details: Record<string, any>;
}

export interface Candidate {
  rank: number;
  period: number;
  depth_ppm: number;
  duration_hours: number;
  sde: number;
  cnn_score: number;
  cnn_passed: boolean;
  classification: Classification;
  confidence: number;
  tests: PhysicalTest[];
  plot_base64: string;
  r_planet_earth?: number;
  r_star_solar?: number;
}

export type ScanStep =
  | 'initializing'
  | 'downloading'
  | 'bls'
  | 'cnn'
  | 'validating'
  | 'done';

export interface ScanJob {
  status: 'running' | 'done' | 'error';
  step: ScanStep;
  progress: number;
  star_name: string;
  quarters: number[];
  created_at: string;
  results: Candidate[] | null;
  error: string | null;
  message?: string;
  n_points?: number;
  duration_days?: number;
  n_candidates?: number;
}

export interface HistoryEntry {
  id: string;
  star_name: string;
  quarters: number[];
  date: string;
  n_candidates: number;
  n_planet_candidates: number;
  best_candidate: {
    period: number;
    classification: Classification;
    confidence: number;
  } | null;
}

export interface HistoryDetail {
  star_name: string;
  quarters: number[];
  results: Candidate[];
}

export const STEP_LABELS: Record<ScanStep, string> = {
  initializing: 'Initialisation...',
  downloading : 'Téléchargement des données Kepler...',
  bls         : 'Détection BLS des signaux périodiques...',
  cnn         : 'Chargement du réseau de neurones...',
  validating  : 'Validation physique des candidats...',
  done        : 'Analyse terminée',
};

export const CLASSIFICATION_CONFIG: Record<Classification, { label: string; color: string; icon: string }> = {
  PLANET_CANDIDATE  : { label: 'Candidat planète', color: '#4CAF50', icon: '🪐' },
  ECLIPSING_BINARY  : { label: 'Binaire à éclipses', color: '#F44336', icon: '⭐' },
  STELLAR_VARIABILITY: { label: 'Variabilité stellaire', color: '#FF9800', icon: '🌟' },
  NOISE             : { label: 'Bruit', color: '#9E9E9E', icon: '〰️' },
  INCONCLUSIVE      : { label: 'Inconclusive', color: '#2196F3', icon: '❓' },
};