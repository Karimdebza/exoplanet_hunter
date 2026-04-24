import { Component, ViewChild, ElementRef } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { NasaService } from './services/nasa.service';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

@Component({
  selector: 'app-root',
  imports: [FormsModule,CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
starName: string = 'Kepler-4';
  results: any = null;
  loading: boolean = false;
  errorMessage: string = '';

  constructor(private nasaService: NasaService) {}
@ViewChild('mapCanvas') canvas!: ElementRef;
bestDetections: any[] = [];
chart: any;

onScan() {
    this.loading = true;
this.nasaService.scanStar(this.starName).subscribe({
  next: (data: any) => {
    this.results = data.result;

    // On trie TOUTES les détections par probabilité décroissante
    // On veut voir les 0.97 en haut de la pile !
    this.bestDetections = [...this.results.detections]
      .sort((a, b) => b.proba - a.proba) // Tri du plus haut au plus bas
      .filter((d: any, index, self) => 
        // Optionnel : on évite les doublons de temps trop proches
        index === self.findIndex((t) => Math.abs(t.time - d.time) < 0.1)
      )
      .slice(0, 10); // On prend les 10 meilleures
      console.log(this.bestDetections, "best detect")

      console.log(data)

    this.loading = false;
    setTimeout(() => this.createChart(), 100);
  }
});
  }

  createChart() {
    if (!this.canvas) {
      console.error("Canvas introuvable !");
      return;
    }

    if (this.chart) this.chart.destroy();

    const ctx = this.canvas.nativeElement.getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.results.time,
// Dans la fonction createChart()
      datasets: [{
        label: 'Luminosité (Flux)',
        data: this.results.flux,
        borderColor: '#00ffcc',
        borderWidth: 0.5, // Ligne plus fine pour mieux voir les détails
        pointRadius: 0,
        tension: 0.1 // Adoucit un peu la ligne
      }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { display: false }, // Trop de points pour afficher les labels X proprement
          y: { ticks: { color: '#fff' } }
        }
      }
    });
  }
}
