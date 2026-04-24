import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class NasaService {
  private apiUrl = 'http://localhost:5000/api/scan';

  constructor(private http: HttpClient) {}

  scanStar(starName: string): Observable<any> {
    // On envoie le nom de l'étoile au format JSON
    return this.http.post(this.apiUrl, { star_name: starName });
  }
 
}

  
