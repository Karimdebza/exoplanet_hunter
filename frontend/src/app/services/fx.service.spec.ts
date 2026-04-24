import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';

import { environment } from '../env/environment';

import { FxService } from './fx.service';

describe('FxService', () => {
  let service: FxService;
  let http: HttpTestingController;
  

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [FxService]
    });
    service = TestBed.inject(FxService);
    http = TestBed.inject(HttpTestingController);
  });
  afterEach(() => {
    http.verify();
  });


  it('should be created', () => {
    expect(service).toBeTruthy();
  });
  it('should call predict API', () => {
    const mockResponse = {
      dates: ['2024-01-01', '2024-01-02'],
      historic: [1.0, 1.1],
      pred_dates: ['2024-01-03'],
      predictions: [1.2],
      lower: [1.15],
      upper: [1.25]
    };
    service.predict('USD', 3).subscribe(response => {
      expect(response).toEqual(mockResponse);
    });
    const req = http.expectOne(`${environment.apiUrl}/predict?devise=USD&days=3`);
    expect(req.request.method).toBe('GET');
    req.flush(mockResponse);
  });
  
  it('should call backtest API', () => {
    const mockBacktest = {
      strategy: 'Test Strategy',
      total_return: 0.1,
      annualized_return: 0.05,
      max_drawdown: 0.02,
      sharpe_ratio: 1.5
    };
    service.backtest('USD').subscribe(response => {
      expect(response).toEqual(mockBacktest);
    });
    const req = http.expectOne(`${environment.apiUrl}/backtest?devise=USD`);
    expect(req.request.method).toBe('GET');
    req.flush(mockBacktest);
  });
});
