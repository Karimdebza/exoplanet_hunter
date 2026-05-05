import { Routes } from '@angular/router';
import { SearchComponent }  from './pages/search/search';
import { ResultsComponent } from './pages/results/results';
import { HistoryComponent } from './pages/history/history';

export const routes: Routes = [
  { path: '',         component: SearchComponent  },
  { path: 'results',  component: ResultsComponent },
  { path: 'history',  component: HistoryComponent },
  { path: '**',       redirectTo: ''              },
];