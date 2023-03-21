import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

const baseUrl = 'http://127.0.0.1:8000/api/user';
const solutionUrl = 'http://127.0.0.1:8000/api/solution';
const analyzeUrl = 'http://127.0.0.1:8000/api/analyze';
const compareUrl = 'http://127.0.0.1:8000/api/compare';
const dashboardUrl = 'http://127.0.0.1:8000/api/dashboard';
const authUrl = 'http://127.0.0.1:8000/api/auth';
const userpageUrl = 'http://127.0.0.1:8000/api/userpage';
const setuserUrl = 'http://127.0.0.1:8000/api/setuser';
const scenarioUrl = 'http://127.0.0.1:8000/api/scenario';
const solutionDetailUrl = 'http://127.0.0.1:8000/api/solution_detail';

@Injectable({
  providedIn: 'root'
})
export class trustcalcService {

  constructor(private http: HttpClient) { }

  login(data): Observable<any> {
    return this.http.get(authUrl, { params: data });
  }

  getAll(): Observable<any> {
    return this.http.get(baseUrl);
  }

  getUser(email): Observable<any> {
    return this.http.get(`${setuserUrl}/${email}`);
  }

  updateUser(data): Observable<any> {
    return this.http.post(setuserUrl, data);
  }

  get(email): Observable<any> {
    return this.http.get(`${baseUrl}/${email}`);
  }

  dashboard(email): Observable<any> {
    return this.http.get(`${dashboardUrl}/${email}`);
  }

  create(data): Observable<any> {
    return this.http.post(scenarioUrl, data);
  }

  uploadsolution(data): Observable<any> {
    return this.http.post(solutionUrl, data);
  }

  register(data): Observable<any> {
    return this.http.post(authUrl, data);
  }

  getsolution(email): Observable<any> {
    return this.http.get(`${solutionUrl}/${email}`);
  }

  analyzesolution(data): Observable<any> {
    return this.http.post(analyzeUrl, data);
  }

  userpageUrl(email): Observable<any> {
    return this.http.get(`${userpageUrl}/${email}`);
  }

  userdetails(data): Observable<any> {
    return this.http.post(userpageUrl, data);
  }

  comparesolution(data): Observable<any> {
    return this.http.post(compareUrl, data);
  }

  update(email, data): Observable<any> {
    return this.http.put(`${baseUrl}/${email}`, data);
  }

  getScenario(scenarioId): Observable<any> {
    return this.http.get(`${scenarioUrl}/${scenarioId}`);
  }

  updateScenario(data): Observable<any> {
    return this.http.put(scenarioUrl, data);
  }

  getSolution(id): Observable<any> {
    return this.http.get(`${solutionDetailUrl}/${id}`);
  }

  updateSolution(data): Observable<any> {
    return this.http.put(solutionDetailUrl, data);
  }
}
