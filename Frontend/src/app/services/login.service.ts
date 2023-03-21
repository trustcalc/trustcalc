import { Injectable } from '@angular/core';
import { ActivatedRouteSnapshot, CanActivate, RouterStateSnapshot, UrlTree } from '@angular/router';
import {concatMap, iif, Observable, of} from 'rxjs';
import {AuthService} from '@auth0/auth0-angular';
import {map} from 'rxjs/operators';


@Injectable({
  providedIn: 'root'
})
export class LoginService implements CanActivate {

  constructor(private authService: AuthService) { }


  canActivate(
    next: ActivatedRouteSnapshot,
    state: RouterStateSnapshot
  ): Observable<boolean> {
      return this.authService.isAuthenticated$.pipe(
          concatMap((result) =>
              iif(
                  () => result,
                  of(true),
                  this.authService.loginWithRedirect({screen_hint: 'login'}).pipe(map((_) => false))
              )
          )
      );
  }
}
