import { Component, OnInit } from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';
import { ToastrService } from 'ngx-toastr';
import { trustcalcService } from 'src/app/services/trustcalc.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {
  public email: string;
  public password: string;

  constructor(public auth: AuthService, private router: Router, public trustcalc: trustcalcService, public toast: ToastrService) { }

  ngOnInit(): void {
  }

  login(): void {
    this.trustcalc.login({
      email: this.email,
      password: this.password,
    }).subscribe((data: { email: string, password: string, is_admin: string }) => {
      localStorage.setItem('email', data.email);
      localStorage.setItem('password', data.password);
      localStorage.setItem('is_admin', data.is_admin);

      this.toast.success('Successfully Logged In', 'Log In');
      this.router.navigate(['/dashboard']);
    }, (error) => {
      this.toast.error('Email or Password is incorrect', 'Login Error!');
    })
  }

}
