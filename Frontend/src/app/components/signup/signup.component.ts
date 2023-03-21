import { Component, OnInit } from '@angular/core';
import { trustcalcService } from "src/app/services/trustcalc.service";
import { Router } from '@angular/router';
import { AuthService } from "@auth0/auth0-angular";

@Component({
  selector: 'app-signup',
  templateUrl: './signup.component.html',
  styleUrls: ['./signup.component.scss']
})
export class SignupComponent implements OnInit {
  trustcalc = {
    fullname: '',
    email: '',
    password: '',
  };

  constructor(private trustcalcservice: trustcalcService, private router: Router, public auth: AuthService) { }

  ngOnInit(): void {
  }

  signUp() {
    const formData = new FormData();
    formData.append('email', this.trustcalc.email);
    formData.append('password', this.trustcalc.password);

    this.trustcalcservice.register(formData)
      .subscribe(
        response => {
          this.router.navigate(['/login']);
        },
        error => {
        }
      );
  }

}
