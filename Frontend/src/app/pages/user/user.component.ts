import { Component, OnInit } from "@angular/core";
import { trustcalcService } from "src/app/services/trustcalc.service";
import { AuthService } from '@auth0/auth0-angular';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: "app-user",
  templateUrl: "user.component.html"
})
export class UserComponent implements OnInit {
  trustcalc = {
    fullname: '',
    email: '',
    password: '',
  };

  constructor(public auth: AuthService, private trustcalcService: trustcalcService, private toastr: ToastrService) { }

  ngOnInit() {
    this.trustcalc.email = localStorage.getItem('email');
    this.trustcalc.password = localStorage.getItem('password');
  }

  saveUser() {
    this.trustcalcService.updateUser({
      email: this.trustcalc.email,
      password: this.trustcalc.password,
    }).subscribe(data => {
      this.toastr.info("password has successfully changed");
    });
  }
}
