import { Component, OnInit } from "@angular/core";
import { AuthService } from "@auth0/auth0-angular";
import { trustcalcService } from "src/app/services/trustcalc.service";

@Component({
  selector: "app-usersetting",
  templateUrl: "usersetting.component.html"
})
export class TypographyComponent implements OnInit {

  ScenarioName: any;
  SolutionName: any;
  TrainnigDatafile: File;
  TestFile: File;
  FactsheetFile: File;
  ModelFile: File;
  Admintag = [];
  users: any;
  userScenario: any;
  userSolution: any;

  trustcalc = {
    userinfo: '',
  };

  constructor(public auth: AuthService, private trustcalcservice: trustcalcService) { }

  ngOnInit() {

    const email = localStorage.getItem('email');
    this.trustcalcservice.get(email).subscribe((data: any) => {
      this.ScenarioName = data.ScenarioName;
    });
    this.trustcalcservice.getsolution(email).subscribe((data: any) => {
      this.SolutionName = data.SolutionName;
    });
    this.trustcalcservice.userpageUrl(email).subscribe((data: any) => {
      this.Admintag = data.Admin;
      this.users = data.users;
    });
  }

  savetrustcalc(): void {
    const formData = new FormData();
    formData.append('Useremail', this.trustcalc.userinfo);

    this.trustcalcservice.userdetails(formData)
      .subscribe(
        response => {
          this.userScenario = response.ScenarioName;
          this.userSolution = response.SolutionName;

        },
        error => {
        }
      );
  };
}
