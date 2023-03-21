import { NgModule } from "@angular/core";
import { HttpClientModule } from "@angular/common/http";
import { RouterModule } from "@angular/router";
import { CommonModule } from "@angular/common";
import { FormsModule } from "@angular/forms";

import { AdminLayoutRoutes } from "./admin-layout.routing";
import { DashboardComponent } from "../../pages/dashboard/dashboard.component";
import { IconsComponent } from "../../pages/solution/solution.component";
import { ScenarioComponent } from "../../pages/scenario/scenario.component";
import { analyzeComponent } from "../../pages/analyze/analyze.component";
import { UserComponent } from "../../pages/user/user.component";
import { compareComponent } from "../../pages/compare/compare.component";
import { TypographyComponent } from "../../pages/usersetting/usersetting.component";
// import { RtlComponent } from "../../pages/rtl/rtl.component";

import { NgbModule } from "@ng-bootstrap/ng-bootstrap";

@NgModule({
  imports: [
    CommonModule,
    RouterModule.forChild(AdminLayoutRoutes),
    FormsModule,
    HttpClientModule,
    NgbModule,
  ],
  declarations: [
    DashboardComponent,
    UserComponent,
    compareComponent,
    IconsComponent,
    TypographyComponent,
    analyzeComponent,
    ScenarioComponent,
    // RtlComponent
  ]
})
export class AdminLayoutModule { }
