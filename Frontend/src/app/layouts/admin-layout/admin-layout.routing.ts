import { Routes } from "@angular/router";

import { DashboardComponent } from "../../pages/dashboard/dashboard.component";
import { IconsComponent } from "../../pages/solution/solution.component";
import { ScenarioComponent } from "../../pages/scenario/scenario.component";
import { analyzeComponent } from "../../pages/analyze/analyze.component";
import { UserComponent } from "../../pages/user/user.component";
import { compareComponent } from "../../pages/compare/compare.component";
import { TypographyComponent } from "../../pages/usersetting/usersetting.component";
// import { RtlComponent } from "../../pages/rtl/rtl.component";

export const AdminLayoutRoutes: Routes = [
  { path: "dashboard", component: DashboardComponent },
  {
    path: "icons", component: IconsComponent, children: [{
      path: '**',
      component: IconsComponent,
    }]
  },
  {
    path: "maps", component: ScenarioComponent, children: [{
      path: '**',
      component: ScenarioComponent,
    }]
  },
  { path: "analyze", component: analyzeComponent },
  { path: "user", component: UserComponent },
  { path: "compare", component: compareComponent },
  { path: "typography", component: TypographyComponent },
  // { path: "rtl", component: RtlComponent }
];
