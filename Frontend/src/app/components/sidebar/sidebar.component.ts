import { Component, OnInit } from "@angular/core";
import { Router } from "@angular/router";
import { AuthGuard } from "@auth0/auth0-angular";
import { AuthService } from "@auth0/auth0-angular";
declare interface RouteInfo {
  path: string;
  title: string;
  rtlTitle: string;
  icon: string;
  class: string;
}
export const ROUTES: RouteInfo[] = [
  {
    path: "/dashboard",
    title: "Dashboard",
    rtlTitle: "لوحة القيادة",
    icon: "icon-chart-pie-36",
    class: ""
  },
  {
    path: "/icons",
    title: "Upload a Solution",
    rtlTitle: "الرموز",
    icon: "icon-atom",
    class: ""
  },
  {
    path: "/maps",
    title: "Create a Scenario",
    rtlTitle: "خرائط",
    icon: "icon-pin",
    class: ""
  },
  // {
  //   path: "/user",
  //   title: "Profile",
  //   rtlTitle: "ملف تعريفي للمستخدم",
  //   icon: "icon-single-02",
  //   class: ""
  // },
  {
    path: "/analyze",
    title: "Analyze a Solution",
    rtlTitle: "إخطارات",
    icon: "icon-bell-55",
    class: ""
  },
  {
    path: "/compare",
    title: "Compare two solutions",
    rtlTitle: "قائمة الجدول",
    icon: "icon-puzzle-10",
    class: ""
  },
  // {
  //   path: "/typography",
  //   title: "User's Page",
  //   rtlTitle: "طباعة",
  //   icon: "icon-align-center",
  //   class: ""
  // },
  // {
  //   path: "/rtl",
  //   title: "RTL Support",
  //   rtlTitle: "ار تي ال",
  //   icon: "icon-world",
  //   class: ""
  // }
];

@Component({
  selector: "app-sidebar",
  templateUrl: "./sidebar.component.html",
  styleUrls: ["./sidebar.component.css"]
})
export class SidebarComponent implements OnInit {
  menuItems: any[];

  constructor(public auth: AuthService, public route: Router) { }

  ngOnInit() {
    this.menuItems = ROUTES.filter(menuItem => menuItem);
  }

  isMobileMenu() {
    if (window.innerWidth > 991) {
      return false;
    }
    return true;
  }

  handleNavigate(event) {
    const email = localStorage.getItem('email');
    if (!email || email.length <= 0) {
      this.route.navigate(['/login']);
    }
  }
}
