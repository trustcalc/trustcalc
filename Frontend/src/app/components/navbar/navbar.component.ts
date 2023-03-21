import { Component, Inject, OnInit, ElementRef, OnDestroy } from "@angular/core";
import { ROUTES } from "../sidebar/sidebar.component";
import { Location } from "@angular/common";
import { Router } from "@angular/router";
import { NgbModal, ModalDismissReasons } from '@ng-bootstrap/ng-bootstrap';
import { DOCUMENT } from '@angular/common';
import { AuthService } from '@auth0/auth0-angular';

@Component({
  selector: "app-navbar",
  templateUrl: "./navbar.component.html",
  styleUrls: ["./navbar.component.css"]
})
export class NavbarComponent implements OnInit, OnDestroy {
  private listTitles: any[];
  location: Location;
  mobile_menu_visible: any = 0;
  private toggleButton: any;
  private sidebarVisible: boolean;

  public isCollapsed = true;
  public isAdmin: Boolean = false;

  closeResult: string;

  constructor(
    location: Location,
    private element: ElementRef,
    private router: Router,
    private modalService: NgbModal,
    public auth: AuthService,
    @Inject(DOCUMENT) private doc: Document
  ) {
    this.location = location;
    this.sidebarVisible = false;
  }

  updateColor = () => {
    var navbar = document.getElementsByClassName('navbar')[0];
    if (window.innerWidth < 993 && !this.isCollapsed) {
      navbar.classList.add('bg-white');
      navbar.classList.remove('navbar-transparent');
    } else {
      navbar.classList.remove('bg-white');
      navbar.classList.add('navbar-transparent');
    }
  };

  ngOnInit() {
    window.addEventListener("resize", this.updateColor);
    this.listTitles = ROUTES.filter(listTitle => listTitle);
    const navbar: HTMLElement = this.element.nativeElement;
    this.toggleButton = navbar.getElementsByClassName("navbar-toggler")[0];
    this.router.events.subscribe(event => {
      this.sidebarClose();
      var $layer: any = document.getElementsByClassName("close-layer")[0];
      if ($layer) {
        $layer.remove();
        this.mobile_menu_visible = 0;
      }
    });

    this.isAdmin = localStorage.getItem('is_admin') == 'true';
  }


  logout(): void {
    localStorage.setItem('email', '');
    localStorage.setItem('password', '');

    this.router.navigate(['/login']);
  }

  sidebarOpen() {
    const toggleButton = this.toggleButton;
    const mainPanel = <HTMLElement>(
      document.getElementsByClassName("main-panel")[0]
    );
    const html = document.getElementsByTagName("html")[0];
    if (window.innerWidth < 991) {
      mainPanel.style.position = "fixed";
    }

    setTimeout(function () {
      toggleButton.classList.add("toggled");
    }, 500);

    html.classList.add("nav-open");

    this.sidebarVisible = true;
  }

  sidebarClose() {
    const html = document.getElementsByTagName("html")[0];
    this.toggleButton.classList.remove("toggled");
    const mainPanel = <HTMLElement>(
      document.getElementsByClassName("main-panel")[0]
    );

    if (window.innerWidth < 991) {
      setTimeout(function () {
        mainPanel.style.position = "";
      }, 500);
    }
    this.sidebarVisible = false;
    html.classList.remove("nav-open");
  }

  sidebarToggle() {
    var $toggle = document.getElementsByClassName("navbar-toggler")[0];

    if (this.sidebarVisible === false) {
      this.sidebarOpen();
    } else {
      this.sidebarClose();
    }
    const html = document.getElementsByTagName("html")[0];

    if (this.mobile_menu_visible == 1) {
      html.classList.remove("nav-open");
      if ($layer) {
        $layer.remove();
      }
      setTimeout(function () {
        $toggle.classList.remove("toggled");
      }, 400);

      this.mobile_menu_visible = 0;
    } else {
      setTimeout(function () {
        $toggle.classList.add("toggled");
      }, 430);

      var $layer = document.createElement("div");
      $layer.setAttribute("class", "close-layer");

      if (html.querySelectorAll(".main-panel")) {
        document.getElementsByClassName("main-panel")[0].appendChild($layer);
      } else if (html.classList.contains("off-canvas-sidebar")) {
        document
          .getElementsByClassName("wrapper-full-page")[0]
          .appendChild($layer);
      }

      setTimeout(function () {
        $layer.classList.add("visible");
      }, 100);

      $layer.onclick = function () {
        html.classList.remove("nav-open");
        this.mobile_menu_visible = 0;
        $layer.classList.remove("visible");
        setTimeout(function () {
          $layer.remove();
          $toggle.classList.remove("toggled");
        }, 400);
      }.bind(this);

      html.classList.add("nav-open");
      this.mobile_menu_visible = 1;
    }
  }

  getTitle() {
    var titlee = this.location.prepareExternalUrl(this.location.path());
    if (titlee.charAt(0) === "#") {
      titlee = titlee.slice(1);
    }

    for (var item = 0; item < this.listTitles.length; item++) {
      if (this.listTitles[item].path === titlee) {
        return this.listTitles[item].title;
      }
    }
    return "Dashboard";
  }

  open(content) {
    this.modalService.open(content, { windowClass: 'modal-search' }).result.then((result) => {
      this.closeResult = `Closed with: ${result}`;
    }, (reason) => {
      this.closeResult = `Dismissed ${this.getDismissReason(reason)}`;
    });
  }

  private getDismissReason(reason: any): string {
    if (reason === ModalDismissReasons.ESC) {
      return 'by pressing ESC';
    } else if (reason === ModalDismissReasons.BACKDROP_CLICK) {
      return 'by clicking on a backdrop';
    } else {
      return `with: ${reason}`;
    }
  }
  ngOnDestroy() {
    window.removeEventListener("resize", this.updateColor);
  }
}
