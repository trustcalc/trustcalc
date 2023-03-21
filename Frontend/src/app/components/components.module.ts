import { BrowserAnimationsModule } from "@angular/platform-browser/animations";
import { HttpClientModule } from "@angular/common/http";
import { AppRoutingModule } from "../app-routing.module";
import { ToastrModule } from 'ngx-toastr';
import { NgModule } from "@angular/core";
import { CommonModule } from "@angular/common";
import { RouterModule } from "@angular/router";
import { NgbModule } from "@ng-bootstrap/ng-bootstrap";
import { FormsModule } from "@angular/forms";
import { FooterComponent } from "./footer/footer.component";
import { NavbarComponent } from "./navbar/navbar.component";
import { SidebarComponent } from "./sidebar/sidebar.component";
import { LoginComponent } from './login/login.component';
import { SignupComponent } from './signup/signup.component';
import { AuthModule } from "@auth0/auth0-angular";

@NgModule({
  imports: [
    CommonModule, 
    RouterModule, 
    NgbModule, 
    FormsModule,
    BrowserAnimationsModule,
    FormsModule,
    HttpClientModule,
    NgbModule,
    RouterModule,
    AppRoutingModule,
    AuthModule.forRoot({
      domain: 'dev-pg885vf5puf75lpx.us.auth0.com',
      clientId: 'WtX01rYykVtUT2N9iT3XxUJT0jTvsyev',
    }),
    ToastrModule.forRoot()
  ],
  declarations: [FooterComponent, NavbarComponent, SidebarComponent, LoginComponent, SignupComponent],
  exports: [FooterComponent, NavbarComponent, SidebarComponent]
})
export class ComponentsModule {}
