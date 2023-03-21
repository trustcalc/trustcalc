import { Component, OnInit } from "@angular/core";
import { AuthService, User } from '@auth0/auth0-angular';
import { trustcalcService } from "src/app/services/trustcalc.service";
import { Router } from "@angular/router";
import Papa from 'papaparse';
import { ToastrService } from "ngx-toastr";

@Component({
  selector: "app-solution",
  templateUrl: "solution.component.html"
})

export class IconsComponent implements OnInit {
  data: string[][];
  headers: string[];
  protectedFeatures: string[] = [];
  uniqueValues: string[] = [];
  favourableOutcomes: string[] = [];
  selectedTargetColumn: string;
  uniqueOutcomes: string[];
  Solutiontype: string;
  selectedFavourableOutcomes: string;
  public showVal1: boolean = true;

  toggleVal(): void {
    this.showVal1 = !this.showVal1;
    if (this.trustcalc.Solutiontype == 'unsupervised') { this.trustcalc.Solutiontype = 'supervised'; }
    else { this.trustcalc.Solutiontype = 'unsupervised'; }
  }

  isEditing: Boolean = false;

  ScenarioName: any;
  TrainnigDatafile: File;
  TestFile: File;
  FactsheetFile: File;
  Outlierdatafile: File;
  ModelFile: File;
  WeightMetricFile: File;
  WeightPillarFile: File;
  MapFile: File;
  ProtectedFeatures: string[];
  Protectedvalues: string[];
  // form: FormGroup;
  trustcalc = {
    SelectScenario: '',
    NameSolution: '',
    DescriptionSolution: '',
    Solutiontype: 'supervised',
    // TrainingFile: '',
    // TestFile: '',
    // FactsheetFile: '',
    // ModelFile: '',
    Targetcolumn: '',
    Favourableoutcome: '',
    Protectedfeatures: '',
    Protectedvalues: '',
    // TrainnigDatafile: File,
    ScenarioName: '',
    ModelLinks: '',
    LinktoDataset: '',
    Description: '',
    emailid: '',
    Userid: '',
  };

  constructor(public auth: AuthService, private trustcalcservice: trustcalcService, private router: Router, public toast: ToastrService) { }

  ngOnInit() {
    if (!this.showVal1) {
      this.trustcalc.Solutiontype = 'unsupervised';
    }
    this.trustcalc.emailid = localStorage.getItem('email');
    this.trustcalcservice.get(localStorage.getItem('email')).subscribe((data: any) => {
      this.ScenarioName = data.ScenarioName;
    });

    const id = this.router.url.substring(7);
    if (id.length <= 0)
      return;

    this.isEditing = true;
    this.trustcalcservice.getSolution(id).subscribe(data => {
      this.trustcalc.NameSolution = data.solution_name;
      this.trustcalc.DescriptionSolution = data.description;
      this.trustcalc.Solutiontype = data.solution_type;
      this.trustcalc.Protectedvalues = data.protected_features;
    })
  }

  selectedForm: string = 'supervised';
  onFormSelectionChange(selectedValue: string): void {
    this.selectedForm = selectedValue;
  }

  onFileChange(event) {
    this.TrainnigDatafile = event.target.files[0];
    const reader = new FileReader();
    reader.readAsText(event.target.files[0]);
    reader.onload = () => {
      const csvData = reader.result as string;
      const allTextLines = csvData.split(/\r|\n|\r/);
      const headers = allTextLines[0].split(',');
      const data = [];
      for (let i = 1; i < allTextLines.length; i++) {
        const lineData = allTextLines[i].split(',');
        if (lineData.length === headers.length) {
          const tarr = [];
          for (let j = 0; j < headers.length; j++) {
            tarr.push(lineData[j]);
          }
          data.push(tarr);
        }
      }
      this.data = data;
      this.headers = headers;
    };
    reader.onerror = function () {
    };
  }

  setUniqueOutcomes() {
    this.uniqueOutcomes = [];
    for (const row of this.data) {
      const value = row[this.headers.indexOf(this.selectedTargetColumn)];
      if (this.uniqueOutcomes.indexOf(value) === -1) {
        this.uniqueOutcomes.push(value);
      }
    }
  }

  toggleFavourableOutcome(outcome) {
    console.log('value:', outcome);
    const index = this.selectedFavourableOutcomes.indexOf(outcome);
    if (index === -1) {
      // this.selectedFavourableOutcomes.push(value);
    } else {
      // this.selectedFavourableOutcomes.splice(index, 1);
    }
  }
  toggleProtectedFeature(feature) {
    const index = this.protectedFeatures.indexOf(feature);
    if (index === -1) {
      this.protectedFeatures.push(feature);
      this.uniqueValues = [];
      for (const row of this.data) {
        const value = row[this.headers.indexOf(feature)];
        if (this.uniqueValues.indexOf(value) === -1) {
          this.uniqueValues.push(value);
        }
      }
    } else {
      this.protectedFeatures.splice(index, 1);
      this.uniqueValues = [];
    }
  }

  onTrainnigDatafile(event: any) {
    // const file = event.target.files[0];
    // this.form.get('profile').setValue(file);
    this.TrainnigDatafile = event.target.files[0];
    const file = event.target.files[0];

    // Use a library like PapaParse to parse the contents of the file
    // into a 2D array of strings
    Papa.parse(file, {
      header: true,
      complete: result => {
        this.headers = result.meta.fields;
        this.data = result.data;
      }
    });
  }
  onTestFile(event: any) {
    // const file = event.target.files[0];
    // this.form.get('profile').setValue(file);
    this.TestFile = event.target.files[0];
  }

  onOutlierdatafile(event: any) {
    // const file = event.target.files[0];
    // this.form.get('profile').setValue(file);
    this.Outlierdatafile = event.target.files[0];
  }


  onFactsheetfile(event: any) {
    // const file = event.target.files[0];
    // this.form.get('profile').setValue(file);
    this.FactsheetFile = event.target.files[0];
  }
  onModelfile(event: any) {
    // const file = event.target.files[0];
    // this.form.get('profile').setValue(file);
    this.ModelFile = event.target.files[0];
  }

  onMapChange(event) {
    this.MapFile = event.target.files[0];
  }


  onWeightMetricChange(event) {
    this.WeightMetricFile = event.target.files[0];
  }

  onWeightPillarChange(event) {
    this.WeightPillarFile = event.target.files[0];
  }

  savetrustcalc(): void {
    let formData = new FormData();
    formData.append('Userid', this.trustcalc.Userid);
    formData.append('emailid', this.trustcalc.emailid);
    formData.append('SelectScenario', this.trustcalc.SelectScenario);
    formData.append('NameSolution', this.trustcalc.NameSolution);
    formData.append('DescriptionSolution', this.trustcalc.DescriptionSolution);
    formData.append('TrainingFile', this.TrainnigDatafile);
    formData.append('TestFile', this.TestFile);
    formData.append('FactsheetFile', this.FactsheetFile);
    formData.append('Solutiontype', this.trustcalc.Solutiontype);


    formData.append('WeightMetric', this.WeightMetricFile);
    formData.append('WeightPillar', this.WeightPillarFile);
    // formData.append('ProtectedFeature', new Blob(this.ProtectedFeatures, { type: 'text/plain' }));
    formData.append('ProtectedFeature', this.trustcalc.Protectedfeatures);
    // formData.append('Protectedvalues', new Blob(this.Protectedvalues, { type: 'text/plain' }));
    formData.append('Protectedvalues', this.trustcalc.Protectedvalues);


    formData.append('Outlierdatafile', this.Outlierdatafile);
    formData.append('MapFile', this.MapFile);
    formData.append('ModelFile', this.ModelFile);
    formData.append('Targetcolumn', this.trustcalc.Targetcolumn);
    formData.append('Favourableoutcome', this.selectedFavourableOutcomes);

    const data = {
      SelectScenario: this.trustcalc.SelectScenario,
      TrainnigDatafile: this.TrainnigDatafile,
      // TrainnigDatafile: this.form.get('profile').value,
      // DatafileName: this.TrainnigDatafile.name,
      WeightMetric: this.WeightPillarFile,
      WeightPillar: this.WeightPillarFile,
      ModelLinks: this.trustcalc.ModelLinks,
      LinktoDataset: this.trustcalc.LinktoDataset,
      Description: this.trustcalc.Description,
      emailid: this.trustcalc.emailid,
      Userid: this.trustcalc.Userid
    };

    this.trustcalcservice.uploadsolution(formData)
      .subscribe(
        response => {
          this.router.navigate(['/dashboard']) //redirecting works.. ok? it seems to not work
        },
        error => {
        }
      );
  };

  validateFields(data: FormData): { errorText: string, hasError: boolean } {
    let errorText = '';
    let hasError = false;

    if (data.get('SelectScenario').toString().length <= 0) {
      errorText = 'Please Select Scenario';
      hasError = true;
      return { errorText, hasError };
    }

    if (data.get('NameSolution').toString().length <= 0) {
      errorText = 'Please input solution name';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('DescriptionSolution').toString().length <= 0) {
      errorText = 'Please input solution description';
      hasError = true;
    }
    if (data.get('TrainingFile').toString() == 'undefined') {
      errorText = 'Please select train file';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('TestFile').toString() == 'undefined') {
      errorText = 'Please select test file';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('FactsheetFile').toString() == 'undefined') {
      errorText = 'Please select factsheet file';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('MapFile').toString() == 'undefined') {
      errorText = 'Please select map file';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('ProtectedFeature').toString().length <= 0) {
      errorText = 'Please select protected feature';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('Protectedvalues').toString().length <= 0) {
      errorText = 'Please select protected values';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('Solutiontype') == 'unsupervised' && data.get('Outlierdatafile').toString() == 'undefined') {
      errorText = 'Please select outlier data file';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('WeightPillar').toString() == 'undefined') {
      errorText = 'Please select weight pillar';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('WeightMetric').toString() == 'undefined') {
      errorText = 'Please select weight metric';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('ModelFile').toString() == 'undefined') {
      errorText = 'Please select model file';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('Targetcolumn').toString() == 'undefined') {
      errorText = 'Please select target column';
      hasError = true;
      return { errorText, hasError };
    }
    if (data.get('Favourableoutcome').toString().length <= 0) {
      errorText = 'Please select favourable outcome';
      hasError = true;
      return { errorText, hasError };
    }

    return {
      errorText,
      hasError,
    };
  }

  changetrustcalc(): void {
    const id = this.router.url.substring(7);
    if (id.length <= 0)
      return;

    let formData = new FormData();
    formData.append('SolutionId', id);
    formData.append('SelectScenario', this.trustcalc.SelectScenario);
    formData.append('NameSolution', this.trustcalc.NameSolution);
    formData.append('DescriptionSolution', this.trustcalc.DescriptionSolution);
    formData.append('TrainingFile', this.TrainnigDatafile);
    formData.append('TestFile', this.TestFile);
    formData.append('FactsheetFile', this.FactsheetFile);
    formData.append('Solutiontype', this.Solutiontype);

    formData.append('MapFile', this.MapFile);
    // formData.append('ProtectedFeature', new Blob(this.ProtectedFeatures, { type: 'text/plain' }));
    formData.append('ProtectedFeature', this.trustcalc.Protectedfeatures);
    // formData.append('Protectedvalues', new Blob(this.Protectedvalues, { type: 'text/plain' }));
    formData.append('Protectedvalues', this.trustcalc.Protectedvalues);
    formData.append('Outlierdatafile', this.Outlierdatafile);
    formData.append('WeightPillar', this.WeightPillarFile);
    formData.append('WeightMetric', this.WeightMetricFile);
    formData.append('ModelFile', this.ModelFile);
    formData.append('Targetcolumn', this.trustcalc.Targetcolumn);
    formData.append('Favourableoutcome', this.trustcalc.Favourableoutcome);

    const validateResult = this.validateFields(formData);
    if (validateResult.hasError) {
      this.toast.error(validateResult.errorText, 'Update Error');
      return;
    }

    this.trustcalcservice.updateSolution(formData)
      .subscribe(
        response => {
          this.router.navigate(['/dashboard'])
        },
        error => {
        }
      );
  }

  Getdata(id): void {
    this.trustcalcservice.get(id)
      .subscribe(
        response => {
        },
        error => {
        }
      );
  }
}
