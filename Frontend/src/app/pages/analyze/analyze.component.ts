import { Component, OnInit, TemplateRef, ViewChild } from "@angular/core";
import { trustcalcService } from "src/app/services/trustcalc.service";
import { NgbModalConfig, NgbModal } from "@ng-bootstrap/ng-bootstrap";

@Component({
  selector: "app-analyze",
  templateUrl: "analyze.component.html",
  providers: [NgbModalConfig, NgbModal],
})
export class analyzeComponent implements OnInit {
  public canvas: any;
  public ctx;
  public datasets: any;
  public data: any;
  public myChartData;
  public clicked: boolean = true;
  public clicked1: boolean = false;
  public clicked2: boolean = false;
  public showVal1: boolean = true;

  @ViewChild("dialogRef") dialogRef: TemplateRef<any>;

  toggleVal(): void {
    this.showVal1 = !this.showVal1;
  }

  convertFunction = (data: any) => {
    for (const keyIt in data) {
      const type = typeof data[keyIt];
      if (type == 'object' || type == 'string') continue;
      data[keyIt] = parseFloat(data[keyIt]).toFixed(2);
    }

    return data;
  }

  ScenarioName: any;
  SolutionName: any;
  TrainnigDatafile: File;
  TestFile: File;
  FactsheetFile: File;
  ModelFile: File;
  // form: FormGroup;
  trustcalc = {
    SelectScenario: '',
    SelectSolution: '',
    NameSolution: '',
    DescriptionSolution: '',
    // TrainingFile: '',
    // TestFile: '',
    // FactsheetFile: '',
    // ModelFile: '',
    Targetcolumn: '',
    // TrainnigDatafile: File,
    ScenarioName: '',
    ModelLinks: '',
    LinktoDataset: '',
    Description: '',
    emailid: '',
    Userid: '',

    accuracy: '',
    classweightedf1score: '',
    classweightedprecision: '',
    classweightedrecall: '',
    globalf1score: '',
    globalprecision: '',
    globalrecall: '',

    ModelType: '',
    NormalizationTechnique: '',
    TrainTestSplit: '',
    RegularizationTechnique: '',
    DataSize: '',
    NumberofFeatures: '',

    modelname: '',
    purposedesc: '',
    trainingdatadesc: '',
    modelinfo: '',
    authors: '',
    contactinfo: '',

    fairness_score: '',
    explainability_score: '',
    methodology_score: '',
    robustness_score: '',

    underfitting: '',
    overfitting: '',
    statistical_parity_difference: '',
    equal_opportunity_difference: '',
    average_odds_difference: '',
    disparate_impact: '',
    class_balance: '',

    algorithm_class: '',
    correlated_features: '',
    model_size: '',
    feature_relevance: '',

    confidence_score: '',
    clique_method: '',
    loss_sensitivity: '',
    clever_score: '',
    er_fast_gradient_attack: '',
    er_carlini_wagner_attack: '',
    er_deepfool_attack: '',

    normalization: '',
    missing_data: '',
    regularization: '',
    train_test_split: '',
    factsheet_completeness: '',

  };

  constructor(private trustcalcservice: trustcalcService, config: NgbModalConfig, public modalService: NgbModal) {
    config.backdrop = 'static';
    config.keyboard = false;
  }

  ngOnInit() {
    this.trustcalc.emailid = localStorage.getItem('email');
    this.trustcalcservice.get(localStorage.getItem('email')).subscribe((data: any) => {
      this.ScenarioName = data.ScenarioName;
    });
    this.trustcalcservice.getsolution(localStorage.getItem('email')).subscribe((data: any) => {
      this.SolutionName = data.SolutionName;
    });

  }

  public updateOptions() {
    this.myChartData.data.datasets[0].data = this.data;
    this.myChartData.update();
  }

  analyze(): void {
    const formData = new FormData();
    formData.append('Userid', this.trustcalc.Userid);
    formData.append('emailid', this.trustcalc.emailid);
    formData.append('SelectScenario', this.trustcalc.SelectScenario);
    formData.append('SelectSolution', this.trustcalc.SelectSolution);

    this.trustcalcservice.analyzesolution(formData)
      .subscribe(
        response => {
          response = this.convertFunction(response);
          this.trustcalc.ScenarioName = response.ScenarioName;
          this.trustcalc.LinktoDataset = response.LinktoDataset;
          this.trustcalc.Description = response.Description;

          this.trustcalc.accuracy = response.accuracy;
          this.trustcalc.classweightedf1score = response.classweightedf1score;
          this.trustcalc.classweightedprecision = response.classweightedprecision;
          this.trustcalc.classweightedrecall = response.classweightedrecall;
          this.trustcalc.globalf1score = response.globalf1score;
          this.trustcalc.globalprecision = response.globalprecision;
          this.trustcalc.globalrecall = response.globalrecall;

          this.trustcalc.ModelType = response.ModelType;
          this.trustcalc.NormalizationTechnique = response.NormalizationTechnique;
          this.trustcalc.TrainTestSplit = response.TrainTestSplit;
          this.trustcalc.RegularizationTechnique = response.RegularizationTechnique;
          this.trustcalc.DataSize = response.DataSize;
          this.trustcalc.NumberofFeatures = response.NumberofFeatures;

          this.trustcalc.modelname = response.modelname;
          this.trustcalc.purposedesc = response.purposedesc;
          this.trustcalc.trainingdatadesc = response.trainingdatadesc;
          this.trustcalc.modelinfo = response.modelinfo;
          this.trustcalc.authors = response.authors;
          this.trustcalc.contactinfo = response.contactinfo;

          this.trustcalc.fairness_score = response.fairness_score;
          this.trustcalc.explainability_score = response.explainability_score;
          this.trustcalc.methodology_score = response.methodology_score;
          this.trustcalc.robustness_score = response.robustness_score;

          this.trustcalc.underfitting = response.underfitting;
          this.trustcalc.overfitting = response.overfitting;
          this.trustcalc.statistical_parity_difference = response.statistical_parity_difference;
          this.trustcalc.equal_opportunity_difference = response.equal_opportunity_difference;
          this.trustcalc.average_odds_difference = response.average_odds_difference;
          this.trustcalc.disparate_impact = response.disparate_impact;
          this.trustcalc.class_balance = response.class_balance;

          this.trustcalc.algorithm_class = response.algorithm_class;
          this.trustcalc.correlated_features = response.correlated_features;
          this.trustcalc.model_size = response.model_size;
          this.trustcalc.feature_relevance = response.feature_relevance;

          this.trustcalc.confidence_score = response.confidence_score;
          this.trustcalc.clique_method = response.clique_method;
          this.trustcalc.loss_sensitivity = response.loss_sensitivity;
          this.trustcalc.clever_score = response.clever_score;
          this.trustcalc.er_fast_gradient_attack = response.er_fast_gradient_attack;
          this.trustcalc.er_carlini_wagner_attack = response.er_carlini_wagner_attack;
          this.trustcalc.er_deepfool_attack = response.er_deepfool_attack;

          this.trustcalc.normalization = response.normalization;
          this.trustcalc.missing_data = response.missing_data;
          this.trustcalc.regularization = response.regularization;
          this.trustcalc.train_test_split = response.train_test_split;
          this.trustcalc.factsheet_completeness = response.factsheet_completeness;
        },
        error => {
          this.modalService.open(this.dialogRef);
        }
      )
  };

  open(dialogRef) {
    this.modalService.open(dialogRef);
  }

  downloadFile() {
    const link = document.createElement('a');
    link.setAttribute('target', '_blank');
    link.setAttribute('href', 'http://127.0.0.1:8000/api/factsheet_download');
    document.body.appendChild(link);
    link.click();
    link.remove();
  }
}
