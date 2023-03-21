import { Component, OnInit } from "@angular/core";
import { AuthService, User } from '@auth0/auth0-angular';
import { trustcalcService } from "src/app/services/trustcalc.service";
import { Chart } from 'chart.js';

@Component({
  selector: "app-compare",
  templateUrl: "compare.component.html"
})
export class compareComponent implements OnInit {
  ScenarioName: any;
  SolutionName: any;

  public chartItemLabels: any = {
    trust: ['FAIRNESS', 'EXPLAINABILITY', 'ROBUSTNESS', 'ACCOUNTABILITY'],
    fair: ['UNDERFITTING', 'Statistical Parity Difference', 'Equal Opportunity Difference', 'Average Odds Difference', 'Disperate Impact', 'Class Balance'],
    explain: ['Algorithm Class', 'Correlated Features', 'Model Size', 'Feature Relevance'],
    robust: ['Confidence Score', 'CLIQUE_METHOD_SCORE', 'LOSS_SENSITIVITY', 'CLEVER_SCORE', 'ER_FAST_GRADIENT_SCORE', 'ER_CARLINI_WAGNER_ATTACK_SCORE', 'ER_DEEPFOOL_ATTACK'],
    account: ['Normalization', 'Missing Data', 'Regularization', 'Train Test Split', 'Factsheet Completeness']
  };

  trustcalc = {
    SelectScenario: '',
    SelectSolution1: '',
    SelectSolution2: '',
    emailid: '',
    Userid: '',

    accuracy: '',
    classweightedf1score: '',
    classweightedprecision: '',
    classweightedrecall: '',
    globalf1score: '',
    globalprecision: '',
    globalrecall: '',

    accuracy2: '',
    classweightedf1score2: '',
    classweightedprecision2: '',
    classweightedrecall2: '',
    globalf1score2: '',
    globalprecision2: '',
    globalrecall2: '',

    modelname1: '',
    purposedesc1: '',
    trainingdatadesc1: '',
    modelinfo1: '',
    authors1: '',
    contactinfo1: '',
    modelname2: '',
    purposedesc2: '',
    trainingdatadesc2: '',
    modelinfo2: '',
    authors2: '',
    contactinfo2: '',


    fairness_score1: '',
    explainability_score1: '',
    methodology_score1: '',
    robustness_score1: '',
    trust_score1: '',

    fairness_score2: '',
    explainability_score2: '',
    methodology_score2: '',
    robustness_score2: '',
    trust_score2: '',

    fair: {
      underfitting1: '',
      overfitting1: '',
      statistical_parity_difference1: '',
      equal_opportunity_difference1: '',
      average_odds_difference1: '',
      disperate_impact1: '',
      class_balance1: '',

      underfitting2: '',
      overfitting2: '',
      statistical_parity_difference2: '',
      equal_opportunity_difference2: '',
      average_odds_difference2: '',
      disperate_impact2: '',
      class_balance2: '',
    },

    explain: {
      algorithm1: '',
      correlated1: '',
      model_size1: '',
      feature_relevance1: '',

      algorithm2: '',
      correlated2: '',
      model_size2: '',
      feature_relevance2: '',
    },

    robust: {
      confidence_score1: '',
      clique_method1: '',
      loss_sensitivity1: '',
      clever_score1: '',
      er_fast_gradient_attack1: '',
      er_carlini_wagner_attack1: '',
      er_deepfool_attack1: '',

      confidence_score2: '',
      clique_method2: '',
      loss_sensitivity2: '',
      clever_score2: '',
      er_fast_gradient_attack2: '',
      er_carlini_wagner_attack2: '',
      er_deepfool_attack2: '',
    },

    account: {
      normalization1: '',
      missing_data1: '',
      regularization1: '',
      train_test_split1: '',
      factsheet_completeness1: '',

      normalization2: '',
      missing_data2: '',
      regularization2: '',
      train_test_split2: '',
      factsheet_completeness2: '',
    }
  };

  public chartOptions = {
    type: 'bar',
    options: {
      scales: {
        yAxes: [
          {
            ticks: {
              beginAtZero: true
            }
          }
        ]
      }
    }
  };

  public dataset = {
    backgroundColor: [
      "rgba(255, 99, 132, 0.2)",
      "rgba(54, 162, 235, 0.2)",
      "rgba(255, 206, 86, 0.2)",
      "rgba(75, 192, 192, 0.2)",
      "rgba(153, 102, 255, 0.2)"
    ],
    borderColor: [
      "rgba(255, 99, 132, 1)",
      "rgba(54, 162, 235, 1)",
      "rgba(255, 206, 86, 1)",
      "rgba(75, 192, 192, 1)",
      "rgba(153, 102, 255, 1)"
    ],
    borderWidth: 1
  };

  public trustChart1: Chart;
  public trustChart2: Chart;
  public fairnessChart1: Chart;
  public fairnessChart2: Chart;
  public explainChart1: Chart;
  public explainChart2: Chart;
  public robustChart1: Chart;
  public robustChart2: Chart;
  public accountChart1: Chart;
  public accountChart2: Chart;

  convertFunction = (data: any) => {
    for (const keyIt in data) {
      const type = typeof data[keyIt];
      if (type == 'object' || type == 'string') continue;
      data[keyIt] = parseFloat(data[keyIt]).toFixed(2);
    }

    return data;
  }

  constructor(public auth: AuthService, private trustcalcservice: trustcalcService) { }

  ngOnInit() {
    this.trustcalcservice.get(localStorage.getItem('email')).subscribe((data: any) => {
      this.ScenarioName = data.ScenarioName;
    });
    this.trustcalcservice.getsolution(localStorage.getItem('email')).subscribe((data: any) => {
      this.SolutionName = data.SolutionName;
    });
  }

  createChart(id: string, labelId: string, label: string, data): Chart {
    return new Chart(id, {
      ...this.chartOptions,
      data: {
        labels: this.chartItemLabels[labelId],
        datasets: [
          {
            label,
            data,
            ...this.dataset,
          }
        ]
      },
    });
  }

  savetrustcalc(): void {
    const formData = new FormData();
    formData.append('Userid', this.trustcalc.Userid);
    formData.append('emailid', localStorage.getItem('email'));
    formData.append('SelectScenario', this.trustcalc.SelectScenario);
    formData.append('SelectSolution1', this.trustcalc.SelectSolution1);
    formData.append('SelectSolution2', this.trustcalc.SelectSolution2);

    this.trustcalcservice.comparesolution(formData)
      .subscribe(
        response => {
          response = this.convertFunction(response);
          this.trustcalc.accuracy = response.accuracy;
          this.trustcalc.classweightedf1score = response.classweightedf1score;
          this.trustcalc.classweightedprecision = response.classweightedprecision;
          this.trustcalc.classweightedrecall = response.classweightedrecall;
          this.trustcalc.globalf1score = response.globalf1score;
          this.trustcalc.globalprecision = response.globalprecision;
          this.trustcalc.globalrecall = response.globalrecall;

          this.trustcalc.accuracy2 = response.accuracy2;
          this.trustcalc.classweightedf1score2 = response.classweightedf1score2;
          this.trustcalc.classweightedprecision2 = response.classweightedprecision2;
          this.trustcalc.classweightedrecall2 = response.classweightedrecall2;
          this.trustcalc.globalf1score2 = response.globalf1score2;
          this.trustcalc.globalprecision2 = response.globalprecision2;
          this.trustcalc.globalrecall2 = response.globalrecall2;

          this.trustcalc.modelname1 = response.modelname1;
          this.trustcalc.purposedesc1 = response.purposedesc1;
          this.trustcalc.trainingdatadesc1 = response.trainingdatadesc1;
          this.trustcalc.modelinfo1 = response.modelinfo1;
          this.trustcalc.authors1 = response.authors1;
          this.trustcalc.contactinfo1 = response.contactinfo1;

          this.trustcalc.modelname2 = response.modelname2;
          this.trustcalc.purposedesc2 = response.purposedesc2;
          this.trustcalc.trainingdatadesc2 = response.trainingdatadesc2;
          this.trustcalc.modelinfo2 = response.modelinfo2;
          this.trustcalc.authors2 = response.authors2;
          this.trustcalc.contactinfo2 = response.contactinfo2;
          // Scores.. for chart.
          this.trustcalc.fairness_score1 = response.fairness_score1;
          this.trustcalc.explainability_score1 = response.explainability_score1;
          this.trustcalc.methodology_score1 = response.methodology_score1;
          this.trustcalc.robustness_score1 = response.robustness_score1;
          this.trustcalc.trust_score1 = response.trust_score1;

          this.trustcalc.fairness_score2 = response.fairness_score2;
          this.trustcalc.explainability_score2 = response.explainability_score2;
          this.trustcalc.methodology_score2 = response.methodology_score2;
          this.trustcalc.robustness_score2 = response.robustness_score2;
          this.trustcalc.trust_score2 = response.trust_score2;

          this.trustcalc.fair.underfitting1 = response.underfitting;
          this.trustcalc.fair.overfitting1 = response.overfitting;
          this.trustcalc.fair.statistical_parity_difference1 = response.statistical_parity_difference;
          this.trustcalc.fair.equal_opportunity_difference1 = response.equal_opportunity_difference;
          this.trustcalc.fair.average_odds_difference1 = response.average_odds_difference;
          this.trustcalc.fair.disperate_impact1 = response.disparate_impact;
          this.trustcalc.fair.class_balance1 = response.class_balance;

          this.trustcalc.fair.underfitting2 = response.underfitting2;
          this.trustcalc.fair.overfitting2 = response.overfitting2;
          this.trustcalc.fair.statistical_parity_difference2 = response.statistical_parity_difference2;
          this.trustcalc.fair.equal_opportunity_difference2 = response.equal_opportunity_difference2;
          this.trustcalc.fair.average_odds_difference2 = response.average_odds_difference2;
          this.trustcalc.fair.disperate_impact2 = response.disparate_impact2;
          this.trustcalc.fair.class_balance2 = response.class_balance2;


          this.trustcalc.explain.algorithm1 = response.algorithm_class;
          this.trustcalc.explain.correlated1 = response.correlated_features;
          this.trustcalc.explain.model_size1 = response.model_size;
          this.trustcalc.fairness_score1 = response.feature_relevance;


          this.trustcalc.explain.algorithm2 = response.algorithm_class2;
          this.trustcalc.explain.correlated2 = response.correlated_features2;
          this.trustcalc.explain.model_size2 = response.model_size2;
          this.trustcalc.fairness_score2 = response.feature_relevance2;


          this.trustcalc.robust.confidence_score1 = response.confidence_score;
          this.trustcalc.robust.clique_method1 = response.clique_method;
          this.trustcalc.robust.loss_sensitivity1 = response.loss_sensitivity;
          this.trustcalc.robust.clever_score1 = response.clever_score;
          this.trustcalc.robust.er_fast_gradient_attack1 = response.er_fast_gradient_attack;
          this.trustcalc.robust.er_carlini_wagner_attack1 = response.er_carlini_wagner_attack;
          this.trustcalc.robust.er_deepfool_attack1 = response.er_deepfool_attack;

          this.trustcalc.robust.confidence_score2 = response.confidence_score2;
          this.trustcalc.robust.clique_method2 = response.clique_method2;
          this.trustcalc.robust.loss_sensitivity2 = response.loss_sensitivity2;
          this.trustcalc.robust.clever_score2 = response.clever_score2;
          this.trustcalc.robust.er_fast_gradient_attack2 = response.er_fast_gradient_attack2;
          this.trustcalc.robust.er_carlini_wagner_attack2 = response.er_carlini_wagner_attack2;
          this.trustcalc.robust.er_deepfool_attack2 = response.er_deepfool_attack2;

          this.trustcalc.account.normalization1 = response.normalization;
          this.trustcalc.account.missing_data1 = response.missing_data;
          this.trustcalc.account.regularization1 = response.regularization;
          this.trustcalc.account.train_test_split1 = response.train_test_split;
          this.trustcalc.account.factsheet_completeness1 = response.factsheet_completeness;

          this.trustcalc.account.normalization2 = response.normalization2;
          this.trustcalc.account.missing_data2 = response.missing_data2;
          this.trustcalc.account.regularization2 = response.regularization2;
          this.trustcalc.account.train_test_split2 = response.train_test_split2;
          this.trustcalc.account.factsheet_completeness2 = response.factsheet_completeness2;

          this.trustChart1 = this.createChart('canvas1', 'trust', 'TRUSTWORTHINESS OVERALL SCORE', [response.fairness_score1, response.explainability_score1, response.robustness_score1, response.methodology_score1]);
          this.trustChart2 = this.createChart('canvas2', 'trust', 'TRUSTWORTHINESS OVERALL SCORE', [response.fairness_score2, response.explainability_score2, response.robustness_score2, response.methodology_score2]);
          this.fairnessChart1 = this.createChart('canvas3', 'fair', 'FAIRNESS SCORE', [
            response.underfitting,
            response.overfitting,
            response.statistical_parity_difference,
            response.equal_opportunity_difference,
            response.average_odds_difference,
            response.disparate_impact,
            response.class_balance,
          ]);
          this.fairnessChart2 = this.createChart('canvas4', 'fair', 'FAIRNESS SCORE', [
            response.underfitting2,
            response.overfitting2,
            response.statistical_parity_difference2,
            response.equal_opportunity_difference2,
            response.average_odds_difference2,
            response.disparate_impact2,
            response.class_balance2,
          ]);
          this.explainChart1 = this.createChart('canvas5', 'explain', 'EXPLAINABILITY SCORE', [
            response.algorithm_class,
            response.correlated_features,
            response.model_size,
            response.feature_relevance,
          ]);
          this.explainChart2 = this.createChart('canvas6', 'explain', 'EXPLAINABILITY SCORE', [
            response.algorithm_class2,
            response.correlated_features2,
            response.model_size2,
            response.feature_relevance2,
          ]);
          this.robustChart1 = this.createChart('canvas7', 'robust', 'ROBUSTNESS SCORE', [
            response.confidence_score,
            response.clique_method,
            response.loss_sensitivity,
            response.clever_score,
            response.er_fast_gradient_attack,
            response.er_carlini_wagner_attack,
            response.er_deepfool_attack,
          ]);
          this.robustChart2 = this.createChart('canvas8', 'robust', 'ROBUSTNESS SCORE', [
            response.confidence_score2,
            response.clique_method2,
            response.loss_sensitivity2,
            response.clever_score2,
            response.er_fast_gradient_attack2,
            response.er_carlini_wagner_attack2,
            response.er_deepfool_attack2,
          ]);
          this.accountChart1 = this.createChart('canvas9', 'account', 'ACCOUNTABILITY', [
            response.normalization,
            response.missing_data,
            response.regularization,
            response.train_test_split,
            response.factsheet_completeness,
          ]);
          this.accountChart2 = this.createChart('canvas10', 'account', 'ACCOUNTABILITY', [
            response.normalization2,
            response.missing_data2,
            response.regularization2,
            response.train_test_split2,
            response.factsheet_completeness2,
          ]);
        },
        error => {
        }
      );
  };

}
