import { Component, OnInit } from "@angular/core";
import { AuthService } from "@auth0/auth0-angular";
import { trustcalcService } from "src/app/services/trustcalc.service";

@Component({
  selector: "app-dashboard",
  templateUrl: "dashboard.component.html"
})
export class DashboardComponent implements OnInit {
  trustcalc = {
    SelectScenario: '',
    SelectSolution1: '',
    SelectSolution2: '',
    Userid: '',

    fairness_score: '',
    explainability_score: '',
    methodology_score: '',
    robustness_score: '',
    trust_score: '',

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


    unsupervised_fairness_score: '',
    unsupervised_explainability_score: '',
    unsupervised_methodology_score: '',
    unsupervised_robustness_score: '',
    unsupervised_trust_score: '',


    unsupervised_underfitting: '',
    unsupervised_overfitting: '',
    unsupervised_statistical_parity_difference: '',
    unsupervised_equal_opportunity_difference: '',
    unsupervised_average_odds_difference: '',
    unsupervised_disparate_impact: '',
    unsupervised_class_balance: '',

    unsupervised_algorithm_class: '',
    unsupervised_correlated_features: '',
    unsupervised_model_size: '',
    unsupervised_permutation_importance: '',

    unsupervised_confidence_score: '',
    unsupervised_clique_method: '',
    unsupervised_loss_sensitivity: '',
    unsupervised_clever_score: '',
    unsupervised_er_fast_gradient_attack: '',
    unsupervised_er_carlini_wagner_attack: '',
    unsupervised_er_deepfool_attack: '',

    unsupervised_normalization: '',
    unsupervised_missing_data: '',
    unsupervised_regularization: '',
    unsupervised_train_test_split: '',
    unsupervised_factsheet_completeness: '',

    scenarioList: [],
    solutionList: [],
  };

  descriptions = [{
    title: 'Underfitting',
    description: 'Metric Scores - Compares the models achieved test accuracy against a baseline. Depends on: Model, Test Data'
  }, {
    title: 'Overfitting',
    description: ' Metric Description: Overfitting is present if the training accuracy is significantly higher than the test accuracy Depends on: Model, Training Data, Test Data',
  }, {
    title: 'Statistical Parity Difference:',
    description: 'Metric Description: The spread between the percentage of observations from the majority group receiving a favorable outcome compared to the protected group.The closes this spread is to zero the better.Depends on: Training Data, Factsheet(Definition of Protected Group and Favorable Outcome) '
  }, {
    title: 'Equal Opportunity Difference:',
    description: 'Metric Description: Difference in true positive rates between protected and unprotected group. Depends on: Model, Test Data, Factsheet(Definition of Protected Group and Favorable Outcome) '
  }, {
    title: 'Average Odds Difference:',
    description: 'Metric Description: Is the average of difference in false positive rates and true positive rates between the protected and unprotected group Depends on: Model, Test Data, Factsheet(Definition of Protected Group and Favorable Outcome)'
  }, {
    title: 'Disparate Impact:',
    description: 'Metric Description: Is quotient of the ratio of samples from the protected group receiving a favorable prediction divided by the ratio of samples from the unprotected group receiving a favorable prediction Depends on: Model, Test Data, Factsheet(Definition of Protected Group and Favorable Outcome) ',
  }, {
    title: 'Class Balance:',
    description: 'Metric Description: Measures how well the training data is balanced or unbalanced Depends on: Training Data',
  }];

  showSpinner: Boolean = true;

  constructor(public auth: AuthService, private trustcalcservice: trustcalcService) { }

  convertFunction = (data: any) => {
    for (const keyIt in data) {
      if (typeof data[keyIt] == 'object') continue;
      data[keyIt] = parseFloat(data[keyIt]).toFixed(2);
    }

    return data;
  }

  pickHex = (weight) => {
    const color1 = [255, 255, 0];
    const color2 = [0, 0, 0];
    var w1 = (weight - 1) * 0.25;
    var w2 = (5 - w1) * 0.25;
    const firstColor = ("0" + Math.round(255 - color1[0] * w1 + color2[0] * w2).toString(16)).slice(-2);
    const secondColor = ("0" + Math.round(color1[1] * w1 + color2[1] * w2).toString(16)).slice(-2);
    const thirdColor = ("0" + Math.round(color1[2] * w1 + color2[2] * w2).toString(16)).slice(-2);
    return `#${firstColor}${secondColor}${thirdColor}`;
  }


  ngOnInit() {
    // this.spinner.show();
    this.trustcalcservice.dashboard(localStorage.getItem('email')).subscribe((data: any) => {
      // this.ScenarioName=data.ScenarioName;
      data = this.convertFunction(data);

      this.trustcalc.fairness_score = data.fairness_score;
      this.trustcalc.explainability_score = data.explainability_score;
      this.trustcalc.methodology_score = data.methodology_score;
      this.trustcalc.robustness_score = data.robustness_score;

      this.trustcalc.underfitting = data.underfitting;
      this.trustcalc.overfitting = data.overfitting;
      this.trustcalc.statistical_parity_difference = data.statistical_parity_difference;
      this.trustcalc.equal_opportunity_difference = data.equal_opportunity_difference;
      this.trustcalc.average_odds_difference = data.average_odds_difference;
      this.trustcalc.disparate_impact = data.disparate_impact;
      this.trustcalc.class_balance = data.class_balance;

      this.trustcalc.algorithm_class = data.algorithm_class;
      this.trustcalc.correlated_features = data.correlated_features;
      this.trustcalc.model_size = data.model_size;
      this.trustcalc.feature_relevance = data.feature_relevance;

      this.trustcalc.confidence_score = data.confidence_score;
      this.trustcalc.clique_method = data.clique_method;
      this.trustcalc.loss_sensitivity = data.loss_sensitivity;
      this.trustcalc.clever_score = data.clever_score;
      this.trustcalc.er_fast_gradient_attack = data.er_fast_gradient_attack;
      this.trustcalc.er_carlini_wagner_attack = data.er_carlini_wagner_attack;
      this.trustcalc.er_deepfool_attack = data.er_deepfool_attack;

      this.trustcalc.normalization = data.normalization;
      this.trustcalc.missing_data = data.missing_data;
      this.trustcalc.regularization = data.regularization;
      this.trustcalc.train_test_split = data.train_test_split;
      this.trustcalc.factsheet_completeness = data.factsheet_completeness;



      this.trustcalc.unsupervised_fairness_score = data.unsupervised_fairness_score;
      this.trustcalc.unsupervised_explainability_score = data.unsupervised_explainability_score;
      this.trustcalc.unsupervised_methodology_score = data.unsupervised_methodology_score;
      this.trustcalc.unsupervised_robustness_score = data.unsupervised_robustness_score;

      this.trustcalc.unsupervised_underfitting = data.unsupervised_underfitting;
      this.trustcalc.unsupervised_overfitting = data.unsupervised_overfitting;
      this.trustcalc.unsupervised_statistical_parity_difference = data.unsupervised_statistical_parity_difference;
      // this.trustcalc.unsupervised_equal_opportunity_difference =data.unsupervised_equal_opportunity_difference;
      // this.trustcalc.unsupervised_average_odds_difference =data.unsupervised_average_odds_difference;
      this.trustcalc.unsupervised_disparate_impact = data.unsupervised_disparate_impact;
      // this.trustcalc.unsupervised_class_balance =data.unsupervised_class_balance;

      // this.trustcalc.unsupervised_algorithm_class =data.unsupervised_algorithm_class;
      this.trustcalc.unsupervised_correlated_features = data.unsupervised_correlated_features;
      this.trustcalc.unsupervised_model_size = data.unsupervised_model_size;
      this.trustcalc.unsupervised_permutation_importance = data.unsupervised_permutation_importance;

      // this.trustcalc.unsupervised_confidence_score =data.unsupervised_confidence_score;
      // this.trustcalc.unsupervised_clique_method =data.unsupervised_clique_method;
      // this.trustcalc.unsupervised_loss_sensitivity =data.unsupervised_loss_sensitivity;
      this.trustcalc.unsupervised_clever_score = data.unsupervised_clever_score;
      // this.trustcalc.unsupervised_er_fast_gradient_attack =data.unsupervised_er_fast_gradient_attack;
      // this.trustcalc.unsupervised_er_carlini_wagner_attack =data.unsupervised_er_carlini_wagner_attack;
      // this.trustcalc.unsupervised_er_deepfool_attack =data.unsupervised_er_deepfool_attack;

      this.trustcalc.unsupervised_normalization = data.unsupervised_normalization;
      this.trustcalc.unsupervised_missing_data = data.unsupervised_missing_data;
      this.trustcalc.unsupervised_regularization = data.unsupervised_regularization;
      this.trustcalc.unsupervised_train_test_split = data.unsupervised_train_test_split;
      this.trustcalc.unsupervised_factsheet_completeness = data.unsupervised_factsheet_completeness;

      this.trustcalc.scenarioList = data.scenarioList;
      this.trustcalc.solutionList = data.solutionList;
    }, error => {

    }, () => {
      // this.spinner.hide();
    });
  }
}
