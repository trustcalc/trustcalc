import os
from rest_framework.views import APIView
from rest_framework.response import Response
from ...models import CustomUser, Scenario, ScenarioSolution
from ...views import BASE_DIR
from ...apiback.public import save_files_return_paths


def unsupervised_FinalScore(model_path, training_dataset_path, test_dataset_path, outliers_dataset_path, facsheet_path, metrics_mappings_path, weights_metrics_path=None, weights_pillars_path=None):
    from algorithms.TrustScore.TrustScore import trusting_AI_scores_unsupervised as finalScore_unsupervised
    import pandas as pd

    dict_trusting_score_unsup = finalScore_unsupervised(model=model_path, training_dataset=training_dataset_path,
                                                        test_dataset=test_dataset_path, outliers_data=outliers_dataset_path, factsheet=facsheet_path, mappings=metrics_mappings_path)

    dict_trusting_score_unsup = dict_trusting_score_unsup[0]
    foo1, foo2, foo3, foo4 = dict_trusting_score_unsup["accountability"], dict_trusting_score_unsup[
        "explainability"], dict_trusting_score_unsup["fairness"], dict_trusting_score_unsup["robustness"]

    factsheetcompletness_score, missing_data_score, normalization_score, regularization_score, train_test_split_score = foo1[
        "factsheet_completeness"], foo1["missing_data"], foo1["normalization"], foo1["regularization"], foo1["train_test_split"]

    correlated_features_score, model_size_score, permutation_feature_importance_score = foo2[
        "correlated_features"], foo2["model_size"], foo2["permutation_feature_importance"]

    disparate_impact_score, overfitting_score, statistical_parity_difference_score, underfitting_score = foo3[
        "disparate_impact"], foo3["overfitting"], foo3["statistical_parity_difference"], foo3["underfitting"]

    clever_score = foo4["clever_score"]

    if (weights_metrics_path is None):
        weights_metrics_unsup = {
            "fairness": {
                "underfitting": 0.35,
                "overfitting": 0.15,
                "statistical_parity_difference": 0.15,
                "disparate_impact": 0.1
            },
            "explainability": {
                "correlated_features": 0.15,
                "model_size": 0.15,
                "permutation_feature_importance": 0.15
            },
            "robustness": {
                "clever_score": 0.2
            },
            "methodology": {
                "normalization": 0.2,
                "missing_data": 0.2,
                "regularization": 0.2,
                "train_test_split": 0.2,
                "factsheet_completeness": 0.2
            },
            "pillars": {
                "fairness": 0.25,
                "explainability": 0.25,
                "robustness": 0.25,
                "methodology": 0.25
            }
        }
    else:
        weights_metrics_unsup = pd.read_json(weights_metrics_path)

    print('path:', weights_pillars_path)
    if (weights_pillars_path is None):
        weights_pillars_unsup = {"pillars": {
            "fairness": 0.25,
            "explainability": 0.25,
            "robustness": 0.25,
            "methodology": 0.25
        }}
    else:
        weights_pillars_unsup = pd.read_json(weights_pillars_path)
    try:
        foo5 = weights_metrics_unsup["accountability"]
    except:
        foo5 = weights_metrics_unsup["methodology"]

    foo6, foo7, foo8 = weights_metrics_unsup["explainability"], weights_metrics_unsup[
        "fairness"], weights_metrics_unsup["robustness"]

    accountability_score_unsupervised = foo5["factsheet_completeness"]*factsheetcompletness_score+foo5["missing_data"]*missing_data_score + \
        foo5["normalization"]*normalization_score+foo5["regularization"] * \
        regularization_score + \
        foo5["train_test_split"]*train_test_split_score
    explainability_score_unsupervised = foo6["correlated_features"]*correlated_features_score+foo6["model_size"] * \
        model_size_score + \
        foo6["permutation_feature_importance"] * \
        permutation_feature_importance_score
    fairness_score_unsupervised = foo7["underfitting"]*underfitting_score+foo7["overfitting"]*overfitting_score + \
        foo7["statistical_parity_difference"]*statistical_parity_difference_score + \
        foo7["disparate_impact"]*disparate_impact_score
    try:
        robustnesss_score_unsupervised = clever_score
    except:
        try:
            if (clever_score is None):
                clever_score = 1
        except:
            pass

        robustnesss_score_unsupervised = clever_score

    weights_pillars_unsup = weights_pillars_unsup["pillars"]
    try:
        weight_accountability = weights_pillars_unsup["accountability"]
    except:
        weight_accountability = weights_pillars_unsup["methodology"]

    trust_score_unsupervised = weight_accountability*accountability_score_unsupervised + weights_pillars_unsup["explainability"]*explainability_score_unsupervised+weights_pillars_unsup["fairness"] * fairness_score_unsupervised + weights_pillars_unsup["robustness"] * robustnesss_score_unsupervised

    dict_accountabiltiy_metric_scores = {"Factsheecompletnessscore": factsheetcompletness_score, "Missingdatascore": missing_data_score,
                                            "Normalizationscore": normalization_score, "Regularizationscore": regularization_score, "Traintestsplitscore": train_test_split_score}
    dict_explainabiltiy_metric_scores = {"Correlatedfeaturesscore": correlated_features_score,
                                            "Modelsizescore": model_size_score, "Permutationfeatureimportancescore": permutation_feature_importance_score}
    dict_fairness_metric_scores = {"Underfittingscore": underfitting_score, "Overfittingscore": overfitting_score,
                                    "Statisticalparitydifferencescore": statistical_parity_difference_score, "Disparateimpactscore": disparate_impact_score}
    dict_robustness_metric_scores = {"Cleverscore": clever_score}

    dict_metric_scores = {"Metricscores": {"Accountabilityscore": dict_accountabiltiy_metric_scores, "Explainabilityscore":
                                            dict_explainabiltiy_metric_scores, "Fairnessscore": dict_fairness_metric_scores, "Robustnessscore": dict_robustness_metric_scores}}

    dict_pillars_scores = {"Accountabilityscore": accountability_score_unsupervised, "Explainabilityscore": explainability_score_unsupervised,
                            "Fairnessscore": fairness_score_unsupervised, "Robustnessscore": robustnesss_score_unsupervised}

    dict_result_unsupervised = {"Metricscores": dict_metric_scores,
                                "Pillarscores": dict_pillars_scores, "Trustscore": trust_score_unsupervised}

    return dict_result_unsupervised

def finalScore_supervised(model_path, training_dataset_path, test_dataset_path, facsheet_path, metrics_mappings_path, weights_metrics_path=None, weights_pillars_path=None):
    from algorithms.TrustScore.TrustScore import trusting_AI_scores_supervised
    import pandas as pd

    dict_trusting_score = trusting_AI_scores_supervised(model=model_path, training_dataset=training_dataset_path,
                                                        test_dataset=test_dataset_path, factsheet=facsheet_path, mappings=metrics_mappings_path)

    dict_trusting_score = dict_trusting_score[0]
    foo1, foo2, foo3, foo4 = dict_trusting_score["accountability"], dict_trusting_score[
        "explainability"], dict_trusting_score["fairness"], dict_trusting_score["robustness"]
    factsheetcompletness_score, missing_data_score, normalization_score, regularization_score, train_test_split_score = foo1[
        "factsheet_completeness"], foo1["missing_data"], foo1["normalization"], foo1["regularization"], foo1["train_test_split"]
    algorithm_class_score, correlated_features_score, feature_relevance_score, model_size_score = foo2[
        "algorithm_class"], foo2["correlated_features"], foo2["feature_relevance"], foo2["model_size"]
    average_odds_difference_score, class_balance_score, disparate_impact_score, equal_opportunity_score, overfitting_score, statistical_parity_difference_score, underfitting_score = foo3[
        "average_odds_difference"], foo3["class_balance"], foo3["disparate_impact"], foo3["equal_opportunity_difference"], foo3["overfitting"], foo3["statistical_parity_difference"], foo3["underfitting"]
    clever_score, clique_method_score, confidence_score, er_carlini_wagner_score, er_deep_fool_attack_score, er_fast_gradient_attack_score, loss_sensitivity_score = foo4[
        "clever_score"], foo4["clique_method"], foo4["confidence_score"], foo4["er_carlini_wagner_attack"], foo4["er_deepfool_attack"], foo4["er_fast_gradient_attack"], foo4["loss_sensitivity"]

    print('weight path:', weights_metrics_path)
    if (weights_metrics_path is None):
        weights_metrics = {
            "fairness": {
                "underfitting": 0.35,
                "overfitting": 0.15,
                "statistical_parity_difference": 0.15,
                "equal_opportunity_difference": 0.2,
                "average_odds_difference": 0.1,
                "disparate_impact": 0.1,
                "class_balance": 0.1
            },
            "explainability": {
                "algorithm_class": 0.55,
                "correlated_features": 0.15,
                "model_size": 0.15,
                "feature_relevance": 0.15
            },
            "robustness": {
                "confidence_score": 0.2,
                "clique_method": 0.2,
                "loss_sensitivity": 0.2,
                "clever_score": 0.2,
                "er_fast_gradient_attack": 0.2,
                "er_carlini_wagner_attack": 0.2,
                "er_deepfool_attack": 0.2
            },
            "methodology": {
                "normalization": 0.2,
                "missing_data": 0.2,
                "regularization": 0.2,
                "train_test_split": 0.2,
                "factsheet_completeness": 0.2
            },
            "pillars": {
                "fairness": 0.25,
                "explainability": 0.25,
                "robustness": 0.25,
                "methodology": 0.25
            }
        }
    else:
        weights_metrics = pd.read_json(weights_metrics_path)

    print('weight path:', weights_pillars_path)
    if (weights_pillars_path is None):
        weights_pillars = {"pillars": {
            "fairness": 0.25,
            "explainability": 0.25,
            "robustness": 0.25,
            "methodology": 0.25
        }}
    else:
        weights_pillars = pd.read_json(weights_pillars_path)

    try:
        foo5 = weights_metrics["accountability"]
    except:
        foo5 = weights_metrics["methodology"]

    foo6, foo7, foo8 = weights_metrics["explainability"], weights_metrics["fairness"], weights_metrics["robustness"]

    accountability_score_supervised = foo5["factsheet_completeness"]*factsheetcompletness_score+foo5["missing_data"]*missing_data_score + \
        foo5["normalization"]*normalization_score+foo5["regularization"] * \
        regularization_score + \
        foo5["train_test_split"]*train_test_split_score
    explainability_score_supervised = foo6["algorithm_class"]*algorithm_class_score+foo6["correlated_features"] * \
        correlated_features_score + \
        foo6["model_size"]*model_size_score + \
        foo6["feature_relevance"]*feature_relevance_score
    fairness_score_supervised = foo7["underfitting"]*underfitting_score+foo7["overfitting"]*overfitting_score+foo7["statistical_parity_difference"]*statistical_parity_difference_score + \
        foo7["equal_opportunity_difference"]*equal_opportunity_score+foo7["average_odds_difference"] * \
        average_odds_difference_score + \
        foo7["disparate_impact"]*disparate_impact_score + \
        foo7["class_balance"]*class_balance_score
    try:
        robustnesss_score_supervised = foo8["clever_score"]*clever_score+foo8["clique_method"]*clique_method_score+foo8["confidence_score"]*confidence_score+foo8["er_carlini_wagner_attack"] * \
            er_carlini_wagner_score+foo8["er_deepfool_attack"]*er_deep_fool_attack_score + \
            foo8["er_fast_gradient_attack"]*er_fast_gradient_attack_score + \
            foo8["loss_sensitivity"]*loss_sensitivity_score

    except:
        try:
            if (clique_method_score is None):
                clique_method_score = 1
        except:
            pass
        if (er_carlini_wagner_score is None):
            er_carlini_wagner_score = 1
        if (er_deep_fool_attack_score is None):
            er_deep_fool_attack_score = 1
        if (er_fast_gradient_attack_score is None):
            er_fast_gradient_attack_score = 1
        if (loss_sensitivity_score is None):
            loss_sensitivity_score = 1
        robustnesss_score_supervised = foo8["clever_score"]*clever_score+foo8["clique_method"]*clique_method_score+foo8["er_carlini_wagner_attack"]*er_carlini_wagner_score + \
            foo8["er_deepfool_attack"]*er_deep_fool_attack_score+foo8["er_fast_gradient_attack"] * \
            er_fast_gradient_attack_score + \
            foo8["loss_sensitivity"]*loss_sensitivity_score

    weights_pillars = weights_pillars["pillars"]
    try:
        weight_accountability = weights_pillars["accountability"]
    except:
        weight_accountability = weights_pillars["methodology"]

    trust_score_supervised = weight_accountability*accountability_score_supervised + \
        weights_pillars["explainability"]*explainability_score_supervised+weights_pillars["fairness"] * \
        fairness_score_supervised + \
        weights_pillars["robustness"]*robustnesss_score_supervised

    dict_accountabiltiy_metric_scores = {"Factsheecompletnessscore": factsheetcompletness_score, "Missingdatascore": missing_data_score,
                                            "Normalizationscore": normalization_score, "Regularizationscore": regularization_score, "Traintestsplitscore": train_test_split_score}
    dict_explainabiltiy_metric_scores = {"Algorithmclassscore": algorithm_class_score, "Correlatedfeaturesscore":
                                            correlated_features_score, "Modelsizescore": model_size_score, "Featurerevelancescore": feature_relevance_score}
    dict_fairness_metric_scores = {"Underfittingscore": underfitting_score, "Overfittingscore": overfitting_score, "Statisticalparitydifferencescore": statistical_parity_difference_score,
                                    "Equalopportunityscore": equal_opportunity_score, "Averageoddsdifferencescore": average_odds_difference_score, "Disparateimpactscore": disparate_impact_score, "Classbalancescore": class_balance_score}
    dict_robustness_metric_scores = {"Cleverscore": clever_score, "Cliquemethodscore": clique_method_score, "Confidencescore": confidence_score, "Ercarliniwagnerscore": er_carlini_wagner_score,
                                        "Erdeepfoolattackscore": er_deep_fool_attack_score, "Erfastgradientattack": er_fast_gradient_attack_score, "Losssensitivityscore": loss_sensitivity_score}

    dict_metric_scores = {"Metricscores": {"Accountabilityscore": dict_accountabiltiy_metric_scores, "Explainabilityscore":
                                            dict_explainabiltiy_metric_scores, "Fairnessscore": dict_fairness_metric_scores, "Robustnessscore": dict_robustness_metric_scores}}

    dict_pillars_scores = {"Accountabilityscore": accountability_score_supervised, "Explainabilityscore": explainability_score_supervised,
                            "Fairnessscore": fairness_score_supervised, "Robustnessscore": robustnesss_score_supervised}

    dict_result_supervised = {"Metricscores": dict_metric_scores,
                                "Pillarscores": dict_pillars_scores, "Trustscore": trust_score_supervised}
    return dict_result_supervised


class dashboard(APIView):
    def get(self, request, email):
        uploaddic = {}
        userexist = CustomUser.objects.get(email=email)
        supscenarioobj = ScenarioSolution.objects.filter(
            user_id=userexist.id, solution_type='supervised').values().order_by('id')

        unsupscenarioobj = ScenarioSolution.objects.filter(
            user_id=userexist.id, solution_type='unsupervised').values().order_by('id')

        scenarios = Scenario.objects.filter(user_id=userexist.id).values()
        scenarioobj = ScenarioSolution.objects.filter(
            user_id=userexist.id).values()
        uploaddic['scenarioList'] = scenarios
        uploaddic['solutionList'] = scenarioobj

        if supscenarioobj:
            for i in supscenarioobj:
                path_testdata = i["test_file"]
                path_module = i["model_file"]
                path_traindata = i["training_file"]
                path_factsheet = i["factsheet_file"]
                path_outliersdata = i['outlier_data_file']
                soulutionType = i['solution_type']
                weights_metrics = i['weights_metrics']
                weights_pillars = i['weights_pillars']
                try:
                    mappings_config = save_files_return_paths(
                        i['metrics_mapping_file'])[0]
                except:
                    mappings_config = os.path.join(
                        BASE_DIR, 'apis/TestValues/Mappings/default.json')

            path_module, path_traindata, path_testdata, path_factsheet, path_outliersdata, weights_metrics, weights_pillars = save_files_return_paths(
                path_module, path_traindata, path_testdata, path_factsheet, path_outliersdata, weights_metrics, weights_pillars)
            if (soulutionType == 'supervised'):
                resultSuper = finalScore_supervised(
                    path_module, path_traindata, path_testdata, path_factsheet, mappings_config, weights_metrics, weights_pillars)

                uploaddic['fairness_score'] = resultSuper['Pillarscores']['Fairnessscore']
                try:
                    uploaddic['methodology_score'] = resultSuper['Pillarscores']['Accountabilityscore']
                except:
                    uploaddic['accountability_score'] = resultSuper['Pillarscores']['Accountabilityscore']

                uploaddic['trust_score'] = resultSuper['Trustscore']

                uploaddic['explainability_score'] = resultSuper['Pillarscores']['Explainabilityscore']
                uploaddic['robustness_score'] = resultSuper['Pillarscores']['Robustnessscore']
                uploaddic['underfitting'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Underfittingscore']
                uploaddic['overfitting'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Overfittingscore']
                uploaddic['statistical_parity_difference'] = resultSuper['Metricscores'][
                    'Metricscores']['Fairnessscore']['Statisticalparitydifferencescore']
                uploaddic['equal_opportunity_difference'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Equalopportunityscore']
                uploaddic['average_odds_difference'] = resultSuper['Metricscores'][
                    'Metricscores']['Fairnessscore']['Averageoddsdifferencescore']
                uploaddic['disparate_impact'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Disparateimpactscore']
                uploaddic['class_balance'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Classbalancescore']
                uploaddic['algorithm_class'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Algorithmclassscore']
                uploaddic['correlated_features'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Correlatedfeaturesscore']
                uploaddic['model_size'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Modelsizescore']
                uploaddic['feature_relevance'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Featurerevelancescore']
                uploaddic['confidence_score'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Confidencescore']
                uploaddic['clique_method'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Cliquemethodscore']
                uploaddic['loss_sensitivity'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Losssensitivityscore']
                uploaddic['clever_score'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Cleverscore']
                uploaddic['er_fast_gradient_attack'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Erfastgradientattack']
                uploaddic['er_carlini_wagner_attack'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Ercarliniwagnerscore']
                uploaddic['er_deepfool_attack'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Erdeepfoolattackscore']
                uploaddic['normalization'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Normalizationscore']
                uploaddic['missing_data'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Missingdatascore']
                uploaddic['regularization'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Regularizationscore']
                uploaddic['train_test_split'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Traintestsplitscore']
                uploaddic['factsheet_completeness'] = resultSuper['Metricscores'][
                    'Metricscores']['Accountabilityscore']['Factsheecompletnessscore']

        if unsupscenarioobj:
            for i in unsupscenarioobj:
                path_testdata = i["test_file"]
                path_module = i["model_file"]
                path_traindata = i["training_file"]
                path_factsheet = i["factsheet_file"]
                path_outliersdata = i['outlier_data_file']
                soulutionType = i['solution_type']
                weights_metrics = i['weights_metrics']
                weights_pillars = i['weights_pillars']
                try:
                    mappings_config = save_files_return_paths(
                        i['metrics_mapping_file'])[0]
                except:
                    mappings_config = os.path.join(
                        BASE_DIR, 'apis/TestValues/Mappings/default.json')
            path_module, path_traindata, path_testdata, path_factsheet, path_outliersdata, weights_metrics, weights_pillars = save_files_return_paths(
                path_module, path_traindata, path_testdata, path_factsheet, path_outliersdata, weights_metrics, weights_pillars)

            if (soulutionType == 'unsupervised'):
                resultUnsuper = unsupervised_FinalScore(
                    path_module, path_traindata, path_testdata, path_outliersdata, path_factsheet, mappings_config, weights_metrics, weights_pillars)

                uploaddic['unsupervised_fairness_score'] = resultUnsuper['Pillarscores']['Fairnessscore']
                try:
                    uploaddic['unsupervised_methodology_score'] = round(resultUnsuper['Pillarscores']['Accountabilityscore'], 2)
                except:
                    uploaddic['accountability_score'] = resultUnsuper['Pillarscores']['Accountabilityscore']

                uploaddic['unsupervised_trust_score'] = resultUnsuper['Trustscore']

                uploaddic['unsupervised_explainability_score'] = resultUnsuper['Pillarscores']['Explainabilityscore']
                uploaddic['unsupervised_robustness_score'] = resultUnsuper['Pillarscores']['Robustnessscore']

                uploaddic['unsupervised_underfitting'] = resultUnsuper['Metricscores']['Metricscores']['Fairnessscore']['Underfittingscore']
                uploaddic['unsupervised_overfitting'] = resultUnsuper['Metricscores']['Metricscores']['Fairnessscore']['Overfittingscore']
                uploaddic['unsupervised_statistical_parity_difference'] = resultUnsuper[
                    'Metricscores']['Metricscores']['Fairnessscore']['Statisticalparitydifferencescore']
                uploaddic['unsupervised_disparate_impact'] = resultUnsuper['Metricscores']['Metricscores']['Fairnessscore']['Disparateimpactscore']
                uploaddic['unsupervised_correlated_features'] = resultUnsuper['Metricscores'][
                    'Metricscores']['Explainabilityscore']['Correlatedfeaturesscore']
                uploaddic['unsupervised_permutation_importance'] = resultUnsuper['Metricscores'][
                    'Metricscores']['Explainabilityscore']['Permutationfeatureimportancescore']
                uploaddic['unsupervised_model_size'] = resultUnsuper['Metricscores']['Metricscores']['Explainabilityscore']['Modelsizescore']
                uploaddic['unsupervised_clever_score'] = resultUnsuper['Metricscores']['Metricscores']['Robustnessscore']['Cleverscore']
                uploaddic['unsupervised_normalization'] = resultUnsuper['Metricscores'][
                    'Metricscores']['Accountabilityscore']['Normalizationscore']
                uploaddic['unsupervised_missing_data'] = resultUnsuper['Metricscores']['Metricscores']['Accountabilityscore']['Missingdatascore']
                uploaddic['unsupervised_regularization'] = resultUnsuper['Metricscores'][
                    'Metricscores']['Accountabilityscore']['Regularizationscore']
                uploaddic['unsupervised_train_test_split'] = resultUnsuper['Metricscores'][
                    'Metricscores']['Accountabilityscore']['Traintestsplitscore']
                uploaddic['unsupervised_factsheet_completeness'] = resultUnsuper['Metricscores'][
                    'Metricscores']['Accountabilityscore']['Factsheecompletnessscore']
        try:
            if (supscenarioobj or unsupscenarioobj):
                print('retured:data', uploaddic)
                a= Response(uploaddic, status=200)
                print("RESPONSE VAL: ",a)
                return a
        except:
            print("ERROR OCURRED")
        if (not supscenarioobj) and (not unsupscenarioobj):
            return Response('No Solution', status=409)

    def post(self, request):
        return Response("Successfully add!")
