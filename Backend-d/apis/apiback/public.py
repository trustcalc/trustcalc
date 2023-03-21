from algorithms.supervised.Functions.Fairness.ClassBalanceScore_supervised import get_class_balance_score_supervised
import sys
import django
import pandas as pd
import json

sys.path.extend([r"Backend", r"Backend/apis"])


def save_files_return_paths(*args):
    fs = django.core.files.storage.FileSystemStorage()
    paths = []
    for arg in args:
        if arg is None or arg == 'undefined':
            paths.append(None)
        else:
            paths.append(fs.path(arg))
    return tuple(paths)


# def analyse_fairness(model, training_dataset, test_dataset, factsheet, config):
#     import numpy as np
#     np.random.seed(0)
#     from apis.FourPillars.Fairness.Always.ClassBalance.ClassBalanceScore import class_balance_score
#     from apis.FourPillars.Fairness.Conditional.DisparateImpact.DisparateImpactScore import disparate_impact_score
#     from apis.FourPillars.Fairness.Conditional.AverageOddsDifference.AverageOddsDifferenceScore import average_odds_difference_score
#     from apis.FourPillars.Fairness.Conditional.EqualOpportunityDifference.EqualOpportunityDifferenceScore import equal_opportunity_difference_score
#     from apis.FourPillars.Fairness.Conditional.StatisticalParityDifference.StatisticalParityDifferenceScore import statistical_parity_difference_score
#     from apis.FourPillars.Fairness.Always.Underfitting.UnderfittingScore import underfitting_score
#     from apis.FourPillars.Fairness.Always.Overfitting.OverfittingScore import overfitting_score

#     model = pd.read_pickle(model)
#     training_dataset = pd.read_csv(training_dataset)
#     test_dataset = pd.read_csv(test_dataset)

#     with open(config, 'r') as f:
#         config = json.loads(f.read())
#     with open(factsheet, 'r') as g:
#         factsheet = json.loads(g.read())

#     import collections
#     result = collections.namedtuple('result', 'score properties')

#     statistical_parity_difference_thresholds = config[
#         "score_statistical_parity_difference"]["thresholds"]["value"]
#     overfitting_thresholds = config["score_overfitting"]["thresholds"]["value"]
#     underfitting_thresholds = config["score_underfitting"]["thresholds"]["value"]
#     equal_opportunity_difference_thresholds = config[
#         "score_equal_opportunity_difference"]["thresholds"]["value"]
#     average_odds_difference_thresholds = config["score_average_odds_difference"]["thresholds"]["value"]
#     disparate_impact_thresholds = config["score_disparate_impact"]["thresholds"]["value"]

#     output = dict(
#         underfitting=underfitting_score(
#             model, training_dataset, test_dataset, factsheet, underfitting_thresholds),
#         overfitting=overfitting_score(
#             model, training_dataset, test_dataset, factsheet, overfitting_thresholds),
#         statistical_parity_difference=statistical_parity_difference_score(
#             model, training_dataset, factsheet, statistical_parity_difference_thresholds),
#         equal_opportunity_difference=equal_opportunity_difference_score(
#             model, test_dataset, factsheet, equal_opportunity_difference_thresholds),
#         average_odds_difference=average_odds_difference_score(
#             model, test_dataset, factsheet, average_odds_difference_thresholds),
#         disparate_impact=disparate_impact_score(
#             model, test_dataset, factsheet, disparate_impact_thresholds),
#         class_balance=class_balance_score(training_dataset, factsheet)
#     )

#     scores = dict((k, v.score) for k, v in output.items())
#     properties = dict((k, v.properties) for k, v in output.items())

#     return result(score=scores, properties=properties)


# def analyse_explainability(clf, train_data, test_data, config, factsheet):
#     import pandas as pd
#     import collections
#     from apis.FourPillars.Explainability.AlgorithmClass.AlgorithmClassScore import algorithm_class_score
#     from apis.FourPillars.Explainability.CorrelatedFeatures.CorrelatedFeaturesScore import correlated_features_score
#     from apis.FourPillars.Explainability.FeatureRelevance.FeatureRelevanceScore import feature_relevance_score
#     from apis.FourPillars.Explainability.ModelSize.ModelSizeScore import model_size_score

#     result = collections.namedtuple('result', 'score properties')
#     info = collections.namedtuple('info', 'description value')

#     # convert path data to values
#     clf = pd.read_pickle(clf)
#     train_data = pd.read_csv(train_data)
#     test_data = pd.read_csv(test_data)
#     config = pd.read_json(config)

#     factsheet = pd.read_json(factsheet)

#     target_column = factsheet["general"].get("target_column")
#     clf_type_score = config["score_algorithm_class"]["clf_type_score"]["value"]
#     ms_thresholds = config["score_model_size"]["thresholds"]["value"]
#     cf_thresholds = config["score_correlated_features"]["thresholds"]["value"]
#     high_cor = config["score_correlated_features"]["high_cor"]["value"]
#     fr_thresholds = config["score_feature_relevance"]["thresholds"]["value"]
#     threshold_outlier = config["score_feature_relevance"]["threshold_outlier"]["value"]
#     penalty_outlier = config["score_feature_relevance"]["penalty_outlier"]["value"]

#     output = dict(
#         algorithm_class=algorithm_class_score(clf, clf_type_score),
#         correlated_features=correlated_features_score(
#             train_data, test_data, thresholds=cf_thresholds, target_column=target_column, high_cor=high_cor),
#         model_size=model_size_score(train_data, ms_thresholds),
#         feature_relevance=feature_relevance_score(clf, train_data, target_column=target_column, thresholds=fr_thresholds,
#                                                   threshold_outlier=threshold_outlier, penalty_outlier=penalty_outlier)
#     )

#     scores = dict((k, v.score) for k, v in output.items())
#     properties = dict((k, v.properties) for k, v in output.items())

#     return result(score=scores, properties=properties)


# def analyse_robustness(model, train_data, test_data, config, factsheet):
#     import collections

#     result = collections.namedtuple('result', 'score properties')
#     from apis.FourPillars.Robustness.ConfidenceScore.ConfidenceScore import confidence_score
#     from apis.FourPillars.Robustness.CleverScore.CleverScore import clever_score
#     from apis.FourPillars.Robustness.CliqueMethod.CliqueMethodScore import clique_method
#     from apis.FourPillars.Robustness.LossSensitivity.LossSensitivityScore import loss_sensitivity_score
#     from apis.FourPillars.Robustness.ERFastGradientMethod.FastGradientAttackScore import fast_gradient_attack_score
#     from apis.FourPillars.Robustness.ERCWAttack.CarliWagnerAttackScore import carlini_wagner_attack_score
#     from apis.FourPillars.Robustness.ERDeepFool.DeepFoolAttackScore import deepfool_attack_score

#     model = pd.read_pickle(model)
#     train_data = pd.read_csv(train_data)
#     test_data = pd.read_csv(test_data)
#     config = pd.read_json(config)

#     factsheet = pd.read_json(factsheet)

#     clique_method_thresholds = config["score_clique_method"]["thresholds"]["value"]
#     print("clique_method_thresholds:", clique_method_thresholds)
#     clever_score_thresholds = config["score_clever_score"]["thresholds"]["value"]
#     loss_sensitivity_thresholds = config["score_loss_sensitivity"]["thresholds"]["value"]
#     confidence_score_thresholds = config["score_confidence_score"]["thresholds"]["value"]
#     fsg_attack_thresholds = config["score_fast_gradient_attack"]["thresholds"]["value"]
#     cw_attack_thresholds = config["score_carlini_wagner_attack"]["thresholds"]["value"]
#     deepfool_thresholds = config["score_carlini_wagner_attack"]["thresholds"]["value"]

#     output = dict(
#         confidence_score=confidence_score(
#             model, train_data, test_data, confidence_score_thresholds),
#         clique_method=clique_method(
#             model, train_data, test_data, clique_method_thresholds, factsheet),
#         loss_sensitivity=loss_sensitivity_score(
#             model, train_data, test_data, loss_sensitivity_thresholds),
#         clever_score=clever_score(
#             model, train_data, test_data, clever_score_thresholds),
#         er_fast_gradient_attack=fast_gradient_attack_score(
#             model, train_data, test_data, fsg_attack_thresholds),
#         er_carlini_wagner_attack=carlini_wagner_attack_score(
#             model, train_data, test_data, cw_attack_thresholds),
#         er_deepfool_attack=deepfool_attack_score(
#             model, train_data, test_data, deepfool_thresholds)
#     )
#     scores = dict((k, v.score) for k, v in output.items())
#     properties = dict((k, v.properties) for k, v in output.items())

#     return result(score=scores, properties=properties)


# def analyse_methodology(model, training_dataset, test_dataset, factsheet, methodology_config):
#     import collections
#     import json
#     import collections
#     from apis.FourPillars.Accountability.FactSheetCompletness.FactSheetCompletnessScore import get_factsheet_completeness_score
#     from apis.FourPillars.Accountability.MissingData.MissingDataScore import missing_data_score
#     from apis.FourPillars.Accountability.Normalization.NormalizationScore import normalization_score
#     from apis.FourPillars.Accountability.Regularization.RegularizationScore import regularization_score
#     from apis.FourPillars.Accountability.TrainTestSplit.TrainTestSplitScore import train_test_split_score
#     result = collections.namedtuple('result', 'score properties')

#     model = pd.read_pickle(model)
#     training_dataset = pd.read_csv(training_dataset)
#     test_dataset = pd.read_csv(test_dataset)

#     with open(methodology_config, 'r') as f:
#         methodology_config = json.loads(f.read())
#     with open(factsheet, 'r') as g:
#         factsheet = json.loads(g.read())
#     normalization_mapping = methodology_config["score_normalization"]["mappings"]["value"]
#     missing_data_mapping = methodology_config["score_missing_data"]["mappings"]["value"]['no_null_values']
#     train_test_split_mapping = methodology_config["score_train_test_split"]["mappings"]["value"]['50-60 95-97']

#     output = dict(
#         normalization=normalization_score(
#             model, training_dataset, test_dataset, factsheet, normalization_mapping),
#         missing_data=missing_data_score(
#             model, training_dataset, test_dataset, factsheet, missing_data_mapping),
#         regularization=regularization_score(
#             model, training_dataset, test_dataset, factsheet, methodology_config),
#         train_test_split=train_test_split_score(
#             model, training_dataset, test_dataset, factsheet, train_test_split_mapping),
#         factsheet_completeness=get_factsheet_completeness_score(
#             model, training_dataset, test_dataset, factsheet, methodology_config)
#     )

#     scores = dict((k, v.score) for k, v in output.items())
#     properties = dict((k, v.properties) for k, v in output.items())

#     return result(score=scores, properties=properties)
