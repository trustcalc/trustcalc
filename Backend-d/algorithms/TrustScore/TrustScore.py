import json
import math
import numpy
import sys
import collections
import os
# from config import *

sys.path.append(r"Backend")
sys.path.append(r"Backend/algorithms")
sys.path.append(r"Backend/algorithms/supervised")
sys.path.append(r"Backend/algorithms/supervised/Functions")
try:
    from ..supervised.Functions.Fairness.FarinessScore_supervised import get_fairness_score_supervised as analyse_fairness_supervised
    from ..supervised.Functions.Explainability.ExplainabilityScore_supervised import get_explainability_score_supervised as analyse_explainability_supervised
    from ..supervised.Functions.Robustness.Robustness_supervised import get_robustness_score_supervised as analyse_robustness_supervised
    from ..supervised.Functions.Accountability.AccountabilityScore_supervised import get_accountability_score_supervised as analyse_accountabiltiy_supervised
except:
    from supervised.Functions.Fairness.FarinessScore_supervised import get_fairness_score_supervised as analyse_fairness_supervised
    from supervised.Functions.Explainability.ExplainabilityScore_supervised import get_explainability_score_supervised as analyse_explainability_supervised
    from supervised.Functions.Robustness.Robustness_supervised import get_robustness_score_supervised as analyse_robustness_supervised
    from supervised.Functions.Accountability.AccountabilityScore_supervised import get_accountability_score_supervised as analyse_accountabiltiy_supervised


sys.path.append(r"Backend")
sys.path.append(r"Backend/algorithms")
sys.path.append(r"Backend/algorithms/unsupervised")
sys.path.append(r"Backend/algorithms/unsupervised/Functions")
try:
    from algorithms.unsupervised.Functions.Fairness.Fairness import analyse as analyse_fairness_supervised_unsupervised
    from algorithms.unsupervised.Functions.Explainability.Explainability import analyse as analyse_explainability_supervised_unsupervised
    from algorithms.unsupervised.Functions.Robustness.Robustness import analyse as analyse_robustness_supervised_unsupervised
    from algorithms.unsupervised.Functions.Accountability.Accountability import analyse as analyse_accountability_unsupervised
except:
    from unsupervised.Functions.Fairness.Fairness import analyse as analyse_fairness_supervised_unsupervised
    from unsupervised.Functions.Explainability.Explainability import analyse as analyse_explainability_supervised_unsupervised
    from unsupervised.Functions.Robustness.Robustness import analyse as analyse_robustness_supervised_unsupervised
    from unsupervised.Functions.Accountability.Accountability import analyse as analyse_accountability_unsupervised

info = collections.namedtuple('info', 'description value')
result = collections.namedtuple('result', 'score properties')


def write_into_factsheet(new_factsheet, solution_set_path):
    factsheet_path = os.path.join(solution_set_path, FACTSHEET_NAME)
    with open(factsheet_path, 'w') as outfile:
        json.dump(new_factsheet, outfile, indent=4)


def trustinAI_scores_api(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, penalty_outlier=None, outlier_percentage=None, high_cor=None, print_details=None):
    return


def trusting_AI_scores_supervised(model=not None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=not None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, penalty_outlier=None, outlier_percentage=None, high_cor=None, print_details=None):
    print('reached here')
    output = dict(
        fairness=analyse_fairness_supervised(model, training_dataset, test_dataset, factsheet, mappings, target_column,
                                             outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details),
        explainability=analyse_explainability_supervised(model, training_dataset, test_dataset, factsheet, mappings,
                                                         target_column, outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details),
        robustness=analyse_robustness_supervised(model, training_dataset, test_dataset, factsheet, mappings,
                                                 target_column, outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details),
        accountability=analyse_accountability_unsupervised(model, training_dataset, test_dataset, factsheet, mappings,
                                                           target_column, outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details)
    )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return result(score=scores, properties=properties)


def calculate_pillar_scores(scores, weights, weights_pillars):
    metric_scores = {}

    for pillar in range(4):
        weighted_scores = [scores[pillar][x] * config[x]
                           for x in scores[pillar].keys()]
        metric_scores[pillar] = sum(weighted_scores)
    pillar_scores = {p: w * sum([metric_scores[m] for m in metric_scores if m in weights[p]])
                     for p, w in weights_pillars.items()}
    return pillar_scores


def trusting_AI_scores_unsupervised(model=not None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=not None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, penalty_outlier=None, outlier_percentage=None, high_cor=None, print_details=None):
    print('reached here')
    output = dict(
        fairness=analyse_fairness_supervised_unsupervised(model, training_dataset, test_dataset, factsheet, mappings, target_column,
                                             outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details),
        explainability=analyse_explainability_supervised_unsupervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings, target_column=target_column,
                                                                      outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor, print_details=print_details),
        robustness=analyse_robustness_supervised_unsupervised(model, training_dataset, test_dataset, factsheet, mappings,
                                                              target_column, outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details),
        accountability=analyse_accountability_unsupervised(model, training_dataset, test_dataset, factsheet, mappings,
                                                           target_column, outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details)
    )

    
    for mainkey in output:
        print("FAIRNESS ITEM: ",output[mainkey])
        for key in output[mainkey].score:
            print('key => ', key, output[mainkey].score[key])
            if numpy.isnan(output[mainkey].score[key]):
                output[mainkey].score[key]=1
            
    print("FAIRNESS WORKING: ",output["fairness"]),
    print("ROBUSTNESS WORKING: ",output["robustness"]),
    print("ACCOUNTABILITY WORKING: ",output["accountability"]),
    print("EXPLAINABILITY WORKING: ",output["explainability"])


    try:
        scores = dict((k, v.score) for k, v in output.items())
    except:
        print("SCORES NOT WORKING")
    try:
        properties = dict((k, v.properties) for k, v in output.items())
    except:
        print("PROPERTIES NOT WORKING")
    print("EVERYTHING FINE")
    print("RESULT",result(score=scores, properties=properties))
    return result(score=scores, properties=properties)


def calculate_pillar_scores(scores, weights, weights_pillars):
    metric_scores = {}

    for pillar in range(4):
        weighted_scores = [scores[pillar][x] * config[x]
                           for x in scores[pillar].keys()]
        metric_scores[pillar] = sum(weighted_scores)
    pillar_scores = {p: w * sum([metric_scores[m] for m in metric_scores if m in weights[p]])
                     for p, w in weights_pillars.items()}
    return pillar_scores



def trusting_AI_scores_unsupervised2(model=not None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=not None, target_column=not None, outliers_data=not None, thresholds=not None, outlier_thresholds=not None, penalty_outlier=None, outlier_percentage=not None, high_cor=not None, print_details=True):
    output = dict(
        #fairness=analyse_fairness_supervised_unsupervised(model, training_dataset, test_dataset, factsheet, mappings,
                                                          #target_column, outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details),
        explainability=analyse_explainability_supervised_unsupervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings, target_column=target_column,
                                                                      outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor, print_details=print_details),
        robustness=analyse_robustness_supervised_unsupervised(model, training_dataset, test_dataset, factsheet, mappings,
                                                              target_column, outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details),
        accountability=analyse_accountability_unsupervised(model, training_dataset, test_dataset, factsheet, mappings,
                                                           target_column, outliers_data, thresholds, outlier_thresholds, outlier_percentage, high_cor, print_details)
    )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return result(score=scores, properties=properties)


def get_trust_score(final_score, config):
    if sum(config.values()) == 0:
        return 0
    return round(numpy.nansum(list(map(lambda x: final_score[x] * config[x], final_score.keys())))/numpy.sum(list(config.values())), 1)


"""
train=r"Backend/algorithms/supervised/TestValues/train.csv"
test=r"Backend/algorithms/supervised/TestValues/test.csv"
model=r"Backend/algorithms/supervised/TestValues/model.pkl"
factsheet=r"Backend/algorithms/supervised/TestValues/factsheet.json"
mapping_metrics_default=r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
outliers=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/outliers.csv"
config_weight=r"Backend\algorithms\supervised\Mapping&Weights\weights_metrics_default.json"
weights_metrics_default=r"Backend/algorithms/supervised/Mapping&Weights/weights_metrics_default.json"
weights_pillars_default=r"Backend/algorithms/supervised/Mapping&Weights/weights_pillars_default.json"

a=trusting_AI_scores_supervised(model=model,training_dataset=train,test_dataset=test,factsheet=factsheet,mappings=mapping_metrics_default,outliers_data=outliers)

c=trusting_AI_scores_unsupervised(model=model,training_dataset=train,test_dataset=test,factsheet=factsheet,mappings=mapping_metrics_default,outliers_data=outliers)
"""
