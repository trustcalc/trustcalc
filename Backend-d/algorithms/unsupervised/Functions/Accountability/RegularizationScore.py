def regularization_score(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    NOT_SPECIFIED="not specified"
    import collections
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    import numpy as np

    def regularization_metric(factsheet):
        if "methodology" in factsheet and "regularization" in factsheet["methodology"]:
            return factsheet["methodology"]["regularization"]
        else:
            return NOT_SPECIFIED
    score = 1
    regularization = regularization_metric(factsheet)
    properties = {"dep" :info('Depends on','Factsheet'),
        "regularization_technique": info("Regularization technique", regularization)}

    if regularization == "elasticnet_regression":
        score = 5
    elif regularization == "lasso_regression" or regularization == "lasso_regression":
        score = 4
    elif regularization == "Other":
        score = 3
    elif regularization == NOT_SPECIFIED:
        score = 1
    else:
        score = 1
    return result(score=score, properties=properties)


########################################TEST VALUES#############################################
"""import pandas as pd

train=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/train.csv"
test=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/test.csv"
outliers=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/outliers.csv"

model=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/model.joblib"
factsheet=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/factsheet.json"

mapping_metrics_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
weights_metrics_feault=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_metrics_default.json"
weights_pillars_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_pillars_default.json"

a=regularization_score(model,train,test,factsheet,mapping_metrics_default)
print(a)"""
