def analyse(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None,target_column=None, outliers_data=not None, thresholds=None, outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    """Triggers the fairness analysis and in a first step all fairness metrics get computed.
    In a second step, the scores for the fairness metrics are then created from
    mapping every metric value to a respective score.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        test_dataset: pd.DataFrame containing the used test data.
        factsheet: json document containing all information about the particular solution.
        config: Config file containing the threshold values for the metrics.

    Returns:
        Returns a result object containing all metric scores
        and matching properties for every metric

    """
    import numpy as np
    import keras
    import collections
    np.random.seed(0)

    import sys
    import os,sys
    sys.path.append(r"Functions_Trust")
    sys.path.append(r"Functions_Trust\Backend")
    sys.path.append(r"Functions_Trust\Backend\algorithms")
    sys.path.append(r"Functions_Trust\Backend\algorithms\unsupervised")
    sys.path.append(r"Functions_Trust\Backend\algorithms\unsupervised\Functions")
    sys.path.append(r"Functions_Trust\Backend\algorithms\unsupervised\Functions\Fairness")


    try:
        from .StatisticalParityDifferenceScore import get_statistical_parity_difference_score_unsupervised
        from .OverfittingScore import overfitting_score
        from .UnderfittingScore import underfitting_score
        from .DisparateImpactScore import disparate_impact_score
    except:
        from unsupervised.Functions.Fairness.UnderfittingScore import underfitting_score
        from unsupervised.Functions.Fairness.StatisticalParityDifferenceScore import get_statistical_parity_difference_score_unsupervised
        from unsupervised.Functions.Fairness.OverfittingScore import overfitting_score

        from unsupervised.Functions.Fairness.DisparateImpactScore import disparate_impact_score

    result = collections.namedtuple('result', 'score properties')
    print_details = True
    outlier_percentage = 0.1

    def isKerasAutoencoder(model):
        return isinstance(model, keras.engine.functional.Functional)

    def get_threshold_mse_iqr(autoencoder,train_data):
        train_predicted = autoencoder.predict(train_data)
        mse = np.mean(np.power(train_data - train_predicted, 2), axis=1)
        iqr = np.quantile(mse,0.75) - np.quantile(mse, 0.25) # interquartile range
        up_bound = np.quantile(mse,0.75) + 1.5*iqr
        bottom_bound = np.quantile(mse,0.25) - 1.5*iqr
        thres = [up_bound,bottom_bound]
        return thres

    if isKerasAutoencoder(model):
        print("train size: ", training_dataset.shape)
        print("test size: ", test_dataset.shape)
        print("outliers size: ", outliers_dataset.shape)
        print("model size: ", model.summary())

        outlier_thresh = get_threshold_mse_iqr(model, training_dataset)
    else:
        outlier_thresh = 0

    import pandas as pd
    mappings2=pd.read_json(mappings)
    statistical_parity_difference_thresholds = mappings2["fairness"]["score_statistical_parity_difference"]["thresholds"]["value"]
    overfitting_thresholds = mappings2["fairness"]["score_overfitting"]["thresholds"]["value"]
    underfitting_thresholds = mappings2["fairness"]["score_underfitting"]["thresholds"]["value"]
    disparate_impact_thresholds = mappings2["fairness"]["score_disparate_impact"]["thresholds"]["value"]

    output = dict(
        statistical_parity_difference = get_statistical_parity_difference_score_unsupervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        overfitting = overfitting_score(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=0, outlier_percentage=0.1, high_cor=high_cor,print_details=print_details),
        underfitting = underfitting_score(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=0, outlier_percentage=0.1, high_cor=high_cor,print_details=print_details),
        disparate_impact=disparate_impact_score(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details)
    )
    
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return result(score=scores, properties=properties)

"""
########################################TEST VALUES#############################################
train=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/train.csv"
test=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/test.csv"
outliers=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/outliers.csv"

model=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/model.joblib"
factsheet=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/factsheet.json"

mapping_metrics_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
weights_metrics_feault=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_metrics_default.json"
weights_pillars_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_pillars_default.json"

a=analyse(model, train, test, outliers, factsheet, mapping_metrics_default)
print(a)
"""