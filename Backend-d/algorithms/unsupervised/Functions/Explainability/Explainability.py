def analyse(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None,target_column=None, outliers_data=not None, thresholds=None, outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    import numpy as np
    import collections
    import keras,sys
    sys.path.append(r'Backend/algorithms/unsupervised/Functions/Explainability')
    from .CorrelatedFeaturesScore import correlated_features_score
    from .ModelSizeScore import model_size_score
    from .PermutationFeatureScore import permutation_feature_importance_score

    result = collections.namedtuple('result', 'score properties')
    info = collections.namedtuple('info', 'description value')

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
    import pandas as pd
    mappings2=pd.read_json(mappings)
    ms_thresholds = mappings2["explainability"]["score_model_size"]["thresholds"]["value"]
    cf_thresholds = mappings2["explainability"]["score_correlated_features"]["thresholds"]["value"]
    try:
        pfi_thresholds = mappings2["explainability"]["score_permutation_feature_importance"]["thresholds"]["value"]
    except:
        pfi_thresholds=[
                    0.2,
                    0.15,
                    0.1,
                    0.05
                ]
    high_cor = mappings2["explainability"]["score_correlated_features"]["high_cor"]["value"]

    print_details = True

    if isKerasAutoencoder(model):
        outlier_thresholds = get_threshold_mse_iqr(model, training_dataset)
    else:
        outlier_thresholds = 0

    output = dict(
        
        correlated_features = correlated_features_score(training_dataset=training_dataset, test_dataset=test_dataset, thresholds=cf_thresholds, target_column=None, high_cor=high_cor),
        model_size          = model_size_score(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        permutation_feature_importance   = permutation_feature_importance_score(model=model,training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, thresholds = pfi_thresholds, outlier_thresholds=outlier_thresholds)
    )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return result(score=scores, properties=properties)


"""
########################################TEST VALUES#############################################
import pandas as pd

train=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/train.csv"
test=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/test.csv"
outliers=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/outliers.csv"

model=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/model.joblib"
factsheet=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/factsheet.json"

mapping_metrics_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
weights_metrics_feault=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_metrics_default.json"
weights_pillars_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_pillars_default.json"


a=analyse(model, train, test, outliers, factsheet, mapping_metrics_default)
print(a)
"""