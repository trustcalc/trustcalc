def get_explainability_score_supervised(model=not None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    import sys,pandas
    sys.path.extend([r"Backend",r"Backend/algorithms",r"Backend/algorithms/supervised", r"Backend/algorithms/supervised/Functions",r"Backend/algorithms/supervised/Functions/Explainability"])
    sys.path.append(r"Backend/algorithms/supervised/Functions")
    from ..helpers_supervised import import_functions_from_folder
    """info,result,explainability_functions=import_functions_from_folder(['Explainability'])
    get_algorithm_class_score = explainability_functions['algorithmclassscoresupervised']
    get_correlated_features_score = explainability_functions['correlatedfeaturesscoresupervised']
    get_feature_relevance_score = explainability_functions['featurerelevancescoresupervised']
    get_model_size_score = explainability_functions['modelsizescoresupervised']
"""
    import collections
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    from  algorithms.supervised.Functions.Explainability.AlgorithmClassScore_supervised import get_algorithm_class_score_supervised
    from  algorithms.supervised.Functions.Explainability.CorrelatedFeaturesScore_supervised import get_correlated_features_score_supervised
    from  algorithms.supervised.Functions.Explainability.FeatureRelevanceScore_supervised import get_feature_relevance_score_supervised
    from  algorithms.supervised.Functions.Explainability.ModelSizeScore_supervised import get_model_size_score_supervised


    #convert path data to values
    factsheet2,mappings2=pandas.read_json(factsheet), pandas.read_json(mappings)
    target_column,high_cor = factsheet2["general"].get("target_column"), mappings2["explainability"]["score_correlated_features"]["high_cor"]["value"]
    
    output = dict(
        algorithm_class     = get_algorithm_class_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        correlated_features = get_correlated_features_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        model_size          = get_model_size_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        feature_relevance   = get_feature_relevance_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details )
                 )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)

"""########################################TEST VALUES#############################################
train=r"Backend/algorithms/supervised/TestValues/train.csv"
test=r"Backend/algorithms/supervised/TestValues/test.csv"
model=r"Backend/algorithms/supervised/TestValues/model.pkl"
factsheet=r"Backend/algorithms\supervised/TestValues/factsheet.json"
mapping_metrics_default=r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
print(get_explainability_score_supervised(model=model,training_dataset=train,test_dataset=test,factsheet=factsheet,mappings=mapping_metrics_default))"""





