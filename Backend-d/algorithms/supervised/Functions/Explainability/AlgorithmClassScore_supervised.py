def get_algorithm_class_score_supervised(model=not None, training_dataset=None, test_dataset=None, factsheet= None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    import collections, pandas, numpy as np
    info,result = collections.namedtuple('info', 'description value'),collections.namedtuple('result', 'score properties')
    model,clf_type_score=pandas.read_pickle(model),pandas.read_json(mappings)["explainability"]["score_algorithm_class"]["clf_type_score"]["value"]

    clf_name = type(model).__name__
    exp_score = clf_type_score.get(clf_name,np.nan)
    properties= {"dep" :info('Depends on','Model'),
        "clf_name": info("model type",clf_name)}
    return  result(score=exp_score, properties=properties)

"""########################################TEST VALUES#############################################
model,mappings=r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
print(get_algorithm_class_score_supervised(model=model,mappings=mappings))"""

