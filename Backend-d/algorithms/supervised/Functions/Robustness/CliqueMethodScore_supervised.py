def get_clique_method_supervised(model=None, training_dataset=None, test_dataset=not None, factsheet=not None, mappings=not None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None, print_details=True):
    import collections
    import pandas
    import numpy as np
    from art.estimators.classification import SklearnClassifier
    from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
    from sklearn.base import BaseEstimator

    info, result = collections.namedtuple(
        'info', 'description value'), collections.namedtuple('result', 'score properties')
    test_data, mappings, factsheet = pandas.read_csv(
        test_dataset), pandas.read_json(mappings), pandas.read_json(factsheet)

    if not thresholds:
        thresholds = mappings["robustness"]["score_clique_method"]["thresholds"]["value"]

    default_map = mappings["robustness"]

    if thresholds == default_map["score_clique_method"]["thresholds"]["value"]:
        if "scores" in factsheet.keys() and "properties" in factsheet.keys():
            score = factsheet["scores"]["robustness"]["clique_method"]
            properties = factsheet["properties"]["robustness"]["clique_method"]
            return result(score=score, properties=properties)

    try:
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1:]
        classifier = SklearnClassifier(BaseEstimator)
        rt = RobustnessVerificationTreeModelsCliqueMethod(
            classifier=classifier, verbose=True)

        bound, error = rt.verify(x=X_test.to_numpy()[100:103], y=y_test[100:103].to_numpy(), eps_init=0.5, norm=1,
                                 nb_search_steps=5, max_clique=2, max_level=2)
        score = np.digitize(bound, thresholds) + 1
        return result(score=int(score), properties={
            "error_bound": info("Average error bound", "{:.2f}".format(bound)),
            "error": info("Error", "{:.1f}".format(error)),
            "depends_on": info("Depends on", "Model")
        })
    except:
        return result(score=np.nan, properties={"non_computable": info("Non Computable Because", "Can only be calculated on Tree-Based models.")})


"""########################################TEST VALUES#############################################
test,model, mappings,factsheet=r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend\algorithms\supervised\Mapping&Weights\mapping_metrics_default.json",r"Backend/algorithms/supervised/TestValues/factsheet.json"
print(get_clique_method_supervised(test_dataset=test, model=model,mappings=mappings,factsheet=factsheet))"""
