def get_clever_score_supervised(model=not None, training_dataset=None, test_dataset=not None, factsheet=None, mappings=not None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None, print_details=None):
    import collections
    import pandas
    import numpy as np
    from art.estimators.classification import KerasClassifier
    from art.metrics import clever_u

    info, result = collections.namedtuple(
        'info', 'description value'), collections.namedtuple('result', 'score properties')
    model = pandas.read_pickle(model)
    test_data = pandas.read_csv(test_dataset)
    mappings = pandas.read_json(mappings)

    if not thresholds:
        thresholds = mappings["robustness"]["score_clever_score"]["thresholds"]["value"]

    try:
        X_test = test_data.iloc[:, :-1]
        classifier = KerasClassifier(model, False)

        min_score = 100
        randomX = X_test.sample(10)
        randomX = np.array(randomX)

        for x in randomX:
            temp = clever_u(classifier=classifier, x=x,
                            nb_batches=1, batch_size=1, radius=500, norm=1)
            if min_score > temp:
                min_score = temp
        score = np.digitize(min_score, thresholds) + 1
        return result(score=int(score), properties={"clever_score": info("CLEVER Score", "{:.2f}".format(min_score)),
                                                    "depends_on": info("Depends on", "Model")})
    except Exception as e:
        print(e)
        return result(score=1, properties={"non_computable": info("Non Computable Because",
                                                                  "Can only be calculated on Keras models.")})


"""########################################TEST VALUES#############################################
test,model, mappings=r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend\algorithms\supervised\Mapping&Weights\mapping_metrics_default.json"
print(get_clever_score_supervised(test_dataset=test, model=model,mappings=mappings))"""
