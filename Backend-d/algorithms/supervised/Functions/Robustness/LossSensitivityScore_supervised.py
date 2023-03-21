def get_loss_sensitivity_score_supervised(model=not None, training_dataset= None, test_dataset=not None, factsheet= None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor= None,print_details=True):
    import collections, pandas,art.metrics, art.estimators.classification, numpy as np

    info,result = collections.namedtuple('info', 'description value'),collections.namedtuple('result', 'score properties')
    model,test_data, mappings=pandas.read_pickle(model),pandas.read_csv(test_dataset),pandas.read_json(mappings)

    if not thresholds:
        thresholds = mappings["robustness"]["score_loss_sensitivity"]["thresholds"]["value"]
    try:
        X_test = test_data.iloc[:,:-1]
        X_test = np.array(X_test)
        y = model.predict(X_test)

        classifier = art.estimators.classification.KerasClassifier(model=model, use_logits=False)
        l_s = art.metrics.loss_sensitivity(classifier, X_test, y)
        score = np.digitize(l_s, thresholds, right=True) + 1
        return result(score=int(score), properties={"loss_sensitivity": info("Average gradient value of the loss function", "{:.2f}".format(l_s)),
                                                    "depends_on": info("Depends on", "Model")})
    except Exception as e:
        print(e)
        return result(score=1, properties={"non_computable": info("Non Computable Because",
                                                                       "Can only be calculated on Keras models.")})
"""########################################TEST VALUES#############################################
test,model, mappings=r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend\algorithms\supervised\Mapping&Weights\mapping_metrics_default.json"
print(get_loss_sensitivity_score_supervised(test_dataset=test, model=model,mappings=mappings))"""