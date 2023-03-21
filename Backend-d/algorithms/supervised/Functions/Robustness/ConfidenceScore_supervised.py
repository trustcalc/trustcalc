def get_confidence_score_supervised(model=None, training_dataset=not None, test_dataset=not None, factsheet=None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor= None,print_details=True):
    import collections, pandas,sklearn.metrics, numpy as np
    info,result = collections.namedtuple('info', 'description value'),collections.namedtuple('result', 'score properties')
    model,test_data, mappings=pandas.read_pickle(model),pandas.read_csv(test_dataset),pandas.read_json(mappings)

    if not thresholds:
        thresholds = mappings["robustness"]["score_confidence_score"]["thresholds"]["value"]
    
    try:
        X_test = test_data.iloc[:,:-1]
        y_test = test_data.iloc[:,-1: ]
        y_pred = model.predict(X_test)

        confidence = sklearn.metrics.confusion_matrix(y_test, y_pred)/sklearn.metrics.confusion_matrix(y_test, y_pred).sum(axis=1) 
        confidence_score = np.average(confidence.diagonal())*100
        score = np.digitize(confidence_score, thresholds, right=True) + 1
        return result(score=int(score), properties={"confidence_score": info("Average confidence score", "{:.2f}%".format(confidence_score)),
                                                    "depends_on": info("Depends on", "Model and Data")})
    except:
        return result(score=np.nan, properties={"non_computable": info("Non Computable Because", "Can only be calculated on models which provide prediction probabilities.")})

"""########################################TEST VALUES#############################################
test,model, mappings=r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend\algorithms\supervised\Mapping&Weights\mapping_metrics_default.json"
print(get_confidence_score_supervised(test_dataset=test, model=model,mappings=mappings))"""