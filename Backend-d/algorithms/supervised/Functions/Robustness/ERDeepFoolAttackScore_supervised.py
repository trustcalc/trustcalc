def get_deepfool_attack_score_supervised(model=not None, training_dataset= None, test_dataset=not None, factsheet= None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor= None,print_details=True):
    import collections, pandas,sklearn.metrics, sklearn.preprocessing,numpy as np
    from art.estimators.classification import SklearnClassifier
    from art.attacks.evasion import DeepFool

    info,result = collections.namedtuple('info', 'description value'),collections.namedtuple('result', 'score properties')
    model,test_data, mappings=pandas.read_pickle(model),pandas.read_csv(test_dataset),pandas.read_json(mappings)
    if not thresholds:
        thresholds = mappings["robustness"]["score_deepfool_attack"]["thresholds"]["value"]

    try:
        randomData = test_data.sample(4)
        randomX = randomData.iloc[:,:-1]
        randomY = randomData.iloc[:,-1: ]

        y_pred = model.predict(randomX)
        before_attack = sklearn.metrics.accuracy_score(randomY,y_pred)

        classifier = SklearnClassifier(model=model)
        attack = DeepFool(classifier)
        x_test_adv = attack.generate(x=randomX)

        enc = sklearn.preprocessingOneHotEncoder(handle_unknown='ignore')
        enc.fit(test_data.iloc[:,-1: ])
        randomY = enc.transform(randomY).toarray()

        predictions = model.predict(x_test_adv)
        predictions = enc.transform(predictions.reshape(-1,1)).toarray()
        after_attack = sklearn.metrics.accuracy_score(randomY,predictions)
        print("Accuracy on before_attacks: {}%".format(before_attack * 100))
        print("Accuracy on after_attack: {}%".format(after_attack * 100))

        score = np.digitize((before_attack - after_attack)/before_attack*100, thresholds) + 1
        return result(score=int(score),
                      properties={"before_attack": info("DF Before attack accuracy", "{:.2f}%".format(100 * before_attack)),
                                  "after_attack": info("DF After attack accuracy", "{:.2f}%".format(100 * after_attack)),
                                  "difference": info("DF Proportional difference (After-Att Acc - Before-Att Acc)/Before-Att Acc", "{:.2f}%".format(100 * (before_attack - after_attack) / before_attack)),
                                  "depends_on": info("Depends on", "Model and Data")})
    except:
        return result(score=1, properties={"non_computable": info("Non Computable Because",
                                                                       "Can be calculated on either SVC or Logistic Regression models.")})

"""########################################TEST VALUES#############################################
test,model, mappings=r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend\algorithms\supervised\Mapping&Weights\mapping_metrics_default.json"
print(get_deepfool_attack_score_supervised(test_dataset=test, model=model,mappings=mappings))"""