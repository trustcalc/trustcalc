def get_underfitting_score_supervised(model=not None, training_dataset=None, test_dataset=not None, factsheet=not None, mappings=None, target_column=None, outliers_data=None, thresholds=[0.05, 0.16, 0.28, 0.4], outlier_thresholds=None, outlier_percentage=None, high_cor=0.9, print_details=None):
    try:
        import collections
        import pandas
        import algorithms.supervised.Functions.Fairness.helpers_fairness_supervised
        import numpy as np
    except:
        import collections
        import pandas
        import numpy as np

    print("GET UNDERFITTING SCORE MODEL: ", model)

    test_dataset = pandas.read_csv(test_dataset)
    print('factsheet:', factsheet)
    factsheet = pandas.read_json(factsheet)
    model = pandas.read_pickle(model)

    info, result = collections.namedtuple(
        'info', 'description value'), collections.namedtuple('result', 'score properties')
    if not high_cor:
        high_cor = 0.9
    if not thresholds:
        thresholds = thresholds

    try:
        properties = {}
        properties['Metric Description'] = "Compares the models achieved test accuracy against a baseline."
        properties['Depends on'] = 'Model, Test Data'
        score = 0
        test_accuracy = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.compute_accuracy(
            model, test_dataset, factsheet)
        score = np.digitize(abs(test_accuracy), thresholds, right=False) + 1

        properties["Test Accuracy"] = "{:.2f}%".format(test_accuracy*100)

        if score == 5:
            properties["Conclusion"] = "Model is not underfitting"
        elif score == 4:
            properties["Conclusion"] = "Model mildly underfitting"
        elif score == 3:
            properties["Conclusion"] = "Model is slighly underfitting"
        elif score == 2:
            properties["Conclusion"] = "Model is underfitting"
        else:
            properties["Conclusion"] = "Model is strongly underfitting"

        properties["Score"] = str(score)
        return result(score=int(score), properties=properties)

    except Exception as e:
        print("ERROR in underfitting_score(): {}".format(e))
        return result(score=1, properties={"Non computable because": str(e)})


"""########################################TEST VALUES#############################################
test,factsheet,model=r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/factsheet.json",r"Backend/algorithms/supervised/TestValues/model.pkl"
print(get_underfitting_score_supervised(test_dataset=test,factsheet=factsheet,model=model))"""
