def get_overfitting_score_supervised(model=not None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=not None, target_column=None, outliers_data=None, thresholds=[0.05, 0.16, 0.28, 0.4], outlier_thresholds=None, outlier_percentage=None, high_cor=0.9, print_details=None):
    try:
        import collections
        import pandas
        import tensorflow
        import algorithms.supervised.Functions.Fairness.helpers_fairness_supervised
        import numpy as np
    except:
        import collections
        import pandas
        import tensorflow
        import Functions.Fairness.helpers_fairness_supervised
        import numpy as np

    training_dataset, test_dataset, factsheet, model, mappings = pandas.read_csv(training_dataset), pandas.read_csv(
        test_dataset), pandas.read_json(factsheet), pandas.read_pickle(model), pandas.read_json(mappings)
    info, result = collections.namedtuple(
        'info', 'description value'), collections.namedtuple('result', 'score properties')

    if not thresholds:
        thresholds = mappings["fairness"]["score_overfitting"]["thresholds"]["value"]

    try:
        properties = {}
        properties['Metric Description'] = "Overfitting is present if the training accuracy is significantly higher than the test accuracy"
        properties['Depends on'] = 'Model, Training Data, Test Data'
        overfitting_score = np.nan
        training_accuracy = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.compute_accuracy(
            model=model, dataset=training_dataset, factsheet=factsheet)
        test_accuracy = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.compute_accuracy(
            model=model, dataset=test_dataset, factsheet=factsheet)
        accuracy_difference = training_accuracy - test_accuracy
        underfitting_score = np.digitize(
            abs(test_accuracy), thresholds, right=False) + 1

        if underfitting_score >= 3:
            overfitting_score = np.digitize(
                abs(accuracy_difference), thresholds, right=False) + 1
            properties["Training Accuracy"] = "{:.2f}%".format(
                training_accuracy*100)
            properties["Test Accuracy"] = "{:.2f}%".format(test_accuracy*100)
            properties["Train Test Accuracy Difference"] = "{:.2f}%".format(
                (training_accuracy - test_accuracy)*100)

            if overfitting_score == 5:
                properties["Conclusion"] = "Model is not overfitting"
            elif overfitting_score == 4:
                properties["Conclusion"] = "Model mildly overfitting"
            elif overfitting_score == 3:
                properties["Conclusion"] = "Model is slighly overfitting"
            elif overfitting_score == 2:
                properties["Conclusion"] = "Model is overfitting"
            else:
                properties["Conclusion"] = "Model is strongly overfitting"

            properties["Score"] = str(overfitting_score)
            return result(int(overfitting_score), properties=properties)
        else:
            properties = {}
            properties['Metric Description'] = "Overfitting is present if the training accuracy is significantly higher than the test accuracy"
            properties['Depends on'] = 'Model, Training Data, Test Data'
            overfitting_score = np.nan
            training_accuracy = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.compute_accuracy(
                model, training_dataset, factsheet)
            test_accuracy = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.compute_accuracy(
                model, test_dataset, factsheet)
            properties["Training Accuracy"] = "{:.2f}%".format(
                training_accuracy*100)
            properties["Test Accuracy"] = "{:.2f}%".format(test_accuracy*100)
            overfitting_score = 0
            properties["Train Test Accuracy Difference"] = "{:.2f}%".format(
                (training_accuracy - test_accuracy)*100)
            properties["Conclusion"] = "Model is not overfitting"
            return result(5, properties=properties)
    except Exception as e:
        properties = {}
        properties['Metric Description'] = "Overfitting is present if the training accuracy is significantly higher than the test accuracy"
        properties['Depends on'] = 'Model, Training Data, Test Data'
        overfitting_score = np.nan
        try:
            training_accuracy = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.compute_accuracy(
                model, training_dataset, factsheet)
        except:
            training_accuracy = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.compute_accuracy(
                model, training_dataset, factsheet)

        try:
            test_accuracy = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.compute_accuracy(
                model, test_dataset, factsheet)
        except:
            test_accuracy = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.compute_accuracy(
                model, test_dataset, factsheet)

        properties["Training Accuracy"] = "{:.2f}%".format(
            training_accuracy*100)
        properties["Test Accuracy"] = "{:.2f}%".format(test_accuracy*100)
        properties["Train Test Accuracy Difference"] = "{:.2f}%".format(
            (training_accuracy - test_accuracy)*100)
        properties["Conclusion"] = "Model is not overfitting"
        return result(5, properties=properties)


"""########################################TEST VALUES#############################################
train,test,factsheet,model,mappings=r"Backend/algorithms/supervised/TestValues/train.csv",r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/factsheet.json",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend/algorithms/supervised/TestValues/mappings.json"
print(get_overfitting_score_supervised(training_dataset=train,test_dataset=test,factsheet=factsheet,model=model,mappings=mappings))"""
