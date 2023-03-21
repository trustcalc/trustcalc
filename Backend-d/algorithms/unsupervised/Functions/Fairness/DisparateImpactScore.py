def disparate_impact_score(model=not None, training_dataset=None, test_dataset=not None, factsheet=not None, mappings=not None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, penalty_outlier=None, outlier_percentage=None, high_cor=None, print_details=None):
    import collections
    import numpy
    import pandas as pd
    try:
        from algorithms.unsupervised.Functions.Fairness.helpers_us_fairness import detect_outliers, get_threshold_mse_iqr, isIsolationForest, isKerasAutoencoder, load_fairness_config, read_model
    except:
        from unsupervised.Functions.Fairness.helpers_us_fairness import detect_outliers, get_threshold_mse_iqr, isIsolationForest, isKerasAutoencoder, load_fairness_config, read_model

    factsheet, mappings, test_dataset = pd.read_json(
        factsheet), pd.read_json(mappings), pd.read_csv(test_dataset)
    model = read_model(model)

    if not thresholds:
        thresholds = mappings["fairness"]["score_disparate_impact"]["thresholds"]["value"]

    result, info = collections.namedtuple(
        'result', 'score properties'), collections.namedtuple('info', 'description value')

    if test_dataset.ndim == 1:
        test_dataset = test_dataset.reshape(1, -1)
    elif test_dataset.shape[0] == 1:
        test_dataset = test_dataset.T

    try:
        print("TRY")
        protected_feature, protected_values = load_fairness_config(factsheet)
        print("PROTECTED FEATURE: ",protected_feature)

        minority = test_dataset[test_dataset[protected_feature].isin(
            protected_values)]
        minority_size = len(minority)
        majority = test_dataset[~test_dataset[protected_feature].isin(
            protected_values)]
        majority_size = len(majority)

        if isKerasAutoencoder(model):
            thresh = get_threshold_mse_iqr(model, test_dataset)
            mad_outliers = detect_outliers(model, test_dataset, thresh)
            outlier_indices = [i for i, el in enumerate(
                mad_outliers[0].tolist()) if el == False]

        elif isIsolationForest(model):
            mad_outliers = model.predict(test_dataset)
            outlier_indices = [i for i, el in enumerate(
                mad_outliers.tolist()) if el == -1]

        else:
            mad_outliers = model.predict(test_dataset)
            outlier_indices = [i for i, el in enumerate(
                mad_outliers.tolist()) if el == 1]

        minority_indices = minority.index.tolist()
        majority_indices = majority.index.tolist()

        # measure num of outliers in majority group by intersection of indices
        num_outliers_minority = len(
            list(set(minority_indices) & set(outlier_indices)))
        num_outliers_majority = len(
            list(set(majority_indices) & set(outlier_indices)))

        print('values:', minority_size, majority_size)
        print("TEST_1")
        print("NUM OUTLIERS MINORIY: ",num_outliers_minority)
        print("MINORIY SIZE: ",minority_size)
        print("NUM OUTLIERS MAJORITY: ",num_outliers_majority)
        print("MAJORITY SIZE: ",majority_size)
        
        """if(minority_size==0):
            print("reached minority size 0")
            return result(score=5, properties="Minorize Size is 0")
"""
        favored_minority_ratio = num_outliers_minority / minority_size
        print("TEST_2")

        favored_majority_ratio = num_outliers_majority / majority_size
        print("TEST_3")

        print("num_outliers_minority", num_outliers_minority)
        print("minority_size", minority_size)
        print("num_outliers_majority", num_outliers_majority)
        print("majority_size", majority_size)
        print("favored_minority_ratio", favored_minority_ratio)
        print("favored_majority_ratio", favored_majority_ratio)

        disparate_impact = abs(favored_minority_ratio / favored_majority_ratio)

        if print_details:
            print("\t protected feature: ", protected_feature)
            print("\t protected values: ", protected_values)
            print("\t group size: ", len(majority_indices), len(minority_indices))
            print("\t num outlier: ", num_outliers_majority, num_outliers_minority)
            print("\t outlier ratios: %.4f " %
                  favored_majority_ratio, "%.4f " % favored_minority_ratio)
            print("\t disparate_impact: %.4f" % disparate_impact)

        properties = {}
        properties[
            "Metric Description"] = "Is quotient of the ratio of samples from the protected group detected as outliers divided by the ratio of samples from the unprotected group detected as outliers"
        properties["Depends on"] = "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        properties['----------'] = ''
        properties["protected feature: "] = protected_feature
        properties["protected values: "] = str(protected_values)
        properties['-----------'] = ''
        properties["Formula"] = "Disparate Impact = Protected Favored Ratio / Unprotected Favored Ratio"
        properties["Disparate Impact"] = "{:.2f}%".format(
            disparate_impact * 100)

        properties["|{x|x is protected, y_true is favorable}|"] = num_outliers_minority
        properties["|{x|x is protected}|"] = minority_size
        properties["Favored Protected Group Ratio"] = "P(y_true is favorable|protected=True) = {:.2f}%".format(
            num_outliers_minority / minority_size * 100)
        properties["|{x|x is not protected, y_true is favorable}|"] = num_outliers_majority
        properties["|{x|x is not protected}|"] = majority_size
        properties["Favored Unprotected Group Ratio"] = "P(y_true is favorable|protected=False) = {:.2f}%".format(
            num_outliers_majority / majority_size * 100)

        score = numpy.digitize(disparate_impact, thresholds, right=True) + 1

        properties["Score"] = str(score)
        return result(score=int(score), properties=properties)

    except Exception as e:
        print('exception:', e)
        properties = {}
        properties[
            "Metric Description"] = "Is quotient of the ratio of samples from the protected group detected as outliers divided by the ratio of samples from the unprotected group detected as outliers"
        properties["Depends on"] = "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        properties['----------'] = ''
        properties["protected feature: "] = protected_feature
        properties["protected values: "] = str(protected_values)
        properties['-----------'] = ''
        disparate_impact = 0
        properties["Formula"] = "Disparate Impact = Protected Favored Ratio / Unprotected Favored Ratio"
        properties["Disparate Impact"] = "{:.2f}%".format(
            disparate_impact * 100)
        try:
            properties["|{x|x is protected, y_true is favorable}|"] = num_outliers_minority
        except:
            properties["|{x|x is protected, y_true is favorable}|"] = 0
        try:
            properties["|{x|x is protected}|"] = minority_size
        except:
            properties["|{x|x is protected}|"] = 0
        try:
            properties["Favored Protected Group Ratio"] = "P(y_true is favorable|protected=True) = {:.2f}%".format(
                0)
        except:
            properties["Favored Protected Group Ratio"] = 0

        try:
            properties["|{x|x is not protected, y_true is favorable}|"] = num_outliers_majority
        except:
            properties["|{x|x is not protected, y_true is favorable}|"] = 0

        try:
            properties["|{x|x is not protected}|"] = majority_size
        except:
            properties["|{x|x is not protected}|"] = 0
        try:
            properties["Favored Unprotected Group Ratio"] = "P(y_true is favorable|protected=False) = {:.2f}%".format(
                num_outliers_majority / majority_size * 100)
        except:
            properties["Favored Unprotected Group Ratio"] = 0

        score = 1

        properties["Score"] = str(score)
        print("ERROR in Disparate Impact Score(): {}".format(e))
        print("RESULTTTTTT: ",result(score=int(score), properties=properties))
        return result(score=int(score), properties=properties)

        # return result(score=np.nan, properties={"Non computable because": str(e)})


"""########################################TEST VALUES#############################################
import pandas as pd

train=r"Backend/algorithms/unsupervised/TestValues/train.csv"
test=r"Backend/algorithms/unsupervised/TestValues/test.csv"
outliers=r"Backend/algorithms/unsupervised/TestValues/outliers.csv"

model=r"Backend/algorithms/unsupervised/TestValues/model.joblib"
factsheet=r"Backend/algorithms/unsupervised/TestValues/factsheet.json"

mapping_metrics_default=r"Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"

a= disparate_impact_score(model, test, factsheet, mapping_metrics_default, True,None)

print(a)
"""

