def normalization_score(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    import pandas as pd
    training_dataset=pd.read_csv(training_dataset)
    test_dataset=pd.read_csv(test_dataset)
    factsheet=pd.read_json(factsheet)
    mappings=pd.read_json(mappings)
    
    print('normalization score called')
    try:
        mappings=mappings["accountability"]["score_normalization"]["mappings"]["value"]
    except:
        mappings=mappings["methodology"]["score_normalization"]["mappings"]["value"]


    import numpy as np
    import collections
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    X_train = training_dataset.iloc[:, :-1]
    X_test = test_dataset.iloc[:, :-1]
    train_mean = np.mean(np.mean(X_train))
    train_std = np.mean(np.std(X_train))
    test_mean = np.mean(np.mean(X_test))
    test_std = np.mean(np.std(X_test))
    from cmath import isclose

    properties = {"dep" :info('Depends on','Training and Testing Data'),
        "Training_mean": info("Mean of the training data", "{:.2f}".format(train_mean)),
                  "Training_std": info("Standard deviation of the training data", "{:.2f}".format(train_std)),
                  "Test_mean": info("Mean of the test data", "{:.2f}".format(test_mean)),
                  "Test_std": info("Standard deviation of the test data", "{:.2f}".format(test_std))
                  }
    if not (any(X_train < 0) or any(X_train > 1)) and not (any(X_test < 0) or any(X_test > 1)):        
        score = mappings["training_and_test_normal"]
        properties["normalization"] = info("Normalization", "Training and Testing data are normalized")
    elif isclose(train_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and isclose(train_std, 1, rel_tol=1e-3, abs_tol=1e-6) and (not isclose(test_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and not isclose(test_std, 1, rel_tol=1e-3, abs_tol=1e-6)):
        score = mappings["training_standardized"]
        properties["normalization"] = info("Normalization", "Training data are standardized")
    elif isclose(train_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and isclose(train_std, 1, rel_tol=1e-3, abs_tol=1e-6) and (isclose(test_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and isclose(test_std, 1, rel_tol=1e-3, abs_tol=1e-6)):
        score = mappings["training_and_test_standardize"]
        properties["normalization"] = info("Normalization", "Training and Testing data are standardized")
    elif any(X_train < 0) or any(X_train > 1):
        score = mappings["None"]
        properties["normalization"] = info("Normalization", "None")
    elif not (any(X_train < 0) or any(X_train > 1)) and (any(X_test < 0) or any(X_test > 1)):
        score = mappings["training_normal"]
        properties["normalization"] = info("Normalization", "Training data are normalized")
    return result(score=score, properties=properties)

"""
########################################TEST VALUES#############################################
import pandas as pd

train=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/train.csv"
test=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/test.csv"
outliers=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/outliers.csv"

model=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/model.joblib"
factsheet=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/factsheet.json"

mapping_metrics_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
weights_metrics_feault=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_metrics_default.json"
weights_pillars_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_pillars_default.json"

a=normalization_score(model,train,test,factsheet,mapping_metrics_default)
print(a)
"""