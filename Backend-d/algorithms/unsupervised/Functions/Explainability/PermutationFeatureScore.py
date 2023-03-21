def permutation_feature_importance_score(model=not None, training_dataset=not None, test_dataset=None, factsheet=None, mappings=None, outliers_data=not None, thresholds=[0.2, 0.15, 0.1, 0.05], outlier_thresholds=None, penalty_outlier=None, outlier_percentage=None, high_cor=None, print_details=None):
    if not thresholds:
        thresholds = [0.2, 0.15, 0.1, 0.05]

    import numpy as np
    import keras
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    import statistics
    import collections
    result = collections.namedtuple('result', 'score properties')
    info = collections.namedtuple('info', 'description value')

    print('outliers_data:', outliers_data)
    outliers_data = pd.read_csv(outliers_data)
    features = list(outliers_data.columns)

    train_data = pd.read_csv(training_dataset)

    if train_data.ndim == 1:
        train_data = train_data.reshape(1, -1)
    elif train_data.shape[0] == 1:
        train_data = train_data.T

    def read_model(solution_set_path):
        print("READ MODEL REACHED")
        import os
        from joblib import load
        MODEL_REGEX = "model.*"
        model_file = solution_set_path
        file_extension = os.path.splitext(model_file)[1]
        print("FILE EXTENSION: ",file_extension)

        # pickle_file_extensions = [".sav", ".pkl", ".pickle"]
        pickle_file_extensions = [".pkl"]
        if file_extension in pickle_file_extensions:
            model = pd.read_pickle(model_file)
            return model
        if (file_extension == ".joblib"):  # Check if a .joblib file needs to be loaded
            print("model_file: ", model_file)
            a=load(model_file)
            print("READ MODEL joblib REACHED")
            print("READ JOBLIB MODEl: ",a)
            return a

    def detect_outliers(autoencoder, df, threshold_mse):
        if (len(threshold_mse) == 2):
            return detect_outliers_range(autoencoder, df, threshold_mse)
        pred = autoencoder.predict(df)
        mse = np.mean(np.power(df - pred, 2), axis=1)
        outliers = [np.array(mse) < threshold_mse]
        return outliers

    def detect_outliers_range(autoencoder, df, threshold_mse):
        import numpy as np
        pred = autoencoder.predict(df)
        mse = np.mean(np.power(df - pred, 2), axis=1)
        up_bound = threshold_mse[0]
        bottom_bound = threshold_mse[1]
        outliers = [(np.array(mse) < up_bound) &
                    (np.array(mse) > bottom_bound)]
        return outliers

    def isKerasAutoencoder(model):
        return isinstance(model, keras.engine.functional.Functional)

    def isIsolationForest(model):
        return isinstance(model, IsolationForest)

    def compute_outlier_matrix(model, data, outlier_thresh, print_details=False):
        if isKerasAutoencoder(model):
            mad_outliers = detect_outliers(model, data, outlier_thresh)[0]
        elif isIsolationForest(model):
            mad_outliers = model.predict(data)
        else:
            try:
                mad_outliers = model.predict(data)
            except:
                mad_outliers = 0
        if print_details:
            # print("\t outlier matrix: ", mad_outliers)
            pass
        return mad_outliers
    import pandas as pd

    def get_threshold_mse_iqr(autoencoder, train_data):
        train_predicted = autoencoder.predict(train_data)
        train_predicted = np.reshape(
            train_predicted, (train_predicted.shape[0], 1))
        mse = np.mean(np.power(train_data - train_predicted, 2), axis=1)
        iqr = np.quantile(mse, 0.75) - np.quantile(mse,
                                                   0.25)  # interquartile range
        up_bound = np.quantile(mse, 0.75) + 1.5*iqr
        bottom_bound = np.quantile(mse, 0.25) - 1.5*iqr
        thres = [up_bound, bottom_bound]
        return thres

    model = read_model(model)
    if not outlier_thresholds:
        try:
            outlier_thresholds = get_threshold_mse_iqr(model, train_data)
        except:
            outlier_thresholds = [
                0.2,
                0.15,
                0.1,
                0.05
            ]

    shuffles = 3
    feature_importance = {}
    num_redundant_feat = 0
    num_datapoints = outliers_data.shape[0]
    accuracy_no_permutation = compute_outlier_matrix(
        model=model, outlier_thresh=outlier_thresholds, print_details=True, data=outliers_data)

    for i, feature in enumerate(features):
        feature_importance[feature] = []
        outliers_data_copy = outliers_data.copy()

        for _ in range(shuffles):
            # compute outlier detection with permutation
            outliers_data_copy[feature] = np.random.permutation(
                outliers_data[feature])
            accuracy_permutation = compute_outlier_matrix(
                model, outliers_data_copy, outlier_thresholds, print_details)

            num_diff_val = np.sum(
                accuracy_no_permutation != accuracy_permutation)

            permutation = num_diff_val / num_datapoints
            # print("permutation: ", permutation)
            feature_importance[feature].append(permutation)

        feature_importance[feature] = statistics.mean(
            feature_importance[feature])
        if (feature_importance[feature] == 0):
            num_redundant_feat += 1

    ratio_redundant_feat = num_redundant_feat / len(feature_importance)
    feature_importance_desc = list(
        dict(sorted(feature_importance.items(), key=lambda item: item[1])).keys())[::-1]
    # print(thresholds)

    score = np.digitize(ratio_redundant_feat, thresholds, right=True)+1
    properties = {
        "dep": info('Depends on', 'Model, Outliers Data'),
        "num_redundant_features": info("number of redundant features", num_redundant_feat),
        "num_features": info("number of features", len(feature_importance)),
        "ratio_redundant_features": info("ratio of redundant features", ratio_redundant_feat),
        "importance": info("feature importance descending", {"value": feature_importance_desc})
    }

    return result(score=int(score), properties=properties)


########################################TEST VALUES#############################################
"""train=r"Backend/algorithms/unsupervised/TestValues/train.csv"
test=r"Backend/algorithms/unsupervised/TestValues/test.csv"
outliers=r"Backend/algorithms/unsupervised/TestValues/outliers.csv"
model=r"Backend/algorithms/unsupervised/TestValues/model.joblib"

a= permutation_feature_importance_score(model=model, training_dataset=train,outliers_data=outliers, thresholds = [0.2,0.15,0.1,0.05], print_details = True,outlier_thresholds=None)
print(a)"""
