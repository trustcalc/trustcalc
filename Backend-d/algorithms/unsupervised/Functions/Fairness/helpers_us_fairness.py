class MissingFairnessDefinitionError(Exception):
    def __init__(self, message="Salary is not in (5000, 15000) range"):
        self.message = message
        super().__init__(self.message)


def load_fairness_config(factsheet):
    message = ""
    protected_feature = factsheet.get(
        "fairness", {}).get("protected_feature", '')
    if not protected_feature:
        message += "Definition of protected feature is missing."
    protected_values = factsheet.get(
        "fairness", {}).get("protected_values", [])
    if not protected_values:
        message += "Definition of protected_values is missing."
    print(message)
    if message:
        raise MissingFairnessDefinitionError(message)
    return protected_feature, protected_values


def read_model(solution_set_path):
    import os
    import pandas
    import joblib
    MODEL_REGEX = "model.*"
    model_file = solution_set_path
    file_extension = os.path.splitext(model_file)[1]
    # pickle_file_extensions = [".sav", ".pkl", ".pickle"]
    pickle_file_extensions = [".pkl"]
    if file_extension in pickle_file_extensions:
        model = pandas.read_pickle(model_file)
        return model

    if file_extension == ".joblib":  # Check if a .joblib file needs to be loaded
        joblib.load(model_file)
        return joblib.load(model_file)


def detect_outliers(autoencoder, df, threshold_mse):
    import numpy
    if (len(threshold_mse) == 2):
        return detect_outliers_range(autoencoder, df, threshold_mse)
    pred = autoencoder.predict(df)
    mse = numpy.mean(numpy.power(df - pred, 2), axis=1)
    outliers = [numpy.array(mse) < threshold_mse]
    return outliers


def detect_outliers_range(autoencoder, df, threshold_mse):
    import numpy as np
    pred = autoencoder.predict(df)
    mse = np.mean(np.power(df - pred, 2), axis=1)
    up_bound = threshold_mse[0]
    bottom_bound = threshold_mse[1]
    outliers = [(np.array(mse) < up_bound) & (np.array(mse) > bottom_bound)]
    return outliers


def isKerasAutoencoder(model):
    import keras
    return isinstance(model, keras.engine.functional.Functional)


def isIsolationForest(model):
    from sklearn.ensemble import IsolationForest
    return isinstance(model, IsolationForest)


def compute_outlier_matrix(model, data, outlier_thresh, print_details=False):
    if isKerasAutoencoder(model):
        mad_outliers = detect_outliers(model, data, outlier_thresh)[0]
    elif isIsolationForest(model):
        mad_outliers = model.predict(data)
    else:
        mad_outliers = model.predict(data)
    if print_details:
        print("\t outlier matrix: ", mad_outliers)
    return mad_outliers


def get_threshold_mse_iqr(autoencoder, train_data):
    import numpy
    train_predicted = autoencoder.predict(train_data)
    train_predicted = numpy.reshape(
        train_predicted, (train_predicted.shape[0], 1))
    mse = numpy.mean(numpy.power(train_data - train_predicted, 2), axis=1)
    iqr = numpy.quantile(mse, 0.75) - numpy.quantile(mse,
                                                     0.25)  # interquartile range
    up_bound = numpy.quantile(mse, 0.75) + 1.5*iqr
    bottom_bound = numpy.quantile(mse, 0.25) - 1.5*iqr
    thres = [up_bound, bottom_bound]
    return thres
