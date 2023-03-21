def isKerasAutoencoder(model):
    import keras.engine.functional
    return isinstance(model, keras.engine.functional.Functional)


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



def get_threshold_mse_iqr(autoencoder, training_dataset):
    import numpy as np
    if training_dataset.ndim == 1:
        training_dataset = training_dataset.reshape(1, -1)
    elif training_dataset.shape[0] == 1:
        training_dataset = training_dataset.T

    train_predicted = autoencoder.predict(training_dataset)
    train_predicted = np.reshape(
        train_predicted, (train_predicted.shape[0], 1))
    mse = np.mean(np.power(training_dataset - train_predicted, 2), axis=1)
    iqr = np.quantile(mse, 0.75) - np.quantile(mse,
                                               0.25)  # interquartile range
    up_bound = np.quantile(mse, 0.75) + 1.5*iqr
    bottom_bound = np.quantile(mse, 0.25) - 1.5*iqr
    thres = [up_bound, bottom_bound]
    return thres


def isIsolationForest(model):
    from sklearn.ensemble import IsolationForest
    return isinstance(model, IsolationForest)


def detect_outliers(autoencoder, df, threshold_mse):
    import numpy as np
    if (len(threshold_mse) == 2):
        return detect_outliers_range(autoencoder, df, threshold_mse)
    pred = autoencoder.predict(df)
    mse = np.mean(np.power(df - pred, 2), axis=1)
    outliers = [np.array(mse) < threshold_mse]
    return outliers


def compute_accuracy(unique_elements, counts_elements, outlier_indicator=False, normal_indicator=True):
    tot_datapoints = 0
    num_outliers = 0
    num_normal = 0

    for i, el in enumerate(unique_elements):
        if el == normal_indicator:
            num_normal = counts_elements.item(i)
            tot_datapoints += num_normal
        if el == outlier_indicator:
            num_outliers = counts_elements.item(i)
            tot_datapoints += num_outliers

    if (tot_datapoints > 0):
        accuracy = num_outliers / tot_datapoints

    else:
        accuracy = 0

    return accuracy


def compute_outlier_ratio(model, data, outlier_thresh, print_details=False):
    print('here called error in this function...', model, data)
    import numpy
    if isKerasAutoencoder(model):
        mad_outliers = detect_outliers(model, data, outlier_thresh)
        unique_elements, counts_elements = numpy.unique(
            mad_outliers, return_counts=True)
        outlier_detection_percentage = compute_accuracy(
            unique_elements, counts_elements)

    elif isIsolationForest(model):
        mad_outliers = model.predict(data)
        unique_elements, counts_elements = numpy.unique(
            mad_outliers, return_counts=True)
        outlier_detection_percentage = compute_accuracy(
            unique_elements, counts_elements, -1, 1)

    else:
        mad_outliers = model.predict(data)
        unique_elements, counts_elements = numpy.unique(
            mad_outliers, return_counts=True)
        outlier_detection_percentage = compute_accuracy(
            unique_elements, counts_elements, 1, 0)

    if print_details:
        print("\t uniqueelements: ", unique_elements)
        print("\t counts elements: ", counts_elements)

    return outlier_detection_percentage


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
