def get_factsheet_completeness_score_unsupervised(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, penalty_outlier=None, outlier_percentage=None, high_cor=None, print_details=None):
    def read_model(solution_set_path):
        import os
        import pandas as pd
        import joblib as jb
        MODEL_REGEX = "model.*"
        model_file = solution_set_path
        file_extension = os.path.splitext(model_file)[1]
        # pickle_file_extensions = [".sav", ".pkl", ".pickle"]
        pickle_file_extensions = [".pkl"]
        if file_extension in pickle_file_extensions:
            with open(model_file, 'rb') as file:
                model = pd.read_pickle(file)
            return model

        if file_extension == ".joblib":  # Check if a .joblib file needs to be loaded
            return jb.load(model_file)
    import collections
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')

    import pandas as pd
    training_dataset = pd.read_csv(training_dataset)
    test_dataset = pd.read_csv(test_dataset)
    factsheet = pd.read_json(factsheet)

    score = 0
    properties = {"dep": info('Depends on', 'Factsheet')}
    GENERAL_INPUTS = ["model_name", "purpose_description", "domain_description",
                      "training_data_description", "model_information", "authors", "contact_information"]

    n = len(GENERAL_INPUTS)
    ctr = 0
    for e in GENERAL_INPUTS:
        if "general" in factsheet and e in factsheet["general"]:
            ctr += 1
            properties[e] = info("Factsheet Property {}".format(
                e.replace("_", " ")), "present")
        else:
            properties[e] = info("Factsheet Property {}".format(
                e.replace("_", " ")), "missing")
    score = round(ctr/n*5)
    if score==0:
        score=1
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

a=get_factsheet_completeness_score(model,train,test,factsheet,mapping_metrics_default)
print(a)
"""
