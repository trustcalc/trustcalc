def analyse(    model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    import collections,sys
    sys.path.extend([r'Backend',r'Backend/algorithms',r'Backend/algorithms/unsupervised',r'Backend/algorithms/unsupervised',r'Backend/algorithms/unsupervised',r'Backend/algorithms/unsupervised/Functions',r'Backend/algorithms/unsupervised/Functions/Robustness',r'Backend/algorithms/unsupervised/Functions/Robustness/CLEVER'])

    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    try:
        from .CleverScore import clever_score
    except:
        from Functions.Robustness.CleverScore import clever_score


    """Reads the thresholds from the config file.
    Calls all robustness metric functions with correct arguments.
    Organizes all robustness metrics in a dict. Then returns the scores and the properties.
        Args:
            model: ML-model.
            training_dataset: pd.DataFrame containing the used training data.
            test_dataset: pd.DataFrame containing the used test data.
            config: Config file containing the threshold values for the metrics.
            factsheet: json document containing all information about the particular solution.

        Returns:
            Returns a result object containing all metric scores
            and matching properties for every metric
    """
    import pandas as pd
    mappings2=pd.read_json(mappings)
    clever_score_thresholds = mappings2["robustness"]["score_clever_score"]["thresholds"]["value"]
    
    output = dict(
        clever_score = clever_score(model, training_dataset, test_dataset, factsheet,mappings,thresholds=None),
        #clever_score = result(score=int(1), properties={}),
    )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)

"""
########################################TEST VALUES#############################################
import pandas as pd

train=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/train.csv"
test=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/test.csv"
outliers=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/outliers.csv"

model=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/model.joblib"
factsheet=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/factsheet.json"

mapping_metrics_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
weights_metrics_feault=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_metrics_default.json"
weights_pillars_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_pillars_default.json"


a=analyse(model, train, test, outliers, factsheet, mapping_metrics_default)
print(a)
"""