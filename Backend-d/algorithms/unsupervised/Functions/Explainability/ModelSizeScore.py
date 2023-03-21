import numpy as np
def model_size_score(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds = np.array([10,30,100,500]), outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    import collections
    import pandas as pd
    test_data=pd.read_csv(test_dataset)
    result = collections.namedtuple('result', 'score properties')
    info = collections.namedtuple('info', 'description value')
    print("THRESHOLD: ",thresholds)
    try:
        if (not thresholds):
            thresholds=np.array([10,30,100,500])
    except:
        pass
    try:
        print("THRESHOLDS: ",thresholds)
        dist_score = 5- np.digitize(test_data.shape[1], thresholds, right=True)
    except:
        print("TEST DATA SHAPE ",test_data.shape[1])
        print("dist score ",dist_score)
    if print_details:
        print("\t MODEL SIZE DETAILS")
        print("\t num of features: ", test_data.shape[1])

    return result(score=int(dist_score), properties={"dep" :info('Depends on','Test Data'),
        "n_features": info("number of features", test_data.shape[1]-1)})

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

a= model_size_score(test, thresholds = np.array([10,30,100,500]))
print(a)
"""