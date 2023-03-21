
def train_test_split_score(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    import collections
    import pandas as pd
    import re
    import numpy as np
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    
    training_dataset=pd.read_csv(training_dataset)
    test_dataset=pd.read_csv(test_dataset)
    factsheet=pd.read_json(factsheet)
    mappings=pd.read_json(mappings)



    def train_test_split_metric(training_dataset, test_dataset):
        n_train = len(training_dataset)
        n_test = len(test_dataset)
        n = n_train + n_test
        return round(n_train/n*100), round(n_test/n*100)
    try:
        mappings=mappings["accountability"]["score_train_test_split"]["mappings"]["value"]
    except:
        mappings=mappings["methodology"]["score_train_test_split"]["mappings"]["value"]

    try:
        training_data_ratio, test_data_ratio = train_test_split_metric(training_dataset, test_dataset)
        properties= {"dep" :info('Depends on','Training and Testing Data'),
            "train_test_split": info("Train test split", "{:.2f}/{:.2f}".format(training_data_ratio, test_data_ratio))}
        for k in mappings.keys():
            thresholds = re.findall(r'\d+-\d+', k)
            for boundary in thresholds:
                [a, b] = boundary.split("-")
                if training_data_ratio >= int(a) and training_data_ratio < int(b):
                    score = mappings[k]
        return result(score=score, properties=properties)
    except Exception as e:
        print(e)
        return result(score=np.nan, properties={})
"""
########################################TEST VALUES#############################################

train=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/train.csv"
test=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/test.csv"
outliers=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/outliers.csv"

model=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/model.joblib"
factsheet=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/factsheet.json"

mapping_metrics_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
weights_metrics_feault=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_metrics_default.json"
weights_pillars_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_pillars_default.json"

a=train_test_split_score(model,train,test,factsheet,mapping_metrics_default)
print(a)
"""