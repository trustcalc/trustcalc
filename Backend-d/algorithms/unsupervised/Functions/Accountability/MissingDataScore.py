def missing_data_score(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    import numpy as np
    import collections
    
    print('called here 1')
    import pandas as pd
    training_dataset=pd.read_csv(training_dataset)
    test_dataset=pd.read_csv(test_dataset)
    factsheet=pd.read_json(factsheet)
    mappings=pd.read_json(mappings)

    print('called here 2')
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    score = 0
    try:
        missing_values = training_dataset.isna().sum().sum() + test_dataset.isna().sum().sum()
        if missing_values > 0:
            print("MISSING VALUES >0")
            try:
                score = mappings["accountability"]["score_missing_data"]["mappings"]["value"]["null_values_exist"]
                print("SCORE MISSING DATA WORKS")
            except:
                score = mappings["methodology"]["score_missing_data"]["mappings"]["value"]["null_values_exist"]
        else:
            print("missing values <=0")
            try:
                score = mappings["accountability"]["score_missing_data"]["mappings"]["value"]["no_null_values"]
                print("ACCOUNTABILITY MISSING DATA")

            except:
                score = mappings["methodology"]["score_missing_data"]["mappings"]["value"]["no_null_values"]
        print('end call')
        return result(score=score,properties={"dep" :info('Depends on','Training Data'),
            "null_values": info("Number of the null values", "{}".format(missing_values))})
    except:
        print("RETURN EXCEPTION Missing data")
        return result(score=np.nan, properties={})
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

a=missing_data_score(model,train,test,factsheet,mapping_metrics_default)
print(a)
"""