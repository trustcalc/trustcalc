def get_model_size_score_supervised(model=None, training_dataset=None, test_dataset=not None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds = [10,30,100,500], outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    import collections, pandas, numpy as np
    info,result = collections.namedtuple('info', 'description value'),collections.namedtuple('result', 'score properties')
    test_data=pandas.read_csv(test_dataset)

    thresholds = np.array([10,30,100,500]) if not thresholds else  np.array(thresholds)

    try:
        dist_score = 5- np.digitize(test_data.shape[1]-1 , thresholds, right=True) 
    except:
        dist_score=1
    return result(score=int(dist_score), properties={"dep" :info('Depends on','Training Data'),
        "n_features": info("number of features", test_data.shape[1]-1)})

"""########################################TEST VALUES#############################################
test=r"Backend/algorithms/supervised/TestValues/test.csv"
print(get_model_size_score_supervised(test_dataset=test))"""