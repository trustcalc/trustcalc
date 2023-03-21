def get_normalization_score_supervised(model=None, training_dataset=not None, test_dataset=not None, factsheet=None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    import sys,inspect,cmath
    sys.path.extend([r"Backend",r"Backend/algorithms",r"Backend/algorithms/supervised", r"Backend/algorithms/supervised/Functions",r"Backend/algorithms/supervised/Functions/Accountability"])
    metric_fname = inspect.currentframe().f_code.co_name
    from algorithms.supervised.Functions.Accountability.helpers_supervised_accountability import accountabiltiy_parameter_file_loader
    foo = accountabiltiy_parameter_file_loader(metric_function_name=metric_fname, training_dataset=training_dataset,test_dataset=test_dataset,mappings=mappings)
    info, np, result, training_dataset,test_dataset, = foo['info'],foo['np'],foo['result'],foo['data']['training_dataset'],foo['data']['test_dataset']
    try:
        normalizationscore_mappings=foo['data']['mappings']["accountability"]["score_normalization"]["mappings"]["value"]
    except:
        normalizationscore_mappings=foo['data']['mappings']["methodology"]["score_normalization"]["mappings"]["value"]
    
    X_train=training_dataset.iloc[:, :-1]
    X_test,train_mean = test_dataset.iloc[:, :-1],np.mean(np.mean(X_train))
    train_std,test_mean,test_std, = np.mean(np.std(X_train)),np.mean(np.mean(X_test)),np.mean(np.std(X_test))


    properties = {"dep" :info('Depends on','Training and Testing Data'),
        "Training_mean": info("Mean of the training data", "{:.2f}".format(train_mean)),
                  "Training_std": info("Standard deviation of the training data", "{:.2f}".format(train_std)),
                  "Test_mean": info("Mean of the test data", "{:.2f}".format(test_mean)),
                  "Test_std": info("Standard deviation of the test data", "{:.2f}".format(test_std))
                  }
    if not (any(X_train < 0) or any(X_train > 1)) and not (any(X_test < 0) or any(X_test > 1)):        
        score = normalizationscore_mappings["training_and_test_normal"]
        properties["normalization"] = info("Normalization", "Training and Testing data are normalized")
    elif cmath.isclose(train_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and cmath.isclose(train_std, 1, rel_tol=1e-3, abs_tol=1e-6) and (not cmath.isclose(test_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and not cmath.isclose(test_std, 1, rel_tol=1e-3, abs_tol=1e-6)):
        score = normalizationscore_mappings["training_standardized"]
        properties["normalization"] = info("Normalization", "Training data are standardized")
    elif cmath.isclose(train_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and cmath.isclose(train_std, 1, rel_tol=1e-3, abs_tol=1e-6) and (cmath.isclose(test_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and cmath.isclose(test_std, 1, rel_tol=1e-3, abs_tol=1e-6)):
        score = normalizationscore_mappings["training_and_test_standardize"]
        properties["normalization"] = info("Normalization", "Training and Testing data are standardized")
    elif any(X_train < 0) or any(X_train > 1):
        score = normalizationscore_mappings["None"]
        properties["normalization"] = info("Normalization", "None")
    elif not (any(X_train < 0) or any(X_train > 1)) and (any(X_test < 0) or any(X_test > 1)):
        score = normalizationscore_mappings["training_normal"]
        properties["normalization"] = info("Normalization", "Training data are normalized")
    return result(score=score, properties=properties)

"""########################################TEST VALUES#############################################
train,test,mappings=r"Backend/algorithms/supervised/TestValues/train.csv",r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
print(get_normalization_score_supervised(training_dataset=train,test_dataset=test,mappings=mappings))"""


