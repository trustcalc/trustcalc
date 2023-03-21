def get_missing_data_score_supervised(model=None, training_dataset=not None, test_dataset=not None, factsheet=None, mappings=not None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None, print_details=None):
    import sys
    import inspect
    sys.path.append(r"Backend/algorithms")
    metric_fname = inspect.currentframe().f_code.co_name
    from algorithms.supervised.Functions.Accountability.helpers_supervised_accountability import accountabiltiy_parameter_file_loader
    foo = accountabiltiy_parameter_file_loader(
        metric_function_name=metric_fname, training_dataset=training_dataset, test_dataset=test_dataset, mappings=mappings)
    info, np, result, training_dataset, test_dataset,  = foo['info'], foo['np'], foo['result'], foo['data'][
        'training_dataset'], foo['data']['test_dataset'] 
    try:
        missing_data_mappings=foo['data']['mappings']["accountability"]["score_missing_data"]["mappings"]["value"]
    except:
        missing_data_mappings=foo['data']['mappings']["methodology"]["score_missing_data"]["mappings"]["value"]

    try:
        missing_values = training_dataset.isna().sum().sum() + \
            test_dataset.isna().sum().sum()
        score = missing_data_mappings["null_values_exist"] if missing_values > 0 else missing_data_mappings["no_null_values"]
        return result(score=score, properties={"dep": info('Depends on', 'Training Data'), "null_values": info("Number of the null values", "{}".format(missing_values))})
    except:
        return result(score=np.nan, properties={})


"""########################################TEST VALUES#############################################
train,test,mappings=r"Backend/algorithms/supervised/TestValues/train.csv",r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
print(get_missing_data_score_supervised(training_dataset=train,test_dataset=test,mappings=mappings))"""
