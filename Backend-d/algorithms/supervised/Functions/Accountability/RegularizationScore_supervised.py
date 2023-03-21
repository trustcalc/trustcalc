def get_regularization_score_supervised(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    import sys,inspect
    sys.path.append(r"Backend/algorithms")
    from algorithms.supervised.Functions.Accountability.helpers_supervised_accountability import accountabiltiy_parameter_file_loader
    metric_fname,NOT_SPECIFIED = inspect.currentframe().f_code.co_name, "not specified"
    foo = accountabiltiy_parameter_file_loader(metric_function_name=metric_fname, factsheet=factsheet)
    np,info,result,factsheet,factsheet2 = foo['np'],foo['info'],foo['result'],foo['data'],foo["data"]["methodology"]["regularization"]

    def regularization_metric(factsheet):
        return factsheet2 if "methodology" in factsheet and "regularization" in factsheet["methodology"] else NOT_SPECIFIED

    regularization,score_map = regularization_metric(factsheet),{"elasticnet_regression": 5,"lasso_regression": 4,"Other": 3,NOT_SPECIFIED: np.nan}
    properties = {"dep" :info('Depends on','Factsheet'),"regularization_technique": info("Regularization technique", regularization)}
    score = score_map.get(regularization, 1)

    return result(score=score, properties=properties)

"""########################################TEST VALUES#############################################
factsheet=r"Backend/algorithms/supervised/TestValues/factsheet.json"
print(get_regularization_score_supervised(factsheet=factsheet))"""