def get_accountability_score_supervised(model=not None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    
    
    import sys
    sys.path.extend([r"Backend",r"Backend/algorithms",r"Backend/algorithms/supervised", r"Backend/algorithms/supervised/Functions",r"Backend/algorithms/supervised/Functions/Accountability"])
    # from algorithms.supervised.Functions.helpers_supervised import import_functions_from_folder
    # info,result,accountability_functions=import_functions_from_folder(['Accountability'])
    # print('functions:', accountability_functions)
    # get_normalization_score,get_missing_data_score, get_regularization_score,get_train_test_split_score,get_factsheetcomplettness_score=accountability_functions['normalizationscoresupervised'],accountability_functions['missingdatascoresupervised'],accountability_functions['regularizationscoresupervised'],accountability_functions['traintestsplitscoresupervised'],accountability_functions['factsheetcompletenessscoresupervised']
    import collections
    result = collections.namedtuple('result', 'score properties')

    from algorithms.supervised.Functions.Accountability.NormalizationScore_supervised import get_normalization_score_supervised
    from algorithms.supervised.Functions.Accountability.MissingDataScore_supervised import get_missing_data_score_supervised
    from algorithms.supervised.Functions.Accountability.RegularizationScore_supervised import get_regularization_score_supervised
    from algorithms.supervised.Functions.Accountability.TrainTestSplitScore_supervised import get_train_test_split_score_supervised
    from algorithms.supervised.Functions.Accountability.FactSheetCompletnessScore_supervised import get_factsheet_completeness_score_supervised
    output = dict(  
        normalization  = get_normalization_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
        missing_data  = get_missing_data_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
        regularization  = get_regularization_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
        train_test_split  = get_train_test_split_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
        factsheet_completeness  = get_factsheet_completeness_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
    )
# is this right? hello... is this right? or worng? nobody? wait
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)

"""########################################TEST VALUES#############################################
model,train,test,factsheet,mappigns=r"Backend/algorithms/supervised/TestValues/model.pkl", r"Backend/algorithms/supervised/TestValues/train.csv", r"Backend/algorithms/supervised/TestValues/test.csv", r"Backend/algorithms/supervised/TestValues/factsheet.json", r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
print(get_accountability_score_supervised(model=model,training_dataset=train,test_dataset=test,factsheet=factsheet,mappings=mappigns))"""
