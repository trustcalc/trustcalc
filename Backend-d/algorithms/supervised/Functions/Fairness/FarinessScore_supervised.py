def get_fairness_score_supervised(model=not None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    from .UnderfittingScore_supervised import get_underfitting_score_supervised
    from .AverageOddsDifferenceScore_supervised import get_average_odds_difference_score_supervised
    from .ClassBalanceScore_supervised import get_class_balance_score_supervised
    from .DisparateImpactScore_supervised import get_disparate_impact_score_supervised
    from .EqualOpportunityDifferenceScore_supervised import get_equal_opportunity_difference_score_supervised
    from .OverfittingScore_supervised import get_overfitting_score_supervised
    from .StatisticalParityDifferenceScore import get_statistical_parity_difference_score_supervised
    
    import collections, numpy
    result = collections.namedtuple('result', 'score properties')
    numpy.random.seed(0)
    
    output = dict(
        underfitting = get_underfitting_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        overfitting = get_overfitting_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        statistical_parity_difference = get_statistical_parity_difference_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        equal_opportunity_difference = get_equal_opportunity_difference_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        average_odds_difference = get_average_odds_difference_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        disparate_impact = get_disparate_impact_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        class_balance = get_class_balance_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details)
    )
    
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return  result(score=scores, properties=properties)
   
"""#######################################TEST VALUES#############################################
model,train,test,factsheet,mappigns=r"Backend/algorithms/supervised/TestValues/model.pkl", r"Backend/algorithms/supervised/TestValues/train.csv", r"Backend/algorithms/supervised/TestValues/test.csv", r"Backend/algorithms/supervised/TestValues/factsheet.json", r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
print(get_fairness_score_supervised(model=model,training_dataset=train,test_dataset=test,factsheet=factsheet,mappings=mappigns))"""
