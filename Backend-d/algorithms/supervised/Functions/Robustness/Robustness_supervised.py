def get_robustness_score_supervised(model=not None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    from .ERCarliniWagnerScore_supervised import get_er_carlini_wagner_score_supervised
    from .CleverScore_supervised import get_clever_score_supervised
    from .CliqueMethodScore_supervised import get_clique_method_supervised
    from .ConfidenceScore_supervised import get_confidence_score_supervised
    from .ERDeepFoolAttackScore_supervised import get_deepfool_attack_score_supervised
    from .ERFastGradientAttackScore_supervised import get_er_fast_gradient_attack_score_supervised
    from .LossSensitivityScore_supervised import get_loss_sensitivity_score_supervised

    import collections
    result = collections.namedtuple('result', 'score properties')

    output = dict(
        confidence_score =get_confidence_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        clique_method = get_clique_method_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        loss_sensitivity = get_loss_sensitivity_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        clever_score = get_clever_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        er_fast_gradient_attack = get_er_fast_gradient_attack_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        er_carlini_wagner_attack = get_er_carlini_wagner_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details),
        er_deepfool_attack = get_deepfool_attack_score_supervised(model=model, training_dataset=training_dataset, test_dataset=test_dataset, factsheet=factsheet, mappings=mappings,target_column=target_column, outliers_data=outliers_data, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, high_cor=high_cor,print_details=print_details)

    )
    
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return  result(score=scores, properties=properties)
"""########################################TEST VALUES#############################################
model,train,test,factsheet,mappigns=r"Backend/algorithms/supervised/TestValues/model.pkl", r"Backend/algorithms/supervised/TestValues/train.csv", r"Backend/algorithms/supervised/TestValues/test.csv", r"Backend/algorithms/supervised/TestValues/factsheet.json", r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
print(get_robustness_score_supervised(model=model,training_dataset=train,test_dataset=test,factsheet=factsheet,mappings=mappigns))"""




