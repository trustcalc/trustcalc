def get_equal_opportunity_difference_score_supervised(model=None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=[0.05, 0.16, 0.28, 0.4], outlier_thresholds=None, outlier_percentage=None, high_cor=0.9,print_details=None):
    try:
        import numpy, collections, pandas, tensorflow, algorithms.supervised.Functions.Fairness.helpers_fairness_supervised
    except:
        import numpy, collections, pandas, tensorflow, Functions.Fairness.helpers_fairness_supervised

    test_dataset,model, factsheet,mappings= pandas.read_csv(test_dataset),pandas.read_pickle(model),pandas.read_json(factsheet), pandas.read_json(mappings)
    info,result = collections.namedtuple('info', 'description value'), collections.namedtuple('result', 'score properties')

    if not thresholds:
        thresholds=  mappings["fairness"]["score_average_odds_difference"]["thresholds"]["value"]

    try:
        properties = {}
        score=numpy.nan
        properties["Metric Description"] = "Difference in true positive rates between protected and unprotected group."
        properties["Depends on"] = "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        tpr_protected, tpr_unprotected, tpr_properties = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.true_positive_rates(model=model, test_dataset=test_dataset, factsheet=factsheet)
        
        properties['----------'] = ''
        properties = properties|tpr_properties 
        equal_opportunity_difference = abs(tpr_protected - tpr_unprotected)
        properties['-----------'] = ''
        
        properties["Formula"] = "Equal Opportunity Difference = |TPR Protected Group - TPR Unprotected Group|"
        properties["Equal Opportunity Difference"] = "{:.2f}%".format(equal_opportunity_difference*100)

        score = numpy.digitize(abs(equal_opportunity_difference), thresholds, right=False) + 1 
        
        properties["Score"] = str(score)
        return result(score=int(score), properties=properties) 
    except Exception as e:
        print("ERROR in equal_opportunity_difference_score(): {}".format(e))
        return result(score=numpy.nan, properties={"Non computable because": str(e)})

"""########################################TEST VALUES#############################################
test,model,factsheet,mappings=r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend/algorithms/supervised/TestValues/factsheet.json",r"Backend/algorithms/supervised/TestValues/mappings.json"
print(get_equal_opportunity_difference_score_supervised(test_dataset=test,model=model,factsheet=factsheet,mappings=mappings))"""
