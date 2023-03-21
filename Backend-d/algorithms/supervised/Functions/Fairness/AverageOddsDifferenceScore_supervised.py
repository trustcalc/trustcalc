def get_average_odds_difference_score_supervised(model=None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=[0.05, 0.16, 0.28, 0.4], outlier_thresholds=None, outlier_percentage=None, high_cor=0.9,print_details=None):
    def read_model(solution_set_path):
        import os
        import pandas as pd
        import joblib as jb
        MODEL_REGEX = "model.*"
        model_file = solution_set_path
        file_extension = os.path.splitext(model_file)[1]
        # pickle_file_extensions = [".sav", ".pkl", ".pickle"]
        pickle_file_extensions = [".pkl"]
        
        if file_extension in pickle_file_extensions:
            with open(model_file, 'rb') as file:
                model = pd.read_pickle(file)
            return model
        try:
            if file_extension == ".joblib":  # Check if a .joblib file needs to be loaded
                return jb.load(model_file)
        except:
            pass
    import numpy, collections, pandas, tensorflow, algorithms.supervised.Functions.Fairness.helpers_fairness_supervised

    print('mapname:', mappings)
    test_dataset,model, factsheet, mappings= pandas.read_csv(test_dataset),pandas.read_pickle(model), pandas.read_json(factsheet), pandas.read_json(mappings)
    info,result = collections.namedtuple('info', 'description value'), collections.namedtuple('result', 'score properties')

    print('map:', mappings)
    if not thresholds:
        thresholds=  mappings["fairness"]["score_average_odds_difference"]["thresholds"]["value"]
   
    try:
        score = numpy.nan
        properties = {}
        properties["Metric Description"] = "Is the average of difference in false positive rates and true positive rates between the protected and unprotected group"
        properties["Depends on"] = "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        
        # model = read_model(model)
        # import pandas
        # test_dataset = pandas.read_csv(test_dataset)
        # factsheet = pandas.read_json(factsheet)
        print('averodd:', model, test_dataset, factsheet)
        fpr_protected, fpr_unprotected, fpr_properties = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.false_positive_rates(model, test_dataset, factsheet)
        tpr_protected, tpr_unprotected, tpr_properties = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.true_positive_rates(model, test_dataset, factsheet)
            
        properties["----------"] = ''
        properties = properties|fpr_properties
        properties = properties|tpr_properties
        properties['-----------'] = ''
        
        average_odds_difference = abs(((tpr_protected - tpr_unprotected) + (fpr_protected - fpr_unprotected))/2)
        properties["Formula"] = "Average Odds Difference = |0.5*(TPR Protected - TPR Unprotected) + 0.5*(FPR Protected - FPR Unprotected)|"
        properties["Average Odds Difference"] = "{:.2f}%".format(average_odds_difference*100)
        
        score = numpy.digitize(abs(average_odds_difference), thresholds, right=False) + 1 
        
        properties["Score"] = str(score)   
        return result(score=int(score), properties=properties) 
    except Exception as e:
        print("ERROR in average_odds_difference_score(): {}".format(e))
        return result(score=numpy.nan, properties={"Non computable because": str(e)})

"""########################################TEST VALUES#############################################
test,model,factsheet=r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend/algorithms/supervised/TestValues/factsheet.json",r"Backend/algorithms/supervised/TestValues/mappings.json"
print(get_average_odds_difference_score_supervised(model=model,factsheet=factsheet, test_dataset=test,mappings=mappings))"""