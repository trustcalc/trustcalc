def get_statistical_parity_difference_score_supervised(model=not None, training_dataset=not None, test_dataset=None, factsheet=not None, mappings=not None,target_column=None, outliers_data=None, thresholds=[0.05, 0.16, 0.28, 0.4], outlier_thresholds=None, outlier_percentage=None, high_cor=0.9,print_details=None):
    try:
        import collections, pandas, tensorflow, algorithms.supervised.Functions.Fairness.helpers_fairness_supervised, numpy as np
    except:
        import collections, pandas, tensorflow, Functions.Fairness.helpers_fairness_supervised, numpy as np

    training_dataset, factsheet, model,mappings= pandas.read_csv(training_dataset),pandas.read_json(factsheet), pandas.read_pickle(model), pandas.read_json(mappings)
    info,result = collections.namedtuple('info', 'description value'), collections.namedtuple('result', 'score properties')

    def statistical_parity_difference_metric(model, training_dataset, factsheet):
        try: 
            properties = {}
            protected_feature, protected_values, target_column, favorable_outcomes = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.load_fairness_config(factsheet)
            
            minority = training_dataset[training_dataset[protected_feature].isin(protected_values)]
            minority_size = len(minority)
            majority = training_dataset[~training_dataset[protected_feature].isin(protected_values)]
            majority_size = len(majority)

            favored_minority = minority[minority[target_column].isin(favorable_outcomes)]
            favored_minority_size = len(favored_minority)

            favored_minority_ratio = favored_minority_size/minority_size

            favored_majority = majority[majority[target_column].isin(favorable_outcomes)]
            favored_majority_size = len(favored_majority)
            favored_majority_ratio = favored_majority_size/majority_size
            
            properties["|{x|x is protected, y_true is favorable}|"] = favored_minority_size
            properties["|{x|x is protected}|"] = minority_size
            properties["Favored Protected Group Ratio"] =  "P(y_true is favorable|protected=True) = {:.2f}%".format(favored_minority_ratio*100)
            properties["|{x|x is not protected, y_true is favorable}|"] = favored_majority_size
            properties["|{x|x is not protected}|"] = majority_size
            properties["Favored Unprotected Group Ratio"] =  "P(y_true is favorable|protected=False) = {:.2f}%".format(favored_majority_ratio*100)
            
            statistical_parity_difference = abs(favored_minority_ratio - favored_majority_ratio)
            return statistical_parity_difference, properties
        except Exception as e:
            print("ERROR in statistical_parity_difference_metric(): {}".format(e))
            raise
    
    if not thresholds:
        thresholds=  mappings["fairness"]["score_statistical_parity_difference"]["thresholds"]["value"]

    try: 
        score = np.nan
        properties = {}
        properties["Metric Description"] = "The spread between the percentage of observations from the majority group receiving a favorable outcome compared to the protected group. The closes this spread is to zero the better."
        properties["Depends on"] = "Training Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        statistical_parity_difference, spdm_properties = statistical_parity_difference_metric(model, training_dataset, factsheet)

        properties['----------'] = ''
        properties = properties|spdm_properties
        properties['-----------'] = ''
        properties["Formula"] =  "Statistical Parity Difference = |Favored Protected Group Ratio - Favored Unprotected Group Ratio|"
        properties["Statistical Parity Difference"] = "{:.2f}%".format(statistical_parity_difference*100)
        
        score = np.digitize(abs(statistical_parity_difference), thresholds, right=False) + 1 
        
        properties["Score"] = str(score)
        return result(score=int(score), properties=properties)
    except Exception as e:
        print("ERROR in statistical_parity_difference_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)})

"""########################################TEST VALUES#############################################
train,factsheet,model,mappings=r"Backend/algorithms/supervised/TestValues/train.csv",r"Backend/algorithms/supervised/TestValues/factsheet.json",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend/algorithms/supervised/TestValues/mappings.json"
print(get_statistical_parity_difference_score_supervised(training_dataset=train,factsheet=factsheet,model=model,mappings=mappings))"""
