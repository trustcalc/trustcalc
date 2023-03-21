def get_disparate_impact_score_supervised(model=None, training_dataset= None, test_dataset=not None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=[0.05, 0.16, 0.28, 0.4], outlier_thresholds=None, outlier_percentage=None, high_cor=0.9,print_details=None):
    try:
        import numpy, collections, pandas, tensorflow, algorithms.supervised.Functions.Fairness.helpers_fairness_supervised
    except:
        import numpy, collections, pandas, tensorflow, algorithms.supervised.Functions.Fairness.helpers_fairness_supervised

    test_data,model, factsheet,mappings= pandas.read_csv(test_dataset),pandas.read_pickle(model),pandas.read_json(factsheet), pandas.read_json(mappings)
    info,result = collections.namedtuple('info', 'description value'), collections.namedtuple('result', 'score properties')
    
    if not thresholds:
        thresholds=  mappings["fairness"]["score_disparate_impact"]["thresholds"]["value"]

    def disparate_impact_metric(model, test_dataset, factsheet):
        import tensorflow,numpy as np
        try: 
            properties = {}
            data = test_dataset.copy(deep=True)
            
            protected_feature, protected_values, target_column, favorable_outcomes = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.load_fairness_config(factsheet)
            
            X_data = data.drop(target_column, axis=1)
            if (isinstance(model, tensorflow.keras.Sequential)):
                y_pred_proba = model.predict(X_data)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_data).flatten()
            data['y_pred'] = y_pred.tolist()

            protected_group = data[data[protected_feature].isin(protected_values)]
            unprotected_group = data[~data[protected_feature].isin(protected_values)]
            protected_group_size = len(protected_group)
            unprotected_group_size = len(unprotected_group)

            protected_favored_group = protected_group[protected_group['y_pred'].isin(favorable_outcomes)]
            unprotected_favored_group = unprotected_group[unprotected_group['y_pred'].isin(favorable_outcomes)]
            protected_favored_group_size = len(protected_favored_group)
            unprotected_favored_group_size = len(unprotected_favored_group)
            
            protected_favored_ratio = protected_favored_group_size / protected_group_size
            unprotected_favored_ratio = unprotected_favored_group_size / unprotected_group_size
            
            properties["|{x|x is protected, y_pred is favorable}"] = protected_favored_group_size
            properties["|{x|x is protected}|"] = protected_group_size
            properties["Protected Favored Ratio"] = "P(y_hat=favorable|protected=True) = {:.2f}%".format(protected_favored_ratio*100)
            properties["|{x|x is not protected, y_pred is favorable}|"] = unprotected_favored_group_size
            properties["|{x|x is not protected}|"] = unprotected_group_size
            properties["Unprotected Favored Ratio"] = "P(y_hat=favorable|protected=False) = {:.2f}%".format(unprotected_favored_ratio*100) 

            disparate_impact = abs(protected_favored_ratio / unprotected_favored_ratio)
            return disparate_impact, properties

        except Exception as e:
            print("ERROR in disparate_impact_metric(): {}".format(e))
            raise

    try:
        score = numpy.nan
        properties = {}
        properties["Metric Description"] = "Is quotient of the ratio of samples from the protected group receiving a favorable prediction divided by the ratio of samples from the unprotected group receiving a favorable prediction"
        properties["Depends on"] = "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        disparate_impact, dim_properties = disparate_impact_metric(model=model, test_dataset=test_data, factsheet=factsheet)
        
        properties["----------"] = ''
        properties = properties|dim_properties
        properties['-----------'] = ''
        
        properties["Formula"] = "Disparate Impact = Protected Favored Ratio / Unprotected Favored Ratio"
        properties["Disparate Impact"] = "{:.2f}".format(disparate_impact)

        score = numpy.digitize(disparate_impact, thresholds, right=False)+1
            
        properties["Score"] = str(score)
        return result(score=int(score), properties=properties) 
    except Exception as e:
        print("ERROR in disparate_impact_score(): {}".format(e))
        return result(score=1, properties={"Non computable because": str(e)})

"""########################################TEST VALUES#############################################
test,model,factsheet=r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/TestValues/model.pkl",r"Backend/algorithms/supervised/TestValues/factsheet.json",r"Backend/algorithms/supervised/TestValues/mappings.json"
print(get_disparate_impact_score_supervised(test_dataset=test,model=model,factsheet=factsheet,mappings=mappings))"""
