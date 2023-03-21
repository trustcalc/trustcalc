def get_class_balance_score_supervised(model=None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=[0.05, 0.16, 0.28, 0.4], outlier_thresholds=None, outlier_percentage=None, high_cor=0.9,print_details=None):
    from algorithms.supervised.Functions.Fairness.helpers_fairness_supervised import load_fairness_config
    try:
        import numpy, collections, pandas, tensorflow, algorithms.supervised.Functions.Fairness.helpers_fairness_supervised
    except:
        import numpy, collections, pandas, tensorflow, Functions.Fairness.helpers_fairness_supervised

    training_data, factsheet= pandas.read_csv(training_dataset),pandas.read_json(factsheet)
    info,result = collections.namedtuple('info', 'description value'), collections.namedtuple('result', 'score properties')

    def class_balance_metric(training_data, factsheet):
        from scipy.stats import chisquare
        try:
            protected_feature, protected_values, target_column, favorable_outcomes = algorithms.supervised.Functions.Fairness.helpers_fairness_supervised.load_fairness_config(factsheet)
            absolute_class_occurences = training_data[target_column].value_counts().sort_index().to_numpy()
            significance_level = 0.05
            p_value = chisquare(absolute_class_occurences, ddof=0, axis=0).pvalue

            if p_value < significance_level:
                #The data does not follow a unit distribution
                return 0
            else:
                #We can not reject the null hypothesis assuming that the data follows a unit distribution"
                return 1
        except Exception as e:
            print("ERROR in class_balance_metric(): {}".format(e))
            raise

    try:
        class_balance = class_balance_metric(training_data, factsheet)
        properties = {}
        properties['Metric Description'] = "Measures how well the training data is balanced or unbalanced"
        properties['Depends on'] = 'Training Data'
        if(class_balance == 1):
            score = 5
        else:
            score = 1
        
        properties["Score"] = str(score)
        return result(score=score, properties=properties)     
    except Exception as e:
        print("ERROR in class_balance_score(): {}".format(e))
        return result(score=numpy.nan, properties={"Non computable because": str(e)}) 

"""########################################TEST VALUES#############################################
train,factsheet=r"Backend/algorithms/supervised/TestValues/train.csv",r"Backend/algorithms/supervised/TestValues/factsheet.json"
print(get_class_balance_score_supervised(training_dataset=train,factsheet=factsheet))"""