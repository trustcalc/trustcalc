def get_train_test_split_score_supervised(model=None, training_dataset=not None, test_dataset=not None, factsheet= None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    import sys,inspect,re
    sys.path.append(r"Backend/algorithms")
    from algorithms.supervised.Functions.Accountability.helpers_supervised_accountability import accountabiltiy_parameter_file_loader
    metric_fname = inspect.currentframe().f_code.co_name
    foo = accountabiltiy_parameter_file_loader(metric_function_name=metric_fname, training_dataset=training_dataset,test_dataset=test_dataset,mappings=mappings)
    info, result, training_dataset,test_dataset = foo['info'],foo['result'],foo['data']['training_dataset'],foo['data']['test_dataset']

    # let me check. perhaps you are wrong
    print('traintestsplitedata_now:', foo["properties"])  
    try:
        traintestsplit_mappings=foo['data']['mappings']["accountability"]["score_train_test_split"]["mappings"]["value"]
    except:
        try:
            traintestsplit_mappings=foo['data']['mappings']["methodology"]["score_train_test_split"]["mappings"]["value"] #shouldn'the try except catch the error it checks first for accoutnability then methodology key
        except:
            traintestsplit_mappings=foo['data']['properties']["methodology"]["score_train_test_split"]["mappings"]["value"] #shouldn'the try except catch the error it checks first for accoutnability then methodology key


    def train_test_split_metric(training_dataset, test_dataset):
        n_train,n_test = len(training_dataset),len(test_dataset) 
        n = n_train + n_test
        return round(n_train/n*100), round(n_test/n*100)
    
    try:
        training_data_ratio, test_data_ratio = train_test_split_metric(training_dataset, test_dataset)
        properties= {"dep" :info('Depends on','Training and Testing Data'),
            "train_test_split": info("Train test split", "{:.2f}/{:.2f}".format(training_data_ratio, test_data_ratio))}
        for k in traintestsplit_mappings.keys():
            thresholds = re.findall(r'\d+-\d+', k)
            for boundary in thresholds:
                [a, b] = boundary.split("-")
                if training_data_ratio >= int(a) and training_data_ratio < int(b):
                    score = traintestsplit_mappings[k]
        return result(score=score, properties=properties)
    except Exception as e:
        print(e)
        return result(score=1, properties={})

"""########################################TEST VALUES#############################################
train,test,mappings=r"Backend/algorithms/supervised/TestValues/train.csv",r"Backend/algorithms/supervised/TestValues/test.csv",r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
print(get_train_test_split_score_supervised(training_dataset=train,test_dataset=test,mappings=mappings))"""