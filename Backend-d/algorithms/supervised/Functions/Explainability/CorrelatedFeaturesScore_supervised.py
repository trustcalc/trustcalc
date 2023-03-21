def get_correlated_features_score_supervised(model=None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=None,target_column=None, outliers_data=None, thresholds=[0.05, 0.16, 0.28, 0.4], outlier_thresholds=None, outlier_percentage=None, high_cor=0.9,print_details=None):
    import collections, pandas, numpy as np
    info,result = collections.namedtuple('info', 'description value'),collections.namedtuple('result', 'score properties')
    train_data,test_data=pandas.read_csv(training_dataset),pandas.read_csv(test_dataset)

    test_data,train_data = test_data.copy(),train_data.copy()
     
    if target_column:
        X_test = test_data.drop(target_column, axis=1)
        X_train = train_data.drop(target_column, axis=1)
    else:
        X_test = test_data.iloc[:,:-1]
        X_train = train_data.iloc[:,:-1]
        
    
    df_comb = pandas.concat([X_test, X_train])
    corr_matrix = df_comb.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > high_cor)]
    
    pct_drop = len(to_drop)/len(df_comb.columns)
    
    bins = thresholds
    try:
        score = 5-np.digitize(pct_drop, bins, right=True) 
    except:
        score=1
    properties= {"dep" :info('Depends on','Training Data'),
        "pct_drop" : info("Percentage of highly correlated features", "{:.2f}%".format(100*pct_drop))}
    
    return  result(score=int(score), properties=properties)

"""########################################TEST VALUES#############################################
train,test=r"Backend/algorithms/supervised/TestValues/train.csv",r"Backend/algorithms/supervised/TestValues/test.csv"
print(get_correlated_features_score_supervised(training_dataset=train, test_dataset=test))"""