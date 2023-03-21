def get_feature_relevance_score_supervised(model=not None, training_dataset=not None, test_dataset=None, factsheet= None, mappings=None,target_column=None, outliers_data=None, thresholds=[0.05, 0.1, 0.2, 0.3], outlier_thresholds=0.03, penalty_outlier=0.5,outlier_percentage=None, high_cor=None,print_details=None):
    import collections, pandas, numpy as np
    info,result = collections.namedtuple('info', 'description value'),collections.namedtuple('result', 'score properties')
    model,train_data=pandas.read_pickle(model),pandas.read_csv(training_dataset)

    if not thresholds:
        thresholds=thresholds=[0.05, 0.1, 0.2, 0.3]
    if not outlier_thresholds:
        outlier_thresholds=0.03
    
    pandas.options.mode.chained_assignment = None  
    train_data = train_data.copy()
    if target_column:
        X_train = train_data.drop(target_column, axis=1)
        y_train = train_data[target_column]
    else:
        X_train = train_data.iloc[:,:-1]
        y_train = train_data.iloc[:,-1: ]
        
    scale_factor = 1.5
    distri_threshold = 0.6
    if (type(model).__name__ == 'LogisticRegression') or (type(model).__name__ == 'LinearRegression'): 
        model.max_iter =1000
        model.fit(X_train, y_train.values.ravel())
        importance = model.coef_[0]
        
    elif  (type(model).__name__ == 'RandomForestClassifier') or (type(model).__name__ == 'DecisionTreeClassifier'):
         importance=model.feature_importances_
         
    else:
        return result(score= 1, properties={"dep" :info('Depends on','Training Data and Model')}) 
   
    # absolut values
    importance = abs(importance)
    
    feat_labels = X_train.columns
    indices = np.argsort(importance)[::-1]
    feat_labels = feat_labels[indices]

    importance = importance[indices]
    
    # calculate quantiles for outlier detection
    q1, q2, q3 = np.percentile(importance, [25, 50 ,75])
    lower_threshold , upper_threshold = q1 - scale_factor*(q3-q1),  q3 + scale_factor*(q3-q1) 
    
    #get the number of outliers defined by the two thresholds
    n_outliers = sum(map(lambda x: (x < lower_threshold) or (x > upper_threshold), importance))
    
    # percentage of features that concentrate distri_threshold percent of all importance
    pct_dist = sum(np.cumsum(importance) < 0.6) / len(importance)
    
    try:
        dist_score = np.digitize(pct_dist, thresholds, right=False) + 1 
    except:
        dist_score=1
    if n_outliers/len(importance) >= outlier_thresholds:
        dist_score -= penalty_outlier
    
    score =  max(dist_score,1)
   
    properties = {"dep" :info('Depends on','Training Data and Model'),
        "n_outliers":  info("number of outliers in the importance distribution",int(n_outliers)),
                  "pct_dist":  info("percentage of feature that make up over 60% of all features importance", "{:.2f}%".format(100*pct_dist)),
                  "importance":  info("feature importance", {"value": list(importance), "labels": list(feat_labels)})
                  }
    
    return result(score=int(score), properties=properties)

"""########################################TEST VALUES#############################################
train,model=r"Backend/algorithms/supervised/TestValues/train.csv",r"Backend/algorithms/supervised/TestValues/model.pkl"
print(get_feature_relevance_score_supervised(training_dataset=train, model=model))"""