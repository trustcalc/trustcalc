

def overfitting_score(model=not None, training_dataset=not None, test_dataset= None, factsheet=None, mappings=not None,target_column=None, outliers_data=not None, thresholds=None, outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    import collections, numpy,sys, pandas as pd
    sys.path.extend([r"Backend",r"Backend/algorithms",r"Backend/algorithms/unsupervised", r"Backend/algorithms/unsupervised/Functions", r"Backend/algorithms/unsupervised/Functions/Accountability",r"Backend/algorithms/unsupervised/Functions/Fairness",r"Backend/algorithms/unsupervised/Functions/Explainability",r"Backend/algorithms/unsupervised/Functions/Robustness"])
    from algorithms.unsupervised.Functions.Fairness.helpers_fairness_unsupervised import compute_outlier_ratio, get_threshold_mse_iqr, isKerasAutoencoder

    info,result = collections.namedtuple('info', 'description value'), collections.namedtuple('result', 'score properties')
    def read_model(solution_set_path):
        print("READ MODEL REACHED")
        import os
        from joblib import load
        MODEL_REGEX = "model.*"
        model_file = solution_set_path
        file_extension = os.path.splitext(model_file)[1]
        print("FILE EXTENSION: ",file_extension)

        # pickle_file_extensions = [".sav", ".pkl", ".pickle"]
        pickle_file_extensions = [".pkl"]
        if file_extension in pickle_file_extensions:
            model = pd.read_pickle(model_file)
            return model

        if (file_extension == ".joblib"):  # Check if a .joblib file needs to be loaded
            print("model_file: ", model_file)
            a=load(model_file)
            print("READ MODEL joblib REACHED")
            print("READ JOBLIB MODEl: ",a)
            return a
    
    training_dataset=pd.read_csv(training_dataset)
    outliers_data=pd.read_csv(outliers_data)
    model=read_model(model)
    mappings=pd.read_json(mappings)
    if not thresholds or type(thresholds)==bool:
        thresholds= mappings["fairness"]["score_overfitting"]["thresholds"]["value"]
    outlier_thresholds= get_threshold_mse_iqr(model, training_dataset) if isKerasAutoencoder(model) else 0
    print_details,outlier_percentage =True,0.1


    try:
        properties = {}
        properties['Metric Description'] = "Overfitting is present if the training accuracy is significantly higher than the test accuracy." \
                                           "this metric computes the mean value of the outlier ratio in the outlier data set and the relative outlier detection accuracy in the test data. Note that the overfitting score is only computet when there is little to no underfitting (underfitting score >= 3)"
        properties['Depends on'] = 'Model, Training Data, Test Data, Outliers Data'

        #compute underfitting score
        detection_ratio_train = compute_outlier_ratio(data=outliers_data,model=model,outlier_thresh=outlier_thresholds,print_details=False)
        detection_ratio_test = compute_outlier_ratio(data=outliers_data,model=model,outlier_thresh=outlier_thresholds,print_details=False)

        perc_diff = abs(detection_ratio_train - detection_ratio_test)
        underfitting_score = numpy.digitize(perc_diff, [0.1,0.05,0.025,0.01], right=False) + 1
        overfitting_score = numpy.nan
        if underfitting_score >= 3:
            # compute outlier ratio in outlier dataset
            detection_ratio_outliers = compute_outlier_ratio(data=outliers_data,model=model, outlier_thresh=outlier_thresholds)
            # compute outlier ratio in train dataset
            detection_ratio_test = compute_outlier_ratio(data=outliers_data,model=model, outlier_thresh=outlier_thresholds)
            perc_diff = abs(outlier_percentage - detection_ratio_test)
            training_accuracy = abs(outlier_percentage - perc_diff) / outlier_percentage
            
            mean = (detection_ratio_outliers + training_accuracy) / 2
            overfitting_score = numpy.digitize(mean, thresholds, right=False) + 1

            properties["Outliers Accuracy"] = "{:.2f}%".format(detection_ratio_outliers*100)
            properties["Test Accuracy"] = "{:.2f}%".format(detection_ratio_test*100)
            properties["Outliers Test Accuracy Difference"] = "{:.2f}%".format(perc_diff*100)

            if print_details:
                print("\t   OVERFITTING DETAILS")
                print("\t   outlier percentage in training data: ", outlier_percentage)
                print("\t   detected outlier ratio in validation dataset: %.4f" % detection_ratio_test)
                print("\t   training accuracy: %.4f" % training_accuracy)
                print("\t   detected outlier ratio in outlier dataset: %.4f" % detection_ratio_outliers)
                print("\t   mean value: %.4f" % mean)

            if overfitting_score == 5:
                properties["Conclusion"] = "Model is not overfitting"
            elif overfitting_score == 4:
                properties["Conclusion"] = "Model mildly overfitting"
            elif overfitting_score == 3:
                properties["Conclusion"] = "Model is slighly overfitting"
            elif overfitting_score == 2:
                properties["Conclusion"] = "Model is overfitting"
            else:
                properties["Conclusion"] = "Model is strongly overfitting"

            properties["Score"] = str(overfitting_score)
            return result(int(overfitting_score), properties=properties)
        else:
            properties = {"Non computable because": "The test accuracy is to low and if the model is underfitting to much it can't be overfitting at the same time."}
            properties["Outliers Detection Accuracy"] = "{:.2f}%".format(compute_outlier_ratio(data=outliers_data,model=model, outlier_thresh=outlier_thresholds)*100)
            return result(overfitting_score, properties= properties )
    except Exception as e:
        print("ERROR in overfitting_score(): {}".format(e))
        return result(score=numpy.nan, properties={"Non computable because": str(e)}) 


"""########################################TEST VALUES#############################################
train=r"Backend/algorithms/unsupervised/TestValues/train.csv"
test=r"Backend/algorithms/unsupervised/TestValues/test.csv"
outliers=r"Backend/algorithms/unsupervised/TestValues/outliers.csv"
model=r"Backend/algorithms/unsupervised/TestValues/model.joblib"
mapping_metrics_default=r"Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
a= overfitting_score(model=model, training_dataset=train, test_dataset=test, outliers_data=outliers, mappings=mapping_metrics_default,outlier_percentage=0.1, outlier_thresholds=0, print_details = False,thresholds=None)
print(a)"""
