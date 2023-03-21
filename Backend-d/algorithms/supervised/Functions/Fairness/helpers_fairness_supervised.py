class MissingFairnessDefinitionError(Exception):
    def __init__(self, message="Salary is not in (5000, 15000) range"):
        self.message = message
        super().__init__(self.message)

def compute_accuracy(model, dataset, factsheet):
    import sklearn, tensorflow,numpy as np
    
    try:
        protected_feature, protected_values, target_column, favorable_outcomes = load_fairness_config(factsheet)
        X_data = dataset.drop(target_column, axis=1)
        y_data = dataset[target_column]

        y_true = y_data.values.flatten()
        if (isinstance(model, tensorflow.keras.Sequential)):
            y_train_pred_proba = model.predict(X_data)
            y_pred = np.argmax(y_train_pred_proba, axis=1)
        else:

            y_pred = model.predict(X_data).flatten()
        return sklearn.metrics.accuracy_score(y_true, y_pred)
    except Exception as e:
        print("ERROR in compute_accuracy(): {}".format(e))
        raise

def false_positive_rates(model, test_dataset, factsheet):
    import tensorflow,numpy as np
    try: 
        properties = {}
        data = test_dataset.copy(deep=True)
        
        protected_feature, protected_values, target_column, favorable_outcomes = load_fairness_config(factsheet)
        
        X_data = data.drop(target_column, axis=1)
        if (isinstance(model, tensorflow.keras.Sequential)):
            y_pred_proba = model.predict(X_data)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_data).flatten()
        data['y_pred'] = y_pred.tolist()

        protected_group = data[data[protected_feature].isin(protected_values)]
        unprotected_group = data[~data[protected_feature].isin(protected_values)]

        #2. Compute the number of negative samples y_true=False for the protected and unprotected group.
        protected_group_true_unfavorable = protected_group[~protected_group[target_column].isin(favorable_outcomes)]
        unprotected_group_true_unfavorable = unprotected_group[~unprotected_group[target_column].isin(favorable_outcomes)]
        protected_group_n_true_unfavorable = len(protected_group_true_unfavorable)
        unprotected_group_n_true_unfavorable = len(unprotected_group_true_unfavorable)

        #3. Calculate the number of false positives for the protected and unprotected group
        protected_group_true_unfavorable_pred_favorable = protected_group_true_unfavorable[protected_group_true_unfavorable['y_pred'].isin(favorable_outcomes)]
        unprotected_group_true_unfavorable_pred_favorable = unprotected_group_true_unfavorable[unprotected_group_true_unfavorable['y_pred'].isin(favorable_outcomes)]
        protected_group_n_true_unfavorable_pred_favorable = len(protected_group_true_unfavorable_pred_favorable)
        unprotected_group_n_true_unfavorable_pred_favorable = len(unprotected_group_true_unfavorable_pred_favorable)

        #4. Calculate fpr for both groups.
        fpr_protected = protected_group_n_true_unfavorable_pred_favorable/protected_group_n_true_unfavorable
        fpr_unprotected = unprotected_group_n_true_unfavorable_pred_favorable/unprotected_group_n_true_unfavorable
        
        #5. Adding properties
        properties["|{x|x is protected, y_true is unfavorable, y_pred is favorable}|"] = protected_group_n_true_unfavorable_pred_favorable
        properties["|{x|x is protected, y_true is Unfavorable}|"] = protected_group_n_true_unfavorable
        properties["FPR Protected Group"] = "P(y_pred is favorable|y_true is unfavorable, protected=True) = {:.2f}%".format(fpr_protected*100) 
        properties["|{x|x is not protected, y_true is unfavorable, y_pred is favorable}|"] = unprotected_group_n_true_unfavorable_pred_favorable
        properties["|{x|x is not protected, y_true is unfavorable}|"] = unprotected_group_n_true_unfavorable
        properties["FPR Unprotected Group"] = "P(y_pred is favorable|y_true is unfavorable, protected=False) = {:.2f}%".format(fpr_unprotected*100)
            
        return fpr_protected, fpr_unprotected, properties

    except Exception as e:
        print("ERROR in false_positive_rates(): {}".format(e))
        raise



def true_positive_rates(model, test_dataset, factsheet):
    import tensorflow,numpy as np

    try: 
        properties = {}
        data = test_dataset.copy(deep=True)
        
        protected_feature, protected_values, target_column, favorable_outcomes = load_fairness_config(factsheet)
        
        X_data = data.drop(target_column, axis=1)
        if (isinstance(model, tensorflow.keras.Sequential)):
            y_pred_proba = model.predict(X_data)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_data).flatten()
        data['y_pred'] = y_pred.tolist()

        favored_samples = data[data[target_column].isin(favorable_outcomes)]
        protected_favored_samples = favored_samples[favored_samples[protected_feature].isin(protected_values)]
        unprotected_favored_samples = favored_samples[~favored_samples[protected_feature].isin(protected_values)]

        num_unprotected_favored_true = len(unprotected_favored_samples)
        num_unprotected_favored_pred = len(unprotected_favored_samples[unprotected_favored_samples['y_pred'].isin(favorable_outcomes)])
        tpr_unprotected = num_unprotected_favored_pred/num_unprotected_favored_true

        num_protected_favored_true = len(protected_favored_samples)
        num_protected_favored_pred = len(protected_favored_samples[protected_favored_samples['y_pred'].isin(favorable_outcomes)])
        tpr_protected = num_protected_favored_pred / num_protected_favored_true 
        
        # Adding properties
        properties["|{x|x is protected, y_true is favorable, y_pred is favorable}|"] = num_protected_favored_pred
        properties["|{x|x is protected, y_true is favorable}|"] = num_protected_favored_true
        properties["TPR Protected Group"] = "P(y_pred is favorable|y_true is favorable, protected=True) = {:.2f}%".format(tpr_protected*100) 
        properties["|{x|x is not protected, y_true is favorable, y_pred is favorable}|"] = num_unprotected_favored_pred
        properties["|{x|x is not protected, y_true is favorable}|"] = num_unprotected_favored_true
        properties["TPR Unprotected Group"] = "P(y_pred is favorable|y_true is favorable, protected=False) = {:.2f}%".format(tpr_unprotected*100)
        
        return tpr_protected, tpr_unprotected, properties

    except Exception as e:
        print("ERROR in true_positive_rates(): {}".format(e))
        raise

def load_fairness_config(factsheet):
    message = ""
    protected_feature = factsheet.get("fairness", {}).get("protected_feature", '')
    if not protected_feature:
        message += "Definition of protected feature is missing."
        
    protected_values = factsheet.get("fairness", {}).get("protected_values", [])
    if not protected_values:
        message += "Definition of protected_values is missing."
        
    target_column = factsheet.get("general", {}).get("target_column", '')
    if not target_column:
        message += "Definition of target column is missing."
        
    favorable_outcomes = factsheet.get("fairness", {}).get("favorable_outcomes", [])
    if not favorable_outcomes:
        message += "Definition of favorable outcomes is missing."
        
    if message:
        raise MissingFairnessDefinitionError(message)
    
    return protected_feature, protected_values, target_column, favorable_outcomes