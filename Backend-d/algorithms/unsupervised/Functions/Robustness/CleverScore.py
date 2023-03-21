def clever_score(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None,penalty_outlier=None, outlier_percentage=None, high_cor=None,print_details=None):
    import pandas as pd
    factsheet=pd.read_json(factsheet)
    mappings=pd.read_json(mappings)

    if not thresholds:
        thresholds = mappings["robustness"]["score_clever_score"]["thresholds"]["value"]
    import collections
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    from art.estimators.classification import KerasClassifier
    import numpy as np
    from art.metrics import clever_u

    """For a given Keras-NN model this function calculates the Untargeted-Clever score.
    It uses clever_u function from IBM art library.
    Returns a score according to the thresholds.
        Args:
            model: ML-model (Keras).
            train_data: pd.DataFrame containing the data.
            test_data: pd.DataFrame containing the data.
            threshold: list of threshold values

        Returns:
            Clever score
    """
    try:
        classifier = KerasClassifier(model, False)

        min_score = 100

        randomX = X_test.sample(10)
        randomX = np.array(randomX)

        for x in randomX:
            temp = clever_u(classifier=classifier, x=x, nb_batches=1, batch_size=1, radius=500, norm=1)
            if min_score > temp:
                min_score = temp
        score = np.digitize(min_score, thresholds) + 1
        return result(score=int(score), properties={"clever_score": info("CLEVER Score", "{:.2f}".format(min_score)),
                                                    "depends_on": info("Depends on", "Model")})
    except Exception as e:
        print(e)
        return result(score=1, properties={"non_computable": info("Non Computable Because",
                                                                       "Can only be calculated on Keras models.")})


"""
########################################TEST VALUES#############################################
import pandas as pd

train=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/train.csv"
test=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/test.csv"
outliers=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/outliers.csv"

model=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/model.joblib"
factsheet=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues/factsheet.json"

mapping_metrics_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
weights_metrics_feault=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_metrics_default.json"
weights_pillars_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_pillars_default.json"

a=clever_score(model,train,test,factsheet,mapping_metrics_default,thresholds=None)
print(a)
"""