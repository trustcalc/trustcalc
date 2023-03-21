def analyse(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, penalty_outlier=None, outlier_percentage=None, high_cor=None, print_details=None):
    import sys
    # general+supervised
    sys.path.extend([r"Backend", r"Backend/algorithms", r"Backend/algorithms/supervised", r"Backend/algorithms/supervised/Functions", r"Backend/algorithms/supervised/Functions/Accountability",
                    r"Backend/algorithms/supervised/Functions/Fairness", r"Backend/algorithms/supervised/Functions/Explainability", r"Backend/algorithms/supervised/Functions/Robustness"])
    # unsupervised
    sys.path.extend([r"Backend/algorithms/unsupervised", r"Backend/algorithms/unsupervised/Functions", r"Backend/algorithms/unsupervised/Functions/Accountability",
                    r"Backend/algorithms/unsupervised/Functions/Fairness", r"Backend/algorithms/unsupervised/Functions/Explainability", r"Backend/algorithms/unsupervised/Functions/Robustness"])

    sys.path.append(
        r"Backend/algorithms/unsupervised/Functions/Accountability")
    from .NormalizationScore import normalization_score

    from .MissingDataScore import missing_data_score
    from .RegularizationScore import regularization_score
    from .TrainTestSplitScore import train_test_split_score
    from .FactSheetCompletnessScore import get_factsheet_completeness_score_unsupervised

    import pandas as pd
    import collections
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')

    print_details = True

    output = dict(
        normalization=normalization_score(
            model, training_dataset, test_dataset, factsheet, mappings),
        missing_data=missing_data_score(
            model, training_dataset, test_dataset, factsheet, mappings),
        regularization=regularization_score(
            model, training_dataset, test_dataset, factsheet, mappings, print_details=False),
        train_test_split=train_test_split_score(
            model, training_dataset, test_dataset, factsheet, mappings),
        factsheet_completeness=get_factsheet_completeness_score_unsupervised(
            model, training_dataset, test_dataset, factsheet, mappings),
    )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return result(score=scores, properties=properties)


"""
########################################TEST VALUES#############################################
import pandas as pd

train=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/train.csv"
test=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/test.csv"
outliers=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/outliers.csv"

model=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/model.joblib"
factsheet=r"Functions_Trust/Backend/algorithms/unsupervised/TestValues2/factsheet.json"

mapping_metrics_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
weights_metrics_feault=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_metrics_default.json"
weights_pillars_default=r"Functions_Trust/Backend/algorithms/unsupervised/Mapping&Weights/weights_pillars_default.json"

a=analyse(model, train, test, outliers, factsheet, mapping_metrics_default)
print(a)
"""
