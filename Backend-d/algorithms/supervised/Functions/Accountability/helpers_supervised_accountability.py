def accountabiltiy_parameter_file_loader(metric_function_name, training_dataset=None, test_dataset=None, model=None, factsheet=None, mappings=None):
    import numpy as np
    import collections
    import pandas as pd

    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    data = {}

    if metric_function_name in ["get_factsheet_completness_score_supervised", "get_regularization_score_supervised"]:
        data = pd.read_json(factsheet)

    elif metric_function_name in ["get_missing_data_score_supervised", "get_normalization_score_supervised", "get_train_test_split_score_supervised"]:
        data.update([('training_dataset', pd.read_csv(training_dataset)), ('test_dataset',
                    pd.read_csv(test_dataset)), ('mappings', pd.read_json(mappings))])

    return {'np': np, 'collections': collections, 'pd': pd, 'info': info, 'result': result, 'data': data}
