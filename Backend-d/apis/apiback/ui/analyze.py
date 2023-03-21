import json
import os
from rest_framework.views import APIView
from ...models import CustomUser, Scenario, ScenarioSolution
from ...views import BASE_DIR
from rest_framework.response import Response
from ..public import save_files_return_paths
from algorithms.supervised.Functions.Fairness.FarinessScore_supervised import get_fairness_score_supervised as analyse_fairness
from algorithms.supervised.Functions.Explainability.ExplainabilityScore_supervised import get_explainability_score_supervised as analyse_explainability
from algorithms.supervised.Functions.Accountability.AccountabilityScore_supervised import get_accountability_score_supervised as analyse_methodology
from algorithms.supervised.Functions.Robustness.Robustness_supervised import get_robustness_score_supervised as analyse_robustness
from sklearn import metrics
import numpy as np
import tensorflow as tf
from apis.apiback.ui.dashboard import unsupervised_FinalScore


class analyze(APIView):
    def get(self, request, id):

        print("User not exist.... Created new")
        # return Response(uploaddic)

    def post(self, request):
        uploaddic = {}

        ScenarioName = []
        LinktoDataset = []
        Description = []

        print("POST analyze request.data request:", request.data)
        if request.data is not None:
            userexist = CustomUser.objects.get(
                email=request.data['emailid'])
            scenario = Scenario.objects.filter(
                scenario_name=request.data['SelectScenario']).values()
            scenarioobj = ScenarioSolution.objects.filter(
                user_id=userexist.id).values()

            if scenario:
                for i in scenario:
                    if (i['scenario_name'] == request.data['SelectScenario']):
                        ScenarioName.append(i['scenario_name']),
                        Description.append(i['description']),

            uploaddic['ScenarioName'] = ScenarioName

            DEFAULT_TARGET_COLUMN_INDEX = -1
            import pandas as pd

            def get_performance_metrics(model, test_data, target_column, train_data, factsheet):
                print("GET PERFORMANCE METRICS REACHED")
                try:
                    model = pd.read_pickle(model)
                    print("PERFORMANCE METRICS MODEL: ", model)
                except:
                    print("MODEL ERROR")
                test_data = pd.read_csv(test_data)
                print("PERFORMANCE METRICS test_data: ", test_data)

                train_data = pd.read_csv(train_data)
                print("PERFORMANCE METRICS train_data: ", train_data)
                import pandas
                factsheet1 = os.path.join(BASE_DIR, factsheet)
                factsheet2 = pandas.read_json(factsheet)
                print("PERFORMANCE METRICS factsheet1: ", factsheet1)

                print("PERFORMANCE METRICS factsheet2: ", factsheet2)

                with open(factsheet, 'r') as g:
                    factsheet = json.loads(g.read())

                # y_test = test_data[target_column]
                # y_true = y_test.values.flatten()

                if target_column:
                    X_test = test_data.drop(target_column, axis=1)
                    y_test = test_data[target_column]
                else:
                    X_test = test_data.iloc[:, :DEFAULT_TARGET_COLUMN_INDEX]
                    y_test = test_data.reset_index(
                        drop=True).iloc[:, DEFAULT_TARGET_COLUMN_INDEX:]
                    y_true = y_test.values.flatten()
                if (isinstance(model, tf.keras.Sequential)):
                    y_pred_proba = model.predict(X_test)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = model.predict(X_test).flatten()
                    labels = np.unique(np.array([y_pred, y_true]).flatten())

                performance_metrics = pd.DataFrame({
                    "accuracy": [metrics.accuracy_score(y_true, y_pred)],
                    "global recall": [metrics.recall_score(y_true, y_pred, labels=labels, average="micro")],
                    "class weighted recall": [metrics.recall_score(y_true, y_pred, average="weighted")],
                    "global precision": [metrics.precision_score(y_true, y_pred, labels=labels, average="micro")],
                    "class weighted precision": [metrics.precision_score(y_true, y_pred, average="weighted")],
                    "global f1 score": [metrics.f1_score(y_true, y_pred, average="micro")],
                    "class weighted f1 score": [metrics.f1_score(y_true, y_pred, average="weighted")],
                }).round(decimals=2)

                uploaddic['accuracy'] = (
                    "%.2f" % metrics.accuracy_score(y_true, y_pred))
                uploaddic['globalrecall'] = ("%.2f" % metrics.recall_score(
                    y_true, y_pred, labels=labels, average="micro"))
                uploaddic['classweightedrecall'] = (
                    "%.2f" % metrics.recall_score(y_true, y_pred, average="weighted"))
                uploaddic['globalprecision'] = ("%.2f" % metrics.precision_score(
                    y_true, y_pred, labels=labels, average="micro"))
                uploaddic['classweightedprecision'] = (
                    "%.2f" % metrics.precision_score(y_true, y_pred, average="weighted"))
                uploaddic['globalf1score'] = (
                    "%.2f" % metrics.f1_score(y_true, y_pred, average="micro"))
                uploaddic['classweightedf1score'] = (
                    "%.2f" % metrics.f1_score(y_true, y_pred, average="weighted"))

                print("ERROR SURE")
                if "properties" in factsheet:
                    factsheet = factsheet["properties"]

                    properties = pd.DataFrame({
                        "Model Type": [factsheet["explainability"]["algorithm_class"]["clf_name"][1]],
                        "Train Test Split": [factsheet["methodology"]["train_test_split"]["train_test_split"][1]],
                        "Train / Test Data Size": str(train_data.shape[0]) + " samples / " + str(test_data.shape[0]) + " samples",
                        "Regularization Technique": [factsheet["methodology"]["regularization"]["regularization_technique"][1]],
                        "Normalization Technique": [factsheet["methodology"]["normalization"]["normalization"][1]],
                        "Number of Features": [factsheet["explainability"]["model_size"]["n_features"][1]],
                    })
                    uploaddic['ModelType'] = factsheet["explainability"]["algorithm_class"]["clf_name"][1]
                    uploaddic['TrainTestSplit'] = factsheet["methodology"]["train_test_split"]["train_test_split"][1]
                    uploaddic['DataSize'] = str(
                        train_data.shape[0]) + " samples / " + str(test_data.shape[0]) + " samples"
                    uploaddic['RegularizationTechnique'] = factsheet["methodology"]["regularization"]["regularization_technique"][1]
                    uploaddic['NormalizationTechnique'] = factsheet["methodology"]["normalization"]["normalization"][1]
                    uploaddic['NumberofFeatures'] = factsheet["explainability"]["model_size"]["n_features"][1]
                    properties = properties.transpose()
                    properties = properties.reset_index()
                    properties['index'] = properties['index'].str.title()
                    properties.rename(
                        columns={"index": "key", 0: "value"}, inplace=True)
                performance_metrics = performance_metrics.transpose()
                performance_metrics = performance_metrics.reset_index()
                performance_metrics['index'] = performance_metrics['index'].str.title(
                )
                performance_metrics.rename(
                    columns={"index": "key", 0: "value"}, inplace=True)
                return performance_metrics

            def detect_outliers_range(mdoel, df, threshold_mse):
                import numpy as np
                pred = mdoel.predict(df)
                mse = np.mean(np.power(df - pred, 2), axis=1)
                up_bound = threshold_mse[0]
                bottom_bound = threshold_mse[1]
                outliers = [(np.array(mse) < up_bound) &
                            (np.array(mse) > bottom_bound)]
                return outliers

            def isKerasAutoencoder(model):
                import keras
                return isinstance(model, keras.engine.functional.Functional)

            def isIsolationForest(model):
                from sklearn.ensemble import IsolationForest
                return isinstance(model, IsolationForest)

            def detect_outliers(mdoel, df, threshold_mse):
                if (len(threshold_mse) == 2):
                    return detect_outliers_range(mdoel, df, threshold_mse)
                pred = mdoel.predict(df)
                mse = np.mean(np.power(df - pred, 2), axis=1)
                outliers = [np.array(mse) < threshold_mse]
                return outliers

            def compute_outlier_matrix(model, data, outlier_thresh, print_details=False):
                if isKerasAutoencoder(model):
                    mad_outliers = detect_outliers(
                        model, data, outlier_thresh)[0]
                elif isIsolationForest(model):
                    mad_outliers = model.predict(data)
                else:
                    mad_outliers = model.predict(data)
                if print_details:
                    print("\t outlier matrix: ", mad_outliers)
                return mad_outliers


#here
            def get_performance_metrics_unsupervised(model, outliers, out_thresh):
                import pandas, joblib
                model=joblib.load(model)
                outliers=pandas.read_csv(outliers)
                performance_metrics = pd.DataFrame({
                    "Outlier Detection Ratio": [compute_outlier_matrix(model, outliers, out_thresh)],
                }).round(decimals=2)

                performance_metrics = performance_metrics.transpose()
                performance_metrics = performance_metrics.reset_index()
                performance_metrics['index'] = performance_metrics['index'].str.title()
                performance_metrics.rename(columns={"index": "key", 0: "value"}, inplace=True)
                print("PERFORMANCE METRICS UNSUPERVISED: ", performance_metrics)
                return performance_metrics





            def get_factsheet_completeness_score(factsheet):
                propdic = {}
                import collections
                info = collections.namedtuple('info', 'description value')
                result = collections.namedtuple('result', 'score properties')

                factsheet = save_files_return_paths(factsheet)[0]
                with open(factsheet, 'r') as g:
                    factsheet = json.loads(g.read())

                score = 0
                properties = {"dep": info('Depends on', 'Factsheet')}
                GENERAL_INPUTS = ["model_name", "purpose_description", "domain_description",
                                  "training_data_description", "model_information", "authors", "contact_information"]

                n = len(GENERAL_INPUTS)
                ctr = 0
                for e in GENERAL_INPUTS:
                    if "general" in factsheet and e in factsheet["general"]:
                        ctr += 1
                        properties[e] = info("Factsheet Property {}".format(
                            e.replace("_", " ")), "present")
                    else:
                        properties[e] = info("Factsheet Property {}".format(
                            e.replace("_", " ")), "missing")
                        score = round(ctr/n*5)

                return result(score=score, properties=properties)

            if scenarioobj:
                for i in scenarioobj:
                    if i['scenario_id'] == scenario[0]['id'] and i['solution_name'] == request.data['SelectSolution']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        path_traindata = i['training_file']
                        path_factsheet = i['factsheet_file']
                        Target = i['target_column']

            completeness_prop = get_factsheet_completeness_score(
                path_factsheet)

            uploaddic['modelname'] = completeness_prop[1]['model_name'][1]
            uploaddic['purposedesc'] = completeness_prop[1]['purpose_description'][1]
            uploaddic['trainingdatadesc'] = completeness_prop[1]['training_data_description'][1]
            uploaddic['modelinfo'] = completeness_prop[1]['model_information'][1]
            uploaddic['authors'] = completeness_prop[1]['authors'][1]
            uploaddic['contactinfo'] = completeness_prop[1]['contact_information'][1]

            def get_final_score(model, train_data, test_data, config_weights, mappings_config, factsheet, recalc=False):
                mappingConfig1 = mappings_config

                with open(mappings_config, 'r') as f:
                    mappings_config = json.loads(f.read())

                config_fairness = mappings_config["fairness"]
                config_explainability = mappings_config["explainability"]
                config_robustness = mappings_config["robustness"]
                config_methodology = mappings_config["methodology"]

                def trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, methodology_config, ):
                    output = dict(
                        fairness=analyse_fairness(
                            model, train_data, test_data, factsheet, config_fairness),
                        explainability=analyse_explainability(
                            model, train_data, test_data, config_explainability, factsheet),
                        robustness=analyse_robustness(
                            model, train_data, test_data, config_robustness, factsheet),
                        methodology=analyse_methodology(
                            model, train_data, test_data, factsheet, methodology_config)
                    )
                    scores = dict((k, v.score) for k, v in output.items())
                    properties = dict((k, v.properties)
                                      for k, v in output.items())

                    return result(score=scores, properties=properties)

                with open(mappingConfig1, 'r') as f:
                    default_map = json.loads(f.read())

                print('path:', factsheet)
                factsheet = save_files_return_paths(factsheet)[0]
                with open(factsheet, 'r') as g:
                    factsheet = json.loads(g.read())

                scores = []
                if default_map == mappings_config:
                    print('default map called')
                    if "scores" in factsheet.keys() and "properties" in factsheet.keys():
                        scores = factsheet["scores"]
                        properties = factsheet["properties"]
                else:
                    print('default map not called')
                    result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness,
                                                config_explainability, config_robustness, config_methodology, solution_type)
                    scores = result.score
                    factsheet["scores"] = scores
                    properties = result.properties
                    factsheet["properties"] = properties

                final_scores = dict()
                with open(config_weights, 'r') as n:
                    config_weights = json.loads(n.read())

                fairness_score = 0
                explainability_score = 0
                robustness_score = 0
                methodology_score = 0
                for pillar in scores.items():

                    if pillar[0] == 'fairness':
                        uploaddic['underfitting'] = int(
                            pillar[1]['underfitting'])
                        uploaddic['overfitting'] = int(
                            pillar[1]['overfitting'])
                        uploaddic['statistical_parity_difference'] = int(
                            pillar[1]['statistical_parity_difference'])
                        uploaddic['equal_opportunity_difference'] = int(
                            pillar[1]['equal_opportunity_difference'])
                        uploaddic['average_odds_difference'] = int(
                            pillar[1]['average_odds_difference'])
                        uploaddic['disparate_impact'] = int(
                            pillar[1]['disparate_impact'])
                        uploaddic['class_balance'] = int(
                            pillar[1]['class_balance'])

                        fairness_score = int(
                            pillar[1]['underfitting'])*0.35 + int(pillar[1]['overfitting'])*0.15
                        + int(pillar[1]['statistical_parity_difference'])*0.15 + \
                            int(pillar[1]['equal_opportunity_difference'])*0.2
                        + int(pillar[1]['average_odds_difference']) * \
                            0.1 + int(pillar[1]['disparate_impact'])*0.1
                        + int(pillar[1]['class_balance'])*0.1

                        uploaddic['fairness_score'] = fairness_score
                        print("Fairness Score is:", fairness_score)

                    if pillar[0] == 'explainability':
                        algorithm_class = 0
                        correlated_features = 0
                        model_size = 0
                        feature_relevance = 0

                        if str(pillar[1]['algorithm_class']) != 'nan':
                            algorithm_class = int(
                                pillar[1]['algorithm_class'])*0.55

                        if str(pillar[1]['correlated_features']) != 'nan':
                            correlated_features = int(
                                pillar[1]['correlated_features'])*0.15

                        if str(pillar[1]['model_size']) != 'nan':
                            model_size = int(pillar[1]['model_size'])*5

                        if str(pillar[1]['feature_relevance']) != 'nan':
                            feature_relevance = int(
                                pillar[1]['feature_relevance'])*0.15

                        explainability_score = algorithm_class + \
                            correlated_features + model_size + feature_relevance

                        uploaddic['algorithm_class'] = algorithm_class
                        uploaddic['correlated_features'] = correlated_features
                        uploaddic['model_size'] = model_size
                        uploaddic['feature_relevance'] = feature_relevance
                        uploaddic['explainability_score'] = explainability_score
                        print("explainability Score is:", explainability_score)

                    if pillar[0] == 'robustness':

                        confidence_score = 0
                        clique_method = 0
                        loss_sensitivity = 0
                        clever_score = 0
                        er_fast_gradient_attack = 0
                        er_carlini_wagner_attack = 0
                        er_deepfool_attack = 0

                        if str(pillar[1]['confidence_score']) != 'nan':
                            confidence_score = int(
                                pillar[1]['confidence_score'])*0.2

                        if str(pillar[1]['clique_method']) != 'nan':
                            clique_method = int(pillar[1]['clique_method'])*0.2

                        if str(pillar[1]['loss_sensitivity']) != 'nan':
                            loss_sensitivity = int(
                                pillar[1]['loss_sensitivity'])*0.2

                        if str(pillar[1]['clever_score']) != 'nan':
                            clever_score = int(pillar[1]['clever_score'])*0.2

                        if str(pillar[1]['er_fast_gradient_attack']) != 'nan':
                            er_fast_gradient_attack = int(
                                pillar[1]['er_fast_gradient_attack'])*0.2

                        if str(pillar[1]['er_carlini_wagner_attack']) != 'nan':
                            er_carlini_wagner_attack = int(
                                pillar[1]['er_carlini_wagner_attack'])*0.2

                        if str(pillar[1]['er_deepfool_attack']) != 'nan':
                            er_deepfool_attack = int(
                                pillar[1]['er_deepfool_attack'])*0.2

                        robustness_score = confidence_score + clique_method + loss_sensitivity + \
                            clever_score + er_fast_gradient_attack + \
                            er_carlini_wagner_attack + er_deepfool_attack

                        uploaddic['confidence_score'] = confidence_score
                        uploaddic['clique_method'] = clique_method
                        uploaddic['loss_sensitivity'] = loss_sensitivity
                        uploaddic['clever_score'] = clever_score
                        uploaddic['er_fast_gradient_attack'] = er_fast_gradient_attack
                        uploaddic['er_carlini_wagner_attack'] = er_carlini_wagner_attack
                        uploaddic['er_deepfool_attack'] = er_deepfool_attack
                        uploaddic['robustness_score'] = robustness_score
                        print("robustness Score is:", robustness_score)

                    if pillar[0] == 'methodology':
                        normalization = 0
                        missing_data = 0
                        regularization = 0
                        train_test_split = 0
                        factsheet_completeness = 0

                        if str(pillar[1]['normalization']) != 'nan':
                            normalization = int(pillar[1]['normalization'])*0.2

                        if str(pillar[1]['missing_data']) != 'nan':
                            missing_data = int(pillar[1]['missing_data'])*0.2

                        if str(pillar[1]['regularization']) != 'nan':
                            regularization = int(
                                pillar[1]['regularization'])*0.2

                        if str(pillar[1]['train_test_split']) != 'nan':
                            train_test_split = int(
                                pillar[1]['train_test_split'])*0.2

                        if str(pillar[1]['factsheet_completeness']) != 'nan':
                            factsheet_completeness = int(
                                pillar[1]['factsheet_completeness'])*0.2

                        methodology_score = normalization + missing_data + \
                            regularization + train_test_split + factsheet_completeness

                        uploaddic['normalization'] = normalization
                        uploaddic['missing_data'] = missing_data
                        uploaddic['regularization'] = regularization
                        uploaddic['train_test_split'] = train_test_split
                        uploaddic['factsheet_completeness'] = factsheet_completeness
                        uploaddic['methodology_score'] = (
                            "%.2f" % methodology_score)
                        print("methodology Score is:", methodology_score)

                trust_score = fairness_score*0.25 + explainability_score * \
                    0.25 + robustness_score*0.25 + methodology_score*0.25
                uploaddic['trust_score'] = trust_score
                print("Trust Score is:", trust_score)

            path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
            path_traindata = os.path.join(
                BASE_DIR, 'apis/TestValues/train.csv')
            path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
            path_factsheet = os.path.join(
                BASE_DIR, 'apis/TestValues/factsheet.json')
            config_weights = os.path.join(
                BASE_DIR, 'apis/TestValues/Weights/default.json')
            mappings_config = os.path.join(
                BASE_DIR, 'apis/TestValues/Mappings/default.json')
            factsheet = os.path.join(
                BASE_DIR, 'apis/TestValues/Mappings/default.json')

            if scenarioobj:
                for i in scenarioobj:
                    if (i['scenario_id'] == scenario[0]['id'] and i['solution_name'] == request.data['SelectSolution']):
                        path_testdata = i["test_file"]
                        path_module = i["model_file"]
                        path_traindata = i["training_file"]
                        path_factsheet = i["factsheet_file"]
                        path_outliersdata = i['outlier_data_file']
                        soulutionType = i['solution_type']
                        weights_metrics = i['weights_metrics']
                        weights_pillars = i['weights_pillars']
                        # target_column = i['target_column']

            path_module, path_traindata, path_testdata, path_factsheet, path_outliersdata, weights_metrics, weights_pillars = save_files_return_paths(
                path_module, path_traindata, path_testdata, path_factsheet, path_outliersdata, weights_metrics, weights_pillars)
            # to show the values you wanted in 3) is here.. if I open downline then the error occured. but I don't know why. you please check it. ok ok?

            #for unsupervised another functions has to be used, will send it per skype ok. sent per skypeokok
            if (soulutionType == 'unsupervised'):
                print("Performance_Metrics reslt:", get_performance_metrics_unsupervised(path_module, path_outliersdata, None))
            else:
                print("Performance_Metrics reslt:", get_performance_metrics(path_module, path_testdata, target_column, path_traindata, path_factsheet))

            if (soulutionType == 'unsupervised'):
                result = unsupervised_FinalScore(
                    path_module, path_traindata, path_testdata, path_outliersdata, path_factsheet, mappings_config, weights_metrics, weights_pillars)
                uploaddic['disparate_impact'] = result['Metricscores']['Metricscores']['Fairnessscore']['Disparateimpactscore']
                uploaddic['underfitting'] = result['Metricscores']['Metricscores']['Fairnessscore']['Underfittingscore']
                uploaddic['overfitting'] = result['Metricscores']['Metricscores']['Fairnessscore']['Overfittingscore']
                uploaddic['statistical_parity_difference'] = result['Metricscores'][
                    'Metricscores']['Fairnessscore']['Statisticalparitydifferencescore']
                uploaddic['fairness_score'] = result['Pillarscores']['Fairnessscore']
                uploaddic['normalization'] = result['Metricscores']['Metricscores']['Accountabilityscore']['Normalizationscore']
                uploaddic['missing_data'] = result['Metricscores']['Metricscores']['Accountabilityscore']['Missingdatascore']
                uploaddic['regularization'] = result['Metricscores']['Metricscores']['Accountabilityscore']['Regularizationscore']
                uploaddic['train_test_split'] = result['Metricscores']['Metricscores']['Accountabilityscore']['Traintestsplitscore']
                uploaddic['factsheet_completeness'] = result['Metricscores']['Metricscores']['Accountabilityscore']['Factsheecompletnessscore']
                uploaddic['methodology_score'] = result['Pillarscores']['Accountabilityscore']
                uploaddic['correlated_features'] = result['Metricscores']['Metricscores']['Explainabilityscore']['Correlatedfeaturesscore']
                uploaddic['model_size'] = result['Metricscores']['Metricscores']['Explainabilityscore']['Modelsizescore']
                uploaddic['explainability_score'] = result['Pillarscores']['Explainabilityscore']
                uploaddic['robustness_score'] = result['Pillarscores']['Robustnessscore']
                uploaddic['trust_score'] = result['Trustscore']

                

            else:
                print("Final Score result:", get_final_score(path_module, path_traindata,path_testdata, config_weights, mappings_config, path_factsheet))
                # result = finalScore_supervised(
                #     path_module, path_traindata, path_testdata, path_factsheet, mappings_config, weights_metrics, weights_pillars)

            # uploaddic['class_balance'] =
            # uploaddic['equal_opportunity_difference'] =
            # uploaddic['average_odds_difference'] =
            # uploaddic['algorithm_class'] =
            # uploaddic['feature_relevance'] = feature_relevance
            # uploaddic['confidence_score'] = confidence_score
            # uploaddic['clique_method'] = clique_method
            # uploaddic['clever_score'] = clever_score
            # uploaddic['loss_sensitivity'] = loss_sensitivity
            # uploaddic['er_fast_gradient_attack'] = er_fast_gradient_attack
            # uploaddic['er_carlini_wagner_attack'] = er_carlini_wagner_attack
            # uploaddic['er_deepfool_attack'] = er_deepfool_attack

        return Response(uploaddic)
