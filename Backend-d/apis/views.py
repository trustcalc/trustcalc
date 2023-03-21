from .apiback.public import save_files_return_paths
from django.http import Http404
from algorithms.unsupervised.Functions.Accountability.NormalizationScore import normalization_score as get_normalization_score_unsupervised
from algorithms.supervised.Functions.Robustness.CleverScore_supervised import get_clever_score_supervised
from algorithms.unsupervised.Functions.Robustness.CleverScore import clever_score as get_clever_score_unsupervised
from algorithms.unsupervised.Functions.Fairness.Fairness import analyse as get_fairness_score_unsupervised
from algorithms.supervised.Functions.Fairness.FarinessScore_supervised import get_fairness_score_supervised
from algorithms.unsupervised.Functions.Explainability.Explainability import analyse as get_explainability_score_unsupervised
from algorithms.supervised.Functions.Explainability.ExplainabilityScore_supervised import get_explainability_score_supervised
from algorithms.unsupervised.Functions.Accountability.Accountability import analyse as get_accountability_score_unsupervised
from algorithms.supervised.Functions.Accountability.AccountabilityScore_supervised import get_accountability_score_supervised
from algorithms.supervised.Functions.Fairness.AverageOddsDifferenceScore_supervised import get_average_odds_difference_score_supervised
from algorithms.supervised.Functions.Fairness.EqualOpportunityDifferenceScore_supervised import get_equal_opportunity_difference_score_supervised
from algorithms.unsupervised.Functions.Fairness.StatisticalParityDifferenceScore import get_statistical_parity_difference_score_unsupervised
from algorithms.supervised.Functions.Fairness.StatisticalParityDifferenceScore import get_statistical_parity_difference_score_supervised
from algorithms.unsupervised.Functions.Fairness.UnderfittingScore import underfitting_score as get_underfitting_score_unsupervised
from algorithms.supervised.Functions.Fairness.UnderfittingScore_supervised import get_underfitting_score_supervised
from algorithms.unsupervised.Functions.Fairness.OverfittingScore import overfitting_score as get_overfitting_score_unsupervised
from algorithms.supervised.Functions.Fairness.OverfittingScore_supervised import get_overfitting_score_supervised
from algorithms.supervised.Functions.Fairness.ClassBalanceScore_supervised import get_class_balance_score_supervised
from algorithms.unsupervised.Functions.Fairness.DisparateImpactScore import disparate_impact_score as get_disparate_impact_score_unsupervised
from algorithms.supervised.Functions.Fairness.DisparateImpactScore_supervised import get_disparate_impact_score_supervised
from algorithms.unsupervised.Functions.Explainability.PermutationFeatureScore import permutation_feature_importance_score as get_permutation_feature_importance_score_unsupervised
from algorithms.supervised.Functions.Explainability.FeatureRelevanceScore_supervised import get_feature_relevance_score_supervised
from algorithms.supervised.Functions.Explainability.AlgorithmClassScore_supervised import get_algorithm_class_score_supervised
from algorithms.unsupervised.Functions.Explainability.CorrelatedFeaturesScore import correlated_features_score as get_correlated_features_score_unsupervised
from algorithms.supervised.Functions.Explainability.CorrelatedFeaturesScore_supervised import get_correlated_features_score_supervised
from algorithms.unsupervised.Functions.Explainability.ModelSizeScore import model_size_score as get_modelsize_score_unsupervised
from algorithms.supervised.Functions.Explainability.ModelSizeScore_supervised import get_model_size_score_supervised as get_modelsize_score_supervised
from algorithms.supervised.Functions.Robustness.LossSensitivityScore_supervised import get_loss_sensitivity_score_supervised
from algorithms.supervised.Functions.Robustness.ERFastGradientAttackScore_supervised import get_er_fast_gradient_attack_score_supervised as get_fast_gradient_attack_score_supervised
from algorithms.supervised.Functions.Robustness.ERDeepFoolAttackScore_supervised import get_deepfool_attack_score_supervised as get_deepfoolattack_score_supervised
from algorithms.supervised.Functions.Robustness.ERCarliniWagnerScore_supervised import get_er_carlini_wagner_score_supervised as get_carliwagnerwttack_score_supervised
from algorithms.supervised.Functions.Robustness.ConfidenceScore_supervised import get_confidence_score_supervised
from algorithms.supervised.Functions.Robustness.CliqueMethodScore_supervised import get_clique_method_supervised as get_clique_method_score_supervised
from algorithms.unsupervised.Functions.Robustness.CleverScore import clever_score as get_clique_method_score_unsupervised
from algorithms.unsupervised.Functions.Accountability.TrainTestSplitScore import train_test_split_score as get_train_test_split_score_unsupervised
from algorithms.unsupervised.Functions.Accountability.RegularizationScore import regularization_score as get_regularization_score_unsupervised
from algorithms.supervised.Functions.Accountability.MissingDataScore_supervised import get_missing_data_score_supervised
from algorithms.unsupervised.Functions.Accountability.MissingDataScore import missing_data_score as get_missing_data_score_unsupervised
from algorithms.supervised.Functions.Accountability.NormalizationScore_supervised import get_normalization_score_supervised
from algorithms.supervised.Functions.Accountability.RegularizationScore_supervised import get_regularization_score_supervised
from algorithms.supervised.Functions.Accountability.TrainTestSplitScore_supervised import get_train_test_split_score_supervised
from algorithms.supervised.Functions.Accountability.FactSheetCompletnessScore_supervised import get_factsheet_completeness_score_supervised
from algorithms.unsupervised.Functions.Accountability.FactSheetCompletnessScore import get_factsheet_completeness_score_unsupervised
from algorithms.supervised.Functions.Robustness.Robustness_supervised import get_robustness_score_supervised
from algorithms.unsupervised.Functions.Robustness.Robustness import analyse as get_robustness_score_unsupervised
from algorithms.TrustScore.TrustScore import trusting_AI_scores_unsupervised
from algorithms.TrustScore.TrustScore import trusting_AI_scores_supervised
from .models import Scenario
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Scenario, ScenarioSolution
from .serilizers import SolutionSerializer, ScenarioSerializer
# import stripe
from rest_framework import status
from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
print("Base dir is:", BASE_DIR)


def handle_score_request(type, detailType,  data, user_id):
    solution_name = data['solution_name']
    # solution_type = data['solution_type']

    print('data:', type, detailType, data['solution_name'], user_id)
    solution = ScenarioSolution.objects.filter(
        solution_name=solution_name,
        user_id=user_id).values().order_by('id')[:1]

    path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
    path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
    path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')

    if solution:
        for i in solution:
            print('i:', i)
            path_testdata = i["test_file"]
            path_traindata = i["training_file"]
            path_factsheet = i["factsheet_file"]
            path_mapping = i["metrics_mappings_file"]
            solutionType = i['solution_type']
            model_file = i['model_file']
            target_column = i['target_column']
            outliers_data = i['outlier_data_file']

        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, outliers_data, target_column = save_files_return_paths(
            model_file, path_traindata, path_testdata, path_factsheet, path_mapping, outliers_data, target_column)

        status = 200
        result = ''

        try:
            if (type == 'account'):
                if (detailType == 'factsheet'):

                    if (solutionType == 'supervised'):
                        result = get_factsheet_completeness_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        result = get_factsheet_completeness_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                elif (detailType == 'missingdata'):
                    if (solutionType == 'supervised'):
                        result = get_missing_data_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        result = get_missing_data_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                elif (detailType == 'normalization'):
                    if (solutionType == 'supervised'):
                        result = get_normalization_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        result = get_normalization_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                elif (detailType == 'regularization'):
                    if (solutionType == 'supervised'):
                        result = get_regularization_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        result = get_regularization_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                elif (detailType == 'train_test'):
                    if (solutionType == 'supervised'):
                        result = get_train_test_split_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        result = get_train_test_split_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                else:
                    result = 'none'
                    status = 201

            elif (type == 'robust'):

                if (detailType == 'clever_score'):
                    if (solutionType == 'supervised'):
                        result = get_clever_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        result = get_clever_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                elif (detailType == 'clique_method_score'):
                    if (solutionType == 'supervised'):
                        result = get_clique_method_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                    else:
                        status = 404
                        result = get_clique_method_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                elif (detailType == 'confidence_score'):
                    if (solutionType == 'supervised'):
                        result = get_confidence_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'carliwagnerwttack_score'):
                    if (solutionType == 'supervised'):
                        result = get_carliwagnerwttack_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'loss_sensitivity_score'):
                    if (solutionType == 'supervised'):
                        result = get_loss_sensitivity_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'deepfoolattack_score'):
                    if (solutionType == 'supervised'):
                        result = get_deepfoolattack_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'fast_gradient_attack_score'):
                    if (solutionType == 'supervised'):
                        result = get_fast_gradient_attack_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"
            elif (type == 'explain'):
                if (detailType == 'modelsize_score'):
                    if (solutionType == 'supervised'):
                        result = get_modelsize_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                    else:
                        result = get_modelsize_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                elif (detailType == 'correlated_features_score'):
                    if (solutionType == 'supervised'):
                        result = get_correlated_features_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                    else:
                        result = get_correlated_features_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                elif (detailType == 'algorithm_class_score'):
                    if (solutionType == 'supervised'):
                        result = get_algorithm_class_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'feature_relevance_score'):
                    if (solutionType == 'supervised'):
                        result = get_feature_relevance_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"
                elif (detailType == 'permutation_feature_importance_score'):
                    if (solutionType == 'supervised'):
                        status = 404
                        result = "The metric function isn't applicable for supervised ML/DL solutions"
                    else:
                        if (outliers_data.find('.') < 0):
                            status = 404
                            result = "The outlier data file is missing"
                        else:
                            result = get_permutation_feature_importance_score_unsupervised(
                                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, outliers_data)

            elif (type == 'fairness'):
                if (detailType == 'disparate_impact_score'):
                    if (solutionType == 'supervised'):
                        result = get_disparate_impact_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        print('before get:')
                        result = get_disparate_impact_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                        print('get result:', result)

                elif (detailType == 'class_balance_score'):
                    if (solutionType == 'supervised'):
                        result = get_class_balance_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'overfitting_score'):
                    if (solutionType == 'supervised'):
                        result = get_overfitting_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        result = get_overfitting_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                elif (detailType == 'underfitting_score'):
                    if (solutionType == 'supervised'):
                        result = get_underfitting_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        result = get_underfitting_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                elif (detailType == 'statistical_parity_difference_score'):
                    if (solutionType == 'supervised'):
                        result = get_statistical_parity_difference_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        result = get_statistical_parity_difference_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                elif (detailType == 'equal_opportunity_difference_score'):
                    if (solutionType == 'supervised'):
                        result = get_equal_opportunity_difference_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"
                elif (detailType == 'average_odds_difference_score'):
                    if (solutionType == 'supervised'):
                        result = get_average_odds_difference_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

            elif (type == 'pillar'):
                if (detailType == 'accountability_score'):
                    if (solutionType == 'supervised'):
                        result = get_accountability_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                    else:
                        result = get_accountability_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                elif (detailType == 'robustnesss_score'):
                    if (solutionType == 'supervised'):
                        result = get_robustness_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                    else:
                        result = get_robustness_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                elif (detailType == 'explainability_score'):
                    print("outdata:", )
                    if (solutionType == 'supervised'):
                        print('super called')
                        result = get_explainability_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                    else:
                        if (outliers_data.find('.') < 0):
                            status = 404
                            result = "The outlier data file is missing"
                        else:
                            result = get_explainability_score_unsupervised(
                                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                elif (detailType == 'fairness_score'):
                    if (solutionType == 'supervised'):
                        result = get_fairness_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                    else:
                        result = get_fairness_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            elif (type == 'trust'):
                print('test:', type, detailType)
                if (detailType == 'trustscore'):
                    if (solutionType == 'supervised'):
                        result = trusting_AI_scores_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data).score
                    else:
                        result = trusting_AI_scores_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data).score
                elif (detailType == 'trusting_AI_scores_supervised'):
                    result = trusting_AI_scores_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                elif (detailType == 'trusting_AI_scores_unsupervised'):
                    result = trusting_AI_scores_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            else:
                result = 'none'
                status = 201

            return Response(result, status)
        except Exception as e:
            print('error:', e)
            return Response("Error while reading corrupted model file", status=409)

    else:
        return Response("Please login / User doesn't exist", status=409)


class SolutionList(APIView):
    """
    List all solutions or create a new solution.
    """

    def get(self, request):
        user = request.user
        print("GET SOLUTIONLIST USER: ", user)
        if not user.is_authenticated:
            return Response({'error': 'Authentication failed'})

        solutions = ScenarioSolution.objects.all()
        serializer = SolutionSerializer(solutions, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = SolutionSerializer(data=request.data)
        if serializer.is_valid():
            user = request.user
            scenario_name = request.data.get('scenario_name', '')
            if not scenario_name:
                # Generate new scenario name
                base_name = f'Scenario_{slugify(user.username)}_'
                count = Scenario.objects.filter(
                    name__startswith=base_name).count()
                scenario_name = f'{base_name}{count + 1}'

            # Check if scenario exists
            scenario = Scenario.objects.filter(
                scenario_name=scenario_name, user_id=user.id).first()
            if not scenario:
                # Create new scenario
                scenario = Scenario.objects.create(
                    scenario_name=scenario_name, user_id=user.id)

            serializer.save(user=user, scenario=scenario)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk):
        solution = get_object_or_404(ScenarioSolution, pk=pk)
        serializer = SolutionSerializer(solution, data=request.data)
        if serializer.is_valid():
            user = request.user
            scenario_name = serializer.validated_data['scenario_name']
            scenario = get_object_or_404(
                Scenario, scenario_name=scenario_name, user=user)
            serializer.save(user=user, scenario=scenario)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        user = request.user
        solution_name = request.data.get('solution_name')
        if not solution_name:
            return Response({'error': 'solution_name is required.'}, status=status.HTTP_400_BAD_REQUEST)
        solution = ScenarioSolution.objects.filter(
            user=user, solution_name=solution_name).first()
        if not solution:
            return Response({'error': 'Solution not found for the given user.'}, status=status.HTTP_404_NOT_FOUND)
        solution.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ScenarioList(APIView):
    """
    List all scenarios or create a new scenario.
    """

    def get_object(self, scenario_name):
        try:
            return Scenario.objects.get(user=self.request.user, scenario_name=scenario_name)
        except Scenario.DoesNotExist:
            raise Http404

    def get(self, request):
        scenarios = Scenario.objects.all()
        serializer = ScenarioSerializer(scenarios, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = ScenarioSerializer(data=request.data)
        if serializer.is_valid():
            user = request.user
            serializer.save(user=user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request):
        user = request.user
        scenario_name = request.data.get('scenario_name')
        if not scenario_name:
            return Response({'error': 'scenario_name is required.'}, status=status.HTTP_400_BAD_REQUEST)
        scenario = Scenario.objects.filter(
            user=user, scenario_name=scenario_name).first()
        if not scenario:
            return Response({'error': 'Scenario not found for the given user.'}, status=status.HTTP_404_NOT_FOUND)
        serializer = ScenarioSerializer(
            scenario, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        user = request.user
        scenario_name = request.data.get('scenario_name')
        if not scenario_name:
            return Response({'error': 'scenario_name is required.'}, status=status.HTTP_400_BAD_REQUEST)
        scenario = Scenario.objects.filter(
            user=user, scenario_name=scenario_name).first()
        if not scenario:
            return Response({'error': 'Scenario not found for the given user.'}, status=status.HTTP_404_NOT_FOUND)
        scenario.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
