from rest_framework.views import APIView
from ...models import ScenarioSolution
from rest_framework.response import Response


class solutiondetail(APIView):
    def get(self, request, id):
        solutionDetail = ScenarioSolution.objects.get(id=id)

        return Response({
            'solution_name': solutionDetail.solution_name,
            'description': solutionDetail.description,
            'solution_type': solutionDetail.solution_type,
            'protected_features': solutionDetail.protected_features,
            'protected_values': solutionDetail.protected_values,
            'target_column': solutionDetail.target_column
        }, status=200)

    def put(self, request):
        solutionDetail = ScenarioSolution.objects.get(
            id=request.data['SolutionId'])
        solutionDetail.solution_name = request.data['NameSolution']
        solutionDetail.description = request.data['DescriptionSolution']
        if (request.data['TrainingFile'] != 'undefined'):
            print('asdfasdfasdf')
        if (request.data['TrainingFile'] != 'undefined'):
            solutionDetail.training_file = request.data['TrainingFile']
        if (request.data['TestFile'] != 'undefined'):
            solutionDetail.test_file = request.data['TestFile']
        if (request.data['FactsheetFile'] != 'undefined'):
            solutionDetail.factsheet_file = request.data['FactsheetFile']
        if (request.data['ModelFile'] != 'undefined'):
            solutionDetail.model_file = request.data['ModelFile']
        if (len(request.data['Targetcolumn']) <= 0):
            solutionDetail.target_column = request.data['Targetcolumn']
        if (request.data['Outlierdatafile'] != 'undefined'):
            solutionDetail.outlier_data_file = request.data['Outlierdatafile']
        if (len(request.data['ProtectedFeature']) <= 0):
            solutionDetail.protected_features = request.data['ProtectedFeature']
        if (len(request.data['Protectedvalues']) <= 0):
            solutionDetail.protected_values = request.data['Protectedvalues']
        if (len(request.data['Favourableoutcome']) <= 0):
            solutionDetail.favourable_outcome = request.data['Favourableoutcome']
        if (len(request.data['WeightMetric']) <= 0):
            solutionDetail.weights_metrics = request.data['WeightMetric']
        if (len(request.data['WeightPillar']) <= 0):
            solutionDetail.weights_pillars = request.data['WeightPillar']
        solutionDetail.save()

        return Response('successfully changed', 200)
