from rest_framework.views import APIView
from rest_framework.response import Response
from ...models import CustomUser, Scenario, ScenarioSolution


class solution(APIView):
    def get(self, request, email):
        uploaddic = {}

        SolutionName = []

        userexist = CustomUser.objects.filter(email=email)
        if userexist:
            userobj = CustomUser.objects.get(email=email)
            scenarioobj = ScenarioSolution.objects.filter(
                user_id=userobj.id).values()

            if scenarioobj:
                for i in scenarioobj:
                    SolutionName.append(i['solution_name'])

            uploaddic['SolutionName'] = SolutionName
            return Response(uploaddic)

        else:
            return Response("User not exist.... Please Sign Up!")

    def post(self, request):
        if request.data is not None:
            mapFile = ''
            if request.data['MapFile'] is None or request.data['MapFile'] == 'undefined':
                mapFile = 'files/mapping_metrics_default.json'
            else:
                mapFile = request.data['MapFile']

            print('req dta:', request.data)

            try:
                userexist = CustomUser.objects.get(
                    email=request.data['emailid'])
                scenario = Scenario.objects.get(
                    scenario_name=request.data['SelectScenario'])
                fileupload = ScenarioSolution.objects.create(
                    user_id=userexist.id,
                    scenario_id=scenario.id,
                    solution_name=request.data['NameSolution'],
                    description=request.data['DescriptionSolution'],
                    training_file=request.data['TrainingFile'],
                    metrics_mappings_file=mapFile,
                    test_file=request.data['TestFile'],
                    factsheet_file=request.data['FactsheetFile'],
                    model_file=request.data['ModelFile'],
                    target_column=request.data['Targetcolumn'],
                    solution_type=request.data['Solutiontype'],

                    outlier_data_file=request.data['Outlierdatafile'],
                    protected_features=request.data['ProtectedFeature'],
                    protected_values=request.data['Protectedvalues'],
                    favourable_outcome=request.data['Favourableoutcome'],
                    weights_metrics=request.data['WeightMetric'],
                    weights_pillars=request.data['WeightPillar']
                )
                fileupload.save()

                return Response("Successfully add!", status=200)

            except Exception as e:
                print('errror:', e)
                return Response("Error occured", status=201)
