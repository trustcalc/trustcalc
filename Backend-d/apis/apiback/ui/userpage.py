from rest_framework.views import APIView
from ...models import CustomUser, Scenario, ScenarioSolution
from rest_framework.response import Response


class userpage(APIView):
    def get(self, request, email):
        uploaddic = {}

        userexist = CustomUser.objects.get(email=email)
        if userexist.is_admin == True:
            uploaddic['Admin'] = "Admin"

            users = []
            userlist = CustomUser.objects.all()
            for i in userlist:
                users.append(i.email)
            uploaddic['users'] = users
            print("Users list:", users)
        else:
            uploaddic['Admin'] = "noad"
        return Response(uploaddic)

    def post(self, request):
        uploaddic = {}
        print('data:', request.data)
        if request.data is not None:
            userexist = CustomUser.objects.get(
                email=request.data['Useremail'])
            scenario = Scenario.objects.filter(user_id=userexist.id).values()
            scenarioobj = ScenarioSolution.objects.filter(
                user_id=userexist.id).values()
            ScenarioName = []
            SolutionName = []

            if scenario:
                for i in scenario:
                    print("Response data ScenarioName:",
                          i['scenario_name']),

                    ScenarioName.append(i['scenario_name'])

            uploaddic['ScenarioName'] = ScenarioName

            if scenarioobj:
                for i in scenarioobj:
                    SolutionName.append(i['solution_name'])
            uploaddic['SolutionName'] = SolutionName

        return Response(uploaddic)
