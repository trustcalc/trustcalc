from rest_framework.views import APIView
from ...models import CustomUser, Scenario
from rest_framework.response import Response
from ...serilizers import UserSerializer


class user(APIView):
    def get(self, request, email):
        uploaddic = {}
        print("email:", email)
        ScenarioName = []
        Description = []

        userexist = CustomUser.objects.filter(email=email)
        if userexist:

            userobj = CustomUser.objects.get(email=email)
            scenarioobj = Scenario.objects.filter(user_id=userobj.id).values()

            if scenarioobj:
                for i in scenarioobj:
                    print('i:', i)
                    ScenarioName.append(i['scenario_name'])
                    Description.append(i['description'])

            uploaddic['ScenarioName'] = ScenarioName
            uploaddic['Description'] = Description
            return Response(uploaddic)
        else:
            print("User not exist.... Created new")
        return Response(uploaddic)

    def post(self, request):
        print('data:', request.data)
        userexist = CustomUser.objects.filter(email=request.data['emailid'])
        if userexist:
            uploaddic = {}

            ScenarioName = ''
            ModelLinks = ''
            LinktoDataset = ''
            Description = ''

            if request.data is not None:
                ScenarioName = request.data['ScenarioName'],
                ModelLinks = request.data['ModelLinks'],
                LinktoDataset = request.data['LinktoDataset'],
                Description = request.data['Description'],

            uploaddic['ScenarioName'] = ScenarioName
            uploaddic['ModelLinks'] = ModelLinks
            uploaddic['LinktoDataset'] = LinktoDataset
            uploaddic['Description'] = Description

            return Response('successfully created scenario', status=200)
        else:
            createuser = CustomUser.objects.create(
                email=request.data['emailid'], is_admin=0)
            createuser.save()

            uploaddic = {}

            ScenarioName = ''
            ModelLinks = ''
            LinktoDataset = ''
            Description = ''

            if request.data is not None:
                ScenarioName = request.data['ScenarioName'],
                ModelLinks = request.data['ModelLinks'],
                LinktoDataset = request.data['LinktoDataset'],
                Description = request.data['Description'],

            uploaddic['ScenarioName'] = ScenarioName
            uploaddic['ModelLinks'] = ModelLinks
            uploaddic['LinktoDataset'] = LinktoDataset
            uploaddic['Description'] = Description

            request.data['user'] = createuser.id
            serializer = UserSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save(response_data=uploaddic)
                print("Nice to Add Second!")
                return Response(uploaddic)

            else:
                print('errors:  ', serializer.errors)

            print("User not exist.... Created new")
            return Response("Successfully add!")
