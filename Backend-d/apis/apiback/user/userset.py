from rest_framework.views import APIView
from ...models import CustomUser
from rest_framework.response import Response
from django.contrib.auth.hashers import make_password


class userset(APIView):
    def get(self, request, email):

        user = CustomUser.objects.get(email=email)
        return Response({
            'email': user.email,
            'password': user.password,
        }, status=200)

    def post(self, request):
        user = CustomUser.objects.get(email=request.data['email'])
        user.password = make_password(request.data['password'])
        user.save()

        return Response('success', status=200)
