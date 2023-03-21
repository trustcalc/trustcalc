from rest_framework.views import APIView
from ...models import CustomUser
from rest_framework.response import Response
from passlib.handlers.django import django_pbkdf2_sha256
from django.contrib.auth.hashers import make_password


class auth(APIView):
    def get(self, request):  # when login
        email = request.query_params['email']
        password = request.query_params['password']
        user = CustomUser.objects.get(email=email)
        password_verify = django_pbkdf2_sha256.verify(password, user.password)

        if (password_verify):
            return Response({
                'email': email,
                'password': password,
                'is_admin': user.is_admin,
            }, status=200)
        else:
            return Response('Login Failed', status=400)

    def post(self, request):  # when register
        print('register:', request.data)
        email = request.data['email']
        password = request.data['password']

        if email is None or password is None:
            return Response('register Error', status=400)

        password = make_password(password)
        newUser = CustomUser.objects.create(
            email=email, password=password, username=email)
        newUser.save()

        return Response('register', status=200)
