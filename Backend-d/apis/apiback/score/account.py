from rest_framework.decorators import authentication_classes, api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from ...authentication import CustomUserAuthentication
from ...views import handle_score_request


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_factsheet_completness_score(request):
    return handle_score_request('account', 'factsheet', request.data, request.user.id)


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_missing_data_score(request):
    return handle_score_request('account', 'missingdata', request.data, request.user.id)


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_normalization_score(request):
    return handle_score_request('account', 'normalization', request.data, request.user.id)


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_regularization_score(request):
    return handle_score_request('account', 'regularization', request.data, request.user.id)


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_train_test_split_score(request):
    return handle_score_request('account', 'train_test', request.data, request.user.id)
