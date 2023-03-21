from rest_framework.decorators import authentication_classes, api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from ...authentication import CustomUserAuthentication
from ...views import handle_score_request


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_clever_score(request):
    return handle_score_request('robust', 'clever_score', request.data, request.user.id)


# 2)CliqueMethodScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_clique_method_score(request):
    return handle_score_request('robust', 'clique_method_score', request.data, request.user.id)


# 3)ConfidenceScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_confidence_score(request):
    return handle_score_request('robust', 'confidence_score', request.data, request.user.id)


# 4)CarliWagnerAttackScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_carliwagnerwttack_score(request):
    return handle_score_request('robust', 'carliwagnerwttack_score', request.data, request.user.id)


# 5)DeepFoolAttackScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_deepfoolattack_score(request):
    return handle_score_request('robust', 'deepfoolattack_score', request.data, request.user.id)


# 6)ERFastGradientAttackScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_fast_gradient_attack_score(request):
    return handle_score_request('robust', 'fast_gradient_attack_score', request.data, request.user.id)


# 7)LossSensitivityScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_loss_sensitivity_score(request):
    return handle_score_request('robust', 'loss_sensitivity_score', request.data, request.user.id)
