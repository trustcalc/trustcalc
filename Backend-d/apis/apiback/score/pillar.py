from rest_framework.decorators import authentication_classes, api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from ...authentication import CustomUserAuthentication
from ...views import handle_score_request


# E)PillarScores
# 1)AccountabilityScore
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_accountability_score(request):
    return handle_score_request('pillar', 'accountability_score', request.data, request.user.id)


# 2)RobustnessScore
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_robustness_score(request):
    print('data:', request.data, request.user.id)
    return handle_score_request('pillar', 'robustnesss_score', request.data, request.user.id)


# 3)ExplainabilityScore
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_explainability_score(request):
    return handle_score_request('pillar', 'explainability_score', request.data, request.user.id)


# 4)FairnessScore
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_fairness_score(request):
    return handle_score_request('pillar', 'fairness_score', request.data, request.user.id)
