from rest_framework.decorators import authentication_classes, api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from ...authentication import CustomUserAuthentication
from ...views import handle_score_request
# 5)TrustScore


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_trust_score(request):
    return handle_score_request('trust', 'trustscore', request.data, request.user.id)


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_trusting_AI_scores_supervised(request):
    return handle_score_request('trust', 'trusting_AI_scores_supervised', request.data, request.user.id)


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_trusting_AI_scores_unsupervised(request):
    return handle_score_request('trust', 'trusting_AI_scores_unsupervised', request.data, request.user.id)
