from rest_framework.decorators import authentication_classes, api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from ...authentication import CustomUserAuthentication
from ...views import handle_score_request

# D)Fairness
# 1)DisparateImpactScore


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_disparate_impact_score(request):
    return handle_score_request('fairness', 'disparate_impact_score', request.data, request.user.id)


# 2)ClassBalanceScore
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_class_balance_score(request):
    return handle_score_request('fairness', 'disparate_impact_score', request.data, request.user.id)


# 3)OverfittingScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_overfitting_score(request):
    return handle_score_request('fairness', 'overfitting_score', request.data, request.user.id)


# 4)UnderfittingScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_underfitting_score(request):
    return handle_score_request('fairness', 'underfitting_score', request.data, request.user.id)


# 5)StatisticalParityDifferenceScore
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_statistical_parity_difference_score(request):
    return handle_score_request('fairness', 'statistical_parity_difference_score', request.data, request.user.id)


# 6)EqualOpportunityDifferenceScore
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_equal_opportunity_difference_score(request):
    return handle_score_request('fairness', 'equal_opportunity_difference_score', request.data, request.user.id)


# 7)AverageOddsDifferenceScore
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_average_odds_difference_score(request):
    return handle_score_request('fairness', 'average_odds_difference_score', request.data, request.user.id)
