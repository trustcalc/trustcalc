from rest_framework.decorators import authentication_classes, api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from ...authentication import CustomUserAuthentication
from ...views import handle_score_request

# 1)ModelSizeScore


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_modelsize_score(request):
    return handle_score_request('explain', 'modelsize_score', request.data, request.user.id)

# 2)CorrelatedFeaturesScore


@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_correlated_features_score(request):
    return handle_score_request('explain', 'correlated_features_score', request.data, request.user.id)


# 3)AlgorithmClassScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_algorithm_class_score(request):
    return handle_score_request('explain', 'algorithm_class_score', request.data, request.user.id)


# 4)FeatureRelevanceScore

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_feature_relevance_score(request):
    return handle_score_request('explain', 'feature_relevance_score', request.data, request.user.id)


# 5)PermutationFeatures

@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_permutation_feature_importance_score(request):
    return handle_score_request('explain', 'permutation_feature_importance_score', request.data, request.user.id)
