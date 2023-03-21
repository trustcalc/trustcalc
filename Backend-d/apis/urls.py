# from apiback.ui.userpage import userpage
# from apiback.ui.solutiondetail import solutiondetail
# from apiback.ui.scenario import scenario
# from apiback.ui.compare import compare
# from apiback.ui.analyze import analyze
# from apiback.ui.solution import solution
# from apiback.ui.dashboard import dashboard
import sys
from .apiback.ui import dashboard, solution, scenario, userpage, compare, solutiondetail, analyze
from .apiback.user import  user, userset, auth
from .apiback.score import account, robust, explain, fair, pillar, trust
from django.urls import path
from .import views
from django.conf import settings
from django.conf.urls.static import static

from .download import download
sys.path.extend([r"Backend", r"Backend/apis"])


urlpatterns = [
          #7)User Mangement
    path('api/user', user.user.as_view(), name="user"),
    path('api/user/<str:email>/', user.user.as_view(), name="user"),
    path('api/dashboard/<str:email>',
         dashboard.dashboard.as_view(), name="dashboard"),

    path('api/solution', solution.solution.as_view()),
    path('api/solution/<str:email>', solution.solution.as_view()),

    path('api/userpage/<str:email>', userpage.userpage.as_view()),
    path('api/userpage', userpage.userpage.as_view()),


    path('api/compare', compare.compare.as_view()),
    path('api/setuser/<str:email>', userset.userset.as_view(), name="userset"),
    path('api/setuser', userset.userset.as_view(), name="userset"),

     

    path('api/solution_detail/<int:id>',
         solutiondetail.solutiondetail.as_view(), name="solutiondetail"),
    path('api/solution_detail', solutiondetail.solutiondetail.as_view(),
         name="solutiondetail"),
    path('api/auth', auth.auth.as_view(), name="auth"),

    path('api/factsheet_download', download.as_view(), name="download"),

    #A) GET, DEL all Scenarios of all users admin

    #B) GET, DEL all Scenarios own, non-admin
    #C) GET, PUT, DEL a Scenario other/own admin
    #D) GET, PUT, DEL a Scenario own, non-admin
    path('api/scenario/', views.ScenarioList.as_view()),
    #1) A Scenario User
    path('api/scenario/<str:scenario_name>',scenario.scenario.as_view(), name="scenario"),
    path('api/scenario', scenario.scenario.as_view(), name="scenario"),

    path('api/solution/', views.SolutionList.as_view()),

    # 1)Accountability Scores
    path('api/factsheet_completness_score/',
         account.get_factsheet_completness_score),
    path('api/missing_data_score/', account.get_missing_data_score),
    path('api/normalization_score/', account.get_normalization_score),
    path('api/regularization_score/', account.get_regularization_score),
    path('api/train_test_split_score/', account.get_train_test_split_score),

    # 2)Robustness Scores
    path('api/clever_score/', robust.get_clever_score),
    path('api/clique_method_score/', robust.get_clique_method_score),
    path('api/confidence_score/', robust.get_confidence_score),
    path('api/carliwagnerwttack_score/', robust.get_carliwagnerwttack_score),
    path('api/deepfoolattack_score/', robust.get_deepfoolattack_score),
    path('api/fast_gradient_attack_score/',
         robust.get_fast_gradient_attack_score),
    path('api/loss_sensitivity_score/', robust.get_loss_sensitivity_score),

    # 3)Explainability Scores
    path('api/modelsize_score/', explain.get_modelsize_score),
    path('api/correlated_features_score/',
         explain.get_correlated_features_score),
    path('api/algorithm_class_score/', explain.get_algorithm_class_score),
    path('api/feature_relevance_score/', explain.get_feature_relevance_score),
    path('api/permutation_feature_importance_score/',
         explain.get_permutation_feature_importance_score),

    # 4)Fairness Scores
    path('api/disparate_impact_score/', fair.get_disparate_impact_score),
    path('api/class_balance_score/', fair.get_class_balance_score),
    path('api/overfitting_score/', fair.get_overfitting_score),
    path('api/underfitting_score/', fair.get_underfitting_score),
    path('api/statistical_parity_difference_score/',
         fair.get_statistical_parity_difference_score),
    path('api/equal_opportunity_difference_score/',
         fair.get_equal_opportunity_difference_score),
    path('api/average_odds_difference_score/',
         fair.get_average_odds_difference_score),

    # 5)Pillar Scores
    path('api/accountability_score/', pillar.get_accountability_score),
    path('api/robustnesss_score/', pillar.get_robustness_score),
    path('api/explainability_score/', pillar.get_explainability_score),
    path('api/fairness_score/', pillar.get_fairness_score),

    # 6)Trust Scores
    path('api/trustscore/', trust.get_trust_score),
    path('api/trusting_AI_scores_supervised/',
         trust.get_trusting_AI_scores_supervised),
    path('api/trusting_AI_scores_unsupervised/',
         trust.get_trusting_AI_scores_unsupervised),
     

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
