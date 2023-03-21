from rest_framework import serializers
from .models import Scenario, ScenarioSolution, CustomUserManager, CustomUser


class AllUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUserManager
        fields = [
            "password",
            "last_login",
            "id",
            "email",
            "is_admin",
            "username"
        ]


class UserSerializer(serializers.ModelSerializer):
    # user = serializers.CharField(source='user',read_only=True)
    class Meta:
        model = CustomUser
        lookup_field = 'slug'
        # fields = ['user','response_data']
        fields = "__all__"


class ScenarioSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Scenario
        lookup_field = 'slug'
        # fields = ['user', 'trainingdata']
        fields = [
            "user_id",
            "scenario_name",
            "description",
        ]


class SolutionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ScenarioSolution
        lookup_field = 'slug'
        fields = [
            "user_id",
            "scenario_id",
            "solution_name",
            "description",
            "solution_type",
            "training_file",
            "test_file",
            "protected_features",
            "protected_values",
            "target_column",
            "outlier_data_file",
            "favourable_outcome",
            "factsheet_file",
            "model_file",
            "metrics_mappings_file"
        ]
