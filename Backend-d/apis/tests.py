from django.test import TestCase

# Create your tests here.
from .models import CustomUser, Scenario, ScenarioSolution


class CustomUserTestCase(TestCase):
    def setUp(self):
        CustomUser.objects.create(
            email="user1", password="user1-password", username="user1", is_admin=False)
        CustomUser.objects.create(
            email="user2", password="user2-password", username="user2", is_admin=True)

    def test_customuser(self):
        """Animals that can speak are correctly identified"""
        user1 = CustomUser.objects.get(email="user1")
        user2 = CustomUser.objects.get(email="user2")
        self.assertEqual(user1.get_user_name(), 'user1')
        self.assertEqual(user1.get_user_password(), 'user1-password')
        self.assertEqual(user2.get_user_name(), 'user2')
        self.assertEqual(user2.get_user_password(), 'user2-password')


class ScenarioTestCase(TestCase):
    def setUp(self):
        tempUser = CustomUser.objects.create(
            email="temp", password="temp", username="temp")
        Scenario.objects.create(
            scenario_name="testscenario", description="testscenario", user_id=tempUser.get_user_id())

    def test_scenario(self):
        tempUser = CustomUser.objects.get(email="temp")
        temp = Scenario.objects.get(scenario_name="testscenario")
        self.assertEqual(temp.get_description(), 'testscenario')
        self.assertEqual(temp.get_user_id(), temp.get_user_id())


class SoltionTestCase(TestCase):
    def setUp(self):
        tempUser = CustomUser.objects.create(
            email="temp", password="temp", username="temp")
        print('id:', tempUser.get_user_id())
        tempScenario = Scenario.objects.create(
            user_id=tempUser.id,
            scenario_name="temp",
            description="temp"
        )
        ScenarioSolution.objects.create(
            user_id=tempUser.id,
            scenario_id=tempScenario.id,
            solution_name="temp",
            description="temp",
            solution_type="supervised",
            training_file=None,
            test_file=None,
            protected_features="temp",
            protected_values="temp",
            target_column="temp",
            outlier_data_file=None,
            favourable_outcome="temp",
            factsheet_file=None,
            model_file=None,
            metrics_mappings_file=None,
            weights_metrics=None,
            weights_pillars=None
        )

    def test_solution(self):
        tempUser = CustomUser.objects.get(email="temp")
        tempScenario = Scenario.objects.get(user_id=tempUser.id)
        tempSolution = ScenarioSolution.objects.get(
            scenario_id=tempScenario.id)
        self.assertEqual(tempSolution.get_description(), 'temp')
        self.assertEqual(tempSolution.get_solution_name(), 'temp')
        self.assertEqual(tempSolution.get_solution_type(), 'supervised')
        print('file:', tempSolution.get_traing_file(), 'asdb')
        self.assertEqual(tempSolution.get_traing_file(), '')
        self.assertEqual(tempSolution.get_test_file(), '')
        self.assertEqual(tempSolution.get_protected_feature(), 'temp')
        self.assertEqual(tempSolution.get_protected_value(), 'temp')
        self.assertEqual(tempSolution.get_target_column(), 'temp')
        self.assertEqual(tempSolution.get_outlier_file(), '')
        self.assertEqual(tempSolution.get_favourable_outcome(), 'temp')
        self.assertEqual(tempSolution.get_factsheet_file(), '')
        self.assertEqual(tempSolution.get_model_file(), '')
        self.assertEqual(tempSolution.get_metrics_mapping_file(), '')
        self.assertEqual(tempSolution.get_weights_metrics(), '')
        self.assertEqual(tempSolution.get_weights_pillars(), '')
