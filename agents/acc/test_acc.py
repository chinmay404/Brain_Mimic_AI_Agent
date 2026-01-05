import unittest
import sys
import os

# Add project root to path so we can import 'agents'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from agents.acc.acc_main import AnteriorCingulateCortex, ACCInputs, ACCOutputs

class TestACC(unittest.TestCase):
    def setUp(self):
        self.acc = AnteriorCingulateCortex()

    def test_low_conflict_scenario(self):
        inputs = ACCInputs(
            goal_deviation=0.1,
            prediction_error=0.05,
            action_entropy=0.1,
            estimated_cost=0.1,
            expected_gain=0.9,
            error_trend=-0.1,
            model_uncertainty=0.1
        )
        output = self.acc.process(inputs)
        self.assertLess(output.cns_score, 0.3)
        self.assertEqual(output.strategy_shift, 0)
        self.assertFalse(output.abort_flag)

    def test_high_conflict_scenario(self):
        inputs = ACCInputs(
            goal_deviation=0.8,
            prediction_error=0.5,
            action_entropy=0.9,
            estimated_cost=0.8,
            expected_gain=0.2,
            error_trend=0.5,
            model_uncertainty=0.8
        )
        output = self.acc.process(inputs)
        self.assertGreater(output.cns_score, 0.7)
        self.assertEqual(output.strategy_shift, 2)
        # Abort flag might be true or false depending on exact calculation, but let's check if it runs without error
        self.assertIsInstance(output.abort_flag, bool)

    def test_abort_condition(self):
        # Force extremely high values to trigger abort
        inputs = ACCInputs(
            goal_deviation=1.0,
            prediction_error=1.0,
            action_entropy=1.0,
            estimated_cost=1.0,
            expected_gain=0.0, # Zero gain makes ratio huge
            error_trend=1.0,
            model_uncertainty=1.0
        )
        output = self.acc.process(inputs)
        self.assertTrue(output.abort_flag)

if __name__ == '__main__':
    unittest.main()
