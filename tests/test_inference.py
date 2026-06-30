"""
Unit tests for the inference module.
"""

import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.confidence_gate import check_confidence

class TestInference(unittest.TestCase):
    
    def test_check_confidence(self):
        # Above threshold
        res1 = check_confidence(0.85, threshold=0.60)
        self.assertTrue(res1["proceed"])
        self.assertEqual(res1["status"], "confident")
        
        # Below threshold
        res2 = check_confidence(0.40, threshold=0.60)
        self.assertFalse(res2["proceed"])
        self.assertEqual(res2["status"], "uncertain")

if __name__ == "__main__":
    unittest.main()
