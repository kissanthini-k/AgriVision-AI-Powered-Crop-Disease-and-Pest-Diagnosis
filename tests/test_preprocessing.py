"""
Unit tests for the preprocessing module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.validate_image import check_resolution, check_format

class TestPreprocessing(unittest.TestCase):
    
    def test_check_format(self):
        self.assertTrue(check_format("image.jpg"))
        self.assertTrue(check_format("IMAGE.PNG"))
        self.assertFalse(check_format("doc.pdf"))
        
    def test_check_resolution(self):
        # Create a dummy high-res image (300x300, 3 channels)
        img_good = np.zeros((300, 300, 3), dtype=np.uint8)
        self.assertTrue(check_resolution(img_good, min_size=224))
        
        # Create a dummy low-res image (150x150)
        img_bad = np.zeros((150, 150, 3), dtype=np.uint8)
        self.assertFalse(check_resolution(img_bad, min_size=224))

if __name__ == "__main__":
    unittest.main()
