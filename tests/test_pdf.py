"""
Unit tests for PDF generation.
"""

import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pdf_generator.generator import generate_pdf

class TestPDF(unittest.TestCase):
    
    def test_generate_pdf(self):
        # Mock result data
        result = {
            "name_en": "Test Disease",
            "name_regional": "परीक्षण रोग",
            "confidence": 0.95,
            "severity_stage": "late",
            "yield_loss_pct": 45,
            "inr_per_acre": 15000,
            "solution": {
                "pesticide": "Test Pesticide",
                "application": "Test App",
                "organic_alternative": "Test Organic",
                "source": "Test Source"
            },
            "img_original": "",
            "img_heatmap": ""
        }
        
        out_path = os.path.join(os.path.dirname(__file__), "test_report.pdf")
        
        try:
            generate_pdf(result, out_path, language="hi")
            self.assertTrue(os.path.exists(out_path))
            # Cleanup
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception as e:
            self.fail(f"PDF generation raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
