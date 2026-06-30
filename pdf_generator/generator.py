"""
PDF Generation module using xhtml2pdf and Jinja2.

Renders a bilingual HTML report template with dynamic data and converts
it into a styled PDF.
"""

import os
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa
import datetime

def generate_pdf(result: dict, output_path: str) -> str:
    """
    Renders a Jinja2 template and generates a PDF using xhtml2pdf.
    
    Args:
        result: Dictionary containing diagnosis, confidence, severity, images, etc.
        output_path: Path where the generated PDF will be saved.
        
    Returns:
        The path to the generated PDF.
        
    Raises:
        RuntimeError: If PDF generation fails.
    """
    # Setup Jinja2 environment
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('report.html')
    
    # Base path for local font resolution in HTML
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(base_dir, 'fonts').replace('\\', '/')
    
    # Current date
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare template context
    context = {
        "font_dir": font_dir,
        "date": date_str,
        "disease_name_en": result["name_en"],
        "confidence": f"{result['confidence'] * 100:.1f}%",
        "severity": result["severity_stage"].capitalize(),
        "yield_loss_pct": f"{result['yield_loss_pct']}%",
        "inr_per_acre": f"₹{result['inr_per_acre']}",
        "crop_type": result.get("crop_type", "Unknown"),
        "state": result.get("state", "Unknown"),
        "zone": result.get("zone", "Unknown"),
        "pesticide": result["solution"].get("pesticide", "N/A"),
        "application": result["solution"].get("application", "N/A"),
        "organic": result["solution"].get("organic_alternative", "N/A"),
        "source": result["solution"].get("source", "N/A"),
        "img_original": result.get("img_original", ""),
        "img_heatmap": result.get("img_heatmap", "")
    }
    
    # Render HTML
    html_out = template.render(context)
    
    # Generate PDF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(
            html_out, 
            dest=result_file,
            encoding='UTF-8'
        )
        
    if pisa_status.err:
        raise RuntimeError(f"xhtml2pdf error creating PDF: {pisa_status.err}")
        
    return output_path
