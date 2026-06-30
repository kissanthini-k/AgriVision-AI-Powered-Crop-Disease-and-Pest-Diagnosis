"""
CLI Entry Point.

Runs the full crop disease detection pipeline on a single image via command line,
outputs results to the console using tabulate, and generates a PDF report.
"""

import argparse
import sys
import os
import json
import cv2
from tabulate import tabulate

from utils.config import MODEL_PATH, CLASS_NAMES
from utils.logger import setup_logger
from utils.helpers import load_image_as_array, array_to_base64
from preprocessing.preprocess import prepare_for_inference
from inference.predict import load_model, predict
from inference.gradcam import generate_gradcam, overlay_heatmap
from inference.confidence_gate import check_confidence
from analysis.damage_assessment import assess
from analysis.zone_filter import get_zone, filter_by_zone
from analysis.recommendation import get_recommendation
from pdf_generator.generator import generate_pdf

logger = setup_logger(__name__)

def run_pipeline(args):
    """
    Executes the pipeline steps sequentially.
    """
    logger.info("Starting Agri Disease Detector Pipeline...")
    
    # 1. Load DB
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'disease_db.json')
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            db = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load disease DB: {e}")
        return

    # 2. Preprocess
    logger.info(f"Processing image: {args.image}")
    try:
        img_tensor = prepare_for_inference(args.image)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    # 3. Predict
    logger.info("Running inference...")
    try:
        model = load_model(MODEL_PATH)
        pred_result = predict(img_tensor, model, CLASS_NAMES)
        disease_key = pred_result["class"]
        confidence = pred_result["confidence"]
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return

    # 4. Confidence Gate
    gate = check_confidence(confidence)
    if not gate["proceed"]:
        logger.warning(f"Confidence Gate failed: {gate['message']} (Confidence: {confidence:.2f})")
        return

    # 5. Zone Filtering
    zone = get_zone(args.state)
    logger.info(f"Identified Zone: {zone}")
    if not filter_by_zone(disease_key, zone, db):
        logger.warning(f"Note: {disease_key} is not typically found in {zone}.")

    # 6. Grad-CAM
    logger.info("Generating Grad-CAM heatmap...")
    heatmap = generate_gradcam(model, img_tensor, pred_result["class_idx"])
    orig_img = load_image_as_array(args.image)
    heatmap_bgr = overlay_heatmap(orig_img, heatmap)

    # 7. Damage Assessment
    severity_data = assess(disease_key, confidence, db)
    
    # 8. Recommendation
    rec_data = get_recommendation(disease_key, args.language, db)

    # --- CLI Output ---
    print("\n" + "="*50)
    print("DIAGNOSIS RESULTS")
    print("="*50)
    
    table_data = [
        ["Disease/Pest (English)", rec_data["name_en"]],
        [f"Regional Name ({args.language})", rec_data["name_regional"]],
        ["Confidence", f"{confidence*100:.1f}%"],
        ["Severity Stage", severity_data["severity_stage"].capitalize()],
        ["Estimated Yield Loss", f"{severity_data['yield_loss_pct']}%"],
        ["Estimated Loss (₹/Acre)", f"₹{severity_data['inr_per_acre']}"],
        ["Zone", zone]
    ]
    print(tabulate(table_data, tablefmt="fancy_grid"))
    
    print("\nRECOMMENDED TREATMENT:")
    sol = rec_data["solution"]
    print(f"Chemical: {sol.get('pesticide', 'N/A')}")
    print(f"Application: {sol.get('application', 'N/A')}")
    print(f"Organic: {sol.get('organic_alternative', 'N/A')}")
    print("="*50 + "\n")

    # 9. Generate PDF
    if args.output:
        logger.info("Generating PDF report...")
        pdf_data = {
            "disease_key": disease_key,
            "confidence": confidence,
            "severity_stage": severity_data["severity_stage"],
            "yield_loss_pct": severity_data["yield_loss_pct"],
            "inr_per_acre": severity_data["inr_per_acre"],
            "name_en": rec_data["name_en"],
            "name_regional": rec_data["name_regional"],
            "solution": rec_data["solution"],
            "crop_type": args.crop,
            "state": args.state,
            "zone": zone,
            "img_original": array_to_base64(orig_img),
            "img_heatmap": array_to_base64(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))
        }
        
        try:
            generate_pdf(pdf_data, args.output, args.language)
            logger.info(f"PDF report saved successfully to: {args.output}")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Disease Detector Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to the input leaf image")
    parser.add_argument("--crop", type=str, default="Unknown", help="Crop type (e.g., rice, wheat)")
    parser.add_argument("--state", type=str, default="Unknown", help="Indian state for zone filtering")
    parser.add_argument("--language", type=str, default="hi", choices=["hi", "ta", "te", "kn", "mr"], help="Regional language code")
    parser.add_argument("--output", type=str, help="Path to save the generated PDF report")
    
    args = parser.parse_args()
    run_pipeline(args)
