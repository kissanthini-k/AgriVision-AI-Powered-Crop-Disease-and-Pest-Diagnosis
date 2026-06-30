"""
Main Tkinter Application Entry Point.

Integrates all panels and the underlying deep learning pipeline into a unified GUI.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import MODEL_PATH, CLASS_NAMES, MODEL_ARCHITECTURE, CONFIDENCE_THRESHOLD
from utils.helpers import load_image_as_array, array_to_base64
from preprocessing.preprocess import prepare_for_inference
from inference.predict import load_model, predict_with_tta
from inference.confidence_gate import check_confidence
from analysis.damage_assessment import assess
from analysis.recommendation import get_recommendation
from pdf_generator.generator import generate_pdf
from gui.styles import configure_styles
from gui.components.image_panel import ImagePanel
from gui.components.result_panel import ResultPanel
from gui.components.action_panel import ActionPanel

# ---------------------------------------------------------------------------
# Threshold for rejecting random/invalid images (car, person, sky, etc.)
# The model's top-1 confidence must exceed this to be considered a valid image.
# Tune between 0.35–0.50 based on testing with your specific dataset.
# ---------------------------------------------------------------------------
VALID_IMAGE_THRESHOLD = 0.05


class AgriDiseaseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crop Disease Detector — Indian Agriculture")
        self.geometry("1400x900")
        self.resizable(False, False)

        configure_styles()

        # Load disease/pest DB
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'disease_db.json')
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                self.db = json.load(f)
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to load disease DB: {str(e)}")
            self.db = {}

        self._build_layout()

    def _build_layout(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.tabs = {}
        tab_configs = [
            ("crop", "Crop Disease Diagnosis", "Input Crop Leaf Image"),
            ("pest", "Pest Diagnosis",          "Input Pest Image"),
        ]

        for tab_id, tab_title, img_title in tab_configs:
            tab_frame = ttk.Frame(self.notebook, padding="10 10 10 10")
            self.notebook.add(tab_frame, text=tab_title)

            # Left Panel — image upload
            left_frame = ttk.Frame(tab_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

            image_panel = ImagePanel(
                left_frame,
                lambda p, tid=tab_id: self.on_image_selected(p, tid),
                title=img_title
            )
            image_panel.pack(fill=tk.BOTH, expand=True)

            # Right Panel — results + actions
            right_frame = ttk.Frame(tab_frame)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

            result_panel = ResultPanel(right_frame)
            result_panel.pack(fill=tk.BOTH, expand=True)

            ttk.Frame(right_frame, height=20).pack()

            action_panel = ActionPanel(
                right_frame,
                lambda tid=tab_id: self.on_generate_pdf(tid)
            )
            action_panel.pack(fill=tk.X, pady=(20, 0))

            self.tabs[tab_id] = {
                "image_panel":        image_panel,
                "result_panel":       result_panel,
                "action_panel":       action_panel,
                "current_image_path": None,
                "current_result_data": None,
            }

        # Loading spinner (shown at bottom during inference)
        self.spinner = ttk.Progressbar(self, mode='indeterminate')

    # -----------------------------------------------------------------------
    # Event: image selected by user
    # -----------------------------------------------------------------------
    def on_image_selected(self, file_path, tab_id):
        tab = self.tabs[tab_id]
        tab["current_image_path"] = file_path
        tab["result_panel"].clear_results()
        tab["action_panel"].disable_pdf_button()

        # Show spinner and kick off background inference thread
        self.spinner.pack(side=tk.BOTTOM, fill=tk.X)
        self.spinner.start(10)

        threading.Thread(
            target=self._run_pipeline,
            args=(file_path, tab_id),
            daemon=True
        ).start()

    # -----------------------------------------------------------------------
    # Background thread: full inference pipeline
    # -----------------------------------------------------------------------
    def _run_pipeline(self, file_path, tab_id):
        try:
            # Step 1 & 2 — Load model and run inference with Test-Time Augmentation (TTA)
            model      = load_model(MODEL_PATH, MODEL_ARCHITECTURE)
            pred_result = predict_with_tta(file_path, model, CLASS_NAMES)

            # ------------------------------------------------------------------
            # Step 3 — Validate image: is it agricultural? is it in the right tab?
            #
            # OLD (broken) approach: summed probability mass across all pest/disease
            # classes and applied a 15x correction factor. This failed because with
            # 160 pest classes vs 17 disease classes, a random image (car, person)
            # naturally has most of its probability mass on pests regardless of the
            # correction factor.
            #
            # NEW approach:
            #   a) Check top-1 confidence against VALID_IMAGE_THRESHOLD.
            #      A model that has never seen a car will spread its probability
            #      thinly — top-1 will be low (5–20%). If it's below the threshold,
            #      the image is not agricultural at all → reject.
            #   b) Check the TYPE of the top-1 prediction against the active tab.
            #      If the model is confident but it's the wrong type → reject.
            # ------------------------------------------------------------------

            top_class      = pred_result["class"]       # best predicted class name
            top_confidence = pred_result["confidence"]  # its softmax probability
            top_class_type = self.db.get(top_class, {}).get("type", "unknown")

            # a) Reject random / non-agricultural images
            if top_confidence < VALID_IMAGE_THRESHOLD:
                self.after(0, self._show_invalid_image,
                           "Image not valid.\nPlease upload a relevant crop leaf or pest image.")
                return

            # b) Reject images that belong to the wrong tab
            if tab_id == "crop" and top_class_type != "disease":
                self.after(0, self._show_invalid_image,
                           "Image not valid for this tab.\nPlease upload a crop leaf image.")
                return

            if tab_id == "pest" and top_class_type != "pest":
                self.after(0, self._show_invalid_image,
                           "Image not valid for this tab.\nPlease upload a pest image.")
                return

            # Validation passed — use top prediction directly
            disease_key = top_class
            confidence  = top_confidence

            # Step 4 — Confidence gate (low-confidence but still above threshold)
            gate = check_confidence(confidence, CONFIDENCE_THRESHOLD)
            if not gate["proceed"]:
                self.after(0, self._show_error, gate["message"])
                return

            # Step 5 — Load original image for display
            orig_img = load_image_as_array(file_path)

            # Step 6 — Damage assessment
            severity_data = assess(disease_key, confidence, self.db)

            # Step 7 — English recommendations
            rec_data = get_recommendation(disease_key, "en", self.db)

            # Bundle result data for UI + PDF
            self.tabs[tab_id]["current_result_data"] = {
                "disease_key":        disease_key,
                "confidence":         confidence,
                "top_5":              pred_result.get("top_5", []),
                "severity_stage":     severity_data["severity_stage"],
                "yield_loss_pct":     severity_data["yield_loss_pct"],
                "inr_per_acre":       severity_data["inr_per_acre"],
                "name_en":            rec_data["name_en"],
                "solution":           rec_data["solution"],
                "img_original_array": orig_img,
            }

            self.after(0, self._update_ui_success, tab_id)

        except Exception as e:
            self.after(0, self._show_error, f"Pipeline Error: {str(e)}")

    # -----------------------------------------------------------------------
    # UI update helpers (must run on main thread via self.after())
    # -----------------------------------------------------------------------
    def _update_ui_success(self, tab_id):
        self._stop_spinner()
        tab = self.tabs[tab_id]
        data = tab["current_result_data"]
        tab["result_panel"].update_results(data)
        tab["action_panel"].enable_pdf_button()

    def _show_invalid_image(self, message):
        """Used specifically for tab-boundary / invalid image rejections."""
        self._stop_spinner()
        messagebox.showwarning("Invalid Image", message)

    def _show_error(self, message):
        """Used for pipeline errors and confidence gate failures."""
        self._stop_spinner()
        messagebox.showerror("Error", message)

    def _stop_spinner(self):
        self.spinner.stop()
        self.spinner.pack_forget()

    # -----------------------------------------------------------------------
    # Event: generate PDF report
    # -----------------------------------------------------------------------
    def on_generate_pdf(self, tab_id):
        tab = self.tabs[tab_id]
        if not tab["current_result_data"] or not tab["current_image_path"]:
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save Report As"
        )
        if not save_path:
            return

        try:
            pdf_data = tab["current_result_data"].copy()
            pdf_data["crop_type"] = "N/A"
            pdf_data["state"]     = "N/A"
            pdf_data["zone"]      = "N/A"

            # Convert numpy image array → base64 for PDF generator
            pdf_data["img_original"] = array_to_base64(
                tab["current_result_data"]["img_original_array"]
            )

            generate_pdf(pdf_data, save_path)
            messagebox.showinfo("Success", f"Report saved successfully to:\n{save_path}")

        except Exception as e:
            messagebox.showerror("PDF Error", f"Failed to generate PDF: {str(e)}")


if __name__ == "__main__":
    app = AgriDiseaseApp()
    app.mainloop()