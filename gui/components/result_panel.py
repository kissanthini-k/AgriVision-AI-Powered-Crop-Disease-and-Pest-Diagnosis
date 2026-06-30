"""
Result Panel Component — CropSense AI
Dark green agricultural theme.
Displays diagnosis name, confidence bar, severity badge, yield loss,
treatment recommendation, and Top-5 predictions with mini bars.
"""

import tkinter as tk
from tkinter import ttk

from gui.styles import (
    BG_PANEL, BG_CARD, BG_INPUT, BORDER_COLOR,
    ACCENT_GREEN, ACCENT_BRIGHT, ACCENT_GOLD, ACCENT_ORANGE, ACCENT_RED,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    FONT_HEADING, FONT_SUBHEADING, FONT_BODY, FONT_BODY_BOLD, FONT_SMALL, FONT_BADGE,
    SEVERITY_COLORS,
)


def _fmt_class_name(raw: str) -> str:
    """
    Convert raw model class name to a readable display name.
    e.g. 'wheat_rust_yellow' → 'Wheat Rust Yellow'
         'fruit_borer_pest'  → 'Fruit Borer Pest'
    """
    return raw.replace("_", " ").replace("-", " ").title()


class ResultPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="Panel.TFrame")
        self._build_ui()

    # -----------------------------------------------------------------------
    def _build_ui(self):
        self.configure(padding=16)

        # ── Section heading ────────────────────────────────────────────────
        ttk.Label(self, text="Diagnosis Results", style="Heading.TLabel").pack(
            anchor="w", pady=(0, 14)
        )

        # ── Card: Detected name + confidence ──────────────────────────────
        card_detect = self._card(self)
        card_detect.pack(fill="x", pady=(0, 8))

        ttk.Label(card_detect, text="DETECTED", style="CardMuted.TLabel").grid(
            row=0, column=0, sticky="w", padx=14, pady=(10, 2)
        )
        self.lbl_disease = tk.Label(
            card_detect,
            text="—",
            font=FONT_SUBHEADING,
            fg=ACCENT_BRIGHT,
            bg=BG_CARD,
        )
        self.lbl_disease.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 4))

        ttk.Separator(card_detect, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=4
        )

        ttk.Label(card_detect, text="CONFIDENCE", style="CardMuted.TLabel").grid(
            row=3, column=0, sticky="w", padx=14, pady=(4, 2)
        )

        conf_row = tk.Frame(card_detect, bg=BG_CARD)
        conf_row.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))
        card_detect.columnconfigure(0, weight=1)

        self.progress_conf = ttk.Progressbar(
            conf_row,
            length=260,
            mode="determinate",
            style="Confidence.Horizontal.TProgressbar",
        )
        self.progress_conf.pack(side="left", padx=(0, 10))

        self.lbl_conf_text = tk.Label(
            conf_row, text="0%", font=FONT_BODY_BOLD,
            fg=ACCENT_GREEN, bg=BG_CARD
        )
        self.lbl_conf_text.pack(side="left")

        # ── Card: Severity + Yield Loss ────────────────────────────────────
        card_sev = self._card(self)
        card_sev.pack(fill="x", pady=(0, 8))
        card_sev.columnconfigure(1, weight=1)

        ttk.Label(card_sev, text="SEVERITY STAGE", style="CardMuted.TLabel").grid(
            row=0, column=0, sticky="w", padx=14, pady=(10, 4)
        )
        ttk.Label(card_sev, text="EST. YIELD LOSS", style="CardMuted.TLabel").grid(
            row=0, column=1, sticky="w", padx=14, pady=(10, 4)
        )

        self.lbl_severity = tk.Label(
            card_sev,
            text="—",
            font=FONT_BADGE,
            fg="#0F1F0F",
            bg=BORDER_COLOR,
            padx=10, pady=4,
        )
        self.lbl_severity.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 12))

        self.lbl_loss = tk.Label(
            card_sev, text="—",
            font=FONT_BODY_BOLD, fg=TEXT_PRIMARY, bg=BG_CARD
        )
        self.lbl_loss.grid(row=1, column=1, sticky="w", padx=14, pady=(0, 12))

        # ── Card: Recommended Treatment ────────────────────────────────────
        card_treat = self._card(self)
        card_treat.pack(fill="x", pady=(0, 8))

        ttk.Label(card_treat, text="RECOMMENDED TREATMENT", style="CardMuted.TLabel").pack(
            anchor="w", padx=14, pady=(10, 4)
        )

        self.text_treatment = tk.Text(
            card_treat,
            height=5,
            font=FONT_BODY,
            bg=BG_INPUT,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            wrap="word",
            padx=10,
            pady=8,
            borderwidth=0,
        )
        self.text_treatment.pack(fill="x", padx=14, pady=(0, 12))
        self._set_treatment_text("Upload an image to see treatment recommendations.")
        self.text_treatment.config(state=tk.DISABLED)

        # ── Card: Top 5 Predictions ────────────────────────────────────────
        card_top5 = self._card(self)
        card_top5.pack(fill="x", pady=(0, 4))

        ttk.Label(card_top5, text="TOP 5 PREDICTIONS", style="CardMuted.TLabel").pack(
            anchor="w", padx=14, pady=(10, 6)
        )

        self.top5_frame = tk.Frame(card_top5, bg=BG_CARD)
        self.top5_frame.pack(fill="x", padx=14, pady=(0, 12))

        # Placeholder rows
        self._top5_rows = []
        for _ in range(5):
            row = self._make_top5_row(self.top5_frame)
            self._top5_rows.append(row)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _card(self, parent) -> tk.Frame:
        """Returns a styled card frame."""
        return tk.Frame(
            parent,
            bg=BG_CARD,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1,
        )

    def _make_top5_row(self, parent) -> dict:
        """Creates one Top-5 row with label + mini bar + pct."""
        frame = tk.Frame(parent, bg=BG_CARD)
        frame.pack(fill="x", pady=2)

        lbl_name = tk.Label(
            frame, text="", width=26, anchor="w",
            font=FONT_SMALL, fg=TEXT_SECONDARY, bg=BG_CARD
        )
        lbl_name.pack(side="left")

        canvas = tk.Canvas(
            frame, height=10, width=160,
            bg=BG_INPUT, highlightthickness=0, bd=0
        )
        canvas.pack(side="left", padx=(6, 6))

        lbl_pct = tk.Label(
            frame, text="", width=6, anchor="e",
            font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_CARD
        )
        lbl_pct.pack(side="left")

        return {"frame": frame, "name": lbl_name, "canvas": canvas, "pct": lbl_pct}

    def _draw_bar(self, canvas, value: float, color: str):
        """Draw a mini progress bar on a canvas. value is 0.0–1.0."""
        canvas.delete("all")
        w = int(canvas.winfo_reqwidth() * value)
        if w > 0:
            canvas.create_rectangle(0, 0, w, 10, fill=color, outline="")

    def _set_treatment_text(self, text: str):
        self.text_treatment.config(state=tk.NORMAL)
        self.text_treatment.delete(1.0, tk.END)
        self.text_treatment.insert(tk.END, text)
        self.text_treatment.config(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # Public API (called by app.py — do not rename)
    # -----------------------------------------------------------------------
    def update_results(self, result: dict):
        # Disease name — use formatted display name
        self.lbl_disease.config(text=result.get("name_en", "Unknown"))

        # Confidence bar
        conf = result.get("confidence", 0.0)
        self.progress_conf["value"] = conf * 100
        pct_text = f"{conf * 100:.1f}%"
        self.lbl_conf_text.config(text=pct_text)

        # Severity badge
        sev = result.get("severity_stage", "unknown").upper()
        bg_col, fg_col = SEVERITY_COLORS.get(sev, SEVERITY_COLORS["UNKNOWN"])
        self.lbl_severity.config(text=sev, bg=bg_col, fg=fg_col)

        # Yield loss
        loss_pct = result.get("yield_loss_pct", 0)
        inr      = result.get("inr_per_acre", 0)
        self.lbl_loss.config(text=f"{loss_pct}%  (≈ ₹{inr:,} / acre)")

        # Treatment text
        sol  = result.get("solution", {})
        pest = sol.get("pesticide",          "N/A")
        app  = sol.get("application",        "N/A")
        org  = sol.get("organic_alternative","N/A")
        self._set_treatment_text(
            f"Chemical:  {pest}\n"
            f"Application:  {app}\n"
            f"Organic:  {org}"
        )

        # Top 5 — formatted names + mini bars
        top5 = result.get("top_5", [])
        max_conf = top5[0]["confidence"] if top5 else 1.0

        for i, row in enumerate(self._top5_rows):
            if i < len(top5):
                pred  = top5[i]
                name  = _fmt_class_name(pred["class"])
                score = pred["confidence"]
                # Truncate long names
                display = name if len(name) <= 28 else name[:25] + "…"
                bar_color = ACCENT_BRIGHT if i == 0 else BORDER_COLOR
                row["name"].config(text=display, fg=ACCENT_BRIGHT if i == 0 else TEXT_SECONDARY)
                row["pct"].config(text=f"{score*100:.1f}%")
                self._draw_bar(row["canvas"], score / max_conf if max_conf > 0 else 0, bar_color)
            else:
                row["name"].config(text="")
                row["pct"].config(text="")
                row["canvas"].delete("all")

    def clear_results(self):
        self.lbl_disease.config(text="—")
        self.progress_conf["value"] = 0
        self.lbl_conf_text.config(text="0%")
        self.lbl_severity.config(text="—", bg=BORDER_COLOR, fg="#0F1F0F")
        self.lbl_loss.config(text="—")
        self._set_treatment_text("Upload an image to see treatment recommendations.")

        for row in self._top5_rows:
            row["name"].config(text="")
            row["pct"].config(text="")
            row["canvas"].delete("all")