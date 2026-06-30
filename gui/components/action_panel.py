"""
Action Panel Component — CropSense AI
Dark green agricultural theme.
Provides the Generate PDF Report button.
"""

import tkinter as tk
from tkinter import ttk

from gui.styles import (
    BG_PANEL, BG_CARD, BORDER_COLOR,
    ACCENT_GREEN, ACCENT_BRIGHT,
    TEXT_PRIMARY, TEXT_MUTED,
    FONT_BODY_BOLD, FONT_SMALL,
)


class ActionPanel(ttk.Frame):
    def __init__(self, parent, on_generate_pdf):
        super().__init__(parent, style="Panel.TFrame")
        self.on_generate_pdf = on_generate_pdf
        self._build_ui()

    # -----------------------------------------------------------------------
    def _build_ui(self):
        self.configure(padding=(0, 8, 0, 0))

        # Divider line above button
        tk.Frame(self, bg=BORDER_COLOR, height=1).pack(fill="x", pady=(0, 14))

        # Button row
        btn_row = tk.Frame(self, bg=BG_PANEL)
        btn_row.pack(fill="x")

        self.btn_pdf = ttk.Button(
            btn_row,
            text="⬇  Generate PDF Report",
            style="Accent.TButton",
            command=self._handle_click,
            state=tk.DISABLED,
        )
        self.btn_pdf.pack(side="right")

        self.lbl_hint = tk.Label(
            btn_row,
            text="Run a diagnosis first to enable the report.",
            font=FONT_SMALL,
            fg=TEXT_MUTED,
            bg=BG_PANEL,
        )
        self.lbl_hint.pack(side="left", anchor="w")

    # -----------------------------------------------------------------------
    def _handle_click(self):
        self.on_generate_pdf()

    # -----------------------------------------------------------------------
    # Public API (called by app.py — do not rename)
    # -----------------------------------------------------------------------
    def enable_pdf_button(self):
        self.btn_pdf.config(state=tk.NORMAL)
        self.lbl_hint.config(text="Report ready — click to save as PDF.")

    def disable_pdf_button(self):
        self.btn_pdf.config(state=tk.DISABLED)
        self.lbl_hint.config(text="Run a diagnosis first to enable the report.")