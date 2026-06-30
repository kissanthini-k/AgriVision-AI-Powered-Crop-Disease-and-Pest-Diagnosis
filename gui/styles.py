"""
GUI Styles Module — CropSense AI
Dark green agricultural theme.
"""

from tkinter import ttk
import tkinter as tk

# ── Palette ────────────────────────────────────────────────────────────────
BG_DARK        = "#0F1F0F"   # near-black forest green — main window bg
BG_PANEL       = "#162816"   # slightly lighter — panel / card bg
BG_CARD        = "#1E3A1E"   # card surfaces
BG_INPUT       = "#1A301A"   # text boxes / listboxes
BORDER_COLOR   = "#2E5A2E"   # subtle green border
ACCENT_GREEN   = "#4CAF50"   # primary action green
ACCENT_BRIGHT  = "#76C442"   # highlight / hover
ACCENT_GOLD    = "#FFB300"   # early severity / warnings
ACCENT_ORANGE  = "#FB8C00"   # mid severity
ACCENT_RED     = "#E53935"   # late severity / errors
TEXT_PRIMARY   = "#E8F5E9"   # near-white for headings
TEXT_SECONDARY = "#A5D6A7"   # muted green for labels
TEXT_MUTED     = "#66BB6A"   # very muted for placeholders

# ── Typography ─────────────────────────────────────────────────────────────
FONT_APP_TITLE   = ("Segoe UI", 11, "bold")
FONT_HEADING     = ("Segoe UI", 15, "bold")
FONT_SUBHEADING  = ("Segoe UI", 11, "bold")
FONT_BODY        = ("Segoe UI", 10)
FONT_BODY_BOLD   = ("Segoe UI", 10, "bold")
FONT_SMALL       = ("Segoe UI", 9)
FONT_MONO        = ("Consolas", 9)
FONT_BADGE       = ("Segoe UI", 9, "bold")

# ── Severity colour map ────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "EARLY":   ("#FFB300", "#0F1F0F"),   # (bg, fg)
    "MID":     ("#FB8C00", "#FFFFFF"),
    "LATE":    ("#E53935", "#FFFFFF"),
    "UNKNOWN": ("#455A64", "#FFFFFF"),
}


def configure_styles():
    """Apply dark green theme to all ttk widgets."""
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    # ── Base ──────────────────────────────────────────────────────────────
    style.configure(".",
        background=BG_DARK,
        foreground=TEXT_PRIMARY,
        font=FONT_BODY,
        borderwidth=0,
        relief="flat",
    )

    # ── Frames ────────────────────────────────────────────────────────────
    style.configure("TFrame",        background=BG_DARK)
    style.configure("Panel.TFrame",  background=BG_PANEL)
    style.configure("Card.TFrame",   background=BG_CARD,  relief="flat")

    # ── Labels ────────────────────────────────────────────────────────────
    style.configure("TLabel",
        background=BG_DARK, foreground=TEXT_PRIMARY, font=FONT_BODY)

    style.configure("Panel.TLabel",
        background=BG_PANEL, foreground=TEXT_PRIMARY, font=FONT_BODY)

    style.configure("Card.TLabel",
        background=BG_CARD, foreground=TEXT_PRIMARY, font=FONT_BODY)

    style.configure("Heading.TLabel",
        background=BG_PANEL, foreground=ACCENT_BRIGHT,
        font=FONT_HEADING)

    style.configure("Subheading.TLabel",
        background=BG_PANEL, foreground=ACCENT_GREEN,
        font=FONT_SUBHEADING)

    style.configure("Muted.TLabel",
        background=BG_PANEL, foreground=TEXT_SECONDARY, font=FONT_SMALL)

    style.configure("CardMuted.TLabel",
        background=BG_CARD, foreground=TEXT_SECONDARY, font=FONT_SMALL)

    style.configure("CardValue.TLabel",
        background=BG_CARD, foreground=TEXT_PRIMARY, font=FONT_BODY_BOLD)

    # ── Notebook (tabs) ───────────────────────────────────────────────────
    style.configure("TNotebook",
        background=BG_DARK, borderwidth=0, tabmargins=[0, 0, 0, 0])

    style.configure("TNotebook.Tab",
        background=BG_PANEL,
        foreground=TEXT_SECONDARY,
        font=FONT_BODY_BOLD,
        padding=[18, 8],
        borderwidth=0,
    )
    style.map("TNotebook.Tab",
        background=[("selected", BG_CARD), ("active", BG_CARD)],
        foreground=[("selected", ACCENT_BRIGHT), ("active", TEXT_PRIMARY)],
    )

    # ── Buttons ───────────────────────────────────────────────────────────
    style.configure("TButton",
        background=BG_CARD, foreground=TEXT_PRIMARY,
        font=FONT_BODY, padding=[10, 6], relief="flat", borderwidth=0)
    style.map("TButton",
        background=[("active", BORDER_COLOR), ("disabled", BG_PANEL)],
        foreground=[("disabled", TEXT_MUTED)])

    style.configure("Accent.TButton",
        background=ACCENT_GREEN, foreground="#0F1F0F",
        font=FONT_BODY_BOLD, padding=[14, 8], relief="flat")
    style.map("Accent.TButton",
        background=[("active", ACCENT_BRIGHT), ("disabled", BORDER_COLOR)],
        foreground=[("disabled", BG_PANEL)])

    style.configure("Upload.TButton",
        background=BG_CARD, foreground=ACCENT_BRIGHT,
        font=FONT_BODY_BOLD, padding=[12, 7],
        relief="flat", borderwidth=1)
    style.map("Upload.TButton",
        background=[("active", BORDER_COLOR)])

    # ── Progressbar ───────────────────────────────────────────────────────
    style.configure("TProgressbar",
        background=ACCENT_GREEN, troughcolor=BG_CARD,
        borderwidth=0, thickness=10)

    style.configure("Confidence.Horizontal.TProgressbar",
        background=ACCENT_GREEN, troughcolor=BG_CARD,
        borderwidth=0, thickness=12)

    # ── Separator ─────────────────────────────────────────────────────────
    style.configure("TSeparator", background=BORDER_COLOR)

    # ── Scrollbar ─────────────────────────────────────────────────────────
    style.configure("TScrollbar",
        background=BG_CARD, troughcolor=BG_PANEL,
        borderwidth=0, arrowcolor=TEXT_MUTED)