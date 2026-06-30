"""
Image Panel Component — CropSense AI
Dark green agricultural theme.
Provides upload button and large image preview with drop-zone feel.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from gui.styles import (
    BG_PANEL, BG_CARD, BG_INPUT, BORDER_COLOR,
    ACCENT_GREEN, ACCENT_BRIGHT, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    FONT_SUBHEADING, FONT_BODY, FONT_BODY_BOLD, FONT_SMALL,
)

PREVIEW_SIZE = 420   # max px for image thumbnail


class ImagePanel(ttk.Frame):
    def __init__(self, parent, on_image_selected, title="Input Image"):
        super().__init__(parent, style="Panel.TFrame")
        self.on_image_selected = on_image_selected
        self.title_text = title
        self.current_image_path = None
        self._photo = None   # keep reference so GC doesn't collect it
        self._build_ui()

    # -----------------------------------------------------------------------
    def _build_ui(self):
        self.configure(padding=16)

        # Section title
        ttk.Label(self, text=self.title_text, style="Subheading.TLabel").pack(
            anchor="w", pady=(0, 12)
        )

        # Drop-zone / preview canvas
        self.canvas_frame = tk.Frame(
            self,
            bg=BG_CARD,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1,
            width=PREVIEW_SIZE + 20,
            height=PREVIEW_SIZE + 20,
        )
        self.canvas_frame.pack(pady=(0, 14))
        self.canvas_frame.pack_propagate(False)

        # Placeholder label (shown before any image is loaded)
        self.lbl_placeholder = tk.Label(
            self.canvas_frame,
            text="📂  Click 'Upload Image' to begin\n\n"
                 "Supported: JPG · JPEG · PNG",
            font=FONT_SMALL,
            fg=TEXT_MUTED,
            bg=BG_CARD,
            justify="center",
        )
        self.lbl_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        # Image label (hidden until image is loaded)
        self.lbl_image = tk.Label(self.canvas_frame, bg=BG_CARD, bd=0)
        # not packed yet — shown on first upload

        # Upload button
        ttk.Button(
            self,
            text="⬆  Upload Image",
            style="Upload.TButton",
            command=self._upload_image,
        ).pack()

        # Filename label
        self.lbl_filename = tk.Label(
            self,
            text="",
            font=FONT_SMALL,
            fg=TEXT_MUTED,
            bg=BG_PANEL,
            wraplength=PREVIEW_SIZE,
        )
        self.lbl_filename.pack(pady=(6, 0))

    # -----------------------------------------------------------------------
    def _upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
        )
        if file_path:
            self.current_image_path = file_path
            self._display_thumbnail(file_path)
            self.on_image_selected(file_path)

    # -----------------------------------------------------------------------
    def _display_thumbnail(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)

            self._photo = ImageTk.PhotoImage(img)

            # Hide placeholder, show image
            self.lbl_placeholder.place_forget()
            self.lbl_image.config(image=self._photo)
            self.lbl_image.place(relx=0.5, rely=0.5, anchor="center")

            # Show short filename below the box
            import os
            self.lbl_filename.config(text=os.path.basename(path))

        except Exception:
            self.lbl_placeholder.config(text="⚠  Could not load image.\nPlease try another file.")
            self.lbl_placeholder.place(relx=0.5, rely=0.5, anchor="center")
            self.lbl_image.place_forget()