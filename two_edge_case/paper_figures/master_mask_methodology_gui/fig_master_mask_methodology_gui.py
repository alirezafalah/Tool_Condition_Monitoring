import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageFont, ImageOps


class MasterMaskFigureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Master Mask Methodology Figure Builder")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_out_name = "master_mask_methodology_figure.png"

        self.path_vars = {
            "frame1": tk.StringVar(),
            "frame2": tk.StringVar(),
            "frame3": tk.StringVar(),
            "master": tk.StringVar(),
        }

        self.label_vars = {
            "frame1": tk.StringVar(value="Frame 0°"),
            "frame2": tk.StringVar(value="Frame 90°"),
            "frame3": tk.StringVar(value="Frame 360°"),
            "master": tk.StringVar(value="Master Mask"),
        }

        self.explanation_var = tk.StringVar(
            value="Master Mask is created by logical OR (union) of all binary mask frames."
        )

        self.target_height_var = tk.IntVar(value=900)
        self.margin_var = tk.IntVar(value=60)
        self.spacing_var = tk.IntVar(value=36)

        self._build_ui()

    def _build_ui(self):
        outer = tk.Frame(self.root, padx=12, pady=12)
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer,
            text="Select 3 input frames and 1 final master mask image",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        self._add_file_row(outer, 1, "Frame 1", "frame1")
        self._add_file_row(outer, 2, "Frame 2", "frame2")
        self._add_file_row(outer, 3, "Frame 3", "frame3")
        self._add_file_row(outer, 4, "Master Mask", "master")

        tk.Label(outer, text="Label Frame 1:").grid(row=5, column=0, sticky="w", pady=(8, 2))
        tk.Entry(outer, textvariable=self.label_vars["frame1"], width=55).grid(row=5, column=1, columnspan=3, sticky="we", pady=(8, 2))

        tk.Label(outer, text="Label Frame 2:").grid(row=6, column=0, sticky="w", pady=2)
        tk.Entry(outer, textvariable=self.label_vars["frame2"], width=55).grid(row=6, column=1, columnspan=3, sticky="we", pady=2)

        tk.Label(outer, text="Label Frame 3:").grid(row=7, column=0, sticky="w", pady=2)
        tk.Entry(outer, textvariable=self.label_vars["frame3"], width=55).grid(row=7, column=1, columnspan=3, sticky="we", pady=2)

        tk.Label(outer, text="Label Master:").grid(row=8, column=0, sticky="w", pady=2)
        tk.Entry(outer, textvariable=self.label_vars["master"], width=55).grid(row=8, column=1, columnspan=3, sticky="we", pady=2)

        tk.Label(outer, text="Explanation text:").grid(row=9, column=0, sticky="w", pady=(8, 2))
        tk.Entry(outer, textvariable=self.explanation_var, width=55).grid(row=9, column=1, columnspan=3, sticky="we", pady=(8, 2))

        tk.Label(outer, text="Target image height (px):").grid(row=10, column=0, sticky="w", pady=(8, 2))
        tk.Entry(outer, textvariable=self.target_height_var, width=10).grid(row=10, column=1, sticky="w", pady=(8, 2))

        tk.Label(outer, text="Outer margin (px):").grid(row=10, column=2, sticky="e", pady=(8, 2))
        tk.Entry(outer, textvariable=self.margin_var, width=10).grid(row=10, column=3, sticky="w", pady=(8, 2))

        tk.Label(outer, text="Element spacing (px):").grid(row=11, column=0, sticky="w", pady=2)
        tk.Entry(outer, textvariable=self.spacing_var, width=10).grid(row=11, column=1, sticky="w", pady=2)

        btn_frame = tk.Frame(outer)
        btn_frame.grid(row=12, column=0, columnspan=4, sticky="we", pady=(14, 2))

        tk.Button(btn_frame, text="Generate + Save PNG", command=self.generate_figure, height=2).pack(side="left", padx=(0, 8))
        tk.Button(btn_frame, text="Quit", command=self.root.destroy, height=2).pack(side="left")

        for col in range(4):
            outer.grid_columnconfigure(col, weight=1)

    def _add_file_row(self, parent, row, title, key):
        tk.Label(parent, text=f"{title}:").grid(row=row, column=0, sticky="w", pady=2)
        tk.Entry(parent, textvariable=self.path_vars[key], width=60).grid(row=row, column=1, columnspan=2, sticky="we", pady=2)
        tk.Button(parent, text="Browse", command=lambda k=key: self._browse_file(k)).grid(row=row, column=3, sticky="e", pady=2)

    def _browse_file(self, key):
        path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
            initialdir=self.base_dir,
        )
        if path:
            self.path_vars[key].set(path)

    def _get_font(self, size):
        candidates = [
            "arial.ttf",
            "segoeui.ttf",
            "calibri.ttf",
            "DejaVuSans.ttf",
        ]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _load_and_resize(self, path, target_height):
        img = Image.open(path).convert("L")
        img = ImageOps.autocontrast(img)
        w, h = img.size
        scale = target_height / float(h)
        new_size = (max(1, int(w * scale)), target_height)
        return img.resize(new_size, Image.Resampling.NEAREST)

    @staticmethod
    def _text_size(draw, text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def generate_figure(self):
        required_keys = ["frame1", "frame2", "frame3", "master"]
        for key in required_keys:
            path = self.path_vars[key].get().strip()
            if not path:
                messagebox.showerror("Missing input", f"Please select {key} image.")
                return
            if not os.path.exists(path):
                messagebox.showerror("Invalid file", f"File not found:\n{path}")
                return

        try:
            target_h = int(self.target_height_var.get())
            margin = int(self.margin_var.get())
            spacing = int(self.spacing_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Invalid values", "Height, margin, and spacing must be integers.")
            return

        if target_h < 200:
            messagebox.showerror("Invalid height", "Target height should be at least 200 px.")
            return

        if margin < 0 or spacing < 0:
            messagebox.showerror("Invalid values", "Margin and spacing must be >= 0.")
            return

        try:
            frame1 = self._load_and_resize(self.path_vars["frame1"].get().strip(), target_h)
            frame2 = self._load_and_resize(self.path_vars["frame2"].get().strip(), target_h)
            frame3 = self._load_and_resize(self.path_vars["frame3"].get().strip(), target_h)
            master = self._load_and_resize(self.path_vars["master"].get().strip(), target_h)
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not load images:\n{exc}")
            return

        label_font = self._get_font(max(20, target_h // 22))
        symbol_font = self._get_font(max(38, target_h // 9))
        explain_font = self._get_font(max(20, target_h // 28))

        temp = Image.new("RGB", (10, 10), "white")
        draw_temp = ImageDraw.Draw(temp)

        symbols = ["+", "+", "...", "+", "="]
        symbol_sizes = [self._text_size(draw_temp, s, symbol_font) for s in symbols]
        symbol_widths = [s[0] for s in symbol_sizes]

        img_w = [frame1.width, frame2.width, frame3.width, master.width]

        # Sequence: frame1 + frame2 + ... + frame3 = master
        canvas_w = (
            2 * margin
            + img_w[0]
            + spacing + symbol_widths[0] + spacing
            + img_w[1]
            + spacing + symbol_widths[1] + spacing
            + symbol_widths[2]
            + spacing + symbol_widths[3] + spacing
            + img_w[2]
            + spacing + symbol_widths[4] + spacing
            + img_w[3]
        )

        label_texts = [
            self.label_vars["frame1"].get().strip(),
            self.label_vars["frame2"].get().strip(),
            self.label_vars["frame3"].get().strip(),
            self.label_vars["master"].get().strip(),
        ]
        label_heights = [self._text_size(draw_temp, txt, label_font)[1] if txt else 0 for txt in label_texts]
        labels_band_h = max(label_heights) + 16 if any(label_texts) else 0

        explanation = self.explanation_var.get().strip()
        exp_h = self._text_size(draw_temp, explanation, explain_font)[1] + 12 if explanation else 0

        canvas_h = margin + target_h + labels_band_h + exp_h + margin
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)

        y_img = margin
        y_symbol = y_img + target_h // 2

        x = margin

        def paste_image_and_label(img, text):
            nonlocal x
            canvas.paste(Image.merge("RGB", (img, img, img)), (x, y_img))
            if text:
                tw, th = self._text_size(draw, text, label_font)
                tx = x + (img.width - tw) // 2
                ty = y_img + target_h + 8
                draw.text((tx, ty), text, fill="black", font=label_font)
            x += img.width

        def draw_symbol(sym, sym_w):
            nonlocal x
            x += spacing
            sw, sh = self._text_size(draw, sym, symbol_font)
            draw.text((x + (sym_w - sw) // 2, y_symbol - sh // 2), sym, fill="black", font=symbol_font)
            x += sym_w + spacing

        paste_image_and_label(frame1, label_texts[0])
        draw_symbol("+", symbol_widths[0])

        paste_image_and_label(frame2, label_texts[1])
        draw_symbol("+", symbol_widths[1])
        draw_symbol("...", symbol_widths[2])
        draw_symbol("+", symbol_widths[3])

        paste_image_and_label(frame3, label_texts[2])
        draw_symbol("=", symbol_widths[4])

        paste_image_and_label(master, label_texts[3])

        if explanation:
            tw, th = self._text_size(draw, explanation, explain_font)
            tx = (canvas_w - tw) // 2
            ty = y_img + target_h + labels_band_h + 6
            draw.text((tx, ty), explanation, fill="black", font=explain_font)

        default_path = os.path.join(self.base_dir, self.default_out_name)
        out_path = filedialog.asksaveasfilename(
            title="Save stitched methodology figure",
            defaultextension=".png",
            initialfile=self.default_out_name,
            initialdir=self.base_dir,
            filetypes=[("PNG image", "*.png")],
        )
        if not out_path:
            out_path = default_path

        try:
            canvas.save(out_path, format="PNG", optimize=False, dpi=(300, 300))
            messagebox.showinfo("Saved", f"Figure saved successfully:\n{out_path}")
        except Exception as exc:
            messagebox.showerror("Save error", f"Could not save output:\n{exc}")


def main():
    root = tk.Tk()
    app = MasterMaskFigureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
