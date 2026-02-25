import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageFont


class MasterMaskEquationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Master Mask Equation Figure Generator")
        self.root.geometry("860x560")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ccd_data_dir = os.path.abspath(os.path.join(self.base_dir, "..", "..", "..", ".."))
        self.data_dir = os.path.join(self.ccd_data_dir, "DATA")
        self.masks_dir = os.path.join(self.data_dir, "masks")
        self.master_masks_dir = os.path.join(
            self.data_dir,
            "threshold_analysis",
            "master_mask_perspective",
            "master_masks",
        )
        self.papers_dir = os.path.join(self.ccd_data_dir, "Papers")
        self.default_output_name = "master_mask_methodology_equation.png"

        self.frame1_path = tk.StringVar()
        self.frame2_path = tk.StringVar()
        self.frame3_path = tk.StringVar()
        self.master_path = tk.StringVar()

        self.label1 = tk.StringVar(value="Frame 0°")
        self.label2 = tk.StringVar(value="Frame 90°")
        self.label3 = tk.StringVar(value="Frame 360°")
        self.label_master = tk.StringVar(value="Master Mask")

        self.target_height = tk.IntVar(value=900)
        self.spacing = tk.IntVar(value=38)
        self.symbol_font_scale = tk.DoubleVar(value=1.6)
        self.label_font_scale = tk.DoubleVar(value=1.8)

        self._build_ui()

    def _build_ui(self):
        container = tk.Frame(self.root, padx=14, pady=14)
        container.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            container,
            text="Create methodology figure: [Frame 1] + [Frame 2] + ... + [Frame 3] = [Master Mask]",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            justify="left",
        )
        title.pack(fill=tk.X, pady=(0, 12))

        self._file_row(container, "Frame 1 image", self.frame1_path, "frame")
        self._file_row(container, "Frame 2 image", self.frame2_path, "frame")
        self._file_row(container, "Frame 3 image", self.frame3_path, "frame")
        self._file_row(container, "Master Mask image", self.master_path, "master")

        labels_box = tk.LabelFrame(container, text="Optional labels under images", padx=10, pady=8)
        labels_box.pack(fill=tk.X, pady=(12, 10))

        self._label_row(labels_box, "Label Frame 1", self.label1)
        self._label_row(labels_box, "Label Frame 2", self.label2)
        self._label_row(labels_box, "Label Frame 3", self.label3)
        self._label_row(labels_box, "Label Master", self.label_master)

        settings_box = tk.LabelFrame(container, text="Figure settings", padx=10, pady=8)
        settings_box.pack(fill=tk.X, pady=(4, 12))

        tk.Label(settings_box, text="Image target height (px):").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        tk.Entry(settings_box, textvariable=self.target_height, width=12).grid(row=0, column=1, sticky="w", pady=4)

        tk.Label(settings_box, text="Horizontal spacing (px):").grid(row=0, column=2, sticky="w", padx=(16, 6), pady=4)
        tk.Entry(settings_box, textvariable=self.spacing, width=12).grid(row=0, column=3, sticky="w", pady=4)

        tk.Label(settings_box, text="Symbol size multiplier:").grid(row=0, column=4, sticky="w", padx=(16, 6), pady=4)
        tk.Entry(settings_box, textvariable=self.symbol_font_scale, width=12).grid(row=0, column=5, sticky="w", pady=4)

        tk.Label(settings_box, text="Label size multiplier:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        tk.Entry(settings_box, textvariable=self.label_font_scale, width=12).grid(row=1, column=1, sticky="w", pady=4)

        output_box = tk.LabelFrame(container, text="Output", padx=10, pady=8)
        output_box.pack(fill=tk.X)

        self.output_path_var = tk.StringVar(value=os.path.join(self.papers_dir, self.default_output_name))
        tk.Entry(output_box, textvariable=self.output_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        tk.Button(output_box, text="Browse", command=self.select_output_path).pack(side=tk.LEFT)

        buttons = tk.Frame(container)
        buttons.pack(fill=tk.X, pady=(14, 4))

        tk.Button(buttons, text="Generate Figure", command=self.generate_figure, height=2, bg="#2E7D32", fg="white").pack(side=tk.LEFT)
        tk.Button(buttons, text="Quit", command=self.root.quit, height=2).pack(side=tk.RIGHT)

        hint = tk.Label(
            container,
            text="Tip: The final PNG is saved at high resolution (300 DPI metadata), suitable for paper figures.",
            fg="#555555",
            anchor="w",
            justify="left",
        )
        hint.pack(fill=tk.X, pady=(8, 0))

    def _file_row(self, parent, title, path_var, picker_type):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=4)
        tk.Label(row, text=title, width=18, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        tk.Button(row, text="Browse", command=lambda: self.select_file(path_var, picker_type)).pack(side=tk.LEFT)

    def _label_row(self, parent, title, var):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text=title, width=18, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _pick_existing_initialdir(self, preferred_dir):
        if preferred_dir and os.path.isdir(preferred_dir):
            return preferred_dir
        return self.base_dir

    def select_file(self, var, picker_type):
        if picker_type == "master":
            initialdir = self._pick_existing_initialdir(self.master_masks_dir)
        else:
            initialdir = self._pick_existing_initialdir(self.masks_dir)

        path = filedialog.askopenfilename(
            title="Select image",
            initialdir=initialdir,
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            var.set(path)

    def select_output_path(self):
        initialdir = self._pick_existing_initialdir(self.papers_dir)
        path = filedialog.asksaveasfilename(
            title="Save output PNG",
            defaultextension=".png",
            initialfile=self.default_output_name,
            initialdir=initialdir,
            filetypes=[("PNG", "*.png")],
        )
        if path:
            self.output_path_var.set(path)

    @staticmethod
    def _safe_open_image(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return Image.open(path).convert("L")

    @staticmethod
    def _resize_to_height(image, target_height):
        if target_height <= 0:
            raise ValueError("Target height must be > 0")
        w, h = image.size
        if h == target_height:
            return image
        new_w = max(1, int(round(w * target_height / h)))
        return image.resize((new_w, target_height), Image.Resampling.NEAREST)

    @staticmethod
    def _load_font(size):
        preferred = ["arial.ttf", "segoeui.ttf", "times.ttf"]
        for name in preferred:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _draw_centered_text(self, draw, text, cx, y, font, fill=0):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw // 2, y), text, fill=fill, font=font)
        return th

    def generate_figure(self):
        try:
            frame_paths = [
                self.frame1_path.get().strip(),
                self.frame2_path.get().strip(),
                self.frame3_path.get().strip(),
            ]
            master_path = self.master_path.get().strip()

            if not all(frame_paths) or not master_path:
                messagebox.showerror("Missing input", "Please select exactly 3 frame images and 1 master mask image.")
                return

            target_h = int(self.target_height.get())
            spacing = int(self.spacing.get())
            symbol_scale = max(0.5, float(self.symbol_font_scale.get()))
            label_scale = max(0.5, float(self.label_font_scale.get()))
            if spacing < 0:
                raise ValueError("Spacing must be >= 0")

            labels = [
                self.label1.get().strip(),
                self.label2.get().strip(),
                self.label3.get().strip(),
                self.label_master.get().strip(),
            ]

            img_frame1 = self._resize_to_height(self._safe_open_image(frame_paths[0]), target_h)
            img_frame2 = self._resize_to_height(self._safe_open_image(frame_paths[1]), target_h)
            img_frame3 = self._resize_to_height(self._safe_open_image(frame_paths[2]), target_h)
            img_master = self._resize_to_height(self._safe_open_image(master_path), target_h)

            images = [img_frame1, img_frame2, img_frame3, img_master]
            image_widths = [img.size[0] for img in images]

            symbol_texts = ["+", "+", "...", "+", "="]

            symbol_font_size = max(70, int(target_h * 0.15 * symbol_scale))
            label_font_size = max(36, int(target_h * 0.07 * label_scale))
            symbol_font = self._load_font(symbol_font_size)
            label_font = self._load_font(label_font_size)

            temp = Image.new("L", (10, 10), 255)
            temp_draw = ImageDraw.Draw(temp)
            symbol_boxes = [temp_draw.textbbox((0, 0), s, font=symbol_font) for s in symbol_texts]
            symbol_widths = [b[2] - b[0] for b in symbol_boxes]

            max_label_h = 0
            label_widths = []
            for label in labels:
                if label:
                    b = temp_draw.textbbox((0, 0), label, font=label_font)
                    label_widths.append(b[2] - b[0])
                    max_label_h = max(max_label_h, b[3] - b[1])
                else:
                    label_widths.append(0)

            top_margin = int(target_h * 0.09)
            bottom_margin = int(target_h * 0.08)
            text_gap = int(target_h * 0.045) if max_label_h > 0 else 0

            content_width = (
                image_widths[0] + image_widths[1] + image_widths[2] + image_widths[3]
                + sum(symbol_widths)
                + spacing * (len(symbol_widths) + 3)
            )

            canvas_h = top_margin + target_h + text_gap + max_label_h + bottom_margin
            canvas_w = content_width

            canvas = Image.new("L", (canvas_w, canvas_h), 255)
            draw = ImageDraw.Draw(canvas)

            y_img = top_margin
            y_symbol = y_img + target_h // 2
            y_label = y_img + target_h + text_gap

            x = 0

            def paste_image_with_label(image, label_text):
                nonlocal x
                w = image.size[0]
                canvas.paste(image, (x, y_img))
                center_x = x + w // 2
                if label_text:
                    self._draw_centered_text(draw, label_text, center_x, y_label, label_font, fill=0)
                x += w

            def draw_symbol(sym_text, sym_width):
                nonlocal x
                x += spacing
                sym_box = draw.textbbox((0, 0), sym_text, font=symbol_font)
                sym_h = sym_box[3] - sym_box[1]
                self._draw_centered_text(draw, sym_text, x + sym_width // 2, y_symbol - sym_h // 2, symbol_font, fill=0)
                x += sym_width
                x += spacing

            paste_image_with_label(img_frame1, labels[0])
            draw_symbol("+", symbol_widths[0])

            paste_image_with_label(img_frame2, labels[1])
            draw_symbol("+", symbol_widths[1])

            draw_symbol("...", symbol_widths[2])

            draw_symbol("+", symbol_widths[3])

            paste_image_with_label(img_frame3, labels[2])
            draw_symbol("=", symbol_widths[4])

            paste_image_with_label(img_master, labels[3])

            out_path = self.output_path_var.get().strip()
            if not out_path:
                out_path = os.path.join(self.base_dir, self.default_output_name)
            if not out_path.lower().endswith(".png"):
                out_path += ".png"

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            canvas.save(out_path, format="PNG", dpi=(300, 300), optimize=True)

            messagebox.showinfo("Success", f"Figure saved successfully:\n{out_path}")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))


def main():
    root = tk.Tk()
    app = MasterMaskEquationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
