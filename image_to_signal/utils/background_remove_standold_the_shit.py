import os
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

def process_image(tool_img, bg_img, thresh_value):
    diff = cv2.absdiff(tool_img, bg_img)
    _, mask = cv2.threshold(diff, thresh_value, 255, cv2.THRESH_BINARY)
    
    # Clean up with morphology
    kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Keep largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        final_mask = mask
    return final_mask

class PreviewApp:
    def __init__(self, root, folder_path, bg_img, initial_thresh=33):
        self.root = root
        self.folder_path = folder_path
        self.root.title("Controls - " + Path(folder_path).name)
        
        self.bg_img = bg_img
        self.result_thresh = None
        self.window_name = "Preview Window (Mask)"
        
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))])
        self.current_idx = 0
        
        self.tool_img = cv2.imread(os.path.join(self.folder_path, self.image_files[self.current_idx]), cv2.IMREAD_GRAYSCALE)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 800)
        
        # Navigation Frame
        nav_frame = tk.Frame(root)
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)
        
        tk.Button(nav_frame, text="<< -10", command=lambda: self.change_frame(-10)).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="< Prev", command=lambda: self.change_frame(-1)).pack(side=tk.LEFT, padx=2)
        
        self.frame_label_var = tk.StringVar()
        self.update_frame_label()
        tk.Label(nav_frame, textvariable=self.frame_label_var, font=("Arial", 10, "bold"), width=15).pack(side=tk.LEFT, padx=10)
        
        tk.Button(nav_frame, text="Next >", command=lambda: self.change_frame(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="+10 >>", command=lambda: self.change_frame(10)).pack(side=tk.LEFT, padx=2)
        
        # Controls Frame
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        tk.Label(control_frame, text="Intensity:", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        
        self.thresh_var = tk.IntVar(value=initial_thresh)
        self.thresh_var.trace_add("write", self.update_preview)
        
        self.scale = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.thresh_var, length=250)
        self.scale.pack(side=tk.LEFT, padx=10)
        
        self.entry = tk.Entry(control_frame, textvariable=self.thresh_var, width=5, font=("Arial", 12))
        self.entry.pack(side=tk.LEFT, padx=10)
        
        # Action Frame
        action_frame = tk.Frame(root)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=15)
        
        tk.Button(action_frame, text="Apply to Folder (Enter)", command=self.apply, bg="green", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=20)
        tk.Button(action_frame, text="Skip Folder (Esc)", command=self.skip, bg="red", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        
        self.root.bind("<Return>", lambda e: self.apply())
        self.root.bind("<Escape>", lambda e: self.skip())
        self.root.bind("<Left>", lambda e: self.change_frame(-1))
        self.root.bind("<Right>", lambda e: self.change_frame(1))
        
        self.after_id = None
        self.render()
        
        self.root.after(50, self.check_cv_events)
        
    def change_frame(self, delta):
        new_idx = self.current_idx + delta
        new_idx = max(0, min(new_idx, len(self.image_files) - 1))
        
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self.tool_img = cv2.imread(os.path.join(self.folder_path, self.image_files[self.current_idx]), cv2.IMREAD_GRAYSCALE)
            self.update_frame_label()
            self.render()

    def update_frame_label(self):
        self.frame_label_var.set(f"Frame {self.current_idx + 1} / {len(self.image_files)}")
        
    def check_cv_events(self):
        cv2.waitKey(10)
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.skip()
                return
        except cv2.error:
            return
        self.root.after(50, self.check_cv_events)
        
    def update_preview(self, *args):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(50, self.render)
        
    def render(self):
        try:
            val = self.thresh_var.get()
        except tk.TclError:
            return
            
        mask = process_image(self.tool_img, self.bg_img, val)
        try:
            cv2.imshow(self.window_name, mask)
        except cv2.error:
            pass

    def apply(self):
        try:
            self.result_thresh = self.thresh_var.get()
        except tk.TclError:
            pass
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        self.root.destroy()

    def skip(self):
        self.result_thresh = None
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        self.root.destroy()

def preview_and_get_threshold(folder_path, bg_img, initial_thresh=33):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
    if not image_files:
        return initial_thresh, False
        
    preview_win = tk.Toplevel()
    app = PreviewApp(preview_win, folder_path, bg_img, initial_thresh)
    
    preview_win.grab_set()
    preview_win.wait_window(preview_win)
    
    if app.result_thresh is not None:
        return app.result_thresh, True
    return initial_thresh, False

def main():
    root = tk.Tk()
    root.withdraw()
    
    messagebox.showinfo("Select Folders", "Select the tool folders you want to process. Cancel when you are done selecting.")
    tool_folders = []
    while True:
        folder = filedialog.askdirectory(title="Select Tool Folder (Cancel when done)")
        if not folder:
            break
        tool_folders.append(folder)
        
    if not tool_folders:
        print("No folders selected.")
        return
        
    bg_path = filedialog.askopenfilename(title="Select Background Image", filetypes=[("TIFF", "*.tiff *.tif")])
    if not bg_path:
        print("No background selected.")
        return
        
    bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    
    mode = messagebox.askquestion("Processing Mode", "Do you want to review and adjust the intensity for EACH folder one-by-one?\n\n(Select 'Yes' to see a visual preview for each folder, or 'No' to process them all blindly)")
    
    for folder in tool_folders:
        folder_path = Path(folder)
        masks_dir = folder_path.parent / f"{folder_path.name}_masks"
        masks_dir.mkdir(exist_ok=True)
        
        thresh_value = 33
        if mode == 'yes':
            thresh_value, accepted = preview_and_get_threshold(folder, bg_img, 33)
            if not accepted:
                print(f"Skipped {folder_path.name}")
                continue
        else:
            if folder == tool_folders[0]:
                thresh_val_str = simpledialog.askinteger("Input", "Enter difference threshold (e.g., 33):", initialvalue=33)
                if thresh_val_str is None:
                    return
                thresh_value = thresh_val_str
        
        print(f"Processing {folder_path.name} with threshold {thresh_value}...")
        
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.tif', '.tiff')):
                img_path = os.path.join(folder, filename)
                out_path = os.path.join(masks_dir, os.path.splitext(filename)[0] + '.png')
                
                tool_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if tool_img is not None:
                    mask = process_image(tool_img, bg_img, thresh_value)
                    cv2.imwrite(out_path, mask)
                    
        print(f"Finished {folder_path.name}.")
        
    messagebox.showinfo("Done", "All selected folders have been processed.")

if __name__ == "__main__":
    main()
