import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# TrueType fonts for native Inkscape editing
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

out_dir = '/home/alifalah/Projects/DATA/temp_fractured_tool_analysis'
os.makedirs(out_dir, exist_ok=True)

f_path = '/home/alifalah/Projects/DATA/masks_tilted/tool014/191.70_degrees.png'
img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)

roi_top, roi_bot = 1122, 1704

# 1. Proposed: 10 rows immediately above ROI [1112, 1122]
above_mids = []
for y in range(roi_top - 10, roi_top):
    w = np.where(img[y, :] == 255)[0]
    if len(w) > 1:
        above_mids.append((w[0] + w[-1]) / 2.0)
cx_above = float(np.mean(above_mids))

# 2. Inside ROI (midpoint of active cutting zone near fracture, y=1400 to 1600)
inside_mids = []
for y in range(1400, 1600):
    w = np.where(img[y, :] == 255)[0]
    if len(w) > 1:
        inside_mids.append((w[0] + w[-1]) / 2.0)
cx_inside = float(np.mean(inside_mids))
skew_val = cx_inside - cx_above

# 3. Row-by-row midpoints from y=1000 to y=1680
ys_all = np.arange(1000, 1680)
mids_all = []
lefts_all = []
rights_all = []
for y in ys_all:
    w = np.where(img[y, :] == 255)[0]
    if len(w) > 1:
        mids_all.append((w[0] + w[-1]) / 2.0)
        lefts_all.append(w[0])
        rights_all.append(w[-1])
    else:
        mids_all.append(np.nan)
        lefts_all.append(np.nan)
        rights_all.append(np.nan)

# Create 3-panel publication figure
fig = plt.figure(figsize=(19, 9), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.25], wspace=0.25)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

zoom_ymin = 1020
zoom_ymax = 1730
zoom_xmin = int(cx_above - 460)
zoom_xmax = int(cx_above + 460)

# PANEL 1: Inside ROI Fit (FLAWED)
ax1.imshow(img, cmap='gray', origin='upper')
ax1.axhline(roi_top, color='gold', linestyle='--', linewidth=2.0, label=f'ROI Top ($y={roi_top}$)')
ax1.axhline(roi_bot, color='gold', linestyle='-', linewidth=2.0, label=f'ROI Bottom ($y={roi_bot}$)')
ax1.axvline(cx_inside, color='red', linestyle='--', linewidth=2.5, label=f'Skewed Centerline ($x={cx_inside:.1f}$)')
ax1.axvline(cx_above, color='cyan', linestyle=':', linewidth=1.8, label=f'True Tool Axis ($x={cx_above:.1f}$)')
# Highlight right half from skewed centerline
rect_inside = Rectangle((cx_inside, roi_top), zoom_xmax - cx_inside, roi_bot - roi_top,
                        facecolor='red', alpha=0.22, edgecolor='red', linestyle='--', linewidth=1.5, label='Biased Right Window')
ax1.add_patch(rect_inside)
ax1.set_xlim(zoom_xmin, zoom_xmax)
ax1.set_ylim(zoom_ymax, zoom_ymin)
ax1.set_title('(a) Flawed: Fitted Inside Dynamic ROI\nCenterline Skewed by Fracture (+45.2 px)', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Pixel X-coordinate', fontsize=11)
ax1.set_ylabel('Pixel Y-coordinate (Axial)', fontsize=11)
ax1.legend(loc='lower left', fontsize=9, framealpha=0.92)

# PANEL 2: Above ROI Fit (PROPOSED)
ax2.imshow(img, cmap='gray', origin='upper')
ax2.axhline(roi_top, color='gold', linestyle='--', linewidth=2.0, label=f'ROI Top ($y={roi_top}$)')
ax2.axhline(roi_bot, color='gold', linestyle='-', linewidth=2.0, label=f'ROI Bottom ($y={roi_bot}$)')
# 10 rows reference band
rect_band = Rectangle((zoom_xmin, roi_top - 10), zoom_xmax - zoom_xmin, 10,
                      facecolor='lime', alpha=0.55, edgecolor='green', linewidth=1.5, label='10-Row Intact Calibration Band')
ax2.add_patch(rect_band)
ax2.axvline(cx_above, color='cyan', linestyle='-', linewidth=2.5, label=f'Robust Centerline ($x={cx_above:.1f}$)')
rect_robust = Rectangle((cx_above, roi_top), zoom_xmax - cx_above, roi_bot - roi_top,
                        facecolor='cyan', alpha=0.22, edgecolor='cyan', linestyle='-', linewidth=1.5, label='True Right Flute Window')
ax2.add_patch(rect_robust)
ax2.set_xlim(zoom_xmin, zoom_xmax)
ax2.set_ylim(zoom_ymax, zoom_ymin)
ax2.set_title('(b) Proposed: Fitted Above Dynamic ROI\nImmune to Damage, Accurately Isolates Defect', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Pixel X-coordinate', fontsize=11)
ax2.set_ylabel('Pixel Y-coordinate (Axial)', fontsize=11)
ax2.legend(loc='lower left', fontsize=9, framealpha=0.92)

# PANEL 3: Midpoint drift curve
ax3.plot(mids_all, ys_all, color='crimson', linewidth=2.5, label='Row Midpoint $x_{mid}(y)$')
ax3.axhline(roi_top, color='gold', linestyle='--', linewidth=2.0, label=f'ROI Top ($y={roi_top}$)')
ax3.axvline(cx_above, color='cyan', linestyle='-', linewidth=2.5, label=f'Above-ROI Baseline ($x={cx_above:.1f}$)')
ax3.axvspan(cx_above, cx_inside, color='red', alpha=0.18, label=f'Fracture-Induced Skew (+{skew_val:.1f} px)')
ax3.fill_betweenx([roi_top - 10, roi_top], cx_above - 100, cx_above + 100, color='lime', alpha=0.35, label='10-Row Intact Zone')
ax3.invert_yaxis()
ax3.set_ylim(zoom_ymax, zoom_ymin)
ax3.set_xlim(cx_above - 50, cx_above + 75)
ax3.set_title('(c) Axial Drift of Row Midpoint $x_{mid}(y)$', fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('Calculated Midpoint X (pixels)', fontsize=11)
ax3.set_ylabel('Pixel Y-coordinate (Axial)', fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='lower right', fontsize=9, framealpha=0.92)

plt.suptitle('Why Reference Calibration Must Occur Above the Dynamic ROI (tool014: Fractured 4-Flute Endmill at 191.7°)', fontsize=14, fontweight='bold', y=0.98)

png_path = os.path.join(out_dir, 'fractured_tool_midpoint_skew_demonstration.png')
pdf_path = os.path.join(out_dir, 'fractured_tool_midpoint_skew_demonstration.pdf')
plt.savefig(png_path, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close(fig)
print(f'Successfully generated:\n  {png_path}\n  {pdf_path}')
