import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = r"c:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA"
TOOL_ID = "tool012"

MASTER_MASK_CANDIDATES = [
    os.path.join(
        BASE_DIR,
        "DATA",
        "threshold_analysis",
        "master_mask_perspective",
        "master_masks",
        f"{TOOL_ID}_MASTER_MASK.tiff",
    ),
    os.path.join(
        BASE_DIR,
        "DATA",
        "threshold_analysis",
        "master_mask_perspective",
        "master_masks",
        f"{TOOL_ID}_MASTER_MASK.tif",
    ),
]

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Tool_Condition_Monitoring",
    "two_edge_case",
    "paper_figures",
    "master_mask_geometry",
)

FIG1_PATH = os.path.join(OUTPUT_DIR, "angle_calculation.png")
FIG2_PATH = os.path.join(OUTPUT_DIR, "centerline_determination.png")

FIG_DPI = 500
WIDTH_PERCENTILE_FOR_FIT = 50.0
MIN_ROWS_FOR_FIT = 20
TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 14
ANGLE_TEXT_FONT_SIZE = 16


def find_existing_master_mask():
    for path in MASTER_MASK_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Master mask file not found. Expected one of:\n"
        + "\n".join(MASTER_MASK_CANDIDATES)
    )


def to_binary(mask_gray):
    _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def get_boundaries(binary_mask):
    h, _ = binary_mask.shape
    ys = []
    left_x = []
    right_x = []

    for y in range(h):
        white = np.where(binary_mask[y, :] == 255)[0]
        if white.size > 0:
            ys.append(y)
            left_x.append(white[0])
            right_x.append(white[-1])

    if len(ys) < 2:
        raise ValueError("Not enough boundary points for line fitting.")

    ys = np.array(ys, dtype=np.float64)
    left_x = np.array(left_x, dtype=np.float64)
    right_x = np.array(right_x, dtype=np.float64)
    return ys, left_x, right_x


def select_widest_rows_for_fit(ys, left_x, right_x, width_percentile=WIDTH_PERCENTILE_FOR_FIT):
    widths = right_x - left_x
    width_threshold = np.percentile(widths, width_percentile)
    keep = widths >= width_threshold

    if np.sum(keep) < MIN_ROWS_FOR_FIT:
        # Fallback: keep the top-N widest rows if percentile selection is too small
        top_idx = np.argsort(widths)[-MIN_ROWS_FOR_FIT:]
        keep = np.zeros_like(widths, dtype=bool)
        keep[top_idx] = True

    return ys[keep], left_x[keep], right_x[keep], widths[keep], width_threshold


def fit_lines(ys, left_x, right_x):
    m_left, b_left = np.polyfit(ys, left_x, 1)
    m_right, b_right = np.polyfit(ys, right_x, 1)

    m_center = (m_left + m_right) / 2.0
    b_center = (b_left + b_right) / 2.0

    return (m_left, b_left), (m_right, b_right), (m_center, b_center)


def compute_tilt_deg(m_center):
    # Angle between center line and vertical axis.
    # x = m*y + b => angle from vertical = arctan(m)
    return float(np.degrees(np.arctan(m_center)))


def render_figure_1(binary_mask, ys, line_left, line_right, line_center, tilt_deg, out_path):
    h, w = binary_mask.shape
    y_plot = np.array([ys.min(), ys.max()])

    m_left, b_left = line_left
    m_right, b_right = line_right
    m_center, b_center = line_center

    x_left = m_left * y_plot + b_left
    x_right = m_right * y_plot + b_right
    x_center = m_center * y_plot + b_center

    vertical_x = w / 2.0

    fig, ax = plt.subplots(figsize=(8, 10), dpi=FIG_DPI)
    ax.imshow(binary_mask, cmap="gray", origin="upper")

    ax.plot(x_left, y_plot, color="red", linewidth=2.5, linestyle="-", label="Outer Boundaries (Left/Right)")
    ax.plot(x_right, y_plot, color="red", linewidth=2.5, linestyle="-")
    ax.plot(x_center, y_plot, color="green", linewidth=2.8, linestyle="-", label="Center Bisector")
    ax.plot([vertical_x, vertical_x], [y_plot.min(), y_plot.max()], color="blue", linewidth=2.8, linestyle="--", label="True Vertical Reference")

    angle_text = f"Tilt angle: {tilt_deg:.3f}°"
    ax.text(
        0.03,
        0.97,
        angle_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=ANGLE_TEXT_FONT_SIZE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
    )

    ax.legend(loc="lower right", frameon=True, fontsize=LEGEND_FONT_SIZE)
    ax.set_title("Tilt Angle Calculation from Master Mask", fontsize=TITLE_FONT_SIZE)
    ax.set_axis_off()

    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def rotate_mask_inverse(binary_mask, tilt_deg):
    h, w = binary_mask.shape
    center = (w / 2.0, h / 2.0)

    # Inverse rotation: counteract measured tilt
    rot_mat = cv2.getRotationMatrix2D(center, -tilt_deg, 1.0)

    rotated = cv2.warpAffine(
        binary_mask,
        rot_mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated


def render_figure_2(rotated_mask, ys, line_left, line_right, out_path):
    h, w = rotated_mask.shape
    y_plot = np.array([ys.min(), ys.max()])

    m_left, b_left = line_left
    m_right, b_right = line_right

    x_left = m_left * y_plot + b_left
    x_right = m_right * y_plot + b_right

    # Exact geometric centerline (midpoint between the two boundary lines)
    center_x_line = (x_left + x_right) / 2.0

    fig, ax = plt.subplots(figsize=(8, 10), dpi=FIG_DPI)
    ax.imshow(rotated_mask, cmap="gray", origin="upper")

    ax.plot(x_left, y_plot, color="red", linewidth=2.5, linestyle="-", label="Outer Boundaries")
    ax.plot(x_right, y_plot, color="red", linewidth=2.5, linestyle="-")
    ax.plot(center_x_line, y_plot, color="magenta", linewidth=3.2, linestyle="--", label="Geometric Centerline")

    ax.legend(loc="lower right", frameon=True, fontsize=LEGEND_FONT_SIZE)
    ax.set_title("Centerline Determination on Straightened Master Mask", fontsize=TITLE_FONT_SIZE)
    ax.set_axis_off()

    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mask_path = find_existing_master_mask()
    print(f"Using master mask: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask image: {mask_path}")

    binary = to_binary(mask)

    ys, left_x, right_x = get_boundaries(binary)
    ys_fit, left_fit, right_fit, widths_fit, width_thr = select_widest_rows_for_fit(ys, left_x, right_x)
    line_left, line_right, line_center = fit_lines(ys_fit, left_fit, right_fit)

    tilt_deg = compute_tilt_deg(line_center[0])
    print(f"Calculated tilt angle: {tilt_deg:.6f} deg")
    print(
        f"Figure 1 fitting rows: {len(ys_fit)}/{len(ys)} "
        f"(width percentile >= {WIDTH_PERCENTILE_FOR_FIT:.1f}, threshold={width_thr:.2f} px)"
    )

    render_figure_1(binary, ys, line_left, line_right, line_center, tilt_deg, FIG1_PATH)
    print(f"Saved Figure 1: {FIG1_PATH}")

    rotated = rotate_mask_inverse(binary, tilt_deg)
    ys2, left_x2, right_x2 = get_boundaries(rotated)
    ys2_fit, left2_fit, right2_fit, widths2_fit, width_thr2 = select_widest_rows_for_fit(ys2, left_x2, right_x2)
    line_left2, line_right2, _ = fit_lines(ys2_fit, left2_fit, right2_fit)
    print(
        f"Figure 2 fitting rows: {len(ys2_fit)}/{len(ys2)} "
        f"(width percentile >= {WIDTH_PERCENTILE_FOR_FIT:.1f}, threshold={width_thr2:.2f} px)"
    )

    render_figure_2(rotated, ys2, line_left2, line_right2, FIG2_PATH)
    print(f"Saved Figure 2: {FIG2_PATH}")


if __name__ == "__main__":
    main()
