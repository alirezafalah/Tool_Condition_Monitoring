import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def process_frame(frame):
    x, y, w, h = 810, 671, 80, 240
    roi = frame[y:y+h, x:x+w]
    mask = np.where((roi[:, :, 0] < 0.8 * roi[:, :, 1]) & (roi[:, :, 0] < 0.8 * roi[:, :, 2]), 0, 1)
    mask = refine_mask(mask)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = mask.astype(roi.dtype)
    result = roi * mask
    return result

import sys

def get_nearest_pixel(matrix):
    nearest_pixel = None
    min_distance = sys.maxsize
    center_y, center_x = matrix.shape[0] // 2, matrix.shape[1] // 2

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x] != 0:
                distance = ((center_y - y) ** 2 + (center_x - x) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_pixel = matrix[y, x]

    return nearest_pixel

def refine_mask(mask):
    refined_mask = np.copy(mask)
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0:
                min_y_up, max_y_down = max(0, y - 25), min(height, y + 26)
                min_x_left, max_x_right = max(0, x - 25), min(width, x + 26)

                up_slice = mask[min_y_up:y, x]
                down_slice = mask[y + 1:max_y_down, x]
                left_slice = mask[y, min_x_left:x]
                right_slice = mask[y, x + 1:max_x_right]

                if np.sum(up_slice != 0) > 0 and np.sum(down_slice != 0) > 0 and np.sum(left_slice != 0) > 0 and np.sum(right_slice != 0) > 0:
                    min_y, max_y = max(0, y - 25), min(height, y + 26)
                    min_x, max_x = max(0, x - 25), min(width, x + 26)
                    matrix = mask[min_y:max_y, min_x:max_x]

                    refined_mask[y, x] = get_nearest_pixel(matrix)

    return refined_mask

def display_roi(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        roi = process_frame(frame)

        if roi.size > 0:  # Check if the frame is not empty
            cv2_imshow(roi)
            cv2.waitKey(1)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Replace 'your_video.mp4' with the name of your video file
display_roi('/content/IMG_3619.MOV', num_frames=30)
