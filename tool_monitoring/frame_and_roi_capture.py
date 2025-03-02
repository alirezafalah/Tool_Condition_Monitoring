import cv2
import os

# Set the video file path, start and end times, and frames per second:
# video_path = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\FLUTEMILL-E4-001\IMG_7713.MOV"
video_path = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\BackgroundData\1 - 15 and_33_and_41\IMG_7710.MOV" #background

#background 
start_time = 3
end_time = 9.01

# start_time = 9
# end_time = 15.01
fps = 59.94

skip_frames = round(fps / 60)
# skip_frames = 90

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Setting The Start Frame
start_frame = start_time * fps
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Create a directory to store the extracted frames
output_dir = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Thesis Data\3\FLUTEMILL-E4-001\BG"

# Define ROI: [x, y, width, height]
roi = [1400, 600, 450, 1200]  # Modify this to your desired region

frame_count = int(start_frame)
saved_frames = 0
target_frames = (end_time - start_time) * fps // skip_frames

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if we've reached the end of the video or saved enough frames
    if not ret or saved_frames >= target_frames:
        break

    # Crop frame to the region of interest
    cropped_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    # Save every 30th frame
    if frame_count % skip_frames == 0:
        output_file = os.path.join(output_dir, f"bg{saved_frames + 1:03}.jpg")
        cv2.imwrite(output_file, cropped_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_frames += 1

    frame_count += 1

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()

print(f"Saved {saved_frames} frames.")
