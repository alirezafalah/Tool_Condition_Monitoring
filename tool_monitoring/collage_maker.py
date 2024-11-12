def main():
    video_path = '/content/IMG_3619.MOV'
    output_dir = 'output/'
    step = 17
    num_frames_in_collage = 7

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for starting_frame in range(0, total_frames):
        frames = []
        i = starting_frame
        remaining_steps = step
        while len(frames) < num_frames_in_collage:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                # Loop back to the beginning of the video and continue with the remaining steps
                i = remaining_steps - (total_frames - i)
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

            frame = mask_function(frame)
            frames.append(frame)

            i += step
            remaining_steps = step

            if len(frames) == num_frames_in_collage:
                collage = create_collage(frames)
                output_path = f'{output_dir}collage_{starting_frame}.png'
                cv2.imwrite(output_path, collage)

    cap.release()

if __name__ == "__main__":
    main()
