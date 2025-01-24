import cv2
import numpy as np
from VFIMamba.inference_wrapper import vfi_infer
import argparse
import imageio.v3 as iio

# Default frames per second
FPS = 25

def load_video(file_path):
    """
    Load an MP4 video file into a NumPy array.

    Parameters:
        file_path (str): Path to the input video file.

    Returns:
        numpy.ndarray: A 4D NumPy array with shape (num_frames, height, width, channels).
    """
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def save_video_from_numpy(video_array, output_path, fps=25):
    """
    Save a video from a NumPy array to an MP4 file.

    Parameters:
        video_array (numpy.ndarray): A 4D NumPy array of shape (num_frames, height, width, channels).
                                     Channels should be 3 (RGB) or 1 (grayscale).
        output_path (str): Path to save the video file.
        fps (int): Frames per second for the output video.

    Raises:
        ValueError: If the input video_array is not valid.
    """
    if not isinstance(video_array, np.ndarray):
        raise ValueError("Input video_array must be a NumPy array.")

    if video_array.ndim != 4 or video_array.shape[-1] not in [1, 3]:
        raise ValueError("Input video_array must have shape (num_frames, height, width, channels). "
                         "Channels must be 1 (grayscale) or 3 (RGB).")

    iio.imwrite(output_path, video_array, fps=fps)
    print(f"Video successfully saved at {output_path}")

def parse_arguments():
    """
    Parse command-line arguments for video interpolation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Video interpolation using VFI.")
    parser.add_argument('--n', default=4, type=int,
                        help='Interpolation factor (e.g., 4 means 3 frames will be interpolated between input frames).')
    parser.add_argument('--input', default='./original.mp4', type=str,
                        help='Path to the input video file.')
    parser.add_argument('--output', default='./output.mp4', type=str,
                        help='Path to save the interpolated output video.')
    return parser.parse_args()

def main():
    """
    Main function to perform video interpolation.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Load input video
    print("Loading video...")
    video_array = load_video(args.input)

    # Define frames to extract segments from the video
    frame_1 = FPS * 16  # Time position for the first frame (16 seconds)
    frame_2 = FPS * 30  # Time position for the second frame (30 seconds)
    indent = 0  # Optional indent duration of the first segment after frame_1, more details in find_indent.ipynb

    # Extract video segments
    video_1 = video_array[frame_1 - FPS * 5:frame_1 + indent]  # 5 seconds before frame_1
    video_2 = video_array[frame_2 - FPS * 5 + indent:frame_2 + FPS * 5]  # 5 seconds after frame_2

    # Get boundary frames for interpolation
    left_frame = video_1[-1]  # Last frame of the first segment
    right_frame = video_2[args.n - 1]  # First meaningful frame of the second segment

    # Perform interpolation using VFI
    print("Performing video interpolation...")
    interpolated_frames = vfi_infer(left_frame, right_frame, n=args.n)

    # Remove redundant frames and combine segments
    video_interpolated = np.array(interpolated_frames[1:])  # Skip the first interpolated frame (same as left_frame)
    video_2 = video_2[args.n - 1:]  # Remove the first n-1 frames from the second segment

    # Concatenate all parts of the video
    video_output = np.concatenate((video_1, video_interpolated, video_2), axis=0)

    # Convert output video to uint8 and reverse color channels for saving (BGR to RGB)
    video_output = video_output.astype(np.uint8)
    video_output = video_output[..., ::-1]

    # Save the output video
    print("Saving interpolated video...")
    save_video_from_numpy(video_output, args.output, fps=FPS)

if __name__ == "__main__":
    main()
