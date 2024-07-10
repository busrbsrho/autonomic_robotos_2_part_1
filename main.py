import cv2 as cv
import numpy as np
import csv
from detection import detect_markers

size = 8  # Size of the marker in centimeters

def process_video(video_source, output_filename, csv_filename):
    # Open the video source (0 for webcam, or file path for video file)
    cap = cv.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Unable to open video source")
        return

    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file

    # Define the output video writer
    out = cv.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Define the camera matrix and distortion coefficients
    camera_matrix = np.array([
        [921.170702, 0.000000, 459.904354],
        [0.000000, 919.018377, 351.238301],
        [0.000000, 0.000000, 1.000000]
    ])
    distortion = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

    MARKER_SIZE = size  # Size of the marker in centimeters

    # Get the predefined dictionary of 4x4 markers with 100 unique markers
    marker_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)

    # Initialize the detector parameters using default values
    param_markers = cv.aruco.DetectorParameters()

    # Open CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame ID", "QR id", "QR 2D", "Dist", "Yaw", "Pitch", "Roll"])

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Call detect_markers with the frame size
            frame, csv_data, movement = detect_markers(frame, camera_matrix, distortion, marker_dict, MARKER_SIZE, param_markers,
                                             frame_id, width, height)

            # Display the movement direction on the frame
            cv.putText(frame, movement, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            out.write(frame)
            cv.imshow("frame", frame)
            key = cv.waitKey(1)
            if key == ord("q"):
                break

            # Write data to CSV
            for data in csv_data:
                writer.writerow(data)

            frame_id += 1

    cap.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    video_source = "challengeTest.mp4"  # Change to 0 for webcam
    #video_source = 0
    output_filename = "output.mp4"  # Output video filename
    csv_filename = "output.csv"  # Output CSV filename
    process_video(video_source, output_filename, csv_filename)
