import cv2 as cv
import numpy as np
import csv
from detection import detect_markers

size = 5  # Size of the marker in centimeters


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
            frame, csv_data, movement, distance, yaw, pitch = detect_markers(frame, camera_matrix, distortion,
                                                                             marker_dict, MARKER_SIZE, param_markers,
                                                                             frame_id, width, height)

            # Display the movement direction on the frame
            cv.putText(frame, movement, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            if distance is not None and yaw is not None and pitch is not None:
                # Display the distance, yaw, and pitch on the frame
                cv.putText(frame, f"Dist: {distance:.2f} cm", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                           cv.LINE_AA)
                cv.putText(frame, f"Yaw: {yaw:.2f} degrees", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                           cv.LINE_AA)
                cv.putText(frame, f"Pitch: {pitch:.2f} degrees", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                           2, cv.LINE_AA)

            # Display the frame
            cv.imshow("frame", frame)
            key = cv.waitKey(1)

            if key == ord("q"):
                break
            elif key == ord("c"):  # Press 'c' to capture a picture
                # Define the filename for the captured image and data
                image_filename = "capture_latest.png"
                text_filename = "capture_latest.txt"

                # Save the frame as an image
                cv.imwrite(image_filename, frame)

                # Save the distance, yaw, and pitch to a text file
                if distance is not None and yaw is not None and pitch is not None:
                    with open(text_filename, 'w') as f:
                        f.write(f"Distance: {distance:.2f} cm\n")
                        f.write(f"Yaw: {yaw:.2f} degrees\n")
                        f.write(f"Pitch: {pitch:.2f} degrees\n")

                print(f"Captured image and data saved as {image_filename} and {text_filename}")

            elif key == ord("v"):  # Press 'v' to capture a special image
                # Define the filename for the great position capture
                great_position_image_filename = "great_position.png"
                great_position_text_filename = "great_position.txt"

                # Save the frame as an image
                cv.imwrite(great_position_image_filename, frame)

                # Save the distance, yaw, and pitch to a text file
                if distance is not None and yaw is not None and pitch is not None:
                    with open(great_position_text_filename, 'w') as f:
                        f.write(f"Distance: {distance:.2f} cm\n")
                        f.write(f"Yaw: {yaw:.2f} degrees\n")
                        f.write(f"Pitch: {pitch:.2f} degrees\n")

                print(
                    f"Great position captured and saved as {great_position_image_filename} and {great_position_text_filename}")

            # Write data to CSV
            for data in csv_data:
                writer.writerow(data)

            out.write(frame)
            frame_id += 1

    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_source = 0  # Change to the path of a video file if needed
    output_filename = "output.mp4"  # Output video filename
    csv_filename = "output.csv"  # Output CSV filename
    process_video(video_source, output_filename, csv_filename)
