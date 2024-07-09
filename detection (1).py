import cv2 as cv
from cv2 import aruco
import numpy as np

wanted_dis=47
wanted_yaw=23
wanted_pitch=-7


def detect_markers(frame, camera_matrix, distortion, marker_dict, marker_size, param_markers, frame_id, frame_width,
                   frame_height):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, _ = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    csv_data = []
    closest_marker = None
    second_closest_marker = None
    min_distance = float('inf')
    second_min_distance = float('inf')

    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, marker_size, camera_matrix, distortion
        )
        for i in range(len(marker_IDs)):
            corners = marker_corners[i].reshape(4, 2)
            corners = corners.astype(int)

            # Calculate distance to the marker
            distance = np.linalg.norm(tVec[i][0])

            # Determine if this marker is closest or second closest
            if distance < min_distance:
                second_closest_marker = closest_marker  # Promote the previous closest to second closest
                second_min_distance = min_distance
                min_distance = distance
                closest_marker = (
                    i, marker_IDs[i], tVec[i], rVec[i], corners
                )
            elif distance < second_min_distance:
                second_min_distance = distance
                second_closest_marker = (
                    i, marker_IDs[i], tVec[i], rVec[i], corners
                )

            # Collect data for CSV
            rmat, _ = cv.Rodrigues(rVec[i])
            _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(np.hstack((rmat, np.zeros((3, 1)))))
            yaw, pitch, roll = euler_angles.flatten()
            csv_data.append([
                frame_id,
                marker_IDs[i][0],
                [
                    f" top_left: {corners[0].tolist()}, top_right: {corners[1].tolist()}, "
                    f"bottom_right: {corners[2].tolist()}, bottom_left: {corners[3].tolist()}"],
                round(distance, 2),
                round(float(yaw), 2), round(float(pitch), 2), round(float(roll), 2)
            ])

    if closest_marker is not None:
        idx, marker_id, tVec_closest, rVec_closest, corners_closest = closest_marker

        # Calculate yaw and pitch relative to frame center
        marker_center_x = np.mean(corners_closest[:, 0])
        frame_center_x = frame_width / 2
        deviation_x = marker_center_x - frame_center_x
        yaw_from_center = (deviation_x / frame_center_x) * 45

        marker_center_y = np.mean(corners_closest[:, 1])
        frame_center_y = frame_height / 2
        deviation_y = marker_center_y - frame_center_y
        pitch_from_center = -1* (deviation_y / frame_center_y) * 45

        # Calculate movement direction based on closest and second closest markers
        if second_closest_marker is not None:
            _, _, tVec_second_closest, _, _ = second_closest_marker
            distance_second_closest = np.linalg.norm(tVec_second_closest[0])

            movement = calculate_movement(distance, distance_second_closest, yaw_from_center, pitch_from_center)
        else:
            # If no second closest marker found, use default
            movement = calculate_movement(distance, None, yaw_from_center, pitch_from_center)

        return frame, csv_data, movement, distance, yaw_from_center, pitch_from_center
    else:
        return frame, csv_data, "No marker detected", None, None, None


def calculate_movement(closest_dis, second_closest_dis, yaw_from_center, pitch_from_center):
    if closest_dis == 0:
        return "QR not detected"

    # Ensure saved parameters are treated as floats
    saved_distance = wanted_dis
    saved_yaw = wanted_yaw
    saved_pitch = wanted_pitch

    print(f"Saved Distance: {saved_distance}")
    print(f"Closest Distance: {closest_dis}")
    print(yaw_from_center)
    print(pitch_from_center)

    # Define the range for movement decisions
    distance_threshold = 10  # Example threshold
    yaw_threshold = 10  # Example threshold
    pitch_threshold = 10  # Example threshold

    if saved_distance + distance_threshold < closest_dis:  # QR is far

        if (saved_yaw + yaw_threshold) > yaw_from_center > (saved_yaw - yaw_threshold) and (
                saved_pitch + 5) > pitch_from_center > (
                saved_pitch - pitch_threshold):
            return "Move forward"
        elif (saved_yaw + yaw_threshold) > yaw_from_center > (saved_yaw - yaw_threshold):
            if pitch_from_center > (saved_pitch + pitch_threshold):
                return "Move Up"
            elif pitch_from_center < (saved_pitch - pitch_threshold):
                return "Move Down"
        elif 5 > pitch_from_center > (saved_pitch - pitch_threshold):
            if yaw_from_center > (saved_yaw + yaw_threshold):
                return "Move right"
            elif yaw_from_center < (saved_yaw - yaw_threshold):
                return "Move left"
        else:
            if pitch_from_center > (saved_pitch + pitch_threshold):
                return "Move Up"
            elif pitch_from_center < (saved_pitch - pitch_threshold):
                return "Move Down"
            if yaw_from_center > (saved_yaw + yaw_threshold):
                return "Move right"
            elif yaw_from_center < (saved_yaw - yaw_threshold):
                return "Move left"
    elif(saved_yaw + yaw_threshold) > yaw_from_center > (saved_yaw - yaw_threshold) and (
                saved_pitch + 5) > pitch_from_center > (
                saved_pitch - pitch_threshold):
            return "great"
    else:
        return "Move back"
