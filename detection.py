import cv2 as cv
from cv2 import aruco
import numpy as np


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
        pitch_from_center = (deviation_y / frame_center_y) * 45

        # Calculate movement direction based on closest and second closest markers
        if second_closest_marker is not None:
            _, _, tVec_second_closest, _, _ = second_closest_marker
            distance_second_closest = np.linalg.norm(tVec_second_closest[0])

            movement = calculate_movement(distance, distance_second_closest, yaw_from_center, pitch_from_center)
        else:
            # If no second closest marker found, use default
            movement = calculate_movement(distance, None, yaw_from_center, pitch_from_center)
    else:
        movement = "No marker detected"

    return frame, csv_data, movement


def calculate_movement(closest_dis, second_closest_dis, yaw, pitch):
    # Determine movement direction based on distances and orientation
    if closest_dis < 80:  # closest QR is too close
        if second_closest_dis is not None:
            # Perform checks based on the second closest QR code
            if second_closest_dis > 100 and 5 > yaw > -5 and 5 > pitch > -5:
                return "Move forward"
            elif 5 > yaw > -5:
                if pitch > 5:
                    return "Move down"
                elif pitch < -5:
                    return "Move up"
            elif 5 > pitch > -5:
                if yaw > 5:
                    return "Move right"
                elif yaw < -5:
                    return "Move left"
            elif not 5 > yaw > -5 and not 5 > pitch > -5:
                if pitch > 5:
                    return "Move down"
                elif pitch < -5:
                    return "Move up"
                if yaw > 5:
                    return "Move right"
                elif yaw < -5:
                    return "Move left"
        else:
            # No second closest QR detected, turn towards opposite direction of closest QR
            if yaw > 0:
                return "Turn left"
            elif yaw <= 0:
                return "Turn right"

    else:
        # Default behavior if closest QR is not too close
        if closest_dis > 100 and 5 > yaw > -5 and 5 > pitch > -5:
            return "Move forward"
        elif closest_dis < 80 and 5 > yaw > -5 and 5 > pitch > -5:
            return "Move backward"
        elif 5 > yaw > -5:
            if pitch > 5:
                return "Move down"
            elif pitch < -5:
                return "Move up"
        elif 5 > pitch > -5:
            if yaw > 5:
                return "Move right"
            elif yaw < -5:
                return "Move left"
        elif not 5 > yaw > -5 and not 5 > pitch > -5:
            if pitch > 5:
                return "Move down"
            elif pitch < -5:
                return "Move up"
            if yaw > 5:
                return "Move right"
            elif yaw < -5:
                return "Move left"

    return "Stay"
