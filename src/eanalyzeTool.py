import os

import cv2
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt

def biggest_interval(sorted_array):
    max_interval = 0
    for i in range(len(sorted_array) - 1):
        interval = sorted_array[i+1] - sorted_array[i]
        if interval > max_interval:
            max_interval = interval
    return max_interval

def estimate_camera_movement(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Feature detection and matching (using SIFT)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match keypoints between the frames
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Estimate transformation (homography)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Decompose transformation
    dx = H[0, 2]  # Translation in x direction
    dy = H[1, 2]  # Translation in y direction
    theta = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi  # Rotation angle (in degrees)

    # Calculate camera movement
    translation_distance = np.sqrt(dx ** 2 + dy ** 2)

    # Determine direction of vertical movement
    vertical_direction = None
    if dy > 0:
        vertical_direction = "down"
    elif dy < 0:
        vertical_direction = "up"

    return translation_distance, theta, vertical_direction


def estimate_camera_movement_concurrently(frames):
    estimator = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            future = executor.submit(estimate_camera_movement, frame1, frame2)
            futures.append((i, i + 1, future))  # Store the frame indexes along with the future object

        # Retrieve results
        for start_index, end_index, future in futures:
            translation_distance, theta, vertical_direction = future.result()
            estimator.append((start_index, end_index, translation_distance, theta, vertical_direction))  # Include the frame indexes in the results
    return estimator

import cv2

import cv2

def detect_drone_movement(frames):
    frames_with_movement = []

    # Variables to track the drone's movement
    ascending = False
    turning_down = False

    for frame in frames:
        # Convert frame to grayscale for easier processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect edges in the frame
        edges = cv2.Canny(gray_frame, 50, 150)

        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, cv2.cv2.CV_PI / 180, threshold=50, minLineLength=50, maxLineGap=30)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Check if drone is ascending (going up)
                if y2 < y1:
                    ascending = True
                else:
                    ascending = False

                # Check if drone is turning its camera down
                if x1 != x2:
                    turning_down = True

        # If the drone has finished ascending and turned its camera down
        if ascending and turning_down:
            frames_with_movement.append(frame.copy())

    return frames_with_movement


def calculate_expected_values(estimated_movements, selected_frames, selected_frame_indexes):
    total_translation_distance = 0
    total_theta = 0

    # Calculate total translation_distance and theta
    for i, frame_index in enumerate(selected_frame_indexes):
        frame = selected_frames[i]
        next_frame_index = selected_frame_indexes[i + 1] if i + 1 < len(selected_frame_indexes) else None

        # If it's not the last frame, calculate the translation_distance and theta
        if next_frame_index is not None:
            # Retrieve the corresponding estimated movement
            movement = estimated_movements[i]

            # Extract translation_distance and theta from the estimated movement
            translation_distance, theta = movement[2], movement[3]

            total_translation_distance += translation_distance
            total_theta += theta

    # Calculate the expected value
    num_frames = len(selected_frame_indexes) - 1  # Subtract 1 because we need pairs of frames for estimation
    expected_translation_distance = total_translation_distance / num_frames
    expected_theta = total_theta / num_frames

    return expected_translation_distance, expected_theta


def plot_translation_distance(expected_translation_distance, estimated_movements,
                              selected_frame_indexes, save_path=None):
    translation_distances = [movement[2] for movement in estimated_movements]  # Extracting translation distances
    frame_numbers = selected_frame_indexes[:-1]  # Last frame doesn't have a corresponding next frame

    max_idx = np.argmax(translation_distances)
    min_idx = np.argmin(translation_distances)

    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, translation_distances, label='Translation Distance', marker='o', linestyle='-')
    plt.axhline(y=expected_translation_distance, color='r', linestyle='--', label='Expected Translation Distance')

    # Annotate maximum and minimum points
    plt.annotate(f'Max ({frame_numbers[max_idx]}, {translation_distances[max_idx]})',
                 xy=(frame_numbers[max_idx], translation_distances[max_idx]),
                 xytext=(frame_numbers[max_idx] - 20, translation_distances[max_idx] + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.annotate(f'Min ({frame_numbers[min_idx]}, {translation_distances[min_idx]})',
                 xy=(frame_numbers[min_idx], translation_distances[min_idx]),
                 xytext=(frame_numbers[min_idx] + 10, translation_distances[min_idx] - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.xlabel('Frame Number')
    plt.ylabel('Translation Distance')
    plt.title('Translation Distance vs Frame Number')
    plt.legend()
    plt.grid(True)

    # Show or save the plot based on save_path
    if save_path:
        plt.savefig(os.path.join(save_path, 'translation_distance_plot.png'))
        plt.close()  # Close the plot after saving
    else:
        plt.show()


def plot_theta(expected_theta, estimated_movements,
               selected_frame_indexes, save_path=None):
    thetas = [movement[3] for movement in estimated_movements]  # Extracting thetas
    frame_numbers = selected_frame_indexes[:-1]  # Last frame doesn't have a corresponding next frame

    max_idx = np.argmax(thetas)
    min_idx = np.argmin(thetas)

    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, thetas, label='Theta', marker='o', linestyle='-')
    plt.axhline(y=expected_theta, color='r', linestyle='--', label='Expected Theta')

    # Annotate maximum and minimum points
    plt.annotate(f'Max ({frame_numbers[max_idx]}, {thetas[max_idx]})',
                 xy=(frame_numbers[max_idx], thetas[max_idx]),
                 xytext=(frame_numbers[max_idx] - 20, thetas[max_idx] + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.annotate(f'Min ({frame_numbers[min_idx]}, {thetas[min_idx]})',
                 xy=(frame_numbers[min_idx], thetas[min_idx]),
                 xytext=(frame_numbers[min_idx] + 10, thetas[min_idx] - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.xlabel('Frame Number')
    plt.ylabel('Theta')
    plt.title('Theta vs Frame Number')
    plt.legend()
    plt.grid(True)

    # Show or save the plot based on save_path
    if save_path:
        plt.savefig(os.path.join(save_path, 'theta_plot.png'))
        plt.close()  # Close the plot after saving
    else:
        plt.show()

def plot_translation_distance_zoom_out(expected_translation_distance, estimated_movements,
                              selected_frame_indexes, save_path=None):
    translation_distances = [movement[2] for movement in estimated_movements]  # Extracting translation distances
    frame_numbers = selected_frame_indexes[:-1]  # Last frame doesn't have a corresponding next frame

    max_idx = np.argmax(translation_distances)
    min_idx = np.argmin(translation_distances)

    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, translation_distances, label='Translation Distance', marker='o', linestyle='-')
    plt.axhline(y=expected_translation_distance, color='r', linestyle='--', label='Expected Translation Distance')

    plt.xlabel('Frame Number')
    plt.ylabel('Translation Distance')
    plt.title('Translation Distance vs Frame Number')
    plt.legend()
    plt.grid(True)

    # Get initial axis limits
    init_xlim = plt.gca().get_xlim()
    init_ylim = plt.gca().get_ylim()

    # Show or save the plot based on save_path
    if save_path:
        plt.savefig(os.path.join(save_path, 'translation_distance_plot.png'))
        plt.close()  # Close the plot after saving
    else:
        plt.show()

    # Check for zoom out after the plot is displayed
    final_xlim = plt.gca().get_xlim()
    final_ylim = plt.gca().get_ylim()

    zoom_out_x = final_xlim[1] < init_xlim[1] and final_xlim[0] > init_xlim[0]
    zoom_out_y = final_ylim[1] < init_ylim[1] and final_ylim[0] > init_ylim[0]

    if zoom_out_x or zoom_out_y:
        print("Zoom out detected!")

def plot_theta_zoom_out(expected_theta, estimated_movements, selected_frame_indexes, save_path=None):
    thetas = [movement[3] for movement in estimated_movements]  # Extracting thetas
    frame_numbers = selected_frame_indexes[:-1]  # Last frame doesn't have a corresponding next frame

    max_idx = np.argmax(thetas)
    min_idx = np.argmin(thetas)

    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, thetas, label='Theta', marker='o', linestyle='-')
    plt.axhline(y=expected_theta, color='r', linestyle='--', label='Expected Theta')

    # Annotate max and min points with indexes
    plt.annotate(f'Max ({frame_numbers[max_idx]}, {thetas[max_idx]})',
                 xy=(frame_numbers[max_idx], thetas[max_idx]),
                 xytext=(frame_numbers[max_idx] + 10, thetas[max_idx] - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Min ({frame_numbers[min_idx]}, {thetas[min_idx]})',
                 xy=(frame_numbers[min_idx], thetas[min_idx]),
                 xytext=(frame_numbers[min_idx] + 10, thetas[min_idx] + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel('Frame Number')
    plt.ylabel('Theta')
    plt.title('Tplot_theta_zoom_out')
    plt.legend()
    plt.grid(True)

    # Get initial axis limits
    init_xlim = plt.gca().get_xlim()
    init_ylim = plt.gca().get_ylim()

    # Show or save the plot based on save_path
    if save_path:
        plt.savefig(os.path.join(save_path, 'plot_theta_zoom_out.png'))
        plt.close()  # Close the plot after saving
    else:
        plt.show()

    # Check for zoom out after the plot is displayed
    final_xlim = plt.gca().get_xlim()
    final_ylim = plt.gca().get_ylim()

    zoom_out_x = final_xlim[1] < init_xlim[1] and final_xlim[0] > init_xlim[0]
    zoom_out_y = final_ylim[1] < init_ylim[1] and final_ylim[0] > init_ylim[0]

    if zoom_out_x or zoom_out_y:
        print("Zoom out detected!")

# Example usage:
# Assuming you have a list of frames named 'frames'



# Example usage:
# if __name__ == '__main__':
    # video_path = "/Users/noyagendelman/Desktop/choosingFrames/v1.mp4"
    # frames = extract_frames(video_path)
    # selected_f = [360, 400, 500, 520, 540, 560, 580, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2100, 2120, 2140, 2180, 2200, 2220, 2240, 2260, 2280, 2300, 2320, 2340, 2360, 2380, 2400, 2420, 2440, 2460, 2480, 2500, 2520, 2540, 2560, 2580, 2600, 2620, 2640, 2660, 2680, 2700, 2720, 2740, 2760, 2780, 2800, 2820, 2840, 2860, 2880, 2900, 2920, 2940, 2960, 2980, 3000, 3020, 3040, 3060, 3080, 3100, 3120, 3140, 3160, 3180, 3200, 3220, 3240, 3260, 3280, 3300, 3320, 3340, 3360, 3380, 3400, 3420, 3440, 3460, 3480, 3500, 3520, 3540, 3560, 3580, 3600, 3620, 3640, 3660, 3680, 3700, 3720, 3740, 3760, 3780, 3800, 3820, 3840, 3860, 3880, 3900, 3920, 3940, 3960, 3980, 4000, 4020, 4040, 4060, 4080, 4100, 5000, 5020, 5040, 5060, 5080, 5100, 5120, 5140, 5160, 5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320, 5340, 5360, 5380, 5400, 5420, 5440, 5460, 5480, 5500, 5520, 5540, 5560, 5580, 5600, 5620, 5640, 5660, 5680, 5700, 5720, 5740, 5760, 5780, 5800, 5820, 5840, 5860, 5880, 5900, 5920, 5940, 5960, 5980, 6000, 6020, 6040, 6060, 6080, 6100, 6120, 6140, 6160, 6180, 6200, 6220, 6240, 6260, 6280, 6300, 6320, 6340, 6360, 6380, 6400, 6420, 6440, 6460, 6480, 6500, 6520, 6540, 6560, 6580, 6600, 6620, 6640, 6660, 6680, 6700, 6720, 6740, 6760, 6780, 6800, 6820, 6840, 6860, 6880, 6900, 6920, 6940, 6960, 6980, 7000, 7020, 7040, 7060, 7080, 7100, 7120, 7140, 7160, 7180, 7200, 7220, 7240, 7260, 7280, 7300, 7320, 7340, 7360, 7380, 7400, 7420, 7440, 7460, 7480, 7500, 7520, 7540, 7560, 7580, 7600, 7620, 7640, 7660, 7680, 7700, 7720, 7740, 7760, 7780, 7800, 7820, 7840, 7860, 7880, 7900, 7920, 7940, 7960, 7980, 8000, 8020, 8040, 8060, 8080, 8100, 8120, 8140, 8160, 8180, 8200, 8220, 8240, 8260, 8280, 8300, 8320, 8340, 8360, 8380, 8400, 8420, 8440, 8460, 8480, 8500, 8520, 8540, 8560, 8580, 8600, 8620, 8640, 8660, 8680]
    # frames_from_v = get_selected_frames(selected_f)
    # detect_drone_movement(detect_drone_movement(frames_with_movement))
    # logging.basicConfig(filename="/Users/noyagendelman/Desktop/choosingFrames/v1p", level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.info(f"detected frames: {len(frames_with_movement)}")
    # # b = biggest_interval( [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2100, 2120, 2140, 2160, 2180, 2200, 2220, 2240, 2260, 2280, 2300, 2320, 2340, 2360, 2380, 2400, 2420, 2440, 2460, 2480, 2500, 2520, 2540, 2560, 2580, 2600, 2620, 2640, 2660, 2680, 2700, 2720, 2740, 2760, 2780, 2800, 2820, 2840, 2860, 2880, 2900, 2920, 2940, 2960, 2980, 3000, 3020, 3040, 3060, 3080, 3100, 3120, 3140, 3160, 3180, 3200, 3220, 3240, 3260, 3280, 3300, 3320, 3340, 3360, 3380, 3400, 3420, 3440, 3460, 3480, 3500, 3520, 3540, 3560, 3580, 3600, 3620, 3640, 3660, 3680, 3700, 3720, 3740, 3760, 3780, 3800, 3820, 3840, 3860, 3880, 3900, 3920, 3940, 3960, 3980, 4000, 4020, 4040, 4060, 4080, 4100, 4120, 4140, 4160, 4180, 4200, 4220, 4240, 4260, 4280, 4300, 4320, 4340, 4360, 4380, 4400, 4420, 4440, 4460, 4480, 4500, 4520, 4540, 4560, 4580, 4600, 4620, 4640, 4660, 4680, 4700, 4720, 4740, 4760, 4780, 4800, 4820, 4840, 4860, 4880, 4900, 4920, 4940, 4960, 4980, 5000, 5020, 5040, 5060, 5080, 5100, 5120, 5140, 5160, 5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320, 5340, 5360, 5380, 5400, 5420, 5440, 5460, 5480, 5500, 5520, 5540, 5560, 5580, 5620, 5640, 5660, 5680, 5700, 5720, 5740, 5760, 5780, 5800, 5820, 5840, 5860, 5880, 5900, 5920, 5940, 5960, 5980, 6000, 6020, 6040, 6060, 6080, 6120, 6300, 6340, 6360, 6380, 6400, 6420, 6440, 6460, 6480, 6500, 6520, 6540, 6560, 6580, 6600, 6620, 6640, 6660, 6680, 6700, 6720, 6740, 6760, 6780, 6800, 6820, 6840, 6860, 6880, 6900, 6920, 6940, 6960, 6980, 7000, 7020, 7040, 7060, 7080, 7100, 7120, 7140, 7160, 7180, 7200, 7220, 7240, 7260, 7280, 7300, 7320, 7360, 7380, 7400, 7420, 7440, 7460, 7480, 7500, 7520, 7540, 7560, 7580, 7600, 7620, 7640, 7660, 7680, 7700, 7720, 7740, 7760, 7780, 7800, 7820, 7840, 7860, 7880, 7900, 7920, 7940, 7960, 7980, 8000, 8020, 8040, 8060, 8080, 8100, 8120, 8140, 8160, 8180, 8200, 8220, 8240, 8260, 8300, 8420, 8440, 8460, 8480, 8500, 8580])
    # # print(b)
    # # print(180/20)
