
import numpy as np
import cv2
from shapely.geometry import Polygon
import cv2
from matplotlib import pyplot as plt
from shapely.validation import explain_validity
from shapely.geometry import Polygon
from shapely.validation import explain_validity


def keypoint_movement_towards_center(img1, img2):
    # Load images
    image1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)
    # image1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    # image2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate movement vectors
    movement_vectors = matched_keypoints2 - matched_keypoints1

    # Calculate movement directions
    movement_directions = np.arctan2(movement_vectors[:, :, 1], movement_vectors[:, :, 0]) * 180 / np.pi

    # Define regions of movement (down, up, left, right)
    down_movement = np.mean(movement_directions < -45)
    up_movement = np.mean(movement_directions > 45)
    left_movement = np.mean(np.logical_and(movement_directions > -135, movement_directions < -45))
    right_movement = np.mean(np.logical_and(movement_directions > 45, movement_directions < 135))

    # Determine dominant movement direction
    if down_movement > 0.5:
        return 0
    elif up_movement > 0.5:
        return 1
    elif left_movement > 0.5:
        return 2
    elif right_movement > 0.5:
        return 3
    else:
        return 4



# def keypoint_movement_towards_center(img1, img2):
#     # Convert images to grayscale
#     image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
#     # Initialize ORB detector
#     orb = cv2.ORB_create()
#
#     # Find keypoints and descriptors
#     keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
#     keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
#
#     # Initialize BFMatcher
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
#     # Match descriptors
#     matches = bf.match(descriptors1, descriptors2)
#
#     # Extract matched keypoints
#     matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
#     matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
#
#     # Calculate movement vectors
#     movement_vectors = matched_keypoints2 - matched_keypoints1
#
#     # Calculate distances from the center of the image
#     center = np.array(image1.shape[::-1]) / 2
#     distances_initial = np.linalg.norm(matched_keypoints1 - center, axis=1)
#     distances_final = np.linalg.norm(matched_keypoints2 - center, axis=1)
#
#     # Calculate the average movement ratio towards/away from center
#     movement_ratio_towards = np.mean(distances_final / distances_initial < 1)
#     movement_ratio_away = np.mean(distances_final / distances_initial > 1)
#     # print(movement_ratio_towards+movement_ratio_away)
#     # Define thresholds for zoom and camera movement
#     zoom_threshold = 0.6  # Adjust this based on your needs
#     movement_threshold = 0.7  # Adjust this based on your needs
#
#     # Determine the type of zoom (in/out)
#     if movement_ratio_towards > zoom_threshold:
#         return (0, -1)  # Zoom In
#     elif movement_ratio_away > zoom_threshold:
#         return (1, -1)  # Zoom Out
#
#     # Calculate movement directions (angles)
#     angles = np.arctan2(movement_vectors[:, 1], movement_vectors[:, 0]) * 180 / np.pi
#
#     # Determine the dominant direction of movement
#     if np.mean(angles > 45) > movement_threshold:
#         return (-1, 1)  # Moving Down (South)
#     elif np.mean(angles < -45) > movement_threshold:
#         return (-1, 0)  # Moving Up (North)
#     elif np.mean(np.logical_and(angles > -45, angles < 45)) > movement_threshold:
#         return (-1, 3)  # Moving Right (East)
#     elif np.mean(np.logical_and(angles > 135, angles < -135)) > movement_threshold:
#         return (-1, 2)  # Moving Left (West)
#
#     return (-1, -1)  # No significant movement detected

def detect_altitude_change(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Extract matched keypoints
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Calculate the homography matrix
    H, _ = cv2.findHomography(matched_keypoints1, matched_keypoints2, cv2.RANSAC)

    if H is None:
        return 'No significant movement detected'

    # Analyze the homography matrix
    # The (2,2) element in the homography matrix is associated with scaling in the z-axis (height)
    z_scale_change = H[2, 2]

    # Define thresholds
    altitude_threshold = 0.02  # You may need to adjust this threshold

    if z_scale_change > altitude_threshold:
        return 'Ascending'  # The drone is ascending
    elif z_scale_change < -altitude_threshold:
        return 'Descending'  # The drone is descending
    else:
        return 'No significant altitude change'  # No significant change in altitude

def process_frames(frames, window_size=10):
    result = []
    i = 0
    n = len(frames)

    while i < n:
        current = frames[i]
        direction = current[2]

        # Skip frames with direction (0, 0), (1, 1), (1, 0), or (0, 1)
        if direction in [(1, -1),(0, -1)]:
            i += 1
            continue

        if direction == (1, 4):
            # Calculate the average direction in the surrounding frames
            window_start = max(0, i - window_size)
            window_end = min(n, i + window_size + 1)
            window = frames[window_start:window_end]
            avg_direction = np.mean([frame[2] for frame in window], axis=0)

            # If the average direction in the surrounding area is similar, keep only one frame
            if np.allclose(avg_direction, direction, atol=0.5):
                result.append(current)
                # Skip all subsequent frames with the same direction
                while i < n - 1 and frames[i + 1][2] == direction:
                    i += 1
        else:
            result.append(current)

        i += 1

    return result



def plot_2_images_with_data(image1,image2,data1="", data2=""):
    plt.figure(figsize=(10, 5))

    # Plot first image
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title('Image 1')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.text(0.5, -0.1, data1, ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)
    # Plot second image
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title('Image 2')
    plt.axis('off')  # Turn off axis numbers and ticks

    plt.text(0.5, -0.1, data2, ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)

    plt.show()


def process_frames2(frames, window_size=10, threshold=0.1):
    result = []
    i = 0
    n = len(frames)

    for i in frames:
        frame1_index, frame2_index, directions = i

        # Extract the values
        values = list(directions.values())

        # Calculate the min and max values
        min_val = min(values)
        max_val = max(values)
        avg =sum(values) / len(values)

        top = directions['top']
        bottom = directions['bottom']




        # Calculate the range of values as a percentage of the max value
        if (top+bottom) <10:
            continue
        if (min_val + max_val) < 15:
             continue
        if avg > 70:
            continue


        # If the values are not within the threshold, add the frame to the result
        result.append((frame1_index, frame2_index, directions))

    return result
def process_frames1(frames, window_size=10):
    result = []
    i = 0
    n = len(frames)

    while i < n:
        current = frames[i]
        direction = current[2]

        # Skip frames with direction (0, 0), (1, 1), (1, 0), or (0, 1)
        if direction in [0]:
            i += 1
            continue
            result.append(current)

        i += 1

    return result


def calculate_overlap(image1, image2, ratio_threshold=0.75, min_good_matches=4, min_inliers=4):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()  # Limiting to 500 keypoints
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Check if descriptors are valid and have the same size
    if des1 is None or des2 is None or des1.shape[1] != des2.shape[1]:
        # print("Invalid or mismatched descriptors.")
        return 0.0

    # Use BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Apply ratio test
    good_matches = []
    for m in matches:
        if m.distance < ratio_threshold * matches[-1].distance:
            good_matches.append(m)
    if len(good_matches) < min_good_matches:
        return 0.0

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None or mask.sum() < min_inliers:
        return 0.0

    # Calculate the transformation of the corners of image1
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    corners_image1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners_image1 = cv2.perspectiveTransform(corners_image1, M)

    # Create polygons from the transformed corners
    poly1 = Polygon(transformed_corners_image1.reshape(-1, 2))
    poly2 = Polygon([[0, 0], [w2, 0], [w2, h2], [0, h2]])

    # Validate and fix polygons if needed
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)

    if not poly1.is_valid or not poly2.is_valid:
        print(f"Invalid polygons: {explain_validity(poly1)}, {explain_validity(poly2)}")
        return 0.0

    # Calculate intersection and union areas
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    # Calculate the overlap percentage
    overlap_percentage = (intersection_area / union_area) * 100 if union_area > 0 else 0.0

    return overlap_percentage

def calculate_region_overlaps(image1, image2, border_size=100):
    h, w = image1.shape[:2]

    # Define regions
    regions = {
        'top': (0, border_size, 0, w),
        'bottom': (h - border_size, h, 0, w),
        'left': (0, h, 0, border_size),
        'right': (0, h, w - border_size, w)
    }

    overlaps = {}

    for region_name, (y1, y2, x1, x2) in regions.items():
        cropped_image1 = image1[y1:y2, x1:x2]
        cropped_image2 = image2[y1:y2, x1:x2]
        overlap = calculate_overlap(cropped_image1, cropped_image2)
        overlaps[region_name] = overlap

    return overlaps

def detect_movement_direction(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Extract matched keypoints
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Calculate movement vectors
    movement_vectors = matched_keypoints2 - matched_keypoints1

    # Calculate movement magnitudes and angles
    magnitudes = np.linalg.norm(movement_vectors, axis=1)
    angles = np.arctan2(movement_vectors[:, 1], movement_vectors[:, 0]) * 180 / np.pi

    # Define thresholds
    inward_threshold = 5  # Minimum inward movement magnitude
    outward_threshold = 5  # Minimum outward movement magnitude
    direction_threshold = 5  # Threshold for considering movement in a specific direction

    # Classify movement as inward, outward, or directional
    inward_movement = (magnitudes > inward_threshold) & (matched_keypoints2[:, 0] < matched_keypoints1[:, 0])
    outward_movement = (magnitudes > outward_threshold) & (matched_keypoints2[:, 0] > matched_keypoints1[:, 0])

    # Count the proportion of movements in each direction
    inward_count = np.sum(inward_movement)
    outward_count = np.sum(outward_movement)

    left_movement = np.sum(angles > 135) + np.sum(angles < -135)
    right_movement = np.sum((angles > -45) & (angles < 45))
    upward_movement = np.sum((angles > 45) & (angles < 135))
    downward_movement = np.sum((angles > -135) & (angles < -45))

    # Determine the dominant movement direction
    if inward_count > outward_count and inward_count > 20:
        return 0
    elif outward_count > inward_count and outward_count > 20:
        return 0
    # elif left_movement > right_movement and left_movement > direction_threshold:
    #     return 'Moving Left'
    # elif right_movement > left_movement and right_movement > direction_threshold:
    #     return 'Moving Right'
    # elif upward_movement > downward_movement and upward_movement > direction_threshold:
    #     return 'Moving Up'
    # elif downward_movement > upward_movement and downward_movement > direction_threshold:
    #     return 'Moving Down'
    else:
        return 1

def is_blurry(image, threshold=100.0):
    """Check if the image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_over_or_under_exposed(image, low_threshold=50, high_threshold=200):
    """Check if the image is overexposed or underexposed."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness < low_threshold or mean_brightness > high_threshold
def count_pixel_movements(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Extract matched keypoints
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Calculate movement vectors
    movement_vectors = matched_keypoints2 - matched_keypoints1

    # Calculate movement angles in degrees
    angles = np.arctan2(movement_vectors[:, 1], movement_vectors[:, 0]) * 180 / np.pi

    # Count movements in each direction
    left_movement = np.sum((angles > 135) | (angles < -135))
    right_movement = np.sum((angles > -45) & (angles < 45))
    up_movement = np.sum((angles > 45) & (angles < 135))
    down_movement = np.sum((angles > -135) & (angles < -45))

    return {
        'Left': left_movement,
        'Right': right_movement,
        'Up': up_movement,
        'Down': down_movement
    }

def estimate_camera_movement(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Feature detection and matching (using ORB)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match keypoints between the frames
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Estimate transformation (homography)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Decompose transformation
    dx = H[0, 2]  # Translation in x direction
    dy = H[1, 2]  # Translation in y direction
    theta = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi  # Rotation angle (in degrees)

    # Calculate camera movement
    translation_distance = np.sqrt(dx ** 2 + dy ** 2)

    return translation_distance, theta

def detect_objects_shrinking_sift(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Use BFMatcher to find matches between the descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Calculate the ratio of the distances between matched points
    size_ratios = []
    for match in matches:
        kp1 = keypoints1[match.queryIdx]
        kp2 = keypoints2[match.trainIdx]
        size_ratios.append(kp2.size / kp1.size)

    # Calculate average size ratio
    avg_ratio = sum(size_ratios) / len(size_ratios)

    if avg_ratio < 0.9:
        return 'Objects are getting smaller'
    elif avg_ratio > 1.1:
        return 'Objects are getting larger'
    else:
        return 'No significant size change'

def detect_objects_shrinking(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Define the central region (for simplicity, a square region)
    h, w = gray1.shape
    cx, cy = w // 2, h // 2
    size = min(h, w) // 4  # You can adjust the size based on your needs

    # Extract the central region from both images
    region1 = gray1[cy - size:cy + size, cx - size:cx + size]
    region2 = gray2[cy - size:cy + size, cx - size:cx + size]

    # Calculate the sum of pixel intensities in the central region
    sum_region1 = np.sum(region1)
    sum_region2 = np.sum(region2)
    print(f"sum_region1: {sum_region1}")
    print(f"sum_region2: {sum_region2}")
    print(f"sum_region2<sum_region1 * 0.9 -> {sum_region2} < { sum_region1 * 0.9}")
    print(f"sum_region2 > sum_region1 * 1.1 -> {sum_region2} > {sum_region1 * 1.1}")

    # Compare the sum of intensities
    if sum_region2 < sum_region1 * 0.8:
        return 'Objects are getting smaller'
    elif sum_region2 > sum_region1 * 1.1:
        return 'Objects are getting larger'
    else:
        return 'No significant size change'
def detect_and_draw_contours(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Apply morphological operations to group close edge points
    kernel = np.ones((5, 5), np.uint8)  # Define the kernel size
    dilated = cv2.dilate(edged, kernel, iterations=1)  # Dilation to connect points
    # You can also apply erosion here if needed:
    # eroded = cv2.erode(dilated, kernel, iterations=1)
    # We'll use 'dilated' as the final edge result

    # Find contours on the processed edge image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    return contour_image, dilated  # Return the contour image and the processed edge image





def detect_and_draw_contours_and_keypoints(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (10, 5), 0)

    # Apply Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect keypoints using ORB
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)

    # Draw keypoints on a copy of the original image
    keypoints_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

    # Draw contours on a copy of the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    return contour_image, edged, keypoints_image


def save_combined_image(img1, img2, output_path):
    # Detect and draw contours, Canny edges, and keypoints for both images
    contour_img1, edged_img1, keypoints_img1 = detect_and_draw_contours_and_keypoints(img1)
    contour_img2, edged_img2, keypoints_img2 = detect_and_draw_contours_and_keypoints(img2)

    # Convert images from BGR to RGB for Matplotlib display
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    contour_img1_rgb = cv2.cvtColor(contour_img1, cv2.COLOR_BGR2RGB)
    keypoints_img1_rgb = cv2.cvtColor(keypoints_img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    contour_img2_rgb = cv2.cvtColor(contour_img2, cv2.COLOR_BGR2RGB)
    keypoints_img2_rgb = cv2.cvtColor(keypoints_img2, cv2.COLOR_BGR2RGB)

    # Create a figure to hold the subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Plot the original images, Canny edges, contour images, and keypoints
    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title("Frame 1")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(edged_img1, cmap='gray')
    axes[0, 1].set_title("Canny Edges of Frame 1")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(contour_img1_rgb)
    axes[0, 2].set_title("Contours of Frame 1")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(keypoints_img1_rgb)
    axes[0, 3].set_title("Key Points of Frame 1")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(img2_rgb)
    axes[1, 0].set_title("Frame 2")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(edged_img2, cmap='gray')
    axes[1, 1].set_title("Canny Edges of Frame 2")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(contour_img2_rgb)
    axes[1, 2].set_title("Contours of Frame 2")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(keypoints_img2_rgb)
    axes[1, 3].set_title("Key Points of Frame 2")
    axes[1, 3].axis("off")

    # Save the combined image
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def apply_canny_and_group_edges(image):
    # Convert the image to grayscale if it's not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define the kernel for morphological operations
    kernel = np.ones((10, 10), np.uint8)

    # Apply dilation to connect nearby points
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Optionally, apply erosion to reduce the size of the connected regions
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return edges, eroded


def apply_canny_and_group_dense_edges(image, density_threshold=0.3, kernel_size=(5, 5)):
    # Convert the image to grayscale if it's not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define the kernel for morphological operations
    kernel = np.ones(kernel_size, np.uint8)

    # Calculate the density of edge points using a sliding window
    density_map = cv2.filter2D(edges.astype(np.float32), -1, kernel) / (kernel_size[0] * kernel_size[1])

    # Create a mask where the density exceeds the threshold
    mask = (density_map > density_threshold).astype(np.uint8) * 255

    # Apply dilation only where the mask is 255 (where density is high)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    dense_edges = cv2.bitwise_and(dilated, mask)

    return edges, dense_edges
def calculate_optical_flow(frame1, frame2, region=None):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # If a specific region is defined, extract it
    if region is not None:
        x, y, w, h = region
        flow = flow[y:y+h, x:x+w]

    # Calculate magnitude and angle of flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create an HSV image based on the region or full frame size
    if region is not None:
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        hsv = np.zeros_like(frame1)

    hsv[..., 1] = 255

    # Normalize angle and magnitude for display
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for visualization
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow, rgb_flow
def detect_objects_shrinking_bounding_boxes(img1, img2, cascade_path='haarcascade_frontalface_default.xml'):
    # Load the Haar Cascade for face detection (or any other object detection cascade)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect objects (e.g., faces) in both images
    objects1 = cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    objects2 = cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Calculate the area of the bounding boxes in the first image
    area1 = sum([w * h for (x, y, w, h) in objects1])
    area2 = sum([w * h for (x, y, w, h) in objects2])

    print(f"Total area in image 1: {area1}")
    print(f"Total area in image 2: {area2}")

    # Compare the areas
    if area2 < area1 * 0.9:
        return 'Objects are getting smaller'
    elif area2 > area1 * 1.1:
        return 'Objects are getting larger'
    else:
        return 'No significant size change'


