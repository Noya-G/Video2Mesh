import cv2
import numpy as np

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