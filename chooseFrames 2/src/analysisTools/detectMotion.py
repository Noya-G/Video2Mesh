import cv2
import numpy as np

def detect_camera_movement(frame1, frame2):
    # Convert frames to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur (optional)
    frame1_blurred = cv2.GaussianBlur(frame1_gray, (5, 5), 0)
    frame2_blurred = cv2.GaussianBlur(frame2_gray, (5, 5), 0)

    # Calculate absolute difference between frames
    diff_frame = cv2.absdiff(frame1_blurred, frame2_blurred)

    # Thresholding
    _, threshold_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analyze contours
    for contour in contours:
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        centroid_x = x + w // 2
        centroid_y = y + h // 2

        # Print centroid position for debugging
        print(f"Contour centroid: ({centroid_x}, {centroid_y})")

        # Check if the centroid has moved vertically
        # You may need to adjust the threshold for your specific application
        if abs(centroid_y - frame1.shape[0] // 2) > 10:  # Assuming frame1.shape[0] gives frame height
            if centroid_y > frame1.shape[0] // 2:
                print("Camera is moving up")
            else:
                print("Camera is moving down")
            break
    else:
        print("No significant vertical movement detected")

# Example usage
if __name__ == "__main__":
    frame1 = cv2.imread("/Users/noyagendelman/Desktop/choosingFrames/v1p/chosen_frames_4/frame_3.jpg")
    frame2 = cv2.imread("/Users/noyagendelman/Desktop/choosingFrames/v1p/chosen_frames_4/frame_4.jpg")
    detect_camera_movement(frame1, frame2)
