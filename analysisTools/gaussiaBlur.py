import cv2

def apply_gaussian_blur(image):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image



if __name__ == "__main__":
    # Read the image
    image = cv2.imread("/Users/noyagendelman/Desktop/choosingFrames/v1p/chosen_frames_4/frame_8.jpg")  # Provide the path to your image

    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(image)

    # Display the original and blurred images
    cv2.imshow("Original Image", image)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
