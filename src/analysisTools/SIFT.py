import matplotlib.pyplot as plt
from skimage import io, transform, color
from skimage.feature import match_descriptors, plot_matches, SIFT

def sift(img1_path, img2_path):
    # Read the images from file paths and convert them to grayscale
    img1 = color.rgb2gray(io.imread(img1_path))
    img2 = color.rgb2gray(io.imread(img2_path))

    # Initialize the SIFT descriptor extractor
    descriptor_extractor = SIFT()

    # Detect keypoints and extract descriptors for the first image
    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    # Detect keypoints and extract descriptors for the second image
    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # Match descriptors between the two images using a ratio test and cross-checking
    matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6, cross_check=True)

    # Create subplots for visualization
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

    # Set the colormap to grayscale
    plt.gray()

    # Plot matches between the two images showing all keypoints and matches
    plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12)
    ax[0].axis('off')
    ax[0].set_title("Image1 vs. Imag2\n"
                    "(all keypoints and matches)")

    # Plot a subset of matches for better visibility
    plot_matches(ax[1], img1, img2, keypoints1, keypoints2, matches12[::15], only_matches=True)
    ax[1].axis('off')
    ax[1].set_title("Image1 vs. Imag2\n"
                    "(subset of matches for visibility)")

    # Adjust layout to prevent overlapping of subplots
    plt.tight_layout()
    # Display the plots
    plt.show()

if __name__ == '__main__':

    # Define paths to the input images
    imag1 = "/Users/noyagendelman/Desktop/choosingFrames/v1p/chosen_frames_4/frame_8.jpg"
    imag2 = "/Users/noyagendelman/Desktop/choosingFrames/v1p/chosen_frames_4/frame_9.jpg"
    # Perform SIFT matching between the two images
    sift(imag1, imag2)
