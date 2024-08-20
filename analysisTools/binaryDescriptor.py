from skimage import transform
from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             plot_matches, BRIEF)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.io import imread

def binaryDescriptor(img1_path, img2_path):
    img1 = rgb2gray(imread(img1_path))
    img2 = rgb2gray(imread(img2_path))

    keypoints1 = corner_peaks(corner_harris(img1), min_distance=5,
                              threshold_rel=0.1)
    keypoints2 = corner_peaks(corner_harris(img2), min_distance=5,
                              threshold_rel=0.1)

    extractor = BRIEF()

    extractor.extract(img1, keypoints1)
    keypoints1 = keypoints1[extractor.mask]
    descriptors1 = extractor.descriptors

    extractor.extract(img2, keypoints2)
    keypoints2 = keypoints2[extractor.mask]
    descriptors2 = extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.gray()

    plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12)
    ax.axis('off')
    ax.set_title("Original Image vs. Transformed Image")

    plt.show()

if __name__ == '__main__':

    imag1 = "/Users/noyagendelman/Desktop/choosingFrames/v1p/chosen_frames_4/frame_8.jpg"
    imag2 = "/Users/noyagendelman/Desktop/choosingFrames/v1p/chosen_frames_4/frame_9.jpg"
    binaryDescriptor(imag1, imag2)
