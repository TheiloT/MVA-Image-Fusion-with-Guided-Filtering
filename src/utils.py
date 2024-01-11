import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_images(im1, im2, title1, title2, gray=False):
    """
    Display two images side by side.

    Parameters:
        :im1 (numpy.ndarray): The first image.
        :im2 (numpy.ndarray): The second image.
        :title1 (str): The title for the first image.
        :title2 (str): The title for the second image.
        :gray (bool, optional): Whether to display the images in grayscale. Defaults to False.
    """
    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(im1, cmap="gray" if gray else None)
    axes[0].axis("off")
    axes[0].set_title(f"Cathedral 1: {title1}")
    axes[1].imshow(im2, cmap="gray" if gray else None)
    axes[1].axis("off")
    axes[1].set_title(f"Cathedral 2: {title2}")
    plt.show()

def show_3_images(im1, im2, im3, title1, title2, title3, gray=False):
    """
    Display two images side by side.

    Parameters:
        :im1 (numpy.ndarray): The first image.
        :im2 (numpy.ndarray): The second image.
        :im3 (numpy.ndarray): The third image.
        :title1 (str): The title for the first image.
        :title2 (str): The title for the second image.
        :title3 (str): The title for the third image.
        :gray (bool, optional): Whether to display the images in grayscale. Defaults to False.
    """
    _, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(im1, cmap="gray" if gray else None)
    axes[0].axis("off")
    axes[0].set_title(f"{title1}")
    axes[1].imshow(im2, cmap="gray" if gray else None)
    axes[1].axis("off")
    axes[1].set_title(f"{title2}")
    axes[2].imshow(im3, cmap="gray" if gray else None)
    axes[2].axis("off")
    axes[2].set_title(f"{title3}")
    plt.show()


def rgb2gray(rgb, use_opencv=False):
    """
    Convert an RGB image to grayscale.

    Parameters:
        :rgb (numpy.ndarray): The RGB image to convert.
        :use_opencv (bool, optional): Whether to use OpenCV for conversion. Defaults to False.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    if use_opencv:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        return np.dot(
            rgb[..., :3], [0.299, 0.587, 0.114]
        )  # Chosen to comply with OpenCV, see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    
def entropy(gray_img):
    """
    Compute entropy value for grayscale image.

    Parameters:
        :gray_img (numpy.ndarray): A grayscale image.
    """
    # Ensure the input images is not RGB
    if len(gray_img.shape) != 2:
        raise ValueError("Input images must be grayscale")
    marg = np.histogram(np.ravel(gray_img), bins = 256)[0]/gray_img.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy

def joint_entropy(gray_img1, gray_img2):
    """
    Compute joint entropy value for 2 grayscale images.

    Parameters:
        :gray_img1 (numpy.ndarray): A grayscale image.
        :gray_img2 (numpy.ndarray): A grayscale image.
    """
    # Ensure the input images have the same shape
    if gray_img1.shape != gray_img2.shape:
        raise ValueError("Input images must have the same shape")

    # Flatten the images
    flat_image1 = np.ravel(gray_img1)
    flat_image2 = np.ravel(gray_img2)

    # Calculate the joint histogram
    joint_histogram, _, _ = np.histogram2d(flat_image1, flat_image2, bins=256)

    # Normalize the joint histogram to obtain joint probability distribution
    joint_prob = joint_histogram / np.sum(joint_histogram)

    # Compute joint entropy
    joint_prob = list(filter(lambda p: p > 0, np.ravel(joint_prob)))
    joint_ent = -np.sum(np.multiply(joint_prob, np.log2(joint_prob)))

    return joint_ent

def normalized_mutual_information_metric(gray_img1, gray_img2, gray_fusion):
    """
    Compute Normalized Mutual Information Metric. It measures how the information from sources images are preserved

    Parameters:
        :gray_img1 (numpy.ndarray): A grayscale image.
        :gray_img2 (numpy.ndarray): A grayscale image.
        :gray_fusion (numpy.ndarray): A grayscale image, result of fusion between img1 and img2.
    """
    entropy1 = entropy(gray_img1)
    entropy2 = entropy(gray_img2)
    entropyf = entropy(gray_fusion)
    mutual_info1 = entropy1 + entropyf - joint_entropy(gray_img1, gray_fusion)
    mutual_info2 = entropy2 + entropyf - joint_entropy(gray_img2, gray_fusion)
    return 2 * (mutual_info1 / (entropy1 + entropyf) + mutual_info2 / (entropy2 + entropyf))