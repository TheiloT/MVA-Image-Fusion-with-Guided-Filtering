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
    axes[0].set_title(f"Image 1: {title1}")
    axes[1].imshow(im2, cmap="gray" if gray else None)
    axes[1].axis("off")
    axes[1].set_title(f"Image 2: {title2}")
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

def zoom_on_detail(img, top_left=[150,150], width=200, height=200):
    #Zooming on details
    plt.imshow(img[top_left[0]:top_left[0]+height, top_left[1]:top_left[1]+width, :])
    plt.title("Zoom on detail")
    plt.axis('off')