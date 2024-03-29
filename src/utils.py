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
    Display three images side by side.

    Parameters:
        :im1 (numpy.ndarray): The first image.
        :im2 (numpy.ndarray): The second image.
        :im3 (numpy.ndarray): The third image.
        :title1 (str): The title for the first image.
        :title2 (str): The title for the second image.
        :title3 (str): The title for the third image.
        :gray (bool, optional): Whether to display the images in grayscale. Defaults to False.
    """
    _, axes = plt.subplots(1, 3, figsize=(16, 8))
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


def rgb2gray(rgb):
    """
    Convert an RGB image to grayscale.

    Parameters:
        :rgb (numpy.ndarray): The RGB image to convert.
        :use_opencv (bool, optional): Whether to use OpenCV for conversion. Defaults to False.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    return np.dot(
        rgb[..., :3], [0.299, 0.587, 0.114]
    ).astype(np.uint8)  # Chosen to comply with OpenCV, see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html


def zoom_on_detail(img, top_left=[150,150], width=200, height=200, grayscale=False):
    #Zooming on details
    plt.figure(figsize=(6,6))
    if grayscale:
        plt.imshow(img[top_left[0]:top_left[0]+height, top_left[1]:top_left[1]+width], cmap='gray')
    else:
        plt.imshow(img[top_left[0]:top_left[0]+height, top_left[1]:top_left[1]+width, :])
    plt.title("Zoom on detail", fontsize=16)
    plt.axis('off')
    
    
def show_multi_images(images, title, gray=False, scale=[None, None], savename=False):
    """
    Display multiple images.

    Parameters:
        :images (list(numpy.ndarray)): The list of images.
        :title (str): The title for the figure.
        :gray (bool, optional): Whether to display the images in grayscale. Defaults to False.
        :scale (list(float), optional): Value range covered by the grayscale colormap. Defaults to [None, None].
        :savename (bool, optional): Name under which to save the figure. Do not save if False. Defaults to False.
    """
    nrows, ncols = 1 + (len(images)-1)//4, min(4, len(images))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, 5*nrows-1))
    fig.suptitle(title)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap="gray" if gray else None, vmin=scale[0], vmax=scale[1])
            ax.axis("off")
            ax.set_title(f"Image {i+1}")
        else:
            ax.axis("off")

    plt.tight_layout()
    if savename:
        plt.savefig(f"{savename}.png")
    else:
        plt.show()
