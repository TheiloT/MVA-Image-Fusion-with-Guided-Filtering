import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

from .utils import show_images
from .guided_filter import (
    average_filter,
    guided_filter_with_colored_guide,
    guided_filter
)


def get_base_detail_layers(im, average_filter_size=31):
    # Compute base layers
    base = average_filter(im, average_filter_size).astype(int)

    # Compute details layers
    detail = im - base  # Remark: how to interpret negative values?

    return base, detail


def apply_laplacian_filter(im, kernel_size=3, sigma=0, local_average_size=7):
    im_blur = cv2.GaussianBlur(im, (kernel_size, kernel_size), sigma)
    if len(im_blur.shape) > 2:
        im_gray = cv2.cvtColor(im_blur, cv2.COLOR_RGB2GRAY)
    else: 
        im_gray = im_blur
    H = cv2.Laplacian(im_gray, ddepth=-1, ksize=kernel_size)
    H = cv2.convertScaleAbs(H)
    H = average_filter(
        H, local_average_size
    )  # Take local average of the absolute value

    return H


def get_saliency_map(
    im,
    laplacian_kernel_size=3,
    laplacian_sigma=0,
    local_average_size=7,
    gaussian_filter_sigma=5,
    gaussian_filter_radius=5,
):
    H = apply_laplacian_filter(
        im, laplacian_kernel_size, laplacian_sigma, local_average_size
    )
    saliency = gaussian_filter(
        H, sigma=gaussian_filter_sigma, radius=gaussian_filter_radius
    )

    return saliency


def get_weight_mask_precursors(saliency_maps):
    if saliency_maps.shape[0] > 2:
        raise NotImplementedError(
            "Weight map computation for more than 2 images is not implemented yet"
        )
    saliency1, saliency2 = saliency_maps
    P1 = (saliency1 >= saliency2).astype(int)
    P2 = 1 - P1

    return P1, P2


def get_weight_masks(saliency_maps, guide1, guide2, r1=45, eps1=0.3, r2=7, eps2=1e-6):
    if saliency_maps.shape[0] > 2:
        raise NotImplementedError(
            "Weight map computation for more than 2 images is not implemented yet"
        )
    P1, P2 = get_weight_mask_precursors(saliency_maps)
    if len(guide1.shape) > 2:
        W1B = guided_filter_with_colored_guide(P1, guide1.astype(np.float32) / 255, r1, eps1)
        W2B = guided_filter_with_colored_guide(P2, guide2.astype(np.float32) / 255, r1, eps1)
        W1D = guided_filter_with_colored_guide(P1, guide1.astype(np.float32) / 255, r2, eps2)
        W2D = guided_filter_with_colored_guide(P2, guide2.astype(np.float32) / 255, r2, eps2)
    else : 
        W1B = guided_filter(P1, guide1.astype(np.float32) / 255, r1, eps1)
        W2B = guided_filter(P2, guide2.astype(np.float32) / 255, r1, eps1)
        W1D = guided_filter(P1, guide1.astype(np.float32) / 255, r2, eps2)
        W2D = guided_filter(P2, guide2.astype(np.float32) / 255, r2, eps2)
    
    eps = 1e-6 # to avoid division by 0
    W1B = np.clip(W1B, eps, None,)
    W2B = np.clip(W2B, eps, None,)
    W1D = np.clip(W1D, eps, None,)
    W2D = np.clip(W2D, eps, None,)    
    W1B = W1B / (W1B + W2B)
    W2B = W2B / (W1B + W2B)
    W1D = W1D / (W1D + W2D)
    W2D = W2D / (W1D + W2D)
    return W1B, W2B, W1D, W2D


def fuse_layers(base1, base2, detail1, detail2, W1B, W2B, W1D, W2D):
    if len(base1.shape) > 2:
        fusedB = W1B[:, :, None] * base1 + W2B[:, :, None] * base2
        fusedD = W1D[:, :, None] * detail1 + W2D[:, :, None] * detail2
    else : 
        fusedB = W1B * base1 + W2B * base2
        fusedD = W1D * detail1 + W2D * detail2
    return fusedB.astype(int), fusedD.astype(int)


def fuse_images(
    im1,
    im2,
    average_filter_size=31,
    laplacian_kernel_size=3,
    gaussian_filter_sigma=5,
    gaussian_filter_radius=5,
    local_average_size=7,
    r1=45,
    eps1=0.3,
    r2=7,
    eps2=1e-6,
    verbose=False,
):
    """
    Fuse two images using guided filtering for image fusion.

    Parameters:
        :im1 (numpy.ndarray): The first input image.
        :im2 (numpy.ndarray): The second input image.
        :average_filter_size (int, optional): The size of the average filter for computing base and detail layers. Defaults to 31.
        :laplacian_kernel_size (int, optional): The size of the Laplacian kernel for computing saliency maps. Defaults to 3.
        :gaussian_filter_sigma (float, optional): The standard deviation of the Gaussian filter for computing saliency maps. Defaults to 5.
        :gaussian_filter_radius (int, optional): The radius of the Gaussian filter for computing saliency maps. Defaults to 5.
        :local_average_size (int, optional): The size of the local average filter for computing saliency maps. Defaults to 7.
        :r1 (int, optional): The radius of the guided filter for computing weight masks. Defaults to 45.
        :eps1 (float, optional): The epsilon parameter of the guided filter for computing weight masks. Defaults to 0.3.
        :r2 (int, optional): The radius of the guided filter for computing weight masks. Defaults to 7.
        :eps2 (float, optional): The epsilon parameter of the guided filter for computing weight masks. Defaults to 1e-6.
        :verbose (bool, optional): Whether to print intermediate results. Defaults to False.

    Returns:
        numpy.ndarray: The fused image.
    """
    # Compute base and detail layers
    if verbose:
        print("Computing base and detail layers...")
    base1, detail1 = get_base_detail_layers(im1, average_filter_size)
    base2, detail2 = get_base_detail_layers(im2, average_filter_size)
    if verbose:
        show_images(base1, base2, "base", "base")
        show_images(detail1, detail2, "detail", "detail")
        print()

    # Compute saliency maps
    if verbose:
        print("Computing saliency maps...")
    saliency1 = get_saliency_map(
        im1,
        laplacian_kernel_size=laplacian_kernel_size,
        local_average_size=local_average_size,
        gaussian_filter_sigma=gaussian_filter_sigma,
        gaussian_filter_radius=gaussian_filter_radius,
    )
    saliency2 = get_saliency_map(
        im2,
        laplacian_kernel_size=laplacian_kernel_size,
        local_average_size=local_average_size,
        gaussian_filter_sigma=gaussian_filter_sigma,
        gaussian_filter_radius=gaussian_filter_radius,
    )
    saliency_maps = np.array([saliency1, saliency2])
    if verbose:
        show_images(saliency1, saliency2, "Saliency", "Saliency", gray=True)
        print()

    # Compute weight maps
    if verbose:
        print("Computing weight maps...")
    W1B, W2B, W1D, W2D = get_weight_masks(saliency_maps, im1, im2, r1, eps1, r2, eps2)
    if verbose:
        show_images(W1B, W2B, "Weight mask for base", "Weight mask for base", gray=True)
        show_images(
            W1D, W2D, "Weight mask for detail", "Weight mask for detail", gray=True
        )
        print()

    # Fuse layers
    if verbose:
        print("Fusing layers...")
    fusedB, fusedD = fuse_layers(base1, base2, detail1, detail2, W1B, W2B, W1D, W2D)
    if verbose:
        show_images(fusedB, fusedD, "Fused base", "Fused detail")
        print()

    fused_image = fusedB + fusedD
    if verbose:
        print("Fused image:")
        show_images(fused_image, fused_image, "Fused image", "Fused image")

    return fused_image
