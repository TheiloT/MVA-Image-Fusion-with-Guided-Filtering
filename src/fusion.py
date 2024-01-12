import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

from .utils import show_images, show_multi_images
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
    argmaxes = np.argmax(saliency_maps, axis=0)
    Ps = []
    for i in range(saliency_maps.shape[0]):
        P = np.zeros(saliency_maps[0].shape)
        P[argmaxes == i] = 1
        Ps.append(P)
    return Ps


def get_weight_masks(saliency_maps, guides, r1=45, eps1=0.3, r2=7, eps2=1e-6):
    Ps = get_weight_mask_precursors(saliency_maps)
    
    WB, WD = [], []
    epsilon = 1e-6 # to avoid divisions by 0
    for i in range(len(Ps)):
        if len(guides[0].shape) > 2:
            WBi = guided_filter_with_colored_guide(Ps[i], guides[i].astype(np.float32) / 255, r1, eps1)
            WDi = guided_filter_with_colored_guide(Ps[i], guides[i].astype(np.float32) / 255, r2, eps2)
        else : 
            WBi = guided_filter(Ps[i], guides[i].astype(np.float32) / 255, r1, eps1)
            WDi = guided_filter(Ps[i], guides[i].astype(np.float32) / 255, r2, eps2)
        WB.append(np.clip(WBi, epsilon, None))
        WD.append(np.clip(WDi, epsilon, None))

    WB = WB/np.sum(WB, axis=0)
    WD = WD/np.sum(WD, axis=0)

    return WB, WD


def fuse_layers(bases, details, WB, WD):
    if len(bases[0].shape) > 2:
        fusedB = np.average(bases, axis=0, weights=np.repeat(np.expand_dims(WB, -1), bases.shape[-1], axis=-1))
        fusedD = np.average(details, axis=0, weights=np.repeat(np.expand_dims(WD, -1), bases.shape[-1], axis=-1))
    else : 
        fusedB = np.average(bases, axis=0, weights=WB)
        fusedD = np.average(details, axis=0, weights=WD)
    return fusedB.astype(int), fusedD.astype(int)


def fuse_images(
    imgs,
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
        :imgs (numpy.ndarray): The images to fuse.
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
    # Enforce that all images are encoded in uint8
    for img in imgs:
        assert img.dtype == np.uint8
    
    # Compute base and detail layers
    if verbose:
        print("Computing base and detail layers...")
    bases, details = [], []
    for img in imgs:
        base, detail = get_base_detail_layers(img, average_filter_size)
        bases.append(base)
        details.append(detail)
    bases, details = np.array(bases), np.array(details)
    if verbose:
        show_multi_images(bases, "Base layers")
        show_multi_images(details, "Detail layers")
        print()

    # Compute saliency maps
    if verbose:
        print("Computing saliency maps...")
    saliency_maps = []
    for img in imgs:
        saliency_maps.append(get_saliency_map(
            img,
            laplacian_kernel_size=laplacian_kernel_size,
            local_average_size=local_average_size,
            gaussian_filter_sigma=gaussian_filter_sigma,
            gaussian_filter_radius=gaussian_filter_radius,
        ))
    saliency_maps = np.array(saliency_maps)
    if verbose:
        show_multi_images(saliency_maps, "Saliency maps", gray=True)
        print()

    # Compute weight maps
    if verbose:
        print("Computing weight maps...")
    WB, WD = get_weight_masks(saliency_maps, imgs, r1, eps1, r2, eps2)
    if verbose:
        show_multi_images(WB, "Weight masks for base layers", gray=True, scale=[0, 1])
        show_multi_images(WD, "Weight masks for detail layers", gray=True, scale=[0, 1])
        print()

    # Fuse layers
    if verbose:
        print("Fusing layers...")
    fusedB, fusedD = fuse_layers(bases, details, WB, WD)
    if verbose:
        show_images(fusedB, fusedD, "Fused base", "Fused detail")
        print()

    fused_image = fusedB + fusedD
    if verbose:
        print("Fused image:")
        plt.imshow(fused_image, cmap="gray" if len(fused_image.shape) > 2 else None)
        plt.axis("off")
        plt.title(f"Fused image")
        plt.show()

    return fused_image
