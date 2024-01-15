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
    base = average_filter(im, average_filter_size//2).astype(int)
    # Compute detail layers
    detail = im - base
    return base, detail


def apply_laplacian_filter(im, kernel_size=3, sigma=3, local_average_size=14):
    im_blur = cv2.GaussianBlur(im, (kernel_size, kernel_size), sigma)
    if len(im_blur.shape) > 2:
        im_gray = cv2.cvtColor(im_blur, cv2.COLOR_RGB2GRAY)
    else: 
        im_gray = im_blur
    H = cv2.Laplacian(im_gray, ddepth=-1, ksize=kernel_size)
    H = cv2.convertScaleAbs(H)
    H = average_filter(
        H, local_average_size//2
    )  # Take local average of the absolute value
    return H


def get_saliency_map(
    im,
    laplacian_kernel_size=3,
    laplacian_sigma=3,
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
    savefigs=False,
):
    """
    Fuse multiple images using guided filtering.

    Parameters:
        :imgs (numpy.ndarray): The images to fuse.
        :average_filter_size (int, optional): The size of the average filter for computing base and detail layers. Defaults to 31.
        :laplacian_kernel_size (int, optional): The size of the Laplacian kernel for computing saliency maps. Defaults to 3.
        :gaussian_filter_sigma (float, optional): The standard deviation of the Gaussian filter for computing saliency maps. Defaults to 5.
        :gaussian_filter_radius (int, optional): The radius of the Gaussian filter for computing saliency maps. Defaults to 5.
        :local_average_size (int, optional): The size of the local average filter for computing saliency maps. Defaults to 7.
        :r1 (int, optional): The radius of the guided filter for computing weight maps. Defaults to 45.
        :eps1 (float, optional): The epsilon parameter of the guided filter for computing weight maps. Defaults to 0.3.
        :r2 (int, optional): The radius of the guided filter for computing weight maps. Defaults to 7.
        :eps2 (float, optional): The epsilon parameter of the guided filter for computing weight maps. Defaults to 1e-6.
        :verbose (bool, optional): Whether to print intermediate results. Defaults to False.
        :savefigs (bool, optional): Whether to save intermediate figures. Defaults to False.

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
    if verbose or savefigs:
        show_multi_images(WB, "Weight maps for base layers", gray=True, scale=[0, 1], savename="WB" if savefigs else None)
        show_multi_images(WD, "Weight maps for detail layers", gray=True, scale=[0, 1], savename="WD" if savefigs else None)
        print()

    # Fuse layers
    if verbose:
        print("Fusing layers...")
    fusedB, fusedD = fuse_layers(bases, details, WB, WD)
    if verbose:
        show_images(fusedB, fusedD, "Fused base", "Fused detail")
        print()

    fused_image = fusedB + fusedD
    if verbose or savefigs:
        if verbose:
            print("Fused image:")
        plt.close("all")
        plt.imshow(fused_image, cmap="gray" if len(fused_image.shape) == 2 else None)
        plt.axis("off")
        plt.title(f"Fused image")
        if savefigs:
            plt.savefig(f"fused.png")
        else:
            plt.show()

    return fused_image


def get_weight_masks_no_decomposition(saliency_maps, guides, r=45, eps=0.3):
    Ps = get_weight_mask_precursors(saliency_maps)
    
    W = []
    epsilon = 1e-6 # to avoid divisions by 0
    for i in range(len(Ps)):
        if len(guides[0].shape) > 2:
            Wi = guided_filter_with_colored_guide(Ps[i], guides[i].astype(np.float32) / 255, r, eps)
        else : 
            Wi = guided_filter(Ps[i], guides[i].astype(np.float32) / 255, r, eps)
        W.append(np.clip(Wi, epsilon, None))

    W = W/np.sum(W, axis=0)

    return W

def fuse_layers_no_decomposition(images, W):
    if len(images[0].shape) > 2:
        fused = np.average(images, axis=0, weights=np.repeat(np.expand_dims(W, -1), images.shape[-1], axis=-1))
    else : 
        fused = np.average(images, axis=0, weights=W)
    return fused.astype(int)


def fuse_images_no_decomposition(
    imgs,
    laplacian_kernel_size=3,
    gaussian_filter_sigma=5,
    gaussian_filter_radius=5,
    local_average_size=7,
    r=45,
    eps=0.3,
    verbose=False,
    savefigs=False,
):
    """
    Fuse multiple images using guided filtering, without decomposition into base & detail layers.

    Parameters:
        :imgs (numpy.ndarray): The images to fuse.
        :laplacian_kernel_size (int, optional): The size of the Laplacian kernel for computing saliency maps. Defaults to 3.
        :gaussian_filter_sigma (float, optional): The standard deviation of the Gaussian filter for computing saliency maps. Defaults to 5.
        :gaussian_filter_radius (int, optional): The radius of the Gaussian filter for computing saliency maps. Defaults to 5.
        :local_average_size (int, optional): The size of the local average filter for computing saliency maps. Defaults to 7.
        :r (int, optional): The radius of the guided filter for computing weight maps. Defaults to 45.
        :eps (float, optional): The epsilon parameter of the guided filter for computing weight maps. Defaults to 0.3.
        :verbose (bool, optional): Whether to print intermediate results. Defaults to False.
        :savefigs (bool, optional): Whether to save intermediate figures. Defaults to False.

    Returns:
        numpy.ndarray: The fused image.
    """
    # Enforce that all images are encoded in uint8
    for img in imgs:
        assert img.dtype == np.uint8

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
    W = get_weight_masks_no_decomposition(saliency_maps, imgs, r, eps)
    if verbose or savefigs:
        show_multi_images(W, "Filtered weight maps", gray=True, scale=[0, 1], savename="WB" if savefigs else None)
        print()

    # Fuse
    if verbose:
        print("Fusing...")
    fused_image = fuse_layers_no_decomposition(np.array(imgs), W)
    if verbose or savefigs:
        if verbose:
            print("Fused image:")
        plt.close("all")
        plt.imshow(fused_image, cmap="gray" if len(fused_image.shape) == 2 else None)
        plt.axis("off")
        plt.title(f"Fused image")
        if savefigs:
            plt.savefig(f"fused.png")
        else:
            plt.show()

    return fused_image


def get_weight_masks_no_filtering(saliency_maps):
    P = get_weight_mask_precursors(saliency_maps)
    return P


def fuse_images_no_filtering(
    imgs,
    laplacian_kernel_size=3,
    gaussian_filter_sigma=5,
    gaussian_filter_radius=5,
    local_average_size=7,
    verbose=False,
    savefigs=False,
):
    """
    Fuse multiple images, wuthout guided filtering.

    Parameters:
        :imgs (numpy.ndarray): The images to fuse.
        :laplacian_kernel_size (int, optional): The size of the Laplacian kernel for computing saliency maps. Defaults to 3.
        :gaussian_filter_sigma (float, optional): The standard deviation of the Gaussian filter for computing saliency maps. Defaults to 5.
        :gaussian_filter_radius (int, optional): The radius of the Gaussian filter for computing saliency maps. Defaults to 5.
        :local_average_size (int, optional): The size of the local average filter for computing saliency maps. Defaults to 7.
        :verbose (bool, optional): Whether to print intermediate results. Defaults to False.
        :savefigs (bool, optional): Whether to save intermediate figures. Defaults to False.

    Returns:
        numpy.ndarray: The fused image.
    """
    # Enforce that all images are encoded in uint8
    for img in imgs:
        assert img.dtype == np.uint8

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
    P = get_weight_masks_no_filtering(saliency_maps)
    if verbose or savefigs:
        show_multi_images(P, "Weight maps", gray=True, scale=[0, 1], savename="P" if savefigs else None)
        print()

    # Fuse
    if verbose:
        print("Fusing...")
    fused_image = fuse_layers_no_decomposition(np.array(imgs), P)
    if verbose or savefigs:
        if verbose:
            print("Fused image:")
        plt.close("all")
        plt.imshow(fused_image, cmap="gray" if len(fused_image.shape) == 2 else None)
        plt.axis("off")
        plt.title(f"Fused image")
        if savefigs:
            plt.savefig(f"fused.png")
        else:
            plt.show()

    return fused_image