import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel
from .guided_filter import average_filter

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

def structural_similarity_metric(img1, img2, fusion, window_size):
    """
    Compute Structural Similarity Metric, as introduced in https://www.sciencedirect.com/science/article/pii/S1566253506000704

    Parameters:
        :img1 (numpy.ndarray): A grayscale image.
        :img2 (numpy.ndarray): A grayscale image.
        :fusion (numpy.ndarray): A grayscale image, result of fusion between img1 and img2.
    """
    # Ensure the input images is not RGB
    if len(img1.shape) != 2 or len(img2.shape) != 2 or len(fusion.shape) != 2:
        raise ValueError("Input images must be grayscale")
    # Ensure the input images have the same shape
    if img1.shape != img2.shape or img1.shape != fusion.shape or img2.shape != fusion.shape:
        raise ValueError("Input images must have the same shape")
    
    #Compute SSIM using function from scikit-image
    ssim_1f = ssim(img1, fusion, win_size=window_size, full=True)[1]
    ssim_2f = ssim(img2, fusion, win_size=window_size, full=True)[1]
    ssim_12 = ssim(img1, img2, win_size=window_size, full=True)[1]

    #Compute variance in each window, for both source images
    r = window_size // 2
    mean_img1 = average_filter(img1, r)
    corr_img1 = average_filter(img1 * img1, r)
    var_img1 = corr_img1 - mean_img1 * mean_img1

    mean_img2 = average_filter(img2, r)
    corr_img2 = average_filter(img2 * img2, r)
    var_img2 = corr_img2 - mean_img2 * mean_img2

    #Compute local weights in each window Lambda_w
    local_weights = var_img1 / (var_img1 + var_img2 + 1e-7)

    #Compute final metric Qy
    Qy1 = np.maximum(ssim_1f, ssim_2f)
    Qy2 = local_weights * ssim_1f + (1 - local_weights) * ssim_2f
    comparaison = (ssim_12 < 0.75)
    Qy = Qy1 * comparaison + Qy2 * (1 - comparaison)
    return np.mean(Qy)

def structural_similarity_metric_colored(img1, img2, fusion, window_size): #Not working yet
    #TODO : try with color images
    ssim_1f = ssim(img1, fusion, win_size=window_size, full=True, channel_axis=2)[1]
    ssim_2f = ssim(img2, fusion, win_size=window_size, full=True, channel_axis=2)[1]
    ssim_12 = np.mean(ssim(img1, img2, win_size=window_size, full=True, channel_axis=2)[1], axis=2)

    #Compute variance in each window
    r = window_size // 2
    mean_img1 = average_filter(img1, r)
    corr_img1 = average_filter(img1 * img1, r)
    var_img1 = corr_img1 - mean_img1 * mean_img1 #This is not the right computation for variance, but I don't know how to deal with (W, H, 3, 3) arrays
    # corr_img1 = np.zeros((img1.shape[0], img1.shape[1], 3, 3))  # (W, H, 3, 3)
    # for i in range(3):
    #     for j in range(3):
    #         corr_img1[:, :, i, j] = average_filter(img1[:, :, i] * img1[:, :, j], r)
    # var_img1 = corr_img1 - mean_img1[:,:,None,:] * mean_img1[:,:,:,None]

    mean_img2 = average_filter(img2, r)
    corr_img2 = average_filter(img2 * img2, r)
    var_img2 = corr_img2 - mean_img2 * mean_img2 
    # corr_img2 = np.zeros((img2.shape[0], img2.shape[1], 3, 3))  # (W, H, 3, 3)
    # for i in range(3):
    #     for j in range(3):
    #         corr_img2[:, :, i, j] = average_filter(img2[:, :, i] * img2[:, :, j], r)
    # var_img2 = corr_img2 - mean_img2[:,:,None,:] * mean_img2[:,:,:,None]

    local_weights = var_img1 / (var_img1 + var_img2 + 1e-5)

    Qy1 = np.maximum(ssim_1f, ssim_2f)
    Qy2 = local_weights * ssim_1f + (1 - local_weights) * ssim_2f
    comparaison = (ssim_12 < 0.75)
    Qy = Qy1 * comparaison[:,:,None] + Qy2 * (1 - comparaison)[:,:,None]
    return np.mean(Qy)

def edge_strength(gray_img):
    """Step for Edge information metric: computes g array"""
    sobel_horizontal = sobel(gray_img,0)
    sobel_vertical = sobel(gray_img,1)
    return np.sqrt(sobel_horizontal**2 + sobel_vertical**2)

def edge_orientation(gray_img):
    """Step for Edge information metric: computes alpha array"""
    sobel_horizontal = sobel(gray_img,0)
    sobel_vertical = sobel(gray_img,1)
    return np.arctan(sobel_vertical / (sobel_horizontal + 1e-7)) 

def relative_strength(gray_img, gray_fusion):
    """Step for Edge information metric: computes G_AF array"""
    A = edge_strength(gray_img)
    F = edge_strength(gray_fusion)
    comparison_array = (A > F)
    return F/(A + 1e-7) * comparison_array + A/(F + 1e-7) * (1 - comparison_array)

def relative_orientation(gray_img, gray_fusion):
    """Step for Edge information metric: computes A_AF array"""
    A = edge_orientation(gray_img)
    F = edge_orientation(gray_fusion)
    return np.abs(np.abs(A - F) - (np.pi/2)) / (np.pi/2)

def edge_strength_preservation(gray_img, gray_fusion, gamma, kappa, sigma):
    """Step for Edge information metric: computes Qg_AF array"""
    G_AF = relative_strength(gray_img, gray_fusion)
    Qg_AF = gamma / (1 + np.exp(kappa * (G_AF - sigma)))
    return Qg_AF

def edge_orientation_preservation(gray_img, gray_fusion, gamma, kappa, sigma):
    """Step for Edge information metric: computes Qo_AF array"""
    A_AF = relative_orientation(gray_img, gray_fusion)
    Qo_AF = gamma / (1 + np.exp(kappa * (A_AF - sigma)))
    return Qo_AF

def edge_information_metric(gray_img1, 
                            gray_img2, 
                            gray_fusion, 
                            gamma_g, 
                            gamma_o, 
                            kappa_g, 
                            kappa_o, 
                            sigma_g, 
                            sigma_o,
                            L):
    """
    Compute Edge Information Metric, as introduced in https://www.researchgate.net/publication/3381966_Objective_image_fusion_performance_measure

    Parameters:
        :gray_img1 (numpy.ndarray): A grayscale image.
        :gray_img2 (numpy.ndarray): A grayscale image.
        :gray_fusion (numpy.ndarray): A grayscale image, result of fusion between img1 and img2
        :gamma_g, gamma_o, kappa_g, kappa_o, sigma_g, sigma_o, L(float): hyperparameters, defined in the article
    """
    # Ensure the input images is not RGB
    if len(gray_img1.shape) != 2 or len(gray_img2.shape) != 2 or len(gray_fusion.shape) != 2:
        raise ValueError("Input images must be grayscale")
    # Ensure the input images have the same shape
    if gray_img1.shape != gray_img2.shape or gray_img1.shape != gray_fusion.shape or gray_img2.shape != gray_fusion.shape:
        raise ValueError("Input images must have the same shape")
    Qg_AF = edge_strength_preservation(gray_img1, gray_fusion, gamma_g, kappa_g, sigma_g)
    Qg_BF = edge_strength_preservation(gray_img2, gray_fusion, gamma_g, kappa_g, sigma_g)
    Qo_AF = edge_orientation_preservation(gray_img1, gray_fusion, gamma_o, kappa_o, sigma_o)
    Qo_BF = edge_orientation_preservation(gray_img2, gray_fusion, gamma_o, kappa_o, sigma_o)

    Q_AF = Qg_AF * Qo_AF
    Q_BF = Qg_BF * Qo_BF

    weight_A = np.power(edge_strength(gray_img1), L)
    weight_B = np.power(edge_strength(gray_img2), L)

    Q = np.sum(Q_AF * weight_A + Q_BF * weight_B)
    Q /= np.sum(weight_A + weight_B)
    return Q