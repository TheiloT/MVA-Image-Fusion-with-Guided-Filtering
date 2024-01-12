import numpy as np
import cv2


# Numpy implementation
def average_filter_no_edge(u, r):
    """
    Average filter on image u with a square (2*r+1)x(2*r+1) window using integral images.
    Credit for this functions goes to Julie Delon, Lucía Bouza and Joan Alexis Glaunès.

    Parameters:
        :u (numpy.ndarray): The input image.
        :r (int): The filter radius.

    Returns:
        numpy.ndarray: The filtered image.
    """
    (nrow, ncol) = u.shape
    big_uint = np.zeros((nrow + 2 * r + 1, ncol + 2 * r + 1))
    big_uint[r + 1 : nrow + r + 1, r + 1 : ncol + r + 1] = u
    big_uint = np.cumsum(np.cumsum(big_uint, 0), 1)  # integral image

    out = (
        big_uint[2 * r + 1 : nrow + 2 * r + 1, 2 * r + 1 : ncol + 2 * r + 1]
        + big_uint[0:nrow, 0:ncol]
        - big_uint[0:nrow, 2 * r + 1 : ncol + 2 * r + 1]
        - big_uint[2 * r + 1 : nrow + 2 * r + 1, 0:ncol]
    )
    out = out / (2 * r + 1) ** 2

    return out


def average_filter_multichannel_no_edge(u, r):
    """
    Average filter on image u with a square (2*r+1)x(2*r+1) window using integral images.
    Credit for this functions goes to Julie Delon, Lucía Bouza and Joan Alexis Glaunès.

    Parameters:
        :u (numpy.ndarray): The input image.
        :r (int): The filter radius.

    Returns:
        numpy.ndarray: The filtered image.
    """
    (nrow, ncol) = u.shape[:2]
    big_uint = np.zeros((nrow + 2 * r + 1, ncol + 2 * r + 1, 3))
    big_uint[r + 1 : nrow + r + 1, r + 1 : ncol + r + 1, :] = u
    big_uint = np.cumsum(np.cumsum(big_uint, 0), 1)  # integral image

    out = (
        big_uint[2 * r + 1 : nrow + 2 * r + 1, 2 * r + 1 : ncol + 2 * r + 1]
        + big_uint[0:nrow, 0:ncol]
        - big_uint[0:nrow, 2 * r + 1 : ncol + 2 * r + 1]
        - big_uint[2 * r + 1 : nrow + 2 * r + 1, 0:ncol]
    )
    out = out / (2 * r + 1) ** 2
    
    return out


def average_filter(u, r):
    """
    Average filter on image u with a square (2*r+1)x(2*r+1) window using integral images.

    Parameters:
        :u (numpy.ndarray): The input image.
        :r (int): The filter radius.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if len(u.shape) == 2:
        C = average_filter_no_edge(np.ones(u.shape), r)  # to avoid image edges pb
        return average_filter_no_edge(u, r) / C
    else:
        C = average_filter_no_edge(np.ones(u.shape[:2]), r)  # to avoid image edges pb
        return average_filter_multichannel_no_edge(u, r) / C[:, :, None]


def guided_filter(u, guide, r, eps):
    """
    Guided filtering on image u using guide, filter radius is r and regularization parameter eps.
    Credit for this functions goes to Julie Delon, Lucía Bouza and Joan Alexis Glaunès.

    Parameters:
        :u (numpy.ndarray): One color channel for the input image.
        :guide (numpy.ndarray): The guide image.
        :r (int): The filter radius.
        :eps (float): The regularization parameter.

    Returns:
        numpy.ndarray: The filtered image.
    """
    mean_u = average_filter(u, r)
    mean_guide = average_filter(guide, r)
    corr_guide = average_filter(guide * guide, r)
    corr_uguide = average_filter(u * guide, r)
    var_guide = corr_guide - mean_guide * mean_guide
    cov_uguide = corr_uguide - mean_u * mean_guide

    alph = cov_uguide / (var_guide + eps)
    beta = mean_u - alph * mean_guide

    mean_alph = average_filter(alph, r)
    mean_beta = average_filter(beta, r)

    q = mean_alph * guide + mean_beta
    return q


def guided_filter_with_colored_guide(u, guide, r, eps):
    """
    Guided filtering on image u using colored guide, with filter radius r and regularization parameter eps.

    Parameters:
        :u (numpy.ndarray): One color channel for the input image.
        :guide (numpy.ndarray): The colored guide image.
        :r (int): The filter radius.
        :eps (float): The regularization parameter.

    Returns:
        numpy.ndarray: The filtered image.
    """
    mean_u = average_filter(u, r)  # (W, H)
    mean_guide = average_filter(guide, r)  # (W, H, 3)

    corr_guide = np.zeros((guide.shape[0], guide.shape[1], 3, 3))  # (W, H, 3, 3)
    for i in range(3):
        for j in range(3):
            corr_guide[:, :, i, j] = (
                average_filter(guide[:, :, i] * guide[:, :, j], r)
            )
    cov_guide = (
        corr_guide - mean_guide[:, :, None, :] * mean_guide[:, :, :, None]
    )  # (W, H, 3, 3)

    corr_uguide = np.zeros((guide.shape[0], guide.shape[1], 3))  # (W, H, 3)
    for i in range(3):
        corr_uguide[:, :, i] = average_filter(u * guide[:, :, i], r)
    cov_uguide = corr_uguide - mean_u[:, :, None] * mean_guide  # (W, H, 3)

    U = eps * np.eye(3)

    alpha = np.einsum(
        "mnij, mnj -> mni",
        np.linalg.inv(cov_guide + U[None, None, :, :]),
        cov_uguide,
    )
    beta = mean_u - np.sum(alpha * mean_guide, axis=2)

    mean_alpha = average_filter(alpha, r)
    mean_beta = average_filter(beta, r)

    return np.sum(mean_alpha * guide, axis=-1) + mean_beta


def guided_filter_with_colored_guide_slow(u, guide, r, eps):
    """
    Guided filtering on image u using colored guide, with filter radius r and regularization parameter eps (slow implementation).

    Parameters:
        u (numpy.ndarray): One color channel for the input image.
        guide (numpy.ndarray): The colored guide image.
        r (int): The filter radius.
        eps (float): The regularization parameter.

    Returns:
        numpy.ndarray: The filtered image.
    """
    (width, height) = u.shape[:2]
    alpha = np.zeros((width, height, 3))
    beta = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            mean_guide = np.zeros(3)
            mean_u = 0
            corr_uguide = np.zeros(3)
            corr_guide = np.zeros((3, 3))
            area = 0
            for k in range(-r, r + 1):
                for l in range(-r, r + 1):
                    if i + k >= 0 and i + k < width and j + l >= 0 and j + l < height:
                        mean_guide += guide[i + k, j + l]
                        mean_u += u[i + k, j + l]
                        corr_uguide += guide[i + k, j + l] * u[i + k, j + l]
                        corr_guide += np.outer(guide[i + k, j + l], guide[i + k, j + l])
                        area += 1
            mean_guide /= area
            mean_u /= area
            corr_uguide /= area
            corr_guide /= area
            cov_uguide = corr_uguide - mean_guide * mean_u
            cov_guide = corr_guide - np.outer(mean_guide, mean_guide)
            matU = eps * np.eye(3)
            temp_alpha = np.linalg.inv(cov_guide + matU)
            alpha[i, j] = np.dot(temp_alpha, cov_uguide)
            beta[i, j] = mean_u - np.dot(alpha[i, j], mean_guide)
    alpha_mean = np.zeros((width, height, 3))
    beta_mean = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            area = 0
            for k in range(-r, r + 1):
                for l in range(-r, r + 1):
                    if i + k >= 0 and i + k < width and j + l >= 0 and j + l < height:
                        alpha_mean[i, j] += alpha[i + k, j + l]
                        beta_mean[i, j] += beta[i + k, j + l]
                        area += 1
            alpha_mean[i, j] /= area
            beta_mean[i, j] /= area
    q = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            q[i, j] = np.dot(alpha_mean[i, j], guide[i, j]) + beta_mean[i, j]
    return q
