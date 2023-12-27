import numpy as np


# Numpy implementation
def average_filter(u, r):
    """Average filter on image u with a square (2*r+1)x(2*r+1) window using integral images.
    Credit for this functions goes to Julie Delon, Lucía Bouza and Joan Alexis Glaunès.
        :param u: 2D image to be filtered.
        :param r: Radius of the filter.
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


def average_filter_multichannel(u, r):
    """Average filter on image u with a square (2*r+1)x(2*r+1) window using integral images.
    Credit for this functions goes to Julie Delon, Lucía Bouza and Joan Alexis Glaunès.
        :param u: Colored 2D image to be filtered.
        :param r: Radius of the filter.
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


def guided_filter(u, guide, r, eps):
    # Credit for these functions goes to Julie Delon, Lucía Bouza and Joan Alexis Glaunès
    """Guided filtering on image u using guide, filter radius is r and regularization parameter eps
    Credit for this functions goes to Julie Delon, Lucía Bouza and Joan Alexis Glaunès.
    """
    C = average_filter(np.ones(u.shape), r)  # to avoid image edges pb
    mean_u = average_filter(u, r) / C
    mean_guide = average_filter(guide, r) / C
    corr_guide = average_filter(guide * guide, r) / C
    corr_uguide = average_filter(u * guide, r) / C
    var_guide = corr_guide - mean_guide * mean_guide
    cov_uguide = corr_uguide - mean_u * mean_guide

    alph = cov_uguide / (var_guide + eps)
    beta = mean_u - alph * mean_guide

    mean_alph = average_filter(alph, r) / C
    mean_beta = average_filter(beta, r) / C

    q = mean_alph * guide + mean_beta
    return q


def guided_filter_with_colored_guide(u, guide, r, eps):
    C = average_filter(np.ones(u.shape), r)  # to avoid image edges pb
    mean_u = average_filter(u, r) / C  # (W, H)
    mean_guide = average_filter_multichannel(guide, r) / C[:, :, None]  # (W, H, 3)

    corr_guide = np.zeros((guide.shape[0], guide.shape[1], 3, 3))  # (W, H, 3, 3)
    for i in range(3):
        for j in range(3):
            corr_guide[:, :, i, j] = (
                average_filter(guide[:, :, i] * guide[:, :, j], r) / C
            )
    cov_guide = (
        corr_guide - mean_guide[:, :, None, :] * mean_guide[:, :, :, None]
    )  # (W, H, 3, 3)

    corr_uguide = np.zeros((guide.shape[0], guide.shape[1], 3))  # (W, H, 3)
    for i in range(3):
        corr_uguide[:, :, i] = average_filter(u * guide[:, :, i], r) / C
    cov_uguide = corr_uguide - mean_u[:, :, None] * mean_guide  # (W, H, 3)

    U = eps * np.eye(3)

    alpha = np.einsum(
        "mnij, mnj -> mni",
        np.linalg.inv(cov_guide + U[None, None, :, :]),
        cov_uguide,
    )
    beta = mean_u - np.sum(alpha * mean_guide, axis=2)

    mean_alpha = average_filter_multichannel(alpha, r) / C[:, :, None]
    mean_beta = average_filter(beta, r) / C[:, :]

    return np.sum(mean_alpha * guide, axis=-1) + mean_beta


def guided_filter_with_colored_guide_slow(u, guide, r, eps):
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    image_folder = os.path.join("dataset", "guided_filter_tests")
    imrgb1 = plt.imread(os.path.join(image_folder, "renoir.jpg")) / 255
    imrgb2 = plt.imread(os.path.join(image_folder, "gauguin.jpg")) / 255
    noisy_imrgb1 = imrgb1 + np.random.normal(0, 0.2, imrgb1.shape)

    # useful if the image is a png with a transparency channel
    imrgb1 = imrgb1[:, :, 0:3]
    imrgb2 = imrgb2[:, :, 0:3]
    guided_filter_with_colored_guide(noisy_imrgb1[:, :, 0], imrgb1, 20, 1e-4)
