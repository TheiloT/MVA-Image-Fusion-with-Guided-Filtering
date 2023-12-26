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


def guided_filter_with_colored_guide_v1(u, guide, r, eps):
    """u is one channel of the image to be filtered."""
    C = average_filter(np.ones(u.shape), r)  # to avoid image edges pb
    mean_u = average_filter(u, r) / C  # shape (w,h)
    mean_guide = np.zeros_like(guide)
    for i in range(guide.shape[2]):
        mean_guide[:, :, i] = average_filter(guide[:, :, i], r) / C

    corr_uguide = np.zeros_like(guide)
    for i in range(guide.shape[2]):
        corr_uguide[:, :, i] = average_filter(guide[:, :, i] * u, r) / C

    cov_uguide = corr_uguide - mean_guide * mean_u[:, :, None]  # shape (w,h,3)

    corr_guide = (guide[:, :, :, None] - mean_guide[:, :, :, None]) * (
        guide[:, :, None, :] - mean_guide[:, :, None, :]
    )  # shape (w,h,3,3)
    cov_guide = np.zeros_like(corr_guide)
    for i in range(guide.shape[2]):
        for j in range(guide.shape[2]):
            cov_guide[:, :, i, j] = average_filter(corr_guide[:, :, i, j], r) / C
    matU = eps * np.eye(3)

    # temp_alpha = cov_guide + matU[None,None,:,:]
    temp_alpha = np.linalg.inv(cov_guide + matU[None, None, :, :])
    alpha = np.einsum("whij,whj-> whi", temp_alpha, cov_uguide)  # shape (w,h,3)

    beta = mean_u - np.einsum("whi,whi->wh", alpha, mean_guide)  # shape (w,h)

    average_alpha = np.zeros_like(alpha)
    for i in range(alpha.shape[2]):
        average_alpha[:, :, i] = average_filter(alpha[:, :, i], r) / C
    average_beta = average_filter(beta, r)

    q = np.einsum("whi, whi -> wh", average_alpha, guide) + average_beta
    return q


def guided_filter_with_colored_guide_v2(u, guide, r, eps):
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
