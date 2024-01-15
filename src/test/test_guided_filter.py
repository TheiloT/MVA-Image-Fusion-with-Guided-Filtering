import os
import matplotlib.pyplot as plt
import numpy as np

from ..guided_filter import (
    guided_filter_with_colored_guide,
    guided_filter_with_colored_guide_slow,
)


image_folder = os.path.join("dataset", "guided_filter_tests")
imrgb1 = plt.imread(os.path.join(image_folder, "kandinsky.jpg")) / 255
noisy_imrgb1 = imrgb1 + np.random.normal(0, 0.2, imrgb1.shape)
imrgb1 = imrgb1[:, :, 0:3]


class TestGuidedFilter:
    def test_guided_filter(self):
        sub_imrgb1 = imrgb1[300:310, 300:310, :]
        noisy_sub_imrgb1 = noisy_imrgb1[300:310, 300:310, :]
        r = 2
        eps = 0.01
        out_slow = np.zeros((10, 10, 3))
        out = np.zeros((10, 10, 3))
        for i in range(3):
            out_slow[:, :, i] = guided_filter_with_colored_guide_slow(
                noisy_sub_imrgb1[:, :, i], sub_imrgb1, r, eps
            )
            out[:, :, i] = guided_filter_with_colored_guide(
                noisy_sub_imrgb1[:, :, i], sub_imrgb1, r, eps
            )
        assert out_slow.shape == out.shape
        assert np.allclose(out_slow, out, atol=1e-5)
