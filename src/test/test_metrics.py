import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import normalized_mutual_information

from ..utils import entropy, joint_entropy


image_folder = os.path.join("dataset")
im1 = plt.imread(os.path.join(image_folder, "multi-focus", "grayscale", "g_01_1.tif")) / 255
im2 = plt.imread(os.path.join(image_folder, "multi-focus", "grayscale", "g_01_2.tif")) / 255


class TestMettrics:
    def test_normalized_mutual_information(self):
        entropy1 = entropy(im1)
        entropy2 = entropy(im2)
        mutual_info = entropy1 + entropy2 - joint_entropy(im1, im2)
        nmi = 2*mutual_info/(entropy1+entropy2)
        assert (normalized_mutual_information(im1, im2) - nmi) < 1e-5
