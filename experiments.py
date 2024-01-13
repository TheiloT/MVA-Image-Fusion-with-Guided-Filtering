import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import rgb2gray
from src.fusion import fuse_images
from src.metrics import normalized_mutual_information_metric, structural_similarity_metric, edge_information_metric

@hydra.main(config_path="experiments", config_name="config", version_base="1.2")
def launch_experiment(config: DictConfig):
    """
    Launch an experiment on a pair of images.

    Parameters:
        :img_pair_nb (int): The number of the image pair to use within the color multi-focus dataset.
        :parameter (str): The name of the parameter to change.
        :value (float): The value of the parameter to use.
    """
    print(OmegaConf.to_yaml(config))
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    print(f"Original working directory  : {hydra.utils.get_original_cwd()}")
        
    r1, eps1, r2, eps2 = 45, 0.3, 7, 1e-6
    if config["parameter"] == "r1":
        r1 = config["value"]
    elif config["parameter"] == "eps1":
        eps1 = config["value"]
    elif config["parameter"] == "r2":
        r2 = config["value"]
    elif config["parameter"] == "eps2":
        eps2 = config["value"]
    else:
        raise ValueError("Invalid parameter name")
    
    # Load images
    im1 = plt.imread(os.path.join(hydra.utils.get_original_cwd(), f"dataset/multi-focus/color/c_{config['image']:02d}_1.tif"))
    im2 = plt.imread(os.path.join(hydra.utils.get_original_cwd(), f"dataset/multi-focus/color/c_{config['image']:02d}_2.tif"))

    # Fuse images and save result
    fused_image = fuse_images(
        [im1, im2],
        r1=r1,
        eps1=eps1,
        r2=r2,
        eps2=eps2,
        verbose=False,
        savefigs=True
    )
    
    # Save metrics
    nmi = normalized_mutual_information_metric(rgb2gray(im1), rgb2gray(im2), rgb2gray(fused_image))
    smm = structural_similarity_metric(rgb2gray(im1), rgb2gray(im2), rgb2gray(fused_image), 7)
    # Hyperparameters for edge information metric chosen as in the article https://www.researchgate.net/publication/3381966_Objective_image_fusion_performance_measure
    # In our studied article, the authors do not specify the values of hyperparameters
    gamma_g=0.9994
    gamma_o=0.9879
    kappa_g=-15
    kappa_o=-22
    sigma_g=0.5
    sigma_o=0.8
    L = 1.5
    eim = edge_information_metric(rgb2gray(im1), rgb2gray(im2), rgb2gray(fused_image), gamma_g, gamma_o, kappa_g, kappa_o, sigma_g, sigma_o, L)
    pickle.dump({"nmi": nmi, "smm": smm, "eim": eim}, open(f"metrics.pkl", "wb"))

def average_metric(metric, parameter, value):
    """ 
    Compute the average metric for a given parameter and value.
    """
    exp_path = os.path.join("experiments/results", parameter, str(value))
    metrics = []
    for image in range(1,11):
        with open(os.path.join(exp_path, f"c_{int(image)}", "metrics.pkl"), "rb") as pickle_file:
            metric_storage = pickle.load(pickle_file)
        metrics.append(metric_storage[metric])
    return np.mean(metrics), np.std(metrics)
    
def plot_metric(parameter, values, save=False):
    """
    Plot the metrics for a given parameter and given values.
    """
    markers = {"nmi": 'o', "smm": 'x', "eim": '*'}
    colors = {"nmi": 'r', "smm": 'g', "eim": 'b'}
    parameter_display = {"r1": r"$r_1$", "eps1": r"$\varepsilon_1$", "r2": r"$r_2$", "eps2": r"$\varepsilon_2$"}
    metric_display = {"nmi": "NMI", "smm": "SS", "eim": "EI"}
    with sns.axes_style("darkgrid"):
        for metric in ["nmi", "smm", "eim"]:
            means, stds = [], []
            for value in values:
                mean, std = average_metric(metric, parameter, value)
                means.append(mean)
                stds.append(std)
            plt.plot(values, means, marker=markers[metric], color=colors[metric], label=metric_display[metric])
            plt.fill_between(values, np.array(means)-np.array(stds), np.array(means)+np.array(stds), color=colors[metric], alpha=0.2)
            plt.xlabel(parameter_display[parameter])
        plt.xscale("log")
        plt.ylabel("Metric")
        plt.legend()
        plt.title("Average metrics for " + parameter_display[parameter])
        if save:
            plt.savefig(os.path.join("experiments/plots", f"plot_metrics_{parameter}.png"))
        plt.show()

if __name__ == "__main__":
    launch_experiment()
    # plot_metric("r1", [1, 3, 5, 10, 20, 30, 50, 70, 100, 300])
    # plot_metric("r2", [1, 3, 5, 10, 20, 30, 50, 70, 100, 300])
    # plot_metric("eps1", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 10])
    # plot_metric("eps2", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 10])