# MVA-Image-Fusion-with-Guided-Filtering
This repository contains an implementation of the article [Image Fusion with Guided Filtering](https://xudongkang.weebly.com/uploads/1/6/4/6/16465750/tip1.pdf) - Shutao Li et al.

It contains a source code for the Guided Filter and the proposed fusion method, along with 4 notebooks to perform experiments. The file ``experiments.py`` allows to run the quantitative experiments, which run the algorithm for various parameters on 10 coloured multi-source images and average the resulting metrics.

In addition to the pair ``cathedral1.jpg`` and ``cathedral2.jpg`` and to the paintings of Kandinsky and Gaugin, 3 datasets are included:
- The Stanford Memorial Church radiance map by Paul E. Debevec and Jitendra Malik;
- The Lytro multi-focus image dataset by Dr. Mansour Nejati et al;
- A fragment of the TNO Multiband Image dataset by Alexander Toet.
