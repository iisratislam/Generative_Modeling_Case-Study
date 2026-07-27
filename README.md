# GAN Case Study — Generative Modelling

Coursework exploring Generative Adversarial Networks (GANs), from building them
from scratch on 2D data through to three real-world applications in medicine,
cybersecurity, and creative AI.

## Overview

- **Part 1** — Building and understanding GANs from scratch (PyTorch).
- **Part 2** — Three real-world GAN applications, each with an extension.

All notebooks are in `notebooks/` and can be opened directly in Google Colab.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `part1_sine_wave_gan.ipynb` | Reproduces the tutorial sine-wave GAN (vanilla fully-connected GAN). |
| `part1_spiral_gan.ipynb` | Models a 2D spiral. Vanilla GAN mode-collapses, so uses **WGAN-GP** with gradient penalty for stable coverage. |
| `part1_architecture_comparison.ipynb` | Compares generator **depth** (shallow vs deep) under fixed WGAN-GP training. |
| `part2_octmnist_gan.ipynb` | **DCGAN** on OCTMNIST retinal images. Includes FID evaluation and a **conditional GAN (cGAN)** extension for per-class generation. |
| `part2_cicids_gan.ipynb` | **Tabular GAN** generating synthetic CICIDS 2017 network-traffic feature vectors. PCA + t-SNE distribution comparison. |
| `part2_quickdraw_gan.ipynb` | **DCGAN** on QuickDraw 'birthday cake' sketches, with FID evaluation and a **cat category** extension comparing sketch complexity. |

## Part 1 — GANs from Scratch

1. **Sine-wave GAN** — reproduction of the tutorial GAN on noisy sine data.
2. **2D spiral (WGAN-GP)** — a harder, self-overlapping manifold. A vanilla GAN
   collapses to a single mode; switching to WGAN-GP (with `n_critic=2` and
   best-coverage checkpointing) produces stable coverage of the full spiral.
3. **Architecture comparison** — a single-variable study of generator depth.
   Across seeds, the shallow generator covered the spiral better than the deep
   one, which trained less stably — illustrating that model capacity should
   match task complexity.

## Part 2 — Real-World Applications

### 2.1 OCTMNIST retinal images (DCGAN)
A DCGAN generates synthetic retinal OCT images (28×28 grayscale).
Stable training, convincing outputs, and **FID ≈ 33.6**. Extension: a
conditional GAN that generates images for a specified class on demand.

### 2.2 CICIDS 2017 network traffic (tabular GAN)
A fully-connected GAN generates synthetic traffic feature vectors from the
Wednesday capture (BENIGN + DoS). Very stable training; **PCA and t-SNE both
show strong overlap** between real and synthetic distributions.

### 2.3 QuickDraw birthday cake sketches (DCGAN)
A DCGAN generates birthday-cake doodles. Recognisable outputs with good
variety; **FID ≈ 39.1**. Extension: training on the 'cat' category to compare
how sketch complexity affects generation quality and training stability.

## Tech Stack

- Python, PyTorch
- NumPy, Matplotlib, scikit-learn (PCA, t-SNE)
- `medmnist` (OCTMNIST), `pytorch-fid` (FID evaluation)
- Trained on Google Colab (T4 GPU)

## Datasets

- **OCTMNIST** — via the `medmnist` package (downloads automatically).
- **CICIDS 2017** — Wednesday file from the
  [Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
  on Kaggle (download required; not included in this repo).
- **QuickDraw** — 'birthday cake' and 'cat' bitmap categories from the
  [Google QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset)
  (downloaded in-notebook).

## References

- Goodfellow et al. (2014), *Generative Adversarial Networks*.
- Radford et al. (2016), *DCGAN*.
- Arjovsky et al. (2017), *Wasserstein GAN*.
- Gulrajani et al. (2017), *Improved Training of Wasserstein GANs* (WGAN-GP).
