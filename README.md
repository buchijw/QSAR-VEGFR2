

# Quantitative structure-activity relationship model for VEGFR-2 inhibitors (QSAR-VEGFR2)

Discovery of VEGFR-2 Inhibitors employing Junction Tree Variational Encoder with Local Latent Space Bayesian Optimization and Gradient Ascent Exploration

![GitHub issues](https://img.shields.io/github/issues/buchijw/QSAR-VEGFR2?style=for-the-badge)
![GitHub pull requests](https://img.shields.io/github/issues-pr/buchijw/QSAR-VEGFR2?style=for-the-badge)
![License](https://img.shields.io/github/license/buchijw/QSAR-VEGFR2?style=for-the-badge)
![Git LFS](https://img.shields.io/badge/GIT%20LFS-8A2BE2?style=for-the-badge)

<!-- TABLE OF CONTENTS -->

<details open>
  <summary><h3>Table of Contents</h3></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This repository contains library files, data, and notebooks for building a quantitative structure-activity relationship (QSAR) model that predicts molecules' $pIC_{50}$ values of Vascular Endothelial Growth Factor Receptor 2 (VEGFR-2) inhibiting activity.

The model is a part of the paper **"Discovery of VEGFR-2 Inhibitors employing Junction Tree Variational Encoder with Local Latent Space Bayesian Optimization and Gradient Ascent Exploration"**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REQUIREMENTS -->

## Requirements

This git repo requires Git LFS installed for large files. To clone this repo, please run:

```bash
git lfs install
git clone https://github.com/buchijw/QSAR-VEGFR2.git
```

We tested on Linux (Ubuntu 22.04 LTS with ROCm 6.0, CUDA 12.1) and MacOS (Sonoma 14.5 with Intel CPU), therefore the code supposes to work on CUDA/HIP/CPU based on type of PyTorch installed.

Packages (versions in brackets were used):

* `Python` (3.11.5)
* `RDKit` (2024.03.3)
* `PyTorch` (2.1.0 with ROCm 6.0)
* `scikit-learn`
* `scikit-posthocs`
* `scipy`
* `py-xgboost` (XGBoost)
* `catboost` (CatBoost)
* `imbalanced-learn`
* `mhfp` (for calculating SECFP fingerprints)
* [`cats2d`](https://github.com/iwatobipen/CATS2D.git) (for CATS2D fingerprints)
* [`tmap`](https://github.com/reymond-group/tmap.git) and [`map4`](https://github.com/reymond-group/map4) (for MAP4 fingerprints)
* [`mol2vec`](https://github.com/guyrosin/mol2vec) (for Mol2vec fingerprints)
* `mordredcommunity` (for Mordred fingerprints)
* `scopy` (for molecular filters)
* `tqdm`
* `networkx`
* `joblib`
* `argparse`
* `pandas` (2.1.4)
* `numpy` (1.26.3)

Since some packages need to be manually built, we also provide environment preparation scripts for CUDA/ROCm/CPU/MacOS platforms in the [env_prepare](env_prepare) folder. You can either follow the installation guide for each package or use our scripts. To use the scripts, make sure to carefully take a look at their content, change the `CC` and `CXX` paths to your compiler, and activate your environment first. RUN SCRIPT FILES AT YOUR OWN RISK. We are not responsible for causing damage to your devices.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->

## Quick Start

The following directories and files contains the implementations of the model:

* [`data/`](data) contains data for model building and SMILES entries of FDA-approved drugs for prediction testing.
* [`raw_data_features/VEGFR2/pipeline/`](raw_data_features/VEGFR2/pipeline/) contains necessary files for our built QSAR model.
* [`utils/`](utils/) contains library files.
* [`1_feature_engineering.ipynb`](1_feature_engineering.ipynb) contains codes for generating and analysing molecular representations.
* [`2_optimization.ipynb`](2_optimization.ipynb) contains codes for building and evaluating QSAR models.
* [`3_external_compare.ipynb`](3_external_compare.ipynb) contains codes for further comparing the five top-performing models.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contributing

- [Gia-Bao Truong](https://github.com/buchijw/)
- Thanh-An Pham
- Van-Thinh To
- Hoang-Son Lai Le
- Phuoc-Chung Van Nguyen
- [The-Chuong Trinh](https://trinhthechuong.github.io)
- [Tieu-Long Phan](https://tieulongphan.github.io/)
- Tuyen Ngoc Truong<sup>*</sup>

<sup>*</sup>Corresponding author

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

This work has received support from the Korea International Cooperation Agency (KOICA) under the project entitled "Education and Research Capacity Building Project at University of Medicine and Pharmacy at Ho Chi Minh City," conducted from 2024 to 2025 (Project No. 2021-00020-3).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
