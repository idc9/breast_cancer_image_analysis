# Joint and individual analysis of breast cancer histologic images and genomic covariates

The code in this repository reproduces the analysis from [Joint and individual analysis of breast cancer histologic images and genomic covariates](https://arxiv.org/abs/1912.00434) using the data from the Carolina Breast Cancer Study, phase 3. Due to patient confidentiality we cannot publicly release the raw data, however, researchers may request permission to access the raw data used in this study by visiting
[https://unclineberger.org/cbcs/for-researchers/](https://unclineberger.org/cbcs/for-researchers/).

Supplementary figures from the paper may be downloaded from [this online archive](https://marronwebfiles.sites.oasis.unc.edu/AJIVE-Hist-Gene/) (note this file is about 1.5Gb).

# Instructions to run the code

### 1. Setup data directories

cbcs_joint/Paths.py has instructions for setting up the data directory once the data has been provided by the CBCS steering committee.

### 2. Install code

Download the github repository,
```
git clone https://github.com/idc9/breast_cancer_image_analysis.git
```
Change the folder path in cbcs_joint/Paths.py to match the data directories on your computer.

Using using python 3.7.2. (e.g. `conda create -n cbcs_joint python=3.7.2`, `conda activate cbcs_joint`) install the package

```
cd cbcs_joint/
pip install .
```

### 3. Image patch feature extraction

```
python scripts/patch_feat_extraction.py
```

This step extracts CNN features from each image patch and may take a few hours. If a GPU is available it will automatically be used. The resulting patch features csv file is about 3.6 Gb.

### 4. AJIVE analysis

```
python scripts/ajive_analysis.py
```

The AJIVE analysis runs in about 30 seconds, but the whole script may take a while due to data loading and saving large figures.

### 5. Image visualizations

```
python image_visualizations.py
```

This may take a couple of hours and the resulting saved figures are a couple of gigabytes.

# Help and support

If you have any questions please reach out to [Iain Carmichael](https://idc9.github.io/) (idc9@uw.edu).