# Human Activity Recognition with Smartphone Sensors

Classifying six everyday activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) from smartphone accelerometer and gyroscope data, using the **UCI HAR** dataset. The project compares **classical machine learning** models trained on 561 engineered features against **LSTM** networks trained on the raw sensor signals.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Notebook](https://img.shields.io/badge/notebook-Jupyter-orange)

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

Human Activity Recognition (HAR) is used in fitness tracking, health monitoring and context-aware apps. This project:

1. Loads and cleans the UCI HAR dataset (duplicate and null checks).
2. Performs exploratory data analysis: activity distribution per subject, acceleration-magnitude separation, gravity-angle features, and t-SNE visualisation.
3. Trains and tunes five classical models with `GridSearchCV`.
4. Trains three LSTM architectures on raw inertial signals.
5. Compares every model on accuracy, precision, recall, F1 and confusion matrices.

## Dataset

**UCI Human Activity Recognition Using Smartphones** (Anguita et al., 2013).

- 30 volunteers (aged 19-48) wearing a waist-mounted Samsung Galaxy S II
- Triaxial accelerometer and gyroscope sampled at 50 Hz
- Signals cut into 2.56 s windows with 50% overlap (128 time steps)
- **561 engineered features** per window (time and frequency domain)
- 7,352 training windows and 2,947 test windows (split by subject)
- 6 classes: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`

The dataset is **not included** in this repo. Download it from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
and extract it so the folder is named `UCI_HAR_Dataset/` in the project root.

## Project Structure

```
.
├── MLPROJECT1.ipynb        # Full analysis: EDA, classical ML, LSTM, comparison
├── UCI_HAR_Dataset/        # Downloaded dataset (not tracked by git)
├── images/                 # Saved plots used in this README (optional)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create an environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
curl -L -o har.zip "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
unzip har.zip
unzip "UCI HAR Dataset.zip"
mv "UCI HAR Dataset" UCI_HAR_Dataset
```
(If the archive layout differs, just make sure the final folder is `UCI_HAR_Dataset/` containing `train/`, `test/`, `features.txt` and `activity_labels.txt`.)

### 4. Run the notebook
```bash
jupyter notebook MLPROJECT1.ipynb
```
Run all cells top to bottom. The LSTM section is faster with a GPU but runs fine on CPU.

## Methodology

### Data preparation
- Feature names from `features.txt` are attached to `X_train` / `X_test`; subject IDs and activity labels are merged in.
- Duplicate rows and missing values are checked for in both splits.
- Duplicate feature names are made unique before model fitting.

### Exploratory analysis
- Per-subject activity counts and overall class balance
- KDE and box plots of `tBodyAccMag-mean()` show that **static** (sitting, standing, laying) and **dynamic** (walking variants) activities separate cleanly
- `angle(X,gravityMean)` separates **laying** from every other activity
- t-SNE at perplexities 2, 5, 10, 20 and 50 shows clear clusters, with sitting and standing overlapping

### Classical models (561 features, `GridSearchCV`)
| Model | Tuned parameters |
|---|---|
| Logistic Regression | `C`, L2 penalty |
| Linear SVC | `C` |
| RBF SVM | `C`, `gamma` |
| Decision Tree | `max_depth` |
| Random Forest | `n_estimators`, `max_depth` |

### LSTM models (raw signals, 128 steps x 9 channels)
Inputs: total acceleration, body acceleration and body gyroscope (x, y, z each). A validation set is carved from the training data with `GroupShuffleSplit` **by subject** (so no subject appears in both train and validation), with `EarlyStopping` on validation loss.

| Model | Architecture |
|---|---|
| LSTM 1-layer | LSTM(32) -> Dropout(0.5) -> Dense(6) |
| LSTM 2-layer (48, 32) | LSTM(48) -> BatchNorm -> Dropout -> LSTM(32) -> Dropout -> Dense(6) |
| LSTM 2-layer (64, 48) | LSTM(64) -> BatchNorm -> Dropout -> LSTM(48) -> Dropout -> Dense(6) |

## Results

Evaluated on the official UCI HAR test set (2,947 samples). Precision, recall and F1 are macro-averaged.

| Model | Accuracy % | Precision % | Recall % | F1 % | Test loss |
|---|---|---|---|---|---|
| Logistic Regression | 96.54 | 96.71 | 96.48 | 96.54 | - |
| **Linear SVC** | **96.67** | **96.96** | **96.63** | **96.70** | - |
| RBF SVM | 96.27 | 96.43 | 96.14 | 96.23 | - |
| Decision Tree | 86.22 | 86.25 | 85.86 | 85.93 | - |
| Random Forest | 91.45 | 91.74 | 91.02 | 91.18 | - |
| LSTM 1-layer (32) | 88.56 | 88.96 | 88.46 | 88.53 | 0.420 |
| LSTM 2-layer (48, 32) | 92.16 | 92.56 | 92.22 | 92.17 | 0.232 |
| LSTM 2-layer (64, 48) | 88.77 | 88.94 | 88.97 | 88.84 | 0.323 |

> LSTM results vary slightly between runs because of random initialisation, so small gaps between LSTM variants should not be over-interpreted.

## Key Findings

- **Linear SVC is the best model (96.67%)**, narrowly ahead of Logistic Regression and RBF SVM. Linear models do very well because the engineered features are already highly informative.
- The **best LSTM (92.16%)** works from raw signals with no feature engineering, beating the Decision Tree and Random Forest but not the linear and kernel models.
- **Laying** is classified almost perfectly by every strong model; the hardest pairs are **sitting vs standing** and, for classical models, **walking upstairs vs downstairs**.
- A deeper LSTM is not automatically better: the (64, 48) model did worse than the (48, 32) model.

## Limitations and Future Work

- Try 1D-CNN, CNN-LSTM or bidirectional LSTM models on the raw signals.
- Add more hyperparameter tuning and multiple seeds with mean +/- std for the LSTMs.
- Cross-validate by subject for the classical models to better estimate generalisation to new people.
- Look at feature selection / importance to find a smaller feature set.
- Export the best model and wrap it in a small demo app (Streamlit / Flask).

## Acknowledgements

Dataset: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
*A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013.

## License

Code in this repository is released under the [MIT License](LICENSE). The UCI HAR dataset has its own terms; please see the UCI repository page.

## Author

**<Your Name>** - [GitHub](https://github.com/<your-username>) - [LinkedIn](https://linkedin.com/in/<your-handle>)
