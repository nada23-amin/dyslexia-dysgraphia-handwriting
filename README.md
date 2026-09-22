# Handwriting-Based Dyslexia and Dysgraphia Screening

Public reproducibility package for the hybrid handwriting-based screening framework described in the manuscript.

## Scope

This repository contains the main experimental pipelines only. The ablation study is a separate experiment and is intentionally not part of the main pipeline.

### Dyslexia pipeline

The pipeline follows the manuscript workflow:

1. Dataset audit and duplicate removal.
2. Handwriting preprocessing.
3. Train/validation/test split.
4. Training-only balancing, with the target automatically derived from the largest training class.
5. Lightweight CNN training.
6. Extraction of the CNN representation.
7. MLP, SVM, Random Forest, LightGBM, and XGBoost classifiers.
8. OOF stacking with Logistic Regression as the meta-learner.

### Dysgraphia pipeline

The pipeline is independent from the dyslexia pipeline and follows the manuscript workflow:

1. Dataset collection and SHA-256 duplicate removal.
2. Image-level train/validation/test split.
3. Training-only resampling and augmentation.
4. Handwriting preprocessing.
5. Lightweight CNN representation learning.
6. CNN feature extraction.
7. Classical classifiers.
8. OOF stacking with Logistic Regression as the meta-learner.

Validation and independent test data are not resampled or augmented.

## Dataset locations

The datasets are intentionally not included in this repository.

Dyslexia default location:

`data/raw/Corrected/`
`data/raw/Normal/`
`data/raw/Reversal/`

Dysgraphia default location:

`data/dysgraphia_raw/Low Potential Dysgraphia/`
`data/dysgraphia_raw/Potential Dysgraphia/`

The dataset locations can be overridden with:

- `DYSLEXIA_DATA_DIR`
- `DYSGRAPHIA_DATA_DIR`

## Running

Install the dependencies from `requirements.txt`.

Run the complete main framework:

`python run_pipeline.py`

Run only Dyslexia:

`python run_pipeline.py --dyslexia`

Run only Dysgraphia:

`python run_pipeline.py --dysgraphia`

## Reproducibility

A fixed random seed is used throughout the pipelines where applicable. Generated models, features, splits, and results are written under `outputs/` and are not tracked by Git.

## Notes

- No raw/private dataset is distributed with the repository.
- No expected numerical outputs are hard-coded in this README.
- The code is intended to document and reproduce the methodology described in the manuscript rather than to distribute the research dataset.
