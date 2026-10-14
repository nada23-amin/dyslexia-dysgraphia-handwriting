# Handwriting-Based Classification of Dyslexia-Related Patterns and Dysgraphia

This repository provides the implementation for two independent handwriting-image classification pipelines: dyslexia-related handwriting-pattern classification (Normal, Reversal, Corrected) and binary dysgraphia-related pattern classification (Potential Dysgraphia, Low Potential Dysgraphia).

## Method overview

- Exact-image duplicate detection using SHA-256 before data partitioning.
- Image-level training, validation, and test partitions, with a nominal 70%/15%/15% split and random seed 42.
- Preprocessing using grayscale conversion, CLAHE, Non-Local Means denoising, adaptive Gaussian thresholding, Hough-based deskewing, resizing to 128×128, and downsampling to 28×28.
- Augmentation and class balancing restricted to the training subset; validation and test samples are not augmented or resampled.
- The primary CNN feature extractor is trained for 20 epochs with the validation split passed to `model.fit`. `ReduceLROnPlateau` monitors validation loss (patience 3 for dyslexia and patience 2 for dysgraphia); it can reduce the learning rate but does not stop training early. The test split remains held out from training and model selection.
- Lightweight CNN feature extractor with a 128-dimensional embedding, followed by MLP, SVM, Random Forest, XGBoost, and LightGBM classifiers.
- Five-fold stratified out-of-fold probability features for stacking, with Logistic Regression as the meta-learner. The CNN is trained in the feature-extraction stage; five-fold cross-validation is then applied to the fixed 128-dimensional embeddings, as described in the manuscript.
- Analysis scripts for grouped SHAP explanations, paired exact McNemar tests, bootstrap confidence intervals, end-to-end CNN baselines, ablation experiments, and Smooth Grad-CAM visualizations.



Within each stacking fold, the MLP is trained for a fixed 50 epochs using the fold-training partition, batch size 64, and Adam learning rate 0.001. No EarlyStopping callback is used in this stacking-fold training. The outer held-out fold is reserved exclusively for OOF prediction generation.

For stacking OOF generation, the CNN embeddings are generated in the dedicated feature-extraction stage and then held fixed. The five-fold procedure fits downstream base classifiers on four folds and predicts the held-out fold; fold-level predictions are used to construct meta-training features. The external test set is not used for fitting or model selection. The stacking fold assignment follows stratified five-fold cross-validation. Because augmented samples may be derived from the same source image, this fold protocol should be interpreted together with the manuscript limitations and should not be described as participant-level validation.

## Relationship between standalone classifiers and stacking

The standalone scripts `src/05_train_mlp.py`, `src/06_train_svm.py`,
`src/07_train_random_forest.py`, `src/09_train_xgboost.py`, and
`src/08_train_lightgbm.py` train and evaluate each classifier independently
on the same saved CNN feature representations. Their purpose is to provide
the individual-classifier results used for the comparative/ablation analysis
and to save each model's own predictions and evaluation artifacts.

`src/10_stacking.py` is a separate ensemble-training procedure, not a loader
or wrapper around the saved standalone models. It reinitializes and trains
the five base learners within a five-fold stratified cross-validation loop
on the training feature set. For each training sample, it records the
probabilities predicted when that sample belongs to the held-out fold; these
out-of-fold (OOF) probabilities form the meta-training features for the
Logistic Regression meta-learner. Probabilities for the validation and test
sets are averaged across the five fold-specific models before being passed
to the meta-learner. This separates the purpose of the standalone evaluation
from the OOF-based ensemble fitting and avoids training the meta-learner on
in-sample predictions from base learners fitted to those same samples.

Consequently, the base learners are intentionally fitted again inside the
stacking workflow. The standalone model files are not reused by stacking,
and the standalone scores should not be interpreted as the fold-level scores
or as the stacking model's final performance. Both workflows use the same
CNN feature files and broadly corresponding classifier configurations, while
their training procedures differ to support their distinct evaluation goals.


For the dyslexia McNemar analysis, run the five standalone classifier scripts as well as `src/10_stacking.py` before running `src/12_dyslexia_mcnemar.py`. The script reads the true labels from `outputs/features/y_test.npy`, Stacking meta-features from `outputs/results/stacking/X_meta_test.npy`, and the fitted meta-learner from `outputs/results/stacking/meta_learner_logistic_regression.pkl`. For each comparator, it reads the saved test predictions from the corresponding standalone output directory (`svm`, `mlp`, `random_forest`, `xgboost`, or `lightgbm`). It then compares correctness on the same held-out test images using exact two-sided McNemar tests. Each classifier writes a `test_sample_manifest.json` containing the ordered test-image paths and hashes of the path sequence and test labels. The McNemar script refuses to run if any model manifest is missing or differs, rather than assuming that matching array lengths prove sample identity. Rerun feature extraction, all standalone classifiers, and stacking from one clean split before running the test. Fold-specific predictions used to train the stacking meta-learner are not used as the standalone comparators.

## Dataset layout

Place the dyslexia-related handwriting classes under:

```text
data/raw/Normal/
data/raw/Reversal/
data/raw/Corrected/
```

Place the dysgraphia classes under:

```text
data/dysgraphia_raw/Low Potential Dysgraphia/
data/dysgraphia_raw/Potential Dysgraphia/
```

Alternatively, configure `DYSLEXIA_DATA_DIR` and `DYSGRAPHIA_DATA_DIR` as described in `src/config.py`. Dataset files are not redistributed with this source-code package; users must follow the original dataset licenses and access conditions.

## Environment

Use Python 3.10 or 3.11 in a clean virtual environment:

```bash
python -m pip install -r requirements.txt
```

For an archival run, record the resolved software environment:

```bash
python -m pip freeze > outputs/environment-freeze.txt
```

## Run the pipelines

From the repository root:

```bash
python run_pipeline.py
python run_pipeline.py --dyslexia
python run_pipeline.py --dysgraphia
```

After the relevant main pipeline completes, run the supplementary analyses as needed:

```bash
python src/11_posthoc_shap.py
python src/12_dyslexia_mcnemar.py
python src/13_dyslexia_end_to_end_cnn.py
python src/14_bootstrap_auc_ci.py
python src/16_validate_saved_outputs.py
python src/17_ablation_experiments.py
python src/18_smooth_gradcam.py
python src/19_dysgraphia_end_to_end_cnn.py
```

The ablation script trains separate variants (`full`, `no_resampling`, and `no_preprocessing`) and may require substantial compute. Analysis scripts consume the saved artifacts produced by the relevant main pipeline.

## Outputs

The `outputs/` directory is created during execution and stores split manifests, duplicate audits, trained models, extracted features, predictions and probabilities, confusion matrices, metric reports, SHAP summaries, bootstrap intervals, and visualization artifacts. Split manifests explicitly record the observed sample counts and whether they match the counts specified in the manuscript. Review these manifests when using a different copy or version of a dataset.

## Interpretation and evaluation protocol

- The dyslexia-related labels denote observable handwriting-pattern categories and are not clinical diagnoses.
- Evaluation is image-level because participant/writer identifiers are not available in the released datasets; the results do not establish participant-independent generalization.
- Bootstrap intervals are computed over test images and do not model within-participant dependence. `src/14_bootstrap_auc_ci.py` reads the test-label and probability files declared in its `TASKS` mapping and writes both the interval estimates and a `bootstrap_source_manifest.json` per task. Each task/model uses a deterministic random stream keyed by its task and model name, so adding or reordering another model does not alter an existing model’s bootstrap interval.
- The dysgraphia split is image-level because participant identifiers are unavailable; it is not a participant-independent evaluation. The resulting estimates do not establish generalization to unseen participants.
- `src/17_ablation_experiments.py` reruns the dyslexia and dysgraphia pipelines separately for `full`, `no_resampling`, and `no_preprocessing`. Each variant writes isolated artifacts and an `ablation_manifest.json` under `outputs/ablation/<variant>/`; its computed results must be checked against the specific manuscript ablation tables before use.
- SHAP and Smooth Grad-CAM are model-attribution methods and should not be interpreted as clinical validation or causal explanations.
- Reported metrics should be associated with the exact dataset copy, split manifest, software environment, and run artifacts used to prepare the manuscript tables and figures.


MLP training in the stacking workflow uses a fixed 50-epoch schedule and does not use EarlyStopping. Any EarlyStopping description applies only to a script or experiment that explicitly implements that callback; it should not be generalized to all MLP training procedures.


### Statistical analysis provenance checks

Bootstrap confidence intervals and paired McNemar tests require current-run `test_sample_manifest.json` files. These manifests bind saved test labels and predictions to the ordered held-out image paths. If manifests are missing or disagree, the analysis stops rather than assuming equal array lengths imply paired samples. Regenerate the split/features and all compared model outputs from one clean run before interpreting these tests. This verifies image-level ordering only; it does not establish participant-level independence.
