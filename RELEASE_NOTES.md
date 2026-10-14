# Release v2.0.0

This package organizes the source code and documentation for the handwriting-based classification framework described in the accompanying manuscript.

## Scope
- Preprocessing, image-level data partitioning, training-set balancing/augmentation, and CNN-based 128-dimensional feature extraction.
- Standalone classical classifiers and a separate five-fold stacking workflow.
- Supplementary evaluation scripts for SHAP, McNemar tests, bootstrap intervals, ablation experiments, end-to-end CNN baselines, and Smooth Grad-CAM.

## Reproducibility notes
- The package does not include the source datasets. Users must obtain them from their original sources and respect their licenses and access conditions.
- Run manifests record observed split counts and bind statistical comparisons to the ordered test samples.
- The repository code uses image-level evaluation because writer/participant identifiers are unavailable; this does not establish writer- or participant-independent generalization.
- No claim is made in this release note that the reported manuscript metrics have been reproduced by executing this package.


## Manuscript-alignment updates in this prepared package
- CNN feature-extraction training now passes the validation subset to `model.fit` and monitors validation loss with `ReduceLROnPlateau` for the stated 20-epoch schedule (patience 3 for dyslexia; patience 2 for dysgraphia).
- Stacking uses stratified five-fold splits, consistent with the manuscript's stated protocol.
- These implementation changes can change model behavior and reported metrics. The package has been syntax-checked, but experiments have not been rerun and manuscript metrics have not been reproduced in this preparation step. Re-run experiments before representing metrics as reproduced by this release.
