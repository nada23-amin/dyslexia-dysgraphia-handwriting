# Release

This package provides the source code and documentation for the handwriting-based classification framework described in the accompanying manuscript.

## Scope
- Preprocessing, image-level data partitioning, training-set balancing and augmentation, and CNN-based 128-dimensional feature extraction.
- Standalone classical classifiers and a separate five-fold stacking workflow.
- Supplementary evaluation scripts for SHAP, McNemar tests, bootstrap confidence intervals, ablation experiments, end-to-end CNN baselines, and Smooth Grad-CAM.

## Reproducibility Notes
- The package does not include the source datasets. Users must obtain them from their original sources and comply with the applicable licenses and access conditions.
- Run manifests record the observed split counts and associate statistical comparisons with the ordered test samples.
- The evaluation uses image-level partitioning because writer/participant identifiers are unavailable. Therefore, writer-independent or participant-independent generalization has not been established.

## Implementation and Manuscript Alignment
- CNN feature-extraction training uses the validation subset in `model.fit` and monitors validation loss with `ReduceLROnPlateau` over the specified 20-epoch schedule (patience 3 for dyslexia and patience 2 for dysgraphia).
- The stacking workflow uses stratified five-fold cross-validation, consistent with the manuscript's stated protocol.
