# Final Hack4Health fusion results

This directory is the self-contained, inference-frozen evidence package for
the selected gated multimodal multi-task fusion model.

The evaluated run is `all_modalities` with probabilistic weak alignment and
modalities `(face, speech, numerical) = (1, 1, 1)`. The diagnostic
`same_class_control` run is not used.

## Reproduce the submitted metrics

From the repository root, after installing `requirements.txt`, run:

```powershell
python final_results/verify_final_fusion_results.py
```

The command reads only `frozen_fusion_test_predictions.csv`. It does not
train a model, select a checkpoint, use validation data, or reconstruct the
weak alignment. It recalculates every classification and regression metric
from the exact frozen 600-sample test outputs and fails if the submitted
reference values do not reproduce.

## Contents

- `frozen_fusion_test_predictions.csv`: true labels, four-class
  probabilities, predicted labels, true regression targets, and predicted
  regression outputs for all 600 frozen test examples.
- `final_fusion_metrics.json`: machine-readable metrics and provenance.
- `verify_final_fusion_results.py`: dependency-light independent verifier.
- `final_test_confusion_matrix_hack4health.png`: high-resolution upload-ready
  confusion matrix.
- `SHA256SUMS.txt`: integrity hashes for the package artifacts and selected
  repository checkpoint.

## Scaling and evaluation convention

During fusion training and inference, regression targets are divided by the
fixed maxima `[34, 24, 39]` for Depression, Anxiety, and Stress. Predictions
and targets in this package are already restored to original score units.
MAE, MSE, and RMSE are therefore calculated after inverse scaling. R-squared
and Explained Variance are calculated independently for each target before
their three target-specific values are averaged.

The selected checkpoint used by the repository is
`models/fusion_all_modalities.pt` and is managed by Git LFS. The frozen CSV is
included directly in Git, so metric verification does not require model or
dataset downloads.

