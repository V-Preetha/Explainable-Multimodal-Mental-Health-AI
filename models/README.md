# Model checkpoints

The selected runtime binaries are included through Git LFS because the ConvNeXt checkpoint exceeds GitHub's normal 100 MB file limit.

Included files:

- `convnext_face_winner.pt`
- `speech_emotion2vec_strict.pt`
- `numerical_multitask.pt`
- `numerical_scaler.joblib`
- `numerical_synthetic_18_xgboost.joblib`
- `fusion_all_modalities.pt`

The synthetic XGBoost supplies the live numerical class and score predictions. The numerical multi-task checkpoint and scaler remain as the 128-D adapter required by the previously validated fusion weights.

Install Git LFS before cloning so these files are materialized instead of remaining pointer files. Use the training commands in the root README to reproduce them.
