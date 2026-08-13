# Model checkpoints

Model binaries are intentionally not committed because the selected ConvNeXt checkpoint exceeds GitHub's normal 100 MB file limit and other checkpoints are generated artifacts.

Expected local files for full inference/training workflows:

- `convnext_face_winner.pt`
- `speech_actor_independent.pt`
- `numerical_original.joblib` or a retrained equivalent
- `multimodal_fusion_best.pt`

Use the training commands in the root README to reproduce them. The frontend runs in clearly labeled demo mode without these files.
