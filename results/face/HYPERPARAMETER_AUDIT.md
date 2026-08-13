# ConvNeXt comparison hyperparameter audit

The apparent learning-rate mismatch was a display mismatch between optimizer parameter groups.

| Field | Real-only | Synthetic from scratch |
|---|---|---|
| Initialization | Fresh torchvision ConvNeXt-Tiny ImageNet-1K V1 plus fresh 256D FER head | Same |
| Backbone initial LR | 2e-5 | 2e-5 |
| Head initial LR | 3e-4 | 3e-4 |
| Optimizer | AdamW | AdamW |
| Weight decay | 5e-4 | 5e-4 |
| Scheduler | CosineAnnealingLR, eta_min 2e-6 | Same |
| Scheduler horizon | 60 epochs | 60 epochs |
| Backbone warmup | Frozen epochs 1-2; unfrozen epoch 3 | Same |
| Batch size | 64 | 64 (45 real + 19 synthetic) |
| Image size | 160 | 160 |
| Augmentation | RandomResizedCrop 0.82-1.0, horizontal flip, rotation 10 degrees, RandAugment 2/7 | Same |
| Normalization | ImageNet mean/std after grayscale-to-RGB | Same |
| Split | Frozen face_deep_split.json | Same; synthetic train-only |
| Class mapping | Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise | Same |
| Seed | 42 | 42 with deterministic epoch-derived mixed-sampler seed |

At epoch 8 the displayed 1.940e-5 was the backbone group. The head group at the same scheduler step was 2.901e-4. The real-only console displayed the head group.
