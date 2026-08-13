# Selected model cards

## Face: ConvNeXt-Tiny real-only

ImageNet-1K initialized ConvNeXt-Tiny with a 256-D embedding and seven-class FER head. Selected by validation Macro F1. Training-only synthetic augmentation was tested as an ablation and did not win.

## Speech: emotion2vec+

Frozen emotion2vec+ frame encoder, attentive statistics pooling, and an MLP classifier. The primary result uses a zero-actor-overlap split.

## Numerical

The original-data baseline is the honest real-data reference. Synthetic-only XGBoost and synthetic-held-out regression are separate generator benchmarks; the original-real transfer result is reported alongside them.

## Fusion

Projects 256-D face, 256-D speech, and 128-D numerical embeddings into a gated shared representation with classification and regression heads. It uses weak class-conditional alignment because participant-paired data are unavailable.
