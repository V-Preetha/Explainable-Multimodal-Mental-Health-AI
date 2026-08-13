# Final Results

This is the authoritative results document for the repository. Numbers elsewhere must agree with the evaluation protocol and values below.

## Selected standalone models

| Branch | Model and evaluation protocol | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Face | **ConvNeXt-Tiny real-only**, unchanged real FER2013 test | **0.6896** | **0.6852** | **0.6878** | **0.9103 macro OvR** |
| Speech — primary | **emotion2vec+**, actor-independent RAVDESS test | **0.7583** | **0.7405** | **0.7587** | **0.9615 macro OvR** |
| Numerical — original baseline | Original-real benchmark | **0.2617** | **0.2245** | — | **0.4612 macro OvR** |

Face model selection used validation Macro F1. The selected checkpoint is epoch 22 with validation accuracy **0.6876** and validation Macro F1 **0.6799**.

## Explicitly labeled secondary benchmarks

| Benchmark | Evaluation domain | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Speech augmented hybrid — **RANDOM SPLIT** | Stratified random RAVDESS split; **not speaker-independent** | 0.8657 | 0.8505 | 0.8663 | 0.9879 macro OvR |
| Numerical synthetic-only XGBoost | Synthetic held-out data | 0.8960 | 0.8962 | — | 0.9845 macro OvR |
| Numerical synthetic-enhanced model | Original-real test data | 0.4000 | 0.1988 | — | — |

The random-split speech result must never be presented as actor- or speaker-independent. The numerical synthetic-only result measures recovery of the synthetic generator's structure and must not be presented as real-world generalization.

Synthetic-held-out numerical regression R²:

- Depression: **0.729**
- Anxiety: **0.717**
- Stress: **0.714**

## Face synthetic-data ablation

The matched synthetic-augmented ConvNeXt ablation did not replace the selected face model. It reached validation Macro F1 **0.6718** and real-test Macro F1 **0.6633**, versus **0.6799** validation and **0.6852** test Macro F1 for real-only ConvNeXt. See `results/face/real_vs_synthetic.csv`.

## Fusion research result

| Model | Validation Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Weighted ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| Gated multimodal multi-task fusion | 0.4976 | 0.6633 | 0.5224 | 0.6539 | 0.8303 |

**Scientific boundary:** face, speech, and numerical datasets do not share verified participant identifiers. Fusion uses weak class-conditional construction (70% same stress category, 20% adjacent, 10% conflicting). These are not participant-paired clinical performance claims. A same-class control is an ablation/shortcut diagnostic and is not a deployable result.

## Intended interpretation

This is a research decision-support prototype, not a diagnostic device. Results are dataset-specific, uncertainty is material, and synthetic-held-out or random-split benchmarks are deliberately separated from strict real-data evaluation.
