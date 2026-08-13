# Limitations and responsible use

- This prototype is not a diagnostic device and must not drive clinical or emergency decisions.
- Fusion is not participant-paired; its metrics evaluate a constructed weak-alignment task.
- FER2013 and RAVDESS labels are proxies for emotion, not mental-health diagnoses.
- Actor-independent speech is the primary generalization result. The higher random-split number permits actor overlap.
- Synthetic numerical performance primarily measures the learnability of the generator's structure.
- The numerical original-real baseline is near chance; synthetic-enhanced training transfers poorly to original-real data.
- Minority classes, especially severe stress and some facial emotions, have limited support.
- Demographic, cultural, recording-device, lighting, and language shifts were not comprehensively evaluated.
- Confidence and explainability outputs are model diagnostics, not causal explanations.
