# Dataset information

## Face

FER2013-style grayscale facial-expression images with seven labels: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The selected model uses only original real images for training; validation and test are unchanged real splits.

## Speech

RAVDESS audio with eight emotions. The primary protocol separates actors across train, validation, and test (16/4/4 actors, zero overlap). A random-split augmented hybrid is retained only as a clearly labeled secondary benchmark.

## Numerical

The original data provide 18 behavioral, facial-summary, speech-summary, and physiological indicators plus four-class status and three regression targets. Synthetic numerical data are generated from documented latent factors for benchmarking; synthetic-held-out metrics are never treated as original-real performance.

## Fusion alignment

No verified participant key joins these datasets. Fusion therefore uses a documented weak class-conditional construction. Raw data and generated datasets are excluded from Git and must be placed locally by the user.
