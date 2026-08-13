import torch

from src.face.model import ConvNeXtEmotion
from src.fusion.model import GatedFusionMultiTask
from src.numerical.models import NumericalMultiTask
from src.speech.models import SpeechEmotionMLP


def test_model_output_shapes():
    face = ConvNeXtEmotion(pretrained=False).eval()
    face_embedding, face_logits = face(torch.zeros(1, 3, 64, 64), return_embedding=True)
    assert face_embedding.shape == (1, 256)
    assert face_logits.shape == (1, 7)

    speech = SpeechEmotionMLP().eval()
    speech_embedding, speech_logits, attention = speech(torch.zeros(2, 5, 768))
    assert speech_embedding.shape == (2, 256)
    assert speech_logits.shape == (2, 8)
    assert attention.shape == (2, 5)

    numerical = NumericalMultiTask().eval()
    numerical_embedding, numerical_logits, numerical_scores = numerical(torch.zeros(2, 18))
    assert numerical_embedding.shape == (2, 128)
    assert numerical_logits.shape == (2, 4)
    assert numerical_scores.shape == (2, 3)

    fusion = GatedFusionMultiTask().eval()
    status, scores, weights, shared = fusion(face_embedding.repeat(2, 1), speech_embedding, numerical_embedding)
    assert status.shape == (2, 4)
    assert scores.shape == (2, 3)
    assert weights.shape == (2, 3)
    assert shared.shape == (2, 128)
