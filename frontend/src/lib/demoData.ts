import type { AssessmentResponse } from "@/types/api";

/**
 * Sample output used only when VITE_DEMO_MODE=true (or as an explicit
 * "Load Sample" convenience). Every consumer of this object must render the
 * "Demo Output" label alongside it -- see components/results/StatusHero.tsx.
 * This is API-schema-shaped example data, not a real model prediction.
 */
export const DEMO_ASSESSMENT_RESPONSE: AssessmentResponse = {
  status: "Moderate_Stress",
  confidence: 0.86,
  scores: {
    depression: 18.4,
    anxiety: 13.2,
    stress: 25.1,
  },
  modalities: {
    face: { emotion: "Sad", confidence: 0.78 },
    speech: { emotion: "Fearful", confidence: 0.81 },
    numerical: { confidence: 0.69 },
  },
  modality_weights: {
    face: 0.41,
    speech: 0.39,
    numerical: 0.2,
  },
  explainability: {
    face: {
      predicted_emotion: "Sad",
      confidence: 0.78,
      face_quality: 0.88,
    },
    speech: {
      predicted_emotion: "Fearful",
      confidence: 0.81,
      speech_quality: 0.82,
      feature_contribution: {
        Pitch: 0.27,
        MFCC: 0.24,
        Energy: 0.19,
        "Speech Rate": 0.17,
        "Pause Characteristics": 0.13,
      },
    },
    numerical: {
      feature_importance: {
        HRV_Index: 0.21,
        GSR_Level: 0.18,
        Sleep_Quality: 0.16,
        Social_Engagement: 0.14,
        Heart_Rate_BPM: 0.12,
        Speech_Rate: 0.1,
        Smile_Intensity: 0.09,
      },
    },
  },
  reliability: {
    overall_confidence: 0.86,
    modality_agreement: 0.74,
    input_quality: { face: 0.88, speech: 0.82, numerical: 0.95 },
    mixed_evidence: false,
  },
  isDemo: true,
};
