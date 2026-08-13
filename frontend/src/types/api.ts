/**
 * Shared API contract between the frontend and the research backend.
 * Every field beyond a bare minimum is optional -- partial or
 * missing data must render a graceful "unavailable" state, never a fake
 * value.
 */

export type MentalHealthStatus =
  | "Healthy"
  | "Mild_Stress"
  | "Moderate_Stress"
  | "Severe_Stress";

export const STATUS_LABELS: Record<MentalHealthStatus, string> = {
  Healthy: "Healthy",
  Mild_Stress: "Mild Stress",
  Moderate_Stress: "Moderate Stress",
  Severe_Stress: "Severe Stress",
};

export const SCORE_RANGES = {
  depression: { min: 0, max: 34, label: "Depression Score" },
  anxiety: { min: 0, max: 24, label: "Anxiety Score" },
  stress: { min: 0, max: 39, label: "Stress Score" },
} as const;

export interface DASScores {
  depression: number;
  anxiety: number;
  stress: number;
}

export interface ModalityEmotionResult {
  emotion: string;
  confidence: number;
}

export interface NumericalModalityResult {
  confidence: number;
}

export interface ModalityResults {
  face?: ModalityEmotionResult;
  speech?: ModalityEmotionResult;
  numerical?: NumericalModalityResult;
}

export interface ModalityWeights {
  face: number;
  speech: number;
  numerical: number;
}

export interface FaceExplainability {
  original_image_url?: string;
  overlay_image_url?: string;
  predicted_emotion?: string;
  confidence?: number;
  face_quality?: number;
}

export interface SpeechExplainability {
  predicted_emotion?: string;
  confidence?: number;
  speech_quality?: number;
  feature_contribution?: Record<string, number>;
  waveform?: number[];
  temporal_attention?: number[];
}

export interface NumericalExplainability {
  feature_importance?: Record<string, number>;
}

export interface Explainability {
  face?: FaceExplainability;
  speech?: SpeechExplainability;
  numerical?: NumericalExplainability;
}

export interface ReliabilityInfo {
  overall_confidence?: number;
  modality_agreement?: number;
  input_quality?: Record<string, number>;
  mixed_evidence?: boolean;
}

export interface AssessmentResponse {
  status: MentalHealthStatus;
  confidence: number;
  scores: DASScores;
  modalities: ModalityResults;
  modality_weights?: ModalityWeights;
  explainability?: Explainability;
  reliability?: ReliabilityInfo;
  /** Set by the client when this payload came from demo mode, never from a real backend. */
  isDemo?: boolean;
}

export interface AssessmentRequestPayload {
  faceImage?: File | null;
  audioSample?: File | null;
  numericalFeatures?: Record<string, number> | null;
}

/** Generic row shape for the Model Insights comparison charts. */
export interface ModelComparisonRow {
  model: string;
  branch: "face" | "speech" | "numerical" | "fusion";
  variant: string;
  accuracy?: number;
  macroF1?: number;
  weightedF1?: number;
  rocAuc?: number;
}

export interface ConfusionMatrixData {
  classes: string[];
  counts: number[][];
  normalized: number[][];
}

export interface ApiError {
  message: string;
  status?: number;
}
