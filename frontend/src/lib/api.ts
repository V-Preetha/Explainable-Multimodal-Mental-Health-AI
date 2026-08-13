import type { AssessmentRequestPayload, AssessmentResponse } from "@/types/api";
import { DEMO_ASSESSMENT_RESPONSE } from "@/lib/demoData";
import { MODEL_BRANCHES, type ModelBranch } from "@/lib/metricsConfig";

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8100";
export const DEMO_MODE: boolean = import.meta.env.VITE_DEMO_MODE === "true";

export class ApiRequestError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

function simulateLatency(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Runs the full multimodal assessment.
 *
 * In demo mode this resolves with a clearly labeled sample response.
 */
export async function runAssessment(payload: AssessmentRequestPayload): Promise<AssessmentResponse> {
  if (DEMO_MODE) {
    await simulateLatency(600);
    return { ...DEMO_ASSESSMENT_RESPONSE, isDemo: true };
  }

  const form = new FormData();
  if (payload.faceImage) form.append("face_file", payload.faceImage);
  if (payload.audioSample) form.append("audio_file", payload.audioSample);
  if (payload.numericalFeatures) form.append("numerical_json", JSON.stringify(payload.numericalFeatures));

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/assess`, { method: "POST", body: form });
  } catch (error) {
    throw new ApiRequestError(
      error instanceof Error ? `Could not reach the backend: ${error.message}` : "Could not reach the backend."
    );
  }
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new ApiRequestError(payload?.detail || `Assessment request failed (HTTP ${response.status}).`, response.status);
  }
  const data = (await response.json()) as AssessmentResponse;
  return { ...data, isDemo: false };
}

/**
 * Model Insights use completed benchmark values mirrored from
 * results/FINAL_RESULTS.md. The API representation takes priority when enabled.
 */
export async function getModelMetrics(): Promise<ModelBranch[]> {
  if (DEMO_MODE) {
    return MODEL_BRANCHES;
  }
  try {
    const response = await fetch(`${API_BASE_URL}/api/model-metrics`);
    if (!response.ok) return MODEL_BRANCHES;
    const data = (await response.json()) as ModelBranch[];
    return Array.isArray(data) && data.length > 0 ? data : MODEL_BRANCHES;
  } catch {
    return MODEL_BRANCHES;
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  if (DEMO_MODE) return false;
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
    return response.ok;
  } catch {
    return false;
  }
}
