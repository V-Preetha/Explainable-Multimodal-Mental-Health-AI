import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import type { AssessmentResponse } from "@/types/api";
import { ApiRequestError, runAssessment } from "@/lib/api";
import { appendHistory, clearHistory, loadHistory, type HistoryEntry } from "@/lib/storage";
import { DEMO_SAMPLE_NUMERICAL_VALUES } from "@/lib/constants";

export type AssessmentStage =
  | "idle"
  | "analyzing_face"
  | "analyzing_speech"
  | "processing_numerical"
  | "fusing"
  | "generating"
  | "done"
  | "error";

interface AssessmentState {
  faceImage: File | null;
  faceImagePreviewUrl: string | null;
  audioSample: File | null;
  audioPreviewUrl: string | null;
  numericalValues: Record<string, number>;
  stage: AssessmentStage;
  result: AssessmentResponse | null;
  error: string | null;
  history: HistoryEntry[];
}

interface AssessmentContextValue extends AssessmentState {
  setFaceImage: (file: File | null) => void;
  setAudioSample: (file: File | null) => void;
  setNumericalValue: (key: string, value: number) => void;
  resetNumericalValues: () => void;
  loadSampleNumericalValues: () => void;
  runFullAssessment: () => Promise<void>;
  reset: () => void;
  clearAllHistory: () => void;
}

const AssessmentContext = createContext<AssessmentContextValue | null>(null);

const STAGE_SEQUENCE: { stage: AssessmentStage; minMs: number }[] = [
  { stage: "analyzing_face", minMs: 550 },
  { stage: "analyzing_speech", minMs: 550 },
  { stage: "processing_numerical", minMs: 450 },
  { stage: "fusing", minMs: 500 },
  { stage: "generating", minMs: 450 },
];

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function AssessmentProvider({ children }: { children: ReactNode }) {
  const [faceImage, setFaceImageState] = useState<File | null>(null);
  const [faceImagePreviewUrl, setFaceImagePreviewUrl] = useState<string | null>(null);
  const [audioSample, setAudioSampleState] = useState<File | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [numericalValues, setNumericalValues] = useState<Record<string, number>>({});
  const [stage, setStage] = useState<AssessmentStage>("idle");
  const [result, setResult] = useState<AssessmentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());

  const faceImagePreviewUrlRef = useRef<string | null>(null);
  const audioPreviewUrlRef = useRef<string | null>(null);

  const setFaceImage = useCallback((file: File | null) => {
    if (faceImagePreviewUrlRef.current) URL.revokeObjectURL(faceImagePreviewUrlRef.current);
    const nextUrl = file ? URL.createObjectURL(file) : null;
    faceImagePreviewUrlRef.current = nextUrl;
    setFaceImageState(file);
    setFaceImagePreviewUrl(nextUrl);
  }, []);

  const setAudioSample = useCallback((file: File | null) => {
    if (audioPreviewUrlRef.current) URL.revokeObjectURL(audioPreviewUrlRef.current);
    const nextUrl = file ? URL.createObjectURL(file) : null;
    audioPreviewUrlRef.current = nextUrl;
    setAudioSampleState(file);
    setAudioPreviewUrl(nextUrl);
  }, []);

  const setNumericalValue = useCallback((key: string, value: number) => {
    setNumericalValues((prev) => ({ ...prev, [key]: value }));
  }, []);

  const resetNumericalValues = useCallback(() => setNumericalValues({}), []);
  const loadSampleNumericalValues = useCallback(() => setNumericalValues({ ...DEMO_SAMPLE_NUMERICAL_VALUES }), []);

  const runFullAssessment = useCallback(async () => {
    setError(null);
    setResult(null);
    try {
      for (const step of STAGE_SEQUENCE) {
        setStage(step.stage);
        await wait(step.minMs);
      }
      const response = await runAssessment({
        faceImage,
        audioSample,
        numericalFeatures: Object.keys(numericalValues).length > 0 ? numericalValues : null,
      });
      setResult(response);
      setStage("done");
      setHistory(appendHistory(response));
    } catch (err) {
      const message = err instanceof ApiRequestError ? err.message : "The assessment could not be completed. Please try again.";
      setError(message);
      setStage("error");
    }
  }, [faceImage, audioSample, numericalValues]);

  const reset = useCallback(() => {
    setStage("idle");
    setResult(null);
    setError(null);
  }, []);

  const clearAllHistory = useCallback(() => {
    clearHistory();
    setHistory([]);
  }, []);

  const value = useMemo<AssessmentContextValue>(
    () => ({
      faceImage,
      faceImagePreviewUrl,
      audioSample,
      audioPreviewUrl,
      numericalValues,
      stage,
      result,
      error,
      history,
      setFaceImage,
      setAudioSample,
      setNumericalValue,
      resetNumericalValues,
      loadSampleNumericalValues,
      runFullAssessment,
      reset,
      clearAllHistory,
    }),
    [
      faceImage,
      faceImagePreviewUrl,
      audioSample,
      audioPreviewUrl,
      numericalValues,
      stage,
      result,
      error,
      history,
      setFaceImage,
      setAudioSample,
      setNumericalValue,
      resetNumericalValues,
      loadSampleNumericalValues,
      runFullAssessment,
      reset,
      clearAllHistory,
    ]
  );

  return <AssessmentContext.Provider value={value}>{children}</AssessmentContext.Provider>;
}

export function useAssessment(): AssessmentContextValue {
  const ctx = useContext(AssessmentContext);
  if (!ctx) throw new Error("useAssessment must be used within an AssessmentProvider");
  return ctx;
}
