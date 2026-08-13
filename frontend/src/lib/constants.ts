import type { MentalHealthStatus } from "@/types/api";

export const APP_NAME = "Multimodal Mental Health Assessment";

export type PageKey =
  | "assessment"
  | "results"
  | "explainability"
  | "insights"
  | "about";

export const NAV_ITEMS: { key: PageKey; label: string }[] = [
  { key: "assessment", label: "Assessment" },
  { key: "results", label: "Results" },
  { key: "explainability", label: "Explainability" },
  { key: "insights", label: "Model Insights" },
  { key: "about", label: "About" },
];

export interface NumericalFieldSpec {
  key: string;
  label: string;
  unit?: string;
  min: number;
  max: number;
  step: number;
  control: "slider" | "number";
  helper?: string;
}

export interface NumericalFieldGroup {
  title: string;
  description: string;
  fields: NumericalFieldSpec[];
}

export const NUMERICAL_FIELD_GROUPS: NumericalFieldGroup[] = [
  {
    title: "Behavioral",
    description: "Sleep, social engagement, and device-usage patterns.",
    fields: [
      { key: "Sleep_Quality", label: "Sleep Quality", min: 1, max: 5, step: 1, control: "slider", helper: "Self-reported quality, 1 (poor) to 5 (excellent)" },
      { key: "Social_Engagement", label: "Social Engagement", min: 1, max: 5, step: 1, control: "slider", helper: "1 (withdrawn) to 5 (highly engaged)" },
      { key: "Daily_App_Usage_Min", label: "Daily App Usage", unit: "min", min: 0, max: 480, step: 5, control: "slider" },
      { key: "Typing_Speed_WPM", label: "Typing Speed", unit: "WPM", min: 10, max: 100, step: 1, control: "number" },
      { key: "Session_Frequency", label: "Session Frequency", unit: "/day", min: 0, max: 25, step: 1, control: "slider" },
      { key: "Idle_Time_Min", label: "Idle Time", unit: "min", min: 0, max: 200, step: 5, control: "slider" },
    ],
  },
  {
    title: "Facial / Visual Indicators",
    description: "Derived from short-term facial video analysis.",
    fields: [
      { key: "Facial_Emotion_Variance", label: "Facial Emotion Variance", min: 0, max: 1, step: 0.01, control: "slider" },
      { key: "Eye_Blink_Rate", label: "Eye Blink Rate", unit: "/min", min: 5, max: 40, step: 1, control: "number" },
      { key: "Smile_Intensity", label: "Smile Intensity", min: 0, max: 1, step: 0.01, control: "slider" },
      { key: "Head_Motion_Index", label: "Head Motion Index", min: 0, max: 1, step: 0.01, control: "slider" },
    ],
  },
  {
    title: "Speech Indicators",
    description: "Acoustic descriptors extracted from voice samples.",
    fields: [
      { key: "MFCC_Mean", label: "MFCC Mean", min: -50, max: 50, step: 0.5, control: "number" },
      { key: "MFCC_Variance", label: "MFCC Variance", min: 0, max: 30, step: 0.5, control: "number" },
      { key: "Pitch_Mean", label: "Pitch Mean", unit: "Hz", min: 60, max: 320, step: 1, control: "number" },
      { key: "Speech_Rate", label: "Speech Rate", unit: "syll/s", min: 1, max: 7, step: 0.1, control: "slider" },
    ],
  },
  {
    title: "Physiological",
    description: "Heart-rate and skin-response indicators from a wearable sensor.",
    fields: [
      { key: "Heart_Rate_BPM", label: "Heart Rate", unit: "BPM", min: 45, max: 130, step: 1, control: "slider" },
      { key: "HRV_Index", label: "HRV Index", min: 0, max: 100, step: 1, control: "slider", helper: "Heart-rate variability, higher is generally healthier" },
      { key: "Skin_Temperature", label: "Skin Temperature", unit: "°C", min: 30, max: 38, step: 0.1, control: "number" },
      { key: "GSR_Level", label: "GSR Level", unit: "µS", min: 0, max: 6, step: 0.1, control: "slider", helper: "Galvanic skin response" },
    ],
  },
];

export const DEMO_SAMPLE_NUMERICAL_VALUES: Record<string, number> = {
  Sleep_Quality: 3,
  Social_Engagement: 3,
  Daily_App_Usage_Min: 210,
  Typing_Speed_WPM: 48,
  Session_Frequency: 11,
  Idle_Time_Min: 85,
  Facial_Emotion_Variance: 0.52,
  Eye_Blink_Rate: 19,
  Smile_Intensity: 0.44,
  Head_Motion_Index: 0.41,
  MFCC_Mean: -8,
  MFCC_Variance: 6.2,
  Pitch_Mean: 188,
  Speech_Rate: 3.6,
  Heart_Rate_BPM: 80,
  HRV_Index: 52,
  Skin_Temperature: 33.6,
  GSR_Level: 2.1,
};

export const STATUS_COLOR: Record<MentalHealthStatus, { text: string; bg: string; ring: string }> = {
  Healthy: { text: "text-[var(--color-status-healthy)]", bg: "bg-[var(--color-status-healthy-bg)]", ring: "ring-[var(--color-status-healthy)]/25" },
  Mild_Stress: { text: "text-[var(--color-status-mild)]", bg: "bg-[var(--color-status-mild-bg)]", ring: "ring-[var(--color-status-mild)]/25" },
  Moderate_Stress: { text: "text-[var(--color-status-moderate)]", bg: "bg-[var(--color-status-moderate-bg)]", ring: "ring-[var(--color-status-moderate)]/25" },
  Severe_Stress: { text: "text-[var(--color-status-severe)]", bg: "bg-[var(--color-status-severe-bg)]", ring: "ring-[var(--color-status-severe)]/25" },
};

export const DISCLAIMER_SHORT =
  "This system is intended as an AI-assisted assessment tool and is not a substitute for professional clinical diagnosis.";

export const DISCLAIMER_LONG =
  "This prototype is intended for research and decision-support demonstrations. It does not provide a medical diagnosis and should not replace evaluation by a qualified mental-health professional.";
