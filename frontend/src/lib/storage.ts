import type { AssessmentResponse } from "@/types/api";

export interface HistoryEntry {
  id: string;
  timestamp: string;
  status: AssessmentResponse["status"];
  confidence: number;
  scores: AssessmentResponse["scores"];
  isDemo?: boolean;
}

const STORAGE_KEY = "mmha.assessment-history.v1";
const MAX_ENTRIES = 20;

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function appendHistory(response: AssessmentResponse): HistoryEntry[] {
  const entry: HistoryEntry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: new Date().toISOString(),
    status: response.status,
    confidence: response.confidence,
    scores: response.scores,
    isDemo: response.isDemo,
  };
  const next = [entry, ...loadHistory()].slice(0, MAX_ENTRIES);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // localStorage unavailable (e.g. private browsing quota) -- fail silently, history is a convenience only.
  }
  return next;
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
