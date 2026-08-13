import { ClipboardList, FileQuestion } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { StatusHero } from "@/components/results/StatusHero";
import { ScoreCard } from "@/components/results/ScoreCard";
import { ModalityContribution } from "@/components/results/ModalityContribution";
import { ReliabilityPanel } from "@/components/results/ReliabilityPanel";
import { HistoryPanel } from "@/components/history/HistoryPanel";
import { EmptyState } from "@/components/ui/EmptyState";
import { Button } from "@/components/ui/Button";
import { useAssessment } from "@/context/AssessmentContext";
import { SCORE_RANGES } from "@/types/api";
import type { PageKey } from "@/lib/constants";

export function ResultsPage({ onNavigate }: { onNavigate: (page: PageKey) => void }) {
  const { result, history, clearAllHistory } = useAssessment();

  return (
    <>
      <Header title="Results" subtitle="Multimodal assessment outcome, confidence, and score breakdown." />

      {!result && (
        <EmptyState
          icon={<FileQuestion size={40} strokeWidth={1.25} />}
          title="No assessment yet"
          description="Run a multimodal assessment first to see results here."
          action={
            <Button icon={<ClipboardList size={15} />} onClick={() => onNavigate("assessment")}>
              Go to Assessment
            </Button>
          }
        />
      )}

      {result && (
        <div className="space-y-5">
          <StatusHero result={result} />

          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-3">Depression / Anxiety / Stress Scores</p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <ScoreCard label="Depression Score" value={result.scores.depression} min={SCORE_RANGES.depression.min} max={SCORE_RANGES.depression.max} helper="Modeled depression-related indicator, not a clinical diagnostic score." />
              <ScoreCard label="Anxiety Score" value={result.scores.anxiety} min={SCORE_RANGES.anxiety.min} max={SCORE_RANGES.anxiety.max} helper="Modeled anxiety-related indicator, not a clinical diagnostic score." />
              <ScoreCard label="Stress Score" value={result.scores.stress} min={SCORE_RANGES.stress.min} max={SCORE_RANGES.stress.max} helper="Modeled stress-related indicator, not a clinical diagnostic score." />
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <ModalityContribution weights={result.modality_weights} />
            <ReliabilityPanel reliability={result.reliability} />
          </div>
        </div>
      )}

      <div className="mt-5">
        <HistoryPanel history={history} onClear={clearAllHistory} />
      </div>
    </>
  );
}
