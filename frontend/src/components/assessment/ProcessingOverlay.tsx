import { Check, Loader2 } from "lucide-react";
import type { AssessmentStage } from "@/context/AssessmentContext";
import { cn } from "@/lib/utils";

const STEPS: { stage: AssessmentStage; label: string }[] = [
  { stage: "analyzing_face", label: "Analyzing facial indicators…" },
  { stage: "analyzing_speech", label: "Analyzing speech…" },
  { stage: "processing_numerical", label: "Processing behavioral indicators…" },
  { stage: "fusing", label: "Fusing modality representations…" },
  { stage: "generating", label: "Generating assessment…" },
];

export function ProcessingOverlay({ stage }: { stage: AssessmentStage }) {
  const currentIndex = STEPS.findIndex((s) => s.stage === stage);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm px-4">
      <div className="card-surface bg-white w-full max-w-sm p-7">
        <p className="text-sm font-semibold text-slate-900 mb-1">Running Multimodal Assessment</p>
        <p className="text-xs text-slate-500 mb-6">This takes a few seconds. Please keep this tab open.</p>
        <ol className="space-y-3.5">
          {STEPS.map((step, index) => {
            const isDone = currentIndex > index || currentIndex === -1;
            const isCurrent = index === currentIndex;
            return (
              <li key={step.stage} className="flex items-center gap-3">
                <span
                  className={cn(
                    "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-white",
                    isDone && !isCurrent ? "bg-[var(--color-accent-600)]" : isCurrent ? "bg-[var(--color-accent-100)] text-[var(--color-accent-700)]" : "bg-slate-100 text-slate-400"
                  )}
                >
                  {isDone && !isCurrent ? <Check size={13} /> : isCurrent ? <Loader2 size={13} className="animate-spin" /> : <span className="text-[10px]">{index + 1}</span>}
                </span>
                <span className={cn("text-sm", isCurrent ? "text-slate-900 font-medium" : isDone ? "text-slate-500" : "text-slate-400")}>
                  {step.label}
                </span>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
