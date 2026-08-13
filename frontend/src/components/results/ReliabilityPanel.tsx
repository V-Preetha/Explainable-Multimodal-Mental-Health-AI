import { ShieldAlert, ShieldCheck } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { EmptyState } from "@/components/ui/EmptyState";
import type { ReliabilityInfo } from "@/types/api";
import { formatPercent } from "@/lib/utils";

export function ReliabilityPanel({ reliability }: { reliability?: ReliabilityInfo }) {
  return (
    <Card>
      <CardHeader title="Assessment Reliability" subtitle="How much to trust this specific result." icon={<ShieldCheck size={18} />} />

      {!reliability ? (
        <EmptyState title="Reliability data unavailable" description="Confidence and agreement diagnostics will appear here once returned by the backend." />
      ) : (
        <div className="space-y-4">
          {reliability.overall_confidence !== undefined && (
            <ProgressBar label="Overall confidence" value={reliability.overall_confidence} valueLabel={formatPercent(reliability.overall_confidence)} tone="accent" />
          )}
          {reliability.modality_agreement !== undefined && (
            <ProgressBar label="Modality agreement" value={reliability.modality_agreement} valueLabel={formatPercent(reliability.modality_agreement)} tone="accent" />
          )}
          {reliability.input_quality &&
            Object.entries(reliability.input_quality).map(([modality, quality]) => (
              <ProgressBar key={modality} label={`${modality} input quality`} value={quality} valueLabel={formatPercent(quality)} tone="neutral" size="sm" />
            ))}

          {reliability.mixed_evidence && (
            <div className="flex items-center gap-2 rounded-lg bg-amber-50 border border-amber-200 px-3 py-2.5 mt-2">
              <ShieldAlert size={14} className="text-amber-600 shrink-0" />
              <p className="text-xs text-amber-700 font-medium">Mixed multimodal evidence</p>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
