import { Scale } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { EmptyState } from "@/components/ui/EmptyState";
import type { ModalityWeights } from "@/types/api";
import { formatPercent } from "@/lib/utils";

const ROWS: { key: keyof ModalityWeights; label: string }[] = [
  { key: "face", label: "Facial" },
  { key: "speech", label: "Speech" },
  { key: "numerical", label: "Behavioral / Physiological" },
];

export function ModalityContribution({ weights }: { weights?: ModalityWeights }) {
  return (
    <Card>
      <CardHeader
        title="Modality Contribution"
        subtitle="Relative weight each modality carried in the final fused prediction."
        icon={<Scale size={18} />}
      />
      {!weights ? (
        <EmptyState
          title="Modality weights unavailable"
          description="This will populate once the fusion model returns gating/attention weights for this assessment."
        />
      ) : (
        <div className="space-y-4">
          {ROWS.map((row) => (
            <ProgressBar key={row.key} label={row.label} value={weights[row.key]} valueLabel={formatPercent(weights[row.key])} tone="accent" />
          ))}
        </div>
      )}
    </Card>
  );
}
