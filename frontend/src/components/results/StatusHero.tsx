import { AlertCircle, FlaskConical } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Gauge } from "@/components/ui/Gauge";
import { STATUS_COLOR } from "@/lib/constants";
import { STATUS_LABELS, type AssessmentResponse } from "@/types/api";
import { DISCLAIMER_SHORT } from "@/lib/constants";

const STATUS_TONE: Record<AssessmentResponse["status"], "healthy" | "mild" | "moderate" | "severe"> = {
  Healthy: "healthy",
  Mild_Stress: "mild",
  Moderate_Stress: "moderate",
  Severe_Stress: "severe",
};

export function StatusHero({ result }: { result: AssessmentResponse }) {
  const tone = STATUS_TONE[result.status];
  const colors = STATUS_COLOR[result.status];

  return (
    <Card className="relative overflow-hidden">
      {result.isDemo && (
        <Badge tone="demo" className="absolute top-5 right-5" icon={<FlaskConical size={12} />}>
          Demo Output
        </Badge>
      )}
      <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6">
        <Gauge value={result.confidence} tone={tone} centerLabel={`${Math.round(result.confidence * 100)}%`} centerSubLabel="Model Confidence" />
        <div className="flex-1 text-center sm:text-left">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400 mb-1.5">Overall Mental Health Status</p>
          <span className={`inline-flex items-center rounded-xl px-4 py-2 text-xl font-semibold ${colors.bg} ${colors.text}`}>
            {STATUS_LABELS[result.status]}
          </span>
          <p className="text-sm text-slate-500 mt-4 max-w-lg">
            The model estimates this individual's overall profile is most consistent with the{" "}
            <strong className="font-medium text-slate-700">{STATUS_LABELS[result.status].toLowerCase()}</strong> category, based on the
            provided modalities.
          </p>
          <div className="flex items-start gap-2 mt-4 rounded-lg bg-slate-50 border border-slate-100 px-3 py-2.5 max-w-lg">
            <AlertCircle size={14} className="text-slate-400 mt-0.5 shrink-0" />
            <p className="text-xs text-slate-500 leading-relaxed">{DISCLAIMER_SHORT}</p>
          </div>
        </div>
      </div>
    </Card>
  );
}
