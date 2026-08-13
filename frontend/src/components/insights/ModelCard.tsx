import { Card, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import type { ModelBranch } from "@/lib/metricsConfig";

function MetricCell({ label, value, suffix = "" }: { label: string; value?: number; suffix?: string }) {
  return (
    <div>
      <p className="text-[11px] text-slate-400">{label}</p>
      <p className="text-base font-semibold text-slate-800 tabular-nums mt-0.5">
        {value !== undefined ? `${(value * 100).toFixed(2)}${suffix}` : "—"}
      </p>
    </div>
  );
}

export function ModelCard({ branch }: { branch: ModelBranch }) {
  return (
    <Card>
      <CardHeader title={branch.title} subtitle={branch.architecture} />
      <p className="text-xs text-slate-400 -mt-3 mb-4">Dataset: {branch.dataset}</p>

      <div className="space-y-4">
        {branch.variants.map((variant) => (
          <div key={variant.key} className="rounded-xl border border-slate-100 p-4 bg-slate-50/50">
            <div className="flex items-center justify-between gap-2 mb-2">
              <Badge tone="accent">{variant.label}</Badge>
            </div>
            <p className="text-xs text-slate-500 mb-3">{variant.description}</p>

            {variant.metrics ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCell label="Accuracy" value={variant.metrics.accuracy} suffix="%" />
                <MetricCell label="Macro F1" value={variant.metrics.macroF1} />
                <MetricCell label="Weighted F1" value={variant.metrics.weightedF1} />
                <MetricCell label="ROC-AUC" value={variant.metrics.rocAuc} />
              </div>
            ) : (
              <p className="text-xs text-slate-400 italic">{variant.note ?? "Metrics not yet available."}</p>
            )}
            {variant.metrics && variant.note && <p className="text-[11px] text-slate-400 mt-3 leading-relaxed">{variant.note}</p>}
          </div>
        ))}
      </div>

      {branch.limitation && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2.5 mt-4 leading-relaxed">
          {branch.limitation}
        </p>
      )}
    </Card>
  );
}
