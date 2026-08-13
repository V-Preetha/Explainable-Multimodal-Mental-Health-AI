import { Card } from "@/components/ui/Card";
import { Gauge } from "@/components/ui/Gauge";
import { Tooltip } from "@/components/ui/Tooltip";
import { Info } from "lucide-react";

export function ScoreCard({
  label,
  value,
  min,
  max,
  helper,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  helper?: string;
}) {
  const fraction = (value - min) / (max - min);

  return (
    <Card className="flex flex-col items-center text-center">
      <div className="flex items-center gap-1.5 mb-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
        {helper && (
          <Tooltip label={helper}>
            <Info size={12} className="text-slate-400" />
          </Tooltip>
        )}
      </div>
      <Gauge value={fraction} size={104} strokeWidth={8} tone="accent" centerLabel={value.toFixed(1)} />
      <p className="text-xs text-slate-400 mt-3">
        Range {min}–{max}
      </p>
    </Card>
  );
}
