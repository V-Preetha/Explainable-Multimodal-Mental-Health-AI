import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import { BarChart2 } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { cn } from "@/lib/utils";
import type { ModelComparisonRow } from "@/types/api";

type MetricKey = "accuracy" | "macroF1" | "weightedF1" | "rocAuc";

const METRIC_OPTIONS: { key: MetricKey; label: string }[] = [
  { key: "accuracy", label: "Accuracy" },
  { key: "macroF1", label: "Macro F1" },
  { key: "rocAuc", label: "ROC-AUC" },
];

export function ComparisonChart({ rows }: { rows: ModelComparisonRow[] }) {
  const [metric, setMetric] = useState<MetricKey>("accuracy");

  const data = useMemo(
    () =>
      rows
        .filter((row) => row[metric] !== undefined)
        .map((row) => ({ name: `${row.model} — ${row.variant}`, value: (row[metric] as number) * 100 })),
    [rows, metric]
  );

  return (
    <Card>
      <CardHeader
        title="Model Comparison"
        subtitle="Completed benchmark results across branches. Metrics come from tracked evaluation configs, not hardcoded values."
        icon={<BarChart2 size={18} />}
        action={
          <div className="flex rounded-lg border border-slate-200 p-0.5">
            {METRIC_OPTIONS.map((option) => (
              <button
                key={option.key}
                type="button"
                onClick={() => setMetric(option.key)}
                className={cn(
                  "px-2.5 py-1 text-[11px] font-medium rounded-md transition-colors",
                  metric === option.key ? "bg-[var(--color-accent-600)] text-white" : "text-slate-500 hover:text-slate-700"
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        }
      />
      {data.length === 0 ? (
        <EmptyState title="No comparable results yet" description="This chart populates automatically as branches complete benchmarking." />
      ) : (
        <div style={{ width: "100%", height: Math.max(220, data.length * 42) }}>
          <ResponsiveContainer>
            <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#eef0f3" />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} unit="%" />
              <YAxis type="category" dataKey="name" width={220} tick={{ fontSize: 11, fill: "#475569" }} axisLine={false} tickLine={false} />
              <RechartsTooltip
                cursor={{ fill: "rgba(47,125,114,0.06)" }}
                contentStyle={{ borderRadius: 10, border: "1px solid #e3e6ea", fontSize: 12 }}
                formatter={(value: number) => `${value.toFixed(2)}%`}
              />
              <Bar dataKey="value" fill="#2f7d72" radius={[0, 6, 6, 0]} maxBarSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}
