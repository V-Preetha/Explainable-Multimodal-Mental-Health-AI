import { useState } from "react";
import { Grid3x3 } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { cn } from "@/lib/utils";
import type { ConfusionMatrixData } from "@/types/api";

type ViewMode = "counts" | "normalized";

function cellColor(value: number, max: number): string {
  const intensity = max > 0 ? value / max : 0;
  const alpha = 0.06 + intensity * 0.55;
  return `rgba(47, 125, 114, ${alpha.toFixed(3)})`;
}

export function ConfusionMatrix({ title, subtitle, data }: { title: string; subtitle?: string; data: ConfusionMatrixData }) {
  const [mode, setMode] = useState<ViewMode>("normalized");
  const matrix = mode === "counts" ? data.counts : data.normalized;
  const max = Math.max(...matrix.flat());

  return (
    <Card>
      <CardHeader
        title={title}
        subtitle={subtitle}
        icon={<Grid3x3 size={18} />}
        action={
          <div className="flex rounded-lg border border-slate-200 p-0.5">
            {(["counts", "normalized"] as ViewMode[]).map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setMode(option)}
                className={cn(
                  "px-2.5 py-1 text-[11px] font-medium rounded-md transition-colors capitalize",
                  mode === option ? "bg-[var(--color-accent-600)] text-white" : "text-slate-500 hover:text-slate-700"
                )}
              >
                {option}
              </button>
            ))}
          </div>
        }
      />
      <div className="overflow-x-auto scrollbar-thin">
        <table className="border-separate border-spacing-[3px] text-[11px]">
          <thead>
            <tr>
              <th className="w-20" />
              {data.classes.map((cls) => (
                <th key={cls} className="px-1 pb-1 font-medium text-slate-500 whitespace-nowrap capitalize">
                  {cls}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.classes.map((rowClass, rowIndex) => (
              <tr key={rowClass}>
                <th className="pr-2 text-right font-medium text-slate-500 whitespace-nowrap capitalize">{rowClass}</th>
                {matrix[rowIndex].map((value, colIndex) => (
                  <td
                    key={colIndex}
                    className="h-9 w-9 text-center rounded-md tabular-nums text-slate-700"
                    style={{ backgroundColor: cellColor(value, max) }}
                  >
                    {mode === "normalized" ? value.toFixed(2) : value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-slate-400 mt-3">Rows: true class &middot; Columns: predicted class</p>
    </Card>
  );
}
