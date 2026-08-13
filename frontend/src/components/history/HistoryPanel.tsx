import { History, Trash2 } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { Button } from "@/components/ui/Button";
import { clearHistory, type HistoryEntry } from "@/lib/storage";
import { STATUS_LABELS } from "@/types/api";

const STATUS_TONE = {
  Healthy: "healthy",
  Mild_Stress: "mild",
  Moderate_Stress: "moderate",
  Severe_Stress: "severe",
} as const;

export function HistoryPanel({ history, onClear }: { history: HistoryEntry[]; onClear: () => void }) {
  return (
    <Card>
      <CardHeader
        title="Session History"
        subtitle="Stored only in this browser (localStorage) &mdash; never sent to a server."
        icon={<History size={18} />}
        action={
          history.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              icon={<Trash2 size={13} />}
              onClick={() => {
                clearHistory();
                onClear();
              }}
            >
              Clear
            </Button>
          )
        }
      />
      {history.length === 0 ? (
        <EmptyState title="No assessments yet" description="Completed assessments in this browser session will be listed here." />
      ) : (
        <div className="space-y-2 max-h-72 overflow-y-auto scrollbar-thin pr-1">
          {history.map((entry) => (
            <div key={entry.id} className="flex items-center justify-between rounded-lg border border-slate-100 px-3 py-2.5">
              <div className="flex items-center gap-2.5">
                <Badge tone={STATUS_TONE[entry.status]}>{STATUS_LABELS[entry.status]}</Badge>
                {entry.isDemo && <span className="text-[10px] text-amber-600 font-medium">demo</span>}
              </div>
              <div className="text-right">
                <p className="text-xs font-medium text-slate-600 tabular-nums">{Math.round(entry.confidence * 100)}% confidence</p>
                <p className="text-[11px] text-slate-400">{new Date(entry.timestamp).toLocaleString()}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
