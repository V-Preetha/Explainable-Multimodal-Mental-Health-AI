import { cn } from "@/lib/utils";

interface ProgressBarProps {
  value: number; // 0-1
  max?: number;
  label?: string;
  valueLabel?: string;
  tone?: "accent" | "healthy" | "mild" | "moderate" | "severe" | "neutral";
  className?: string;
  size?: "sm" | "md";
}

const TONE_BG: Record<NonNullable<ProgressBarProps["tone"]>, string> = {
  accent: "bg-[var(--color-accent-500)]",
  healthy: "bg-[var(--color-status-healthy)]",
  mild: "bg-[var(--color-status-mild)]",
  moderate: "bg-[var(--color-status-moderate)]",
  severe: "bg-[var(--color-status-severe)]",
  neutral: "bg-slate-400",
};

export function ProgressBar({ value, label, valueLabel, tone = "accent", className, size = "md" }: ProgressBarProps) {
  const percent = Math.min(100, Math.max(0, value * 100));
  return (
    <div className={cn("w-full", className)}>
      {(label || valueLabel) && (
        <div className="flex items-center justify-between mb-1.5">
          {label && <span className="text-xs font-medium text-slate-600">{label}</span>}
          {valueLabel && <span className="text-xs font-semibold text-slate-800 tabular-nums">{valueLabel}</span>}
        </div>
      )}
      <div className={cn("w-full rounded-full bg-slate-100 overflow-hidden", size === "sm" ? "h-1.5" : "h-2.5")}>
        <div
          className={cn("h-full rounded-full transition-[width] duration-500 ease-out", TONE_BG[tone])}
          style={{ width: `${percent}%` }}
          role="progressbar"
          aria-valuenow={Math.round(percent)}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}
