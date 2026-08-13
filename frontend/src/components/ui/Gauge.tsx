import { cn } from "@/lib/utils";

interface GaugeProps {
  value: number; // 0-1
  size?: number;
  strokeWidth?: number;
  tone?: "accent" | "healthy" | "mild" | "moderate" | "severe" | "neutral";
  centerLabel?: string;
  centerSubLabel?: string;
  className?: string;
}

const TONE_STROKE: Record<NonNullable<GaugeProps["tone"]>, string> = {
  accent: "#2f7d72",
  healthy: "#2f8f5b",
  mild: "#b8860b",
  moderate: "#c26a2b",
  severe: "#b6432f",
  neutral: "#94a3b8",
};

export function Gauge({ value, size = 132, strokeWidth = 10, tone = "accent", centerLabel, centerSubLabel, className }: GaugeProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(1, Math.max(0, value));
  const offset = circumference * (1 - clamped);

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)} style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#eef0f3" strokeWidth={strokeWidth} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={TONE_STROKE[tone]}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 600ms ease-out" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {centerLabel && <span className="text-xl font-semibold text-slate-900 tabular-nums">{centerLabel}</span>}
        {centerSubLabel && <span className="text-[11px] text-slate-500 mt-0.5">{centerSubLabel}</span>}
      </div>
    </div>
  );
}
