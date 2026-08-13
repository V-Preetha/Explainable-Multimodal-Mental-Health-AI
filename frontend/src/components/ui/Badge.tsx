import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type BadgeTone = "accent" | "neutral" | "healthy" | "mild" | "moderate" | "severe" | "demo";

const TONE_CLASSES: Record<BadgeTone, string> = {
  accent: "bg-[var(--color-accent-50)] text-[var(--color-accent-700)] ring-1 ring-inset ring-[var(--color-accent-100)]",
  neutral: "bg-slate-100 text-slate-600 ring-1 ring-inset ring-slate-200",
  healthy: "bg-[var(--color-status-healthy-bg)] text-[var(--color-status-healthy)] ring-1 ring-inset ring-[var(--color-status-healthy)]/20",
  mild: "bg-[var(--color-status-mild-bg)] text-[var(--color-status-mild)] ring-1 ring-inset ring-[var(--color-status-mild)]/20",
  moderate: "bg-[var(--color-status-moderate-bg)] text-[var(--color-status-moderate)] ring-1 ring-inset ring-[var(--color-status-moderate)]/20",
  severe: "bg-[var(--color-status-severe-bg)] text-[var(--color-status-severe)] ring-1 ring-inset ring-[var(--color-status-severe)]/20",
  demo: "bg-amber-50 text-amber-700 ring-1 ring-inset ring-amber-200",
};

export function Badge({ children, tone = "neutral", className, icon }: { children: ReactNode; tone?: BadgeTone; className?: string; icon?: ReactNode }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium", TONE_CLASSES[tone], className)}>
      {icon}
      {children}
    </span>
  );
}
