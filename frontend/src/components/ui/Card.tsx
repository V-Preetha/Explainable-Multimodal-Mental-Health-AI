import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  hoverable?: boolean;
  padded?: boolean;
}

export function Card({ children, className, hoverable, padded = true, ...rest }: CardProps) {
  return (
    <div
      className={cn("card-surface", padded && "p-5 sm:p-6", hoverable && "card-surface-hover", className)}
      {...rest}
    >
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, icon, action }: { title: ReactNode; subtitle?: ReactNode; icon?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 mb-4">
      <div className="flex items-start gap-3">
        {icon && <div className="mt-0.5 shrink-0 text-[var(--color-accent-600)]">{icon}</div>}
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-slate-900">{title}</h3>
          {subtitle && <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{subtitle}</p>}
        </div>
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}
