import type { ReactNode } from "react";
import { Badge } from "@/components/ui/Badge";
import { Layers, Sparkles } from "lucide-react";

export function Header({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <header className="mb-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2.5">
            <Badge tone="accent" icon={<Layers size={12} />}>
              Multimodal AI
            </Badge>
            <Badge tone="neutral" icon={<Sparkles size={12} />}>
              Explainable
            </Badge>
          </div>
          <h1 className="text-2xl sm:text-[28px] font-semibold tracking-tight text-slate-900">{title}</h1>
          {subtitle && <p className="text-sm text-slate-500 mt-1.5 max-w-2xl">{subtitle}</p>}
        </div>
        {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
      </div>
    </header>
  );
}
