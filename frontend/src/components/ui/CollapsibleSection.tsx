import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export function CollapsibleSection({
  title,
  description,
  defaultOpen = true,
  children,
}: {
  title: string;
  description?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border border-[var(--color-border)] rounded-xl overflow-hidden bg-white">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="w-full flex items-center justify-between gap-3 px-4 py-3.5 text-left hover:bg-slate-50/80 transition-colors"
      >
        <div>
          <h4 className="text-sm font-semibold text-slate-800">{title}</h4>
          {description && <p className="text-xs text-slate-500 mt-0.5">{description}</p>}
        </div>
        <ChevronDown size={16} className={cn("shrink-0 text-slate-400 transition-transform duration-200", open && "rotate-180")} />
      </button>
      {open && <div className="px-4 pb-4 pt-1 grid grid-cols-1 sm:grid-cols-2 gap-x-5 gap-y-4">{children}</div>}
    </div>
  );
}
