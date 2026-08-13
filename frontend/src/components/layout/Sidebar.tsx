import { ClipboardList, FileBarChart2, Info, LayoutDashboard, Microscope, type LucideIcon } from "lucide-react";
import { NAV_ITEMS, type PageKey } from "@/lib/constants";
import { cn } from "@/lib/utils";

const ICONS: Record<PageKey, LucideIcon> = {
  assessment: ClipboardList,
  results: LayoutDashboard,
  explainability: Microscope,
  insights: FileBarChart2,
  about: Info,
};

export function Sidebar({ active, onNavigate }: { active: PageKey; onNavigate: (page: PageKey) => void }) {
  return (
    <aside className="hidden lg:flex lg:w-60 shrink-0 flex-col border-r border-[var(--color-border)] bg-white/70 backdrop-blur-sm px-3 py-6">
      <div className="px-3 mb-8">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-[var(--color-accent-600)] flex items-center justify-center text-white font-semibold text-sm">
            M
          </div>
          <div className="leading-tight">
            <p className="text-sm font-semibold text-slate-900">MindSignal</p>
            <p className="text-[11px] text-slate-400">Multimodal Assessment</p>
          </div>
        </div>
      </div>
      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map((item) => {
          const Icon = ICONS[item.key];
          const isActive = item.key === active;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => onNavigate(item.key)}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                "flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors text-left",
                isActive ? "bg-[var(--color-accent-50)] text-[var(--color-accent-700)]" : "text-slate-500 hover:bg-slate-100 hover:text-slate-700"
              )}
            >
              <Icon size={17} className={cn(isActive ? "text-[var(--color-accent-600)]" : "text-slate-400")} />
              {item.label}
            </button>
          );
        })}
      </nav>
      <div className="mt-auto px-3 pt-6">
        <p className="text-[11px] leading-relaxed text-slate-400">
          Decision-support prototype. Not a substitute for professional clinical diagnosis.
        </p>
      </div>
    </aside>
  );
}
