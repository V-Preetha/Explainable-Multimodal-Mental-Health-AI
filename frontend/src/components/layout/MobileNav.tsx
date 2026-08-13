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

export function MobileNav({ active, onNavigate }: { active: PageKey; onNavigate: (page: PageKey) => void }) {
  return (
    <nav className="lg:hidden fixed bottom-0 inset-x-0 z-30 border-t border-[var(--color-border)] bg-white/95 backdrop-blur-sm flex justify-around px-1 py-1.5">
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
              "flex flex-col items-center gap-0.5 rounded-lg px-2.5 py-1.5 text-[10px] font-medium min-w-[56px]",
              isActive ? "text-[var(--color-accent-700)]" : "text-slate-400"
            )}
          >
            <Icon size={18} className={isActive ? "text-[var(--color-accent-600)]" : "text-slate-400"} />
            {item.label}
          </button>
        );
      })}
    </nav>
  );
}
