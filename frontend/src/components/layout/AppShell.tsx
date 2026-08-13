import type { ReactNode } from "react";
import { FlaskConical } from "lucide-react";
import { Sidebar } from "@/components/layout/Sidebar";
import { MobileNav } from "@/components/layout/MobileNav";
import type { PageKey } from "@/lib/constants";
import { DEMO_MODE } from "@/lib/api";

export function AppShell({ active, onNavigate, children }: { active: PageKey; onNavigate: (page: PageKey) => void; children: ReactNode }) {
  return (
    <div className="min-h-screen flex">
      <Sidebar active={active} onNavigate={onNavigate} />
      <div className="flex-1 flex flex-col min-w-0">
        {DEMO_MODE && (
          <div className="flex items-center justify-center gap-2 bg-amber-50 border-b border-amber-200 text-amber-800 text-xs font-medium py-1.5 px-4">
            <FlaskConical size={13} />
            Demo mode &mdash; showing clearly labeled sample output. Connect a backend and set VITE_DEMO_MODE=false for live predictions.
          </div>
        )}
        <main className="flex-1 px-5 sm:px-8 lg:px-10 py-8 pb-24 lg:pb-8 max-w-6xl w-full mx-auto">{children}</main>
      </div>
      <MobileNav active={active} onNavigate={onNavigate} />
    </div>
  );
}
