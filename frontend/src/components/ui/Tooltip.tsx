import { useId, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Tooltip({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();

  return (
    <span
      className={cn("relative inline-flex", className)}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <span aria-describedby={id} tabIndex={0} className="inline-flex focus-visible:outline-none">
        {children}
      </span>
      {open && (
        <span
          id={id}
          role="tooltip"
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-20 w-max max-w-[220px] rounded-lg bg-slate-900 px-2.5 py-1.5 text-[11px] leading-snug text-white shadow-lg"
        >
          {label}
        </span>
      )}
    </span>
  );
}
