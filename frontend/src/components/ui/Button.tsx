import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: ReactNode;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary:
    "bg-[var(--color-accent-600)] text-white hover:bg-[var(--color-accent-700)] shadow-sm disabled:bg-slate-300",
  secondary:
    "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50 disabled:text-slate-400",
  ghost: "bg-transparent text-slate-600 hover:bg-slate-100 disabled:text-slate-300",
  danger: "bg-white text-red-600 border border-red-200 hover:bg-red-50 disabled:text-slate-300",
};

const SIZE_CLASSES: Record<Size, string> = {
  sm: "text-xs px-3 py-1.5 gap-1.5",
  md: "text-sm px-4 py-2.5 gap-2",
  lg: "text-base px-6 py-3 gap-2.5",
};

export function Button({ variant = "primary", size = "md", loading, icon, className, children, disabled, ...rest }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-xl font-medium transition-colors duration-150",
        "focus-visible:outline-none disabled:cursor-not-allowed",
        VARIANT_CLASSES[variant],
        SIZE_CLASSES[size],
        className
      )}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Loader2 className="animate-spin" size={16} /> : icon}
      {children}
    </button>
  );
}
