import { useId } from "react";
import type { NumericalFieldSpec } from "@/lib/constants";
import { Tooltip } from "@/components/ui/Tooltip";
import { Info } from "lucide-react";

interface FieldControlProps {
  spec: NumericalFieldSpec;
  value: number;
  onChange: (value: number) => void;
}

export function FieldControl({ spec, value, onChange }: FieldControlProps) {
  const id = useId();

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={id} className="text-xs font-medium text-slate-600 flex items-center gap-1.5">
          {spec.label}
          {spec.helper && (
            <Tooltip label={spec.helper}>
              <Info size={12} className="text-slate-400" />
            </Tooltip>
          )}
        </label>
        <span className="text-xs font-semibold text-slate-800 tabular-nums">
          {value}
          {spec.unit ? ` ${spec.unit}` : ""}
        </span>
      </div>
      {spec.control === "slider" ? (
        <input
          id={id}
          type="range"
          min={spec.min}
          max={spec.max}
          step={spec.step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full accent-[var(--color-accent-600)] h-1.5 cursor-pointer"
          aria-valuetext={`${value}${spec.unit ? ` ${spec.unit}` : ""}`}
        />
      ) : (
        <input
          id={id}
          type="number"
          min={spec.min}
          max={spec.max}
          step={spec.step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-300)]"
        />
      )}
    </div>
  );
}
