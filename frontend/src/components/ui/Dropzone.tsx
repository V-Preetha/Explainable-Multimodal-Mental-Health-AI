import { useCallback, useId, useRef, useState, type DragEvent, type ReactNode } from "react";
import { UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";

interface DropzoneProps {
  accept: string;
  acceptLabel: string;
  icon?: ReactNode;
  title: string;
  onFileSelected: (file: File) => void;
  onError?: (message: string) => void;
  maxSizeMb?: number;
  disabled?: boolean;
}

export function Dropzone({ accept, acceptLabel, icon, title, onFileSelected, onError, maxSizeMb = 25, disabled }: DropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();

  const acceptedTypes = accept.split(",").map((t) => t.trim());

  const validateAndEmit = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const matchesType = acceptedTypes.some((type) => {
        if (type.endsWith("/*")) return file.type.startsWith(type.replace("/*", "/"));
        return file.type === type || file.name.toLowerCase().endsWith(type.replace("*", ""));
      });
      if (!matchesType) {
        onError?.(`Unsupported file type. Expected ${acceptLabel}.`);
        return;
      }
      if (file.size > maxSizeMb * 1024 * 1024) {
        onError?.(`File is too large. Maximum size is ${maxSizeMb} MB.`);
        return;
      }
      onFileSelected(file);
    },
    [acceptLabel, acceptedTypes, maxSizeMb, onError, onFileSelected]
  );

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    validateAndEmit(event.dataTransfer.files?.[0]);
  }

  return (
    <div
      className={cn(
        "relative rounded-xl border-2 border-dashed transition-colors duration-150 px-5 py-8 text-center cursor-pointer",
        isDragging ? "border-[var(--color-accent-500)] bg-[var(--color-accent-50)]" : "border-slate-200 bg-slate-50/60 hover:bg-slate-50",
        disabled && "opacity-50 pointer-events-none"
      )}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label={title}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
    >
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={accept}
        className="sr-only"
        disabled={disabled}
        onChange={(e) => validateAndEmit(e.target.files?.[0])}
      />
      <div className="flex flex-col items-center gap-2 text-slate-500">
        {icon ?? <UploadCloud size={26} strokeWidth={1.5} />}
        <p className="text-sm font-medium text-slate-700">{title}</p>
        <p className="text-xs text-slate-400">Drag &amp; drop, or click to browse &middot; {acceptLabel}</p>
      </div>
    </div>
  );
}
