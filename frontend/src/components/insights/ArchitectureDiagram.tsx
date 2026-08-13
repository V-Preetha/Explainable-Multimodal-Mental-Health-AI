import { ArrowRight, GitMerge, Layers } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";

function EncoderNode({ label, sub, dim }: { label: string; sub: string; dim: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-center w-full">
      <p className="text-sm font-semibold text-slate-800">{label}</p>
      <p className="text-[11px] text-slate-500 mt-0.5">{sub}</p>
      <p className="text-[11px] font-medium text-[var(--color-accent-600)] mt-1">{dim}</p>
    </div>
  );
}

export function ArchitectureDiagram() {
  return (
    <Card>
      <CardHeader title="System Architecture" subtitle="Three modality encoders feed a gated fusion trunk that produces both outputs." icon={<Layers size={18} />} />

      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr_auto_1fr] items-center gap-4">
        <div className="space-y-3">
          <p className="text-[11px] font-medium text-slate-400 text-center md:text-left">Facial Image</p>
          <EncoderNode label="ConvNeXt Encoder" sub="ImageNet-pretrained backbone" dim="256-D embedding" />
          <p className="text-[11px] font-medium text-slate-400 text-center md:text-left mt-4">Speech</p>
          <EncoderNode label="emotion2vec+ Encoder" sub="Attentive statistics pooling" dim="256-D embedding" />
          <p className="text-[11px] font-medium text-slate-400 text-center md:text-left mt-4">Numerical</p>
          <EncoderNode label="Numerical Encoder" sub="18 -> 256 -> 256 -> 128 MLP" dim="128-D embedding" />
        </div>

        <div className="hidden md:flex items-center justify-center text-slate-300">
          <ArrowRight size={22} />
        </div>

        <div className="flex justify-center">
          <div className="rounded-xl border border-[var(--color-accent-200,#bfe0da)] bg-[var(--color-accent-50)] px-5 py-8 text-center w-full">
            <GitMerge size={22} className="mx-auto text-[var(--color-accent-600)] mb-2" />
            <p className="text-sm font-semibold text-[var(--color-accent-700)]">Gated Fusion</p>
            <p className="text-[11px] text-[var(--color-accent-600)] mt-1">Modality attention weighting</p>
            <div className="h-px bg-[var(--color-accent-100)] my-3" />
            <p className="text-sm font-semibold text-[var(--color-accent-700)]">Shared Representation</p>
          </div>
        </div>

        <div className="hidden md:flex items-center justify-center text-slate-300">
          <ArrowRight size={22} />
        </div>

        <div className="space-y-3">
          <EncoderNode label="Mental Health Status" sub="4-class classification head" dim="Healthy / Mild / Moderate / Severe" />
          <EncoderNode label="D / A / S Regression" sub="Depression, Anxiety, Stress" dim="3 continuous outputs" />
        </div>
      </div>
    </Card>
  );
}
