import { Activity, AudioLines, GitMerge, Microscope, ScanFace } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Card, CardHeader } from "@/components/ui/Card";
import { DISCLAIMER_LONG } from "@/lib/constants";

const SECTIONS = [
  {
    icon: ScanFace,
    title: "Facial Branch",
    body: "A ConvNeXt-Tiny image encoder analyzes a facial photo and produces a 256-dimensional embedding summarizing visual emotional expression, alongside a 7-class emotion prediction (FER2013 taxonomy).",
  },
  {
    icon: AudioLines,
    title: "Speech Branch",
    body: "An emotion2vec+ encoder processes a short voice recording. Frame-level representations are combined with attentive statistics pooling into a 256-dimensional embedding, alongside an 8-class emotion prediction (RAVDESS taxonomy).",
  },
  {
    icon: Activity,
    title: "Numerical Branch",
    body: "Eighteen behavioral, facial-summary, speech-summary and physiological indicators are encoded by a compact multitask MLP into a 128-dimensional embedding, jointly trained for status classification and Depression/Anxiety/Stress regression.",
  },
  {
    icon: GitMerge,
    title: "Gated Fusion",
    body: "The three modality embeddings are combined through a learned gating mechanism that weighs each modality's contribution per-sample, producing a shared representation used for the final status classification and D/A/S regression heads.",
  },
  {
    icon: Microscope,
    title: "Explainability",
    body: "Facial predictions are accompanied by Grad-CAM visualizations highlighting influential image regions; speech predictions expose acoustic feature contribution; numerical predictions expose ranked feature importance. None of these are claims of causal mechanism.",
  },
];

export function AboutPage() {
  return (
    <>
      <Header title="About" subtitle="What this system is, how it works, and what it is not." />

      <div className="space-y-5">
        <Card>
          <CardHeader title="Multimodal Framework" />
          <p className="text-sm text-slate-600 leading-relaxed">
            This system estimates a mental-health-related status and continuous Depression, Anxiety and Stress scores by combining
            three independent modalities &mdash; a facial image, a speech sample, and a set of behavioral/physiological indicators
            &mdash; through a gated multimodal fusion model. Each modality is analyzed by its own dedicated encoder before being
            combined.
          </p>
        </Card>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {SECTIONS.map((section) => (
            <Card key={section.title}>
              <CardHeader title={section.title} icon={<section.icon size={18} />} />
              <p className="text-sm text-slate-600 leading-relaxed">{section.body}</p>
            </Card>
          ))}
        </div>

        <Card className="border-amber-200 bg-amber-50/60">
          <CardHeader title="Disclaimer" />
          <p className="text-sm text-amber-800 leading-relaxed">{DISCLAIMER_LONG}</p>
        </Card>
      </div>
    </>
  );
}
