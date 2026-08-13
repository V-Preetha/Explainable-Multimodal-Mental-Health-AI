import { Header } from "@/components/layout/Header";
import { FaceExplainability } from "@/components/explainability/FaceExplainability";
import { SpeechExplainability } from "@/components/explainability/SpeechExplainability";
import { NumericalExplainability } from "@/components/explainability/NumericalExplainability";

export function ExplainabilityPage() {
  return (
    <>
      <Header title="Explainability" subtitle="How each modality arrived at its individual prediction." />
      <div className="space-y-5">
        <FaceExplainability />
        <SpeechExplainability />
        <NumericalExplainability />
      </div>
    </>
  );
}
