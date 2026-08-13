import { PlayCircle } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { FaceInput } from "@/components/assessment/FaceInput";
import { AudioInput } from "@/components/assessment/AudioInput";
import { NumericalForm } from "@/components/assessment/NumericalForm";
import { ProcessingOverlay } from "@/components/assessment/ProcessingOverlay";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useAssessment } from "@/context/AssessmentContext";
import type { PageKey } from "@/lib/constants";

export function AssessmentPage({ onNavigate }: { onNavigate: (page: PageKey) => void }) {
  const { faceImage, audioSample, numericalValues, stage, error, runFullAssessment } = useAssessment();

  const isProcessing = ["analyzing_face", "analyzing_speech", "processing_numerical", "fusing", "generating"].includes(stage);
  const hasAnyInput = !!faceImage || !!audioSample || Object.keys(numericalValues).length > 0;

  async function handleRun() {
    await runFullAssessment();
    onNavigate("results");
  }

  return (
    <>
      <Header
        title="New Assessment"
        subtitle="AI-assisted analysis of facial, vocal, behavioral and physiological indicators"
      />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <FaceInput />
        <AudioInput />
      </div>

      <div className="mt-5">
        <NumericalForm />
      </div>

      {error && (
        <Card className="mt-5 border-red-200 bg-red-50/60">
          <p className="text-sm font-medium text-red-700">Assessment failed</p>
          <p className="text-xs text-red-600 mt-1">{error}</p>
        </Card>
      )}

      <div className="sticky bottom-4 mt-6 flex justify-center lg:justify-end">
        <div className="card-surface bg-white/95 backdrop-blur-sm px-5 py-3.5 flex items-center gap-4">
          <p className="text-xs text-slate-500 hidden sm:block max-w-[220px]">
            {hasAnyInput ? "At least one modality provided." : "Provide at least one modality to run the assessment."}
          </p>
          <Button size="lg" icon={<PlayCircle size={18} />} loading={isProcessing} onClick={handleRun} disabled={!hasAnyInput}>
            Run Multimodal Assessment
          </Button>
        </div>
      </div>

      {isProcessing && <ProcessingOverlay stage={stage} />}
    </>
  );
}
