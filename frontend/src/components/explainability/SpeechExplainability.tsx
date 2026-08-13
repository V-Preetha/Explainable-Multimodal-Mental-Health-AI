import { AudioLines } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { RankedBarChart } from "@/components/explainability/RankedBarChart";
import { useAssessment } from "@/context/AssessmentContext";

export function SpeechExplainability() {
  const { audioSample, result } = useAssessment();
  const speech = result?.explainability?.speech;

  return (
    <Card>
      <CardHeader title="Speech Analysis" subtitle="Acoustic feature contribution behind the predicted vocal emotion." icon={<AudioLines size={18} />} />

      {!audioSample ? (
        <EmptyState icon={<AudioLines size={30} strokeWidth={1.25} />} title="No speech sample provided" description="Upload or record a speech sample on the Assessment page to see analysis here." />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <Metric label="Predicted emotion" value={speech?.predicted_emotion} />
            <Metric label="Emotion confidence" value={speech?.confidence !== undefined ? `${Math.round(speech.confidence * 100)}%` : undefined} />
          </div>
          <p className="text-[11px] font-medium text-slate-500 mb-2">Acoustic feature contribution</p>
          <RankedBarChart
            data={speech?.feature_contribution}
            emptyTitle="Feature contribution unavailable"
            emptyDescription="Pitch, MFCC, energy, speech-rate and pause-characteristic contributions will appear here once the backend supplies them."
            color="#2f7d72"
          />
        </>
      )}
    </Card>
  );
}

function Metric({ label, value }: { label: string; value?: string }) {
  return (
    <div>
      <p className="text-[11px] text-slate-400">{label}</p>
      <p className="text-sm font-medium text-slate-700 mt-0.5">{value ?? "—"}</p>
    </div>
  );
}
