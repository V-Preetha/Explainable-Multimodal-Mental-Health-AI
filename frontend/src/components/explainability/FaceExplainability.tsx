import { ImageOff, ScanFace } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { useAssessment } from "@/context/AssessmentContext";

export function FaceExplainability() {
  const { faceImagePreviewUrl, result } = useAssessment();
  const face = result?.explainability?.face;

  return (
    <Card>
      <CardHeader title="Facial Explainability" subtitle="Grad-CAM highlights the regions the facial model weighted most heavily." icon={<ScanFace size={18} />} />

      {!faceImagePreviewUrl ? (
        <EmptyState icon={<ImageOff size={30} strokeWidth={1.25} />} title="No facial image provided" description="Upload a facial image on the Assessment page to see Grad-CAM explainability here." />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4">
            <ImageTile label="Original image" src={faceImagePreviewUrl} />
            <ImageTile label="Grad-CAM overlay" src={face?.overlay_image_url} emptyHint="Returned by the backend once available" />
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-100">
            <Metric label="Predicted facial emotion" value={face?.predicted_emotion} />
            <Metric label="Confidence" value={face?.confidence !== undefined ? `${Math.round(face.confidence * 100)}%` : undefined} />
          </div>

          <p className="text-xs text-slate-400 leading-relaxed mt-4">
            Highlighted regions indicate areas that contributed most strongly to the facial model's prediction. This does not prove a
            causal relationship between those regions and the predicted emotion.
          </p>
        </>
      )}
    </Card>
  );
}

function ImageTile({ label, src, emptyHint }: { label: string; src?: string; emptyHint?: string }) {
  return (
    <div>
      <p className="text-[11px] font-medium text-slate-500 mb-1.5">{label}</p>
      <div className="aspect-square rounded-xl bg-slate-100 overflow-hidden flex items-center justify-center">
        {src ? (
          <img src={src} alt={label} className="h-full w-full object-cover" />
        ) : (
          <p className="text-[11px] text-slate-400 px-3 text-center">{emptyHint ?? "Unavailable"}</p>
        )}
      </div>
    </div>
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
