import { useRef, useState } from "react";
import { RefreshCw, ScanFace, X } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Dropzone } from "@/components/ui/Dropzone";
import { Button } from "@/components/ui/Button";
import { useAssessment } from "@/context/AssessmentContext";

export function FaceInput() {
  const { faceImage, faceImagePreviewUrl, setFaceImage, result } = useAssessment();
  const [error, setError] = useState<string | null>(null);
  const replaceInputRef = useRef<HTMLInputElement>(null);

  const faceResult = result?.explainability?.face;

  return (
    <Card>
      <CardHeader
        title="Facial Image"
        subtitle="Upload a clear, front-facing photo (JPEG or PNG)."
        icon={<ScanFace size={18} />}
        action={<Badge tone="accent">Facial Analysis</Badge>}
      />

      {!faceImage ? (
        <Dropzone
          accept="image/jpeg,image/png"
          acceptLabel="JPEG / PNG"
          title="Upload facial image"
          onFileSelected={(file) => {
            setError(null);
            setFaceImage(file);
          }}
          onError={setError}
        />
      ) : (
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative w-full sm:w-40 aspect-square rounded-xl overflow-hidden bg-slate-100 shrink-0">
            {faceImagePreviewUrl && (
              <img src={faceImagePreviewUrl} alt="Uploaded facial photo preview" className="h-full w-full object-cover" />
            )}
          </div>
          <div className="flex-1 flex flex-col justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-slate-800 truncate max-w-[220px]">{faceImage.name}</p>
              <p className="text-xs text-slate-400 mt-0.5">{(faceImage.size / 1024).toFixed(0)} KB</p>
            </div>
            <div className="flex gap-2">
              <input
                ref={replaceInputRef}
                type="file"
                accept="image/jpeg,image/png"
                className="sr-only"
                onChange={(e) => e.target.files?.[0] && setFaceImage(e.target.files[0])}
              />
              <Button variant="secondary" size="sm" icon={<RefreshCw size={13} />} onClick={() => replaceInputRef.current?.click()}>
                Replace
              </Button>
              <Button variant="danger" size="sm" icon={<X size={13} />} onClick={() => setFaceImage(null)}>
                Remove
              </Button>
            </div>
          </div>
        </div>
      )}
      {error && <p className="text-xs text-red-600 mt-2">{error}</p>}

      <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-3 gap-3">
        <ReservedMetric label="Detected emotion" value={faceResult?.predicted_emotion} />
        <ReservedMetric label="Confidence" value={faceResult?.confidence !== undefined ? `${Math.round(faceResult.confidence * 100)}%` : undefined} />
        <ReservedMetric label="Face quality" value={faceResult?.face_quality !== undefined ? `${Math.round(faceResult.face_quality * 100)}%` : undefined} />
      </div>
    </Card>
  );
}

function ReservedMetric({ label, value }: { label: string; value?: string }) {
  return (
    <div>
      <p className="text-[11px] text-slate-400">{label}</p>
      <p className="text-sm font-medium text-slate-700 mt-0.5">{value ?? "—"}</p>
    </div>
  );
}
