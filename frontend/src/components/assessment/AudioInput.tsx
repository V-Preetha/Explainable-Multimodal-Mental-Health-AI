import { useEffect, useRef, useState } from "react";
import { AudioLines, Mic, Square, X } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Dropzone } from "@/components/ui/Dropzone";
import { Button } from "@/components/ui/Button";
import { useAssessment } from "@/context/AssessmentContext";

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function AudioInput() {
  const { audioSample, audioPreviewUrl, setAudioSample, result } = useAssessment();
  const [error, setError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [recordingSupported] = useState(() => typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const speechResult = result?.explainability?.speech;

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const file = new File([blob], `recording-${Date.now()}.webm`, { type: "audio/webm" });
        setAudioSample(file);
        streamRef.current?.getTracks().forEach((track) => track.stop());
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((v) => v + 1), 1000);
    } catch {
      setError("Microphone access was denied or is unavailable.");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
  }

  return (
    <Card>
      <CardHeader
        title="Speech Sample"
        subtitle="Upload a short voice recording (WAV or MP3), or record one directly."
        icon={<AudioLines size={18} />}
        action={<Badge tone="accent">Speech Analysis</Badge>}
      />

      {!audioSample ? (
        <div className="space-y-3">
          <Dropzone
            accept="audio/wav,audio/mpeg,audio/mp3,.wav,.mp3"
            acceptLabel="WAV / MP3"
            icon={<AudioLines size={26} strokeWidth={1.5} />}
            title="Upload audio file"
            onFileSelected={(file) => {
              setError(null);
              setAudioSample(file);
            }}
            onError={setError}
          />

          {recordingSupported && (
            <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3">
              <div className="flex items-center gap-3">
                <div className={isRecording ? "text-red-500" : "text-slate-400"}>
                  <Mic size={18} />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-700">{isRecording ? "Recording…" : "Or record directly"}</p>
                  {isRecording && <p className="text-xs text-slate-400 tabular-nums">{formatDuration(elapsed)}</p>}
                </div>
              </div>
              {isRecording ? (
                <Button variant="danger" size="sm" icon={<Square size={13} />} onClick={stopRecording}>
                  Stop Recording
                </Button>
              ) : (
                <Button variant="secondary" size="sm" icon={<Mic size={13} />} onClick={startRecording}>
                  Start Recording
                </Button>
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3.5">
            <Waveform active={false} />
            {audioPreviewUrl && (
              // eslint-disable-next-line jsx-a11y/media-has-caption
              <audio controls src={audioPreviewUrl} className="w-full mt-3 h-9" />
            )}
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-800 truncate max-w-[220px]">{audioSample.name}</p>
              <p className="text-xs text-slate-400 mt-0.5">{(audioSample.size / 1024).toFixed(0)} KB</p>
            </div>
            <Button variant="danger" size="sm" icon={<X size={13} />} onClick={() => setAudioSample(null)}>
              Remove
            </Button>
          </div>
        </div>
      )}
      {error && <p className="text-xs text-red-600 mt-2">{error}</p>}

      <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-3 gap-3">
        <ReservedMetric label="Predicted emotion" value={speechResult?.predicted_emotion} />
        <ReservedMetric label="Confidence" value={speechResult?.confidence !== undefined ? `${Math.round(speechResult.confidence * 100)}%` : undefined} />
        <ReservedMetric label="Speech quality" value={speechResult?.speech_quality !== undefined ? `${Math.round(speechResult.speech_quality * 100)}%` : undefined} />
      </div>
    </Card>
  );
}

function Waveform({ active }: { active: boolean }) {
  const bars = Array.from({ length: 40 });
  return (
    <div className="flex items-center gap-[3px] h-8">
      {bars.map((_, i) => {
        const height = 20 + Math.round(Math.abs(Math.sin(i * 0.7)) * 60);
        return (
          <span
            key={i}
            className={active ? "bg-[var(--color-accent-500)]" : "bg-[var(--color-accent-300)]"}
            style={{ height: `${height}%`, width: 2, borderRadius: 2, opacity: 0.75 }}
          />
        );
      })}
    </div>
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
