import { useEffect, useState } from "react";
import { Header } from "@/components/layout/Header";
import { ModelCard } from "@/components/insights/ModelCard";
import { ArchitectureDiagram } from "@/components/insights/ArchitectureDiagram";
import { ComparisonChart } from "@/components/insights/ComparisonChart";
import { ConfusionMatrix } from "@/components/insights/ConfusionMatrix";
import { SkeletonText } from "@/components/ui/Skeleton";
import { getModelMetrics } from "@/lib/api";
import { NUMERICAL_CONFUSION_MATRIX, SPEECH_CONFUSION_MATRIX, toComparisonRows, type ModelBranch } from "@/lib/metricsConfig";

export function ModelInsightsPage() {
  const [branches, setBranches] = useState<ModelBranch[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    getModelMetrics().then((data) => {
      if (!cancelled) setBranches(data);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <>
      <Header title="Model Insights" subtitle="Architecture, current benchmark results, and per-branch evaluation detail." />

      <div className="space-y-5">
        <ArchitectureDiagram />

        {!branches ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="card-surface p-6">
                <SkeletonText lines={4} />
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {branches.map((branch) => (
              <ModelCard key={branch.branch} branch={branch} />
            ))}
          </div>
        )}

        {branches && <ComparisonChart rows={toComparisonRows(branches)} />}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <ConfusionMatrix
            title="Speech Emotion Confusion Matrix"
            subtitle="RAVDESS, 8 classes — strict actor-independent test split"
            data={SPEECH_CONFUSION_MATRIX}
          />
          <ConfusionMatrix
            title="Numerical Status Confusion Matrix"
            subtitle="4 classes — synthetic-enhanced model, synthetic held-out test split"
            data={NUMERICAL_CONFUSION_MATRIX}
          />
        </div>
      </div>
    </>
  );
}
