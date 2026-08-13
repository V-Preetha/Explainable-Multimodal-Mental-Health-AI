import { Activity } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { RankedBarChart } from "@/components/explainability/RankedBarChart";
import { useAssessment } from "@/context/AssessmentContext";

export function NumericalExplainability() {
  const { result } = useAssessment();
  const numerical = result?.explainability?.numerical;

  return (
    <Card>
      <CardHeader
        title="Behavioral &amp; Physiological Insights"
        subtitle="Top contributing indicators, ranked by estimated importance for this prediction."
        icon={<Activity size={18} />}
      />
      <RankedBarChart
        data={numerical?.feature_importance}
        emptyTitle="Feature importance unavailable"
        emptyDescription="Once the backend supplies SHAP or feature-importance values for this assessment, ranked indicators (e.g. Sleep Quality, HRV, GSR, Social Engagement) will appear here."
        color="#256a61"
        height={280}
      />
    </Card>
  );
}
