import { Activity, RotateCcw, Sparkles } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { CollapsibleSection } from "@/components/ui/CollapsibleSection";
import { FieldControl } from "@/components/ui/FieldControl";
import { NUMERICAL_FIELD_GROUPS } from "@/lib/constants";
import { useAssessment } from "@/context/AssessmentContext";

export function NumericalForm() {
  const { numericalValues, setNumericalValue, resetNumericalValues, loadSampleNumericalValues } = useAssessment();

  return (
    <Card>
      <CardHeader
        title="Behavioral &amp; Physiological Indicators"
        subtitle="Grouped by domain. Adjust the values you have, or load a demo sample profile."
        icon={<Activity size={18} />}
        action={<Badge tone="accent">Numerical Analysis</Badge>}
      />

      <div className="space-y-3">
        {NUMERICAL_FIELD_GROUPS.map((group) => (
          <CollapsibleSection key={group.title} title={group.title} description={group.description}>
            {group.fields.map((field) => (
              <FieldControl
                key={field.key}
                spec={field}
                value={numericalValues[field.key] ?? field.min}
                onChange={(value) => setNumericalValue(field.key, value)}
              />
            ))}
          </CollapsibleSection>
        ))}
      </div>

      <div className="flex items-center justify-between mt-5 pt-4 border-t border-slate-100">
        <Button variant="ghost" size="sm" icon={<RotateCcw size={13} />} onClick={resetNumericalValues}>
          Reset
        </Button>
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-slate-400">Demo Sample</span>
          <Button variant="secondary" size="sm" icon={<Sparkles size={13} />} onClick={loadSampleNumericalValues}>
            Load Sample
          </Button>
        </div>
      </div>
    </Card>
  );
}
