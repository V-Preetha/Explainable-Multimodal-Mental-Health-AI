import { useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { AssessmentProvider } from "@/context/AssessmentContext";
import { AssessmentPage } from "@/pages/AssessmentPage";
import { ResultsPage } from "@/pages/ResultsPage";
import { ExplainabilityPage } from "@/pages/ExplainabilityPage";
import { ModelInsightsPage } from "@/pages/ModelInsightsPage";
import { AboutPage } from "@/pages/AboutPage";
import type { PageKey } from "@/lib/constants";

function PageRouter({ page, onNavigate }: { page: PageKey; onNavigate: (page: PageKey) => void }) {
  switch (page) {
    case "assessment":
      return <AssessmentPage onNavigate={onNavigate} />;
    case "results":
      return <ResultsPage onNavigate={onNavigate} />;
    case "explainability":
      return <ExplainabilityPage />;
    case "insights":
      return <ModelInsightsPage />;
    case "about":
      return <AboutPage />;
    default:
      return <AssessmentPage onNavigate={onNavigate} />;
  }
}

export default function App() {
  const [page, setPage] = useState<PageKey>("assessment");

  return (
    <AssessmentProvider>
      <AppShell active={page} onNavigate={setPage}>
        <PageRouter page={page} onNavigate={setPage} />
      </AppShell>
    </AssessmentProvider>
  );
}
