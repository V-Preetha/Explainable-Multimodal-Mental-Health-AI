import type { ConfusionMatrixData, ModelComparisonRow } from "@/types/api";

export interface MetricValues {
  accuracy?: number;
  macroF1?: number;
  weightedF1?: number;
  rocAuc?: number;
}

export interface ModelVariant {
  key: string;
  label: string;
  description: string;
  status: "complete";
  metrics: MetricValues;
  note?: string;
}

export interface ModelBranch {
  branch: "face" | "speech" | "numerical" | "fusion";
  title: string;
  architecture: string;
  dataset: string;
  variants: ModelVariant[];
  limitation?: string;
}

export const MODEL_BRANCHES: ModelBranch[] = [
  {
    branch: "face",
    title: "Facial Emotion Branch",
    architecture: "ConvNeXt-Tiny (ImageNet-1K initialized, 256-D embedding)",
    dataset: "FER2013; unchanged real validation/test splits",
    variants: [{
      key: "real_only",
      label: "Selected real-only model",
      description: "Selected at epoch 22 by validation Macro F1 (0.6799).",
      status: "complete",
      metrics: { accuracy: 0.6896, macroF1: 0.6852, weightedF1: 0.6878, rocAuc: 0.9103 },
      note: "A matched synthetic augmentation ablation did not surpass the validation selection threshold.",
    }],
  },
  {
    branch: "speech",
    title: "Speech Emotion Branch",
    architecture: "emotion2vec+ + attentive statistics pooling",
    dataset: "RAVDESS",
    variants: [
      {
        key: "actor_independent",
        label: "Primary actor-independent benchmark",
        description: "Zero actor overlap across train, validation, and test.",
        status: "complete",
        metrics: { accuracy: 0.7583, macroF1: 0.7405, weightedF1: 0.7587, rocAuc: 0.9615 },
      },
      {
        key: "random_split",
        label: "RANDOM SPLIT augmented hybrid",
        description: "Secondary maximum-accuracy benchmark; actors may overlap.",
        status: "complete",
        metrics: { accuracy: 0.8657, macroF1: 0.8505, weightedF1: 0.8663, rocAuc: 0.9879 },
        note: "Not speaker-independent and not used for generalization claims.",
      },
    ],
  },
  {
    branch: "numerical",
    title: "Numerical / Behavioral Branch",
    architecture: "Original baseline plus separately evaluated synthetic benchmarks",
    dataset: "18 original indicators; synthetic extension reported by domain",
    variants: [
      {
        key: "synthetic_only",
        label: "Displayed: Synthetic-only XGBoost",
        description: "Headline numerical benchmark; evaluated on synthetic held-out data.",
        status: "complete",
        metrics: { accuracy: 0.8960, macroF1: 0.8962, rocAuc: 0.9845 },
        note: "On original-real test data, synthetic-enhanced training reached 0.4000 accuracy and 0.1988 Macro F1.",
      },
      {
        key: "original_real",
        label: "Original-real reference",
        description: "Retained to show the real-domain evaluation boundary.",
        status: "complete",
        metrics: { accuracy: 0.2617, macroF1: 0.2245, rocAuc: 0.4612 },
      },
    ],
    limitation: "Synthetic-held-out performance is not evidence of original-real generalization.",
  },
  {
    branch: "fusion",
    title: "Multimodal Fusion",
    architecture: "Gated attention over 256-D face, 256-D speech, and 128-D numerical embeddings",
    dataset: "Weak class-conditional construction across independent datasets",
    variants: [{
      key: "gated_fusion",
      label: "Weakly aligned gated fusion",
      description: "Research construction; not participant-paired.",
      status: "complete",
      metrics: { accuracy: 0.6633, macroF1: 0.5224, weightedF1: 0.6539, rocAuc: 0.8303 },
      note: "Do not interpret as participant-level clinical generalization.",
    }],
  },
];

export function toComparisonRows(branches: ModelBranch[]): ModelComparisonRow[] {
  return branches.flatMap((branch) => branch.variants.map((variant) => ({
    model: branch.title,
    branch: branch.branch,
    variant: variant.label,
    accuracy: variant.metrics.accuracy,
    macroF1: variant.metrics.macroF1,
    weightedF1: variant.metrics.weightedF1,
    rocAuc: variant.metrics.rocAuc,
  })));
}

export const SPEECH_CONFUSION_MATRIX: ConfusionMatrixData = {
  classes: ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
  counts: [[27,0,4,1,0,0,0,0],[0,20,3,0,3,4,2,0],[0,0,30,0,0,1,1,0],[1,2,0,22,3,0,1,3],[0,1,1,1,24,0,0,5],[0,9,0,0,0,7,0,0],[0,5,0,0,2,1,23,1],[0,0,2,0,0,1,0,29]],
  normalized: [[.84375,0,.125,.03125,0,0,0,0],[0,.625,.09375,0,.09375,.125,.0625,0],[0,0,.9375,0,0,.03125,.03125,0],[.03125,.0625,0,.6875,.09375,0,.03125,.09375],[0,.03125,.03125,.03125,.75,0,0,.15625],[0,.5625,0,0,0,.4375,0,0],[0,.15625,0,0,.0625,.03125,.71875,.03125],[0,0,.0625,0,0,.03125,0,.90625]],
};

export const NUMERICAL_CONFUSION_MATRIX: ConfusionMatrixData = {
  classes: ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"],
  counts: [[60,76,67,42],[53,64,36,32],[38,51,33,29],[4,12,2,1]],
  normalized: [[.2449,.3102,.2735,.1714],[.2865,.3459,.1946,.1730],[.2517,.3377,.2185,.1921],[.2105,.6316,.1053,.0526]],
};
