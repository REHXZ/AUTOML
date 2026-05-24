import {
  BarChart3,
  Brain,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  Wrench
} from "lucide-react";

export const PHASES = [
  { id: "business_understanding", title: "Business" },
  { id: "data_understanding", title: "Data" },
  { id: "data_preparation", title: "Preparation" },
  { id: "modeling", title: "Modeling" },
  { id: "evaluation", title: "Evaluation" },
  { id: "iteration", title: "Iteration" }
];

export const AGENTS = [
  { id: "scientist", title: "Scientist", icon: Brain, tone: "agent-scientist" },
  { id: "researcher", title: "Researcher", icon: Search, tone: "agent-researcher" },
  { id: "eda", title: "EDA", icon: BarChart3, tone: "agent-eda" },
  {
    id: "feature_engineering",
    title: "Feature Engineering",
    icon: Wrench,
    tone: "agent-feature"
  },
  { id: "modeling", title: "Modeling", icon: Target, tone: "agent-modeling" },
  { id: "review", title: "Review", icon: ShieldCheck, tone: "agent-review" },
  { id: "fine_tuning", title: "Fine Tuning", icon: SlidersHorizontal, tone: "agent-tuning" }
];

export const STEP_LABELS = {
  thought: "Reasoning",
  tool_call: "Tool Call",
  tool_result: "Tool Result",
  chart: "Chart",
  ask: "Question",
  new_dataset: "Dataset",
  training: "Training",
  summary: "Summary",
  observation: "Observation",
  agent_start: "Started",
  agent_end: "Finished",
  review: "Review",
  phase_transition: "Phase"
};

