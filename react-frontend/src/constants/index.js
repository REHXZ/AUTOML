import {
  Binary,
  Brain,
  Compass,
  Globe,
  ScanEye,
  ScatterChart,
  SlidersVertical,
  WandSparkles
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
  {
    id: "scientist",
    title: "Scientist",
    short: "sci",
    role: "Orchestrator",
    color: "#818CF8",
    icon: Compass
  },
  {
    id: "researcher",
    title: "Researcher",
    short: "res",
    role: "Web research",
    color: "#00A9BD",
    icon: Globe
  },
  {
    id: "eda",
    title: "EDA",
    short: "eda",
    role: "Profile & visualise",
    color: "#06D7E8",
    icon: ScatterChart
  },
  {
    id: "feature_engineering",
    title: "Feature engineering",
    short: "fe",
    role: "Transform datasets",
    color: "#F59E0B",
    icon: SlidersVertical
  },
  {
    id: "modeling",
    title: "Modeling",
    short: "mod",
    role: "AutoML training",
    color: "#10B981",
    icon: Binary
  },
  {
    id: "review",
    title: "Review",
    short: "rev",
    role: "Critique results",
    color: "#E91E63",
    icon: ScanEye
  },
  {
    id: "fine_tuning",
    title: "Fine tuning",
    short: "ft",
    role: "Iterate on review",
    color: "#A5B0FB",
    icon: WandSparkles
  }
];

export const AGENT_BY_ID = AGENTS.reduce((map, agent) => {
  map[agent.id] = agent;
  return map;
}, {});

export const AGENT_ORDER = AGENTS.map((agent) => agent.id);

export const FALLBACK_AGENT = {
  id: "scientist",
  title: "Scientist",
  short: "sci",
  role: "Orchestrator",
  color: "#818CF8",
  icon: Brain
};

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
