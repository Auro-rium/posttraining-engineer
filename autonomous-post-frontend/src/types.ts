export type AgentId = 
  | 'benchmark' 
  | 'failure_analyst' 
  | 'research' 
  | 'data_curator' 
  | 'training_designer' 
  | 'sagemaker' 
  | 'evaluation' 
  | 'promotion';

export type AgentStatus = 'idle' | 'running' | 'completed' | 'blocked' | 'failed';

export interface AgentVisualAccent {
  name: string;
  hex: string;
  glow: string;
  border: string;
  bg: string;
  text: string;
  badge: string;
}

export interface AgentDefinition {
  id: AgentId;
  index: number;
  name: string;
  shortName: string;
  role: string;
  stageName: string;
  accent: AgentVisualAccent;
  avatarStyle: 'visor' | 'scanner' | 'antenna' | 'dual-eye' | 'matrix' | 'core' | 'prism' | 'halo';
  specialty: string;
  inputsRequired: string[];
  outputsProduced: string[];
  metricsMonitored: string[];
}

export type StageId = 
  | 'stage_launcher'
  | 'baseline_benchmark'
  | 'failure_analysis'
  | 'research_hypotheses'
  | 'training_curation'
  | 'qlora_design'
  | 'sagemaker_training'
  | 'heldout_eval'
  | 'deterministic_promotion';

export interface StageDefinition {
  id: StageId;
  stepNumber: number;
  label: string;
  agentId: AgentId | null;
  description: string;
  status: 'idle' | 'running' | 'completed' | 'blocked' | 'failed';
  progress: number; // 0 to 100
  durationSec: number;
  keyMetricLabel?: string;
  keyMetricValue?: string;
}

export type TelemetryEventType = 
  | 'run_started'
  | 'phase_started'
  | 'phase_completed'
  | 'gpu_quota_checked'
  | 'cost_validated'
  | 'human_approval_required'
  | 'human_approval_granted'
  | 'sagemaker_job_submitted'
  | 'training_progress'
  | 'training_completed'
  | 'eval_started'
  | 'eval_completed'
  | 'promotion_decision'
  | 'artifact_retention'
  | 'cleanup_completed'
  | 'blocked'
  | 'warning'
  | 'error';

export interface TelemetryEvent {
  id: string;
  timestamp: string; // ISO or HH:mm:ss.SSS
  stageId: StageId;
  agentId?: AgentId;
  type: TelemetryEventType;
  severity: 'info' | 'success' | 'warning' | 'error';
  summary: string;
  metadata: {
    targetModel?: string;
    reasoningModel?: string;
    instanceType?: string;
    quotaStatus?: string;
    jobArn?: string;
    loss?: number;
    step?: string;
    astPassRate?: number;
    baselinePassRate?: number;
    candidatePassRate?: number;
    improvement?: number;
    regressionCount?: number;
    manifestHash?: string;
    checkpointUri?: string;
    costUsd?: number;
    blockedReason?: string;
  };
}

export interface PreflightItem {
  id: string;
  name: string;
  description: string;
  status: 'passed' | 'checking' | 'failed' | 'blocked';
  detail: string;
}

export interface RunConfig {
  targetModel: string;
  reasoningModel: string;
  benchmarkSuite: string;
  benchmarkVersion: string;
  seed: number;
  gpuInstanceType: string;
  estimatedCostUsd: number;
  approvalStatus: 'NOT_REQUIRED' | 'PENDING' | 'APPROVED' | 'REJECTED';
  qloraRank: number;
  qloraAlpha: number;
  learningRate: string;
  epochs: number;
}

export interface SequentialRunRecord {
  runNumber: number;
  runId: string;
  label: string;
  targetModel: string;
  baselineScore: number; // e.g. 71.4%
  candidateScore: number | null; // e.g. 78.9%
  improvementPct: number | null; // e.g. +7.5%
  regressionStatus: 'ZERO_REGRESSIONS' | 'MARGINAL_REGRESSION' | 'CRITICAL_REGRESSION' | 'UNVERIFIED';
  promotionDecision: 'PROMOTED' | 'REJECTED' | 'BLOCKED' | 'RUNNING' | 'DISCARDED' | 'PENDING';
  trainingJobId: string;
  evalJobId: string;
  manifestHash: string;
  checkpointArtifact: string;
  totalCostUsd: number;
  runtimeMinutes: number;
  isVerifiable: boolean;
  unverifiableReason?: string;
  completedAt: string;
}

export interface ActiveHandoff {
  fromAgent: AgentDefinition;
  toAgent: AgentDefinition;
  artifactTransferred: string;
  payloadHash: string;
  timestamp: string;
}
