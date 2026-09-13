/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  AgentDefinition, 
  AgentStatus, 
  StageId, 
  TelemetryEvent, 
  RunConfig, 
  SequentialRunRecord, 
  PreflightItem, 
  ActiveHandoff 
} from './types';
import { AGENT_REGISTRY } from './data/agentDefinitions';
import { INITIAL_RUNS, INITIAL_PREFLIGHT_CHECKS } from './data/mockRuns';
import { ControlRoomHeader } from './components/ControlRoomHeader';
import { ExecutionBench } from './components/ExecutionBench';
import { RunLauncher } from './components/RunLauncher';
import { TelemetryPanel } from './components/TelemetryPanel';
import { RunComparison } from './components/RunComparison';
import { PerformanceChart } from './components/PerformanceChart';
import { AgentModal } from './components/AgentModal';

export default function App() {
  // Global settings
  const [reducedMotion, setReducedMotion] = useState(false);
  const [executionSpeed, setExecutionSpeed] = useState<number>(1);

  // Agent State Map
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({
    benchmark: 'idle',
    failure_analyst: 'idle',
    research: 'idle',
    data_curator: 'idle',
    training_designer: 'idle',
    sagemaker: 'idle',
    evaluation: 'idle',
    promotion: 'idle',
  });

  // Active Runner State
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentStageId, setCurrentStageId] = useState<StageId>('stage_launcher');
  const [activeAgentId, setActiveAgentId] = useState<string | null>(null);
  const [activeHandoff, setActiveHandoff] = useState<ActiveHandoff | null>(null);
  const [overallProgress, setOverallProgress] = useState(0);

  // Selected agent for modal inspection
  const [selectedAgent, setSelectedAgent] = useState<AgentDefinition | null>(null);

  // Blocked state tracking
  const [blockedDetails, setBlockedDetails] = useState<{
    stage: string;
    agentName: string;
    reason: string;
    actionRequired: string;
  } | null>(null);

  // Simulation scenario
  const [scenarioMode, setScenarioMode] = useState<
    'normal' | 'blocked_quota' | 'blocked_approval' | 'regression_reject'
  >('normal');

  // Preflight Checks
  const [preflights, setPreflights] = useState<PreflightItem[]>(INITIAL_PREFLIGHT_CHECKS);

  // Configuration for Active Run
  const [config, setConfig] = useState<RunConfig>({
    targetModel: 'FunctionGemma-2.6B',
    reasoningModel: 'NVIDIA Nemotron Super 3 120B',
    benchmarkSuite: 'FuncBench-Core',
    benchmarkVersion: 'v2.4.1',
    seed: 4242,
    gpuInstanceType: 'ml.g5.12xlarge (4x A10G)',
    estimatedCostUsd: 18.20,
    approvalStatus: 'NOT_REQUIRED',
    qloraRank: 64,
    qloraAlpha: 128,
    learningRate: '2e-4',
    epochs: 3,
  });

  // Sequential Runs Comparison Data
  const [runs, setRuns] = useState<SequentialRunRecord[]>(INITIAL_RUNS);
  const [activeChampionRunNumber, setActiveChampionRunNumber] = useState<number>(4);

  // Live Telemetry Event Stream
  const [events, setEvents] = useState<TelemetryEvent[]>([
    {
      id: 'init-1',
      timestamp: '23:28:10.104',
      stageId: 'stage_launcher',
      type: 'run_started',
      severity: 'info',
      summary: 'Autonomous Post-Training Bench initialized. FuncBench Core v2.4.1 sealed.',
      metadata: {
        targetModel: 'FunctionGemma-2.6B',
        reasoningModel: 'NVIDIA Nemotron Super 3 120B',
      },
    },
    {
      id: 'init-2',
      timestamp: '23:28:10.250',
      stageId: 'stage_launcher',
      type: 'gpu_quota_checked',
      severity: 'success',
      summary: 'AWS Service Quotas L-F678F1 verified for ml.g5.12xlarge in us-west-2.',
      metadata: {
        instanceType: 'ml.g5.12xlarge',
        quotaStatus: 'CONFIRMED_4_AVAILABLE',
      },
    },
  ]);

  // Helper to add telemetry event
  const addTelemetry = (
    stageId: StageId,
    type: TelemetryEvent['type'],
    severity: TelemetryEvent['severity'],
    summary: string,
    metadata: TelemetryEvent['metadata'] = {},
    agentId?: TelemetryEvent['agentId']
  ) => {
    const now = new Date();
    const timeStr = `${String(now.getUTCHours()).padStart(2, '0')}:${String(
      now.getUTCMinutes()
    ).padStart(2, '0')}:${String(now.getUTCSeconds()).padStart(2, '0')}.${String(
      now.getUTCMilliseconds()
    ).padStart(3, '0')}`;

    const newEvt: TelemetryEvent = {
      id: `evt-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
      timestamp: timeStr,
      stageId,
      agentId,
      type,
      severity,
      summary,
      metadata,
    };

    setEvents((prev) => [...prev, newEvt]);
  };

  // Execution runner step index
  const stepRef = useRef<number>(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Clear all running timers
  const clearExecutionTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  // Start Autonomous Run Flow
  const handleStartRun = () => {
    clearExecutionTimer();
    setIsRunning(true);
    setIsPaused(false);
    setBlockedDetails(null);
    stepRef.current = 0;
    setOverallProgress(5);

    // Reset agent statuses
    const resetStatus: Record<string, AgentStatus> = {};
    AGENT_REGISTRY.forEach((a) => {
      resetStatus[a.id] = 'idle';
    });
    setAgentStatuses(resetStatus);
    setActiveAgentId(null);
    setActiveHandoff(null);

    // Update Run #5 in comparison registry
    setRuns((prev) =>
      prev.map((r) =>
        r.runNumber === 5
          ? {
              ...r,
              candidateScore: null,
              improvementPct: null,
              regressionStatus: 'UNVERIFIED',
              promotionDecision: 'RUNNING',
              isVerifiable: false,
              unverifiableReason: 'Training & Held-out evaluation currently executing...',
            }
          : r
      )
    );

    addTelemetry(
      'stage_launcher',
      'run_started',
      'info',
      'Run #5 initiated: Autonomous pipeline dispatched for FunctionGemma-2.6B with Nemotron 120B reasoning.',
      {
        targetModel: config.targetModel,
        reasoningModel: config.reasoningModel,
        instanceType: config.gpuInstanceType,
        costUsd: config.estimatedCostUsd,
      }
    );

    // If approval scenario is selected, trigger human approval
    if (scenarioMode === 'blocked_approval') {
      setConfig((prev) => ({ ...prev, approvalStatus: 'PENDING' }));
    }

    // Schedule first step
    scheduleNextStep(800);
  };

  // Human approval handlers
  const handleGrantApproval = () => {
    setConfig((prev) => ({ ...prev, approvalStatus: 'APPROVED' }));
    setBlockedDetails(null);
    addTelemetry(
      'qlora_design',
      'human_approval_granted',
      'success',
      'Human Approval Granted: SageMaker on-demand GPU training compute envelope ($18.20) authorized.',
      { costUsd: config.estimatedCostUsd }
    );
    if (isRunning && currentStageId === 'qlora_design') {
      scheduleNextStep(400);
    }
  };

  const handleRejectApproval = () => {
    setConfig((prev) => ({ ...prev, approvalStatus: 'REJECTED' }));
    setIsRunning(false);
    clearExecutionTimer();
    setBlockedDetails({
      stage: 'Human Approval Gate',
      agentName: 'Training Designer Agent',
      reason: 'Human Operator explicitly rejected the compute budget envelope of $18.20.',
      actionRequired: 'Review instance selection or re-authorize with revised parameters.',
    });
    addTelemetry(
      'qlora_design',
      'blocked',
      'warning',
      'Pipeline Halted: Human operator rejected SageMaker training job dispatch.',
      { blockedReason: 'Operator Rejected Approval' }
    );
  };

  // Stop / Cleanup Handler
  const handleStopCleanup = () => {
    clearExecutionTimer();
    setIsRunning(false);
    setActiveAgentId(null);
    setActiveHandoff(null);
    setBlockedDetails(null);

    addTelemetry(
      currentStageId,
      'cleanup_completed',
      'warning',
      'Pipeline Stop Triggered: Active AWS SageMaker job terminated. Temporary S3 staging buckets cleaned up.',
      { jobArn: 'arn:aws:sagemaker:us-west-2:8912:training-job/fgemma-qlora-r64-005-terminated' }
    );

    setRuns((prev) =>
      prev.map((r) =>
        r.runNumber === 5
          ? {
              ...r,
              candidateScore: null,
              improvementPct: null,
              regressionStatus: 'UNVERIFIED',
              promotionDecision: 'DISCARDED',
              isVerifiable: false,
              unverifiableReason: 'User initiated emergency cleanup. Run aborted before held-out eval.',
            }
          : r
      )
    );
  };

  // Schedule Next Step with speed scaling
  const scheduleNextStep = (baseDelayMs: number) => {
    clearExecutionTimer();
    const delay = Math.max(300, baseDelayMs / executionSpeed);
    timerRef.current = setTimeout(() => {
      executeStep();
    }, delay);
  };

  // Execution Step Sequence Machine
  const executeStep = () => {
    const step = stepRef.current;

    switch (step) {
      // Step 0: Preflight checks & Quota Check
      case 0: {
        setCurrentStageId('stage_launcher');
        if (scenarioMode === 'blocked_quota') {
          // Truthful observer behavior: AWS GPU quota blocked!
          setPreflights((prev) =>
            prev.map((p) =>
              p.id === 'quota'
                ? {
                    ...p,
                    status: 'blocked',
                    detail: 'Quota exhausted: 0 active instances available for ml.g5.12xlarge in us-west-2.',
                  }
                : p
            )
          );
          setBlockedDetails({
            stage: 'GPU Quota Confirmation',
            agentName: 'Preflight Gatekeeper',
            reason: 'AWS Service Quotas L-F678F1 indicates zero ml.g5.12xlarge capacity in us-west-2.',
            actionRequired: 'Request quota increase in AWS Console or select alternate GPU tier.',
          });
          setIsRunning(false);
          addTelemetry(
            'stage_launcher',
            'blocked',
            'error',
            'BLOCKED STATE: AWS GPU Quota verification failed for ml.g5.12xlarge. Execution held.',
            { blockedReason: 'AWS Quota Code L-F678F1 Exhausted' }
          );
          return;
        }

        // Normal preflight pass
        setOverallProgress(12);
        addTelemetry(
          'stage_launcher',
          'gpu_quota_checked',
          'success',
          'Preflight 6/6 checks passed. AWS IAM, Quotas (L-F678F1), and checksums validated.',
          { quotaStatus: 'PASS_4_INSTANCES' }
        );

        stepRef.current = 1;
        scheduleNextStep(1200);
        break;
      }

      // Step 1: Agent 1 - Benchmark Agent (Baseline Benchmark)
      case 1: {
        setCurrentStageId('baseline_benchmark');
        setActiveAgentId('benchmark');
        setOverallProgress(20);
        setAgentStatuses((prev) => ({ ...prev, benchmark: 'running' }));

        addTelemetry(
          'baseline_benchmark',
          'phase_started',
          'info',
          'Benchmark Agent running: Testing FunctionGemma-2.6B zero-shot baseline on FuncBench-Core (420 AST scenarios).',
          {},
          'benchmark'
        );

        stepRef.current = 2;
        scheduleNextStep(2200);
        break;
      }

      // Step 2: Agent 1 completes -> Handoff to Agent 2 (Failure Analyst)
      case 2: {
        setAgentStatuses((prev) => ({ ...prev, benchmark: 'completed' }));
        setOverallProgress(32);

        addTelemetry(
          'baseline_benchmark',
          'phase_completed',
          'success',
          'Benchmark Agent completed: Baseline pass rate confirmed at 71.2%. Generated 108 AST error traces.',
          {
            baselinePassRate: 71.2,
            manifestHash: 'sha256:8f4c2194b18de02fa910bb3a5518b',
          },
          'benchmark'
        );

        // Handoff to Failure Analyst
        const fromA = AGENT_REGISTRY[0];
        const toA = AGENT_REGISTRY[1];
        setActiveHandoff({
          fromAgent: fromA,
          toAgent: toA,
          artifactTransferred: '108 AST error trace telemetry dump',
          payloadHash: 'sha256:d82e11fa08',
          timestamp: 'Just now',
        });

        setCurrentStageId('failure_analysis');
        setActiveAgentId('failure_analyst');
        setAgentStatuses((prev) => ({ ...prev, failure_analyst: 'running' }));

        addTelemetry(
          'failure_analysis',
          'phase_started',
          'info',
          'Failure Analyst Agent running: Dissecting 108 schema failures into actionable error clusters.',
          {},
          'failure_analyst'
        );

        stepRef.current = 3;
        scheduleNextStep(2000);
        break;
      }

      // Step 3: Agent 2 completes -> Handoff to Agent 3 (Research Agent)
      case 3: {
        setAgentStatuses((prev) => ({ ...prev, failure_analyst: 'completed' }));
        setOverallProgress(44);

        addTelemetry(
          'failure_analysis',
          'phase_completed',
          'success',
          'Failure Analyst completed: Isolated 3 primary clusters: 1) Nested schemas (48%), 2) Missing parameters (32%), 3) Hallucinated keys (20%).',
          {},
          'failure_analyst'
        );

        // Handoff to Research Agent
        const fromA = AGENT_REGISTRY[1];
        const toA = AGENT_REGISTRY[2];
        setActiveHandoff({
          fromAgent: fromA,
          toAgent: toA,
          artifactTransferred: 'Targeted Failure Defect Vector Matrix',
          payloadHash: 'sha256:91c280bb41',
          timestamp: 'Just now',
        });

        setCurrentStageId('research_hypotheses');
        setActiveAgentId('research');
        setAgentStatuses((prev) => ({ ...prev, research: 'running' }));

        addTelemetry(
          'research_hypotheses',
          'phase_started',
          'info',
          'Research Agent running: Querying NVIDIA Nemotron Super 3 120B reasoning model to synthesize fine-tuning curriculum.',
          { reasoningModel: config.reasoningModel },
          'research'
        );

        stepRef.current = 4;
        scheduleNextStep(2400);
        break;
      }

      // Step 4: Agent 3 completes -> Handoff to Agent 4 (Data Curator)
      case 4: {
        setAgentStatuses((prev) => ({ ...prev, research: 'completed' }));
        setOverallProgress(55);

        addTelemetry(
          'research_hypotheses',
          'phase_completed',
          'success',
          'Research Agent completed: Formulated 4 synthetic hypotheses targeting multi-schema AST tool generation.',
          {},
          'research'
        );

        // Handoff to Data Curator
        const fromA = AGENT_REGISTRY[2];
        const toA = AGENT_REGISTRY[3];
        setActiveHandoff({
          fromAgent: fromA,
          toAgent: toA,
          artifactTransferred: 'Synthetic Curriculum Specification (4 Hypotheses)',
          payloadHash: 'sha256:4a7e93011c',
          timestamp: 'Just now',
        });

        setCurrentStageId('training_curation');
        setActiveAgentId('data_curator');
        setAgentStatuses((prev) => ({ ...prev, data_curator: 'running' }));

        addTelemetry(
          'training_curation',
          'phase_started',
          'info',
          'Data Curator Agent running: Generating and validating 4,800 gold input/output tool calling samples.',
          {},
          'data_curator'
        );

        stepRef.current = 5;
        scheduleNextStep(2200);
        break;
      }

      // Step 5: Agent 4 completes -> Handoff to Agent 5 (Training Designer)
      case 5: {
        setAgentStatuses((prev) => ({ ...prev, data_curator: 'completed' }));
        setOverallProgress(65);

        addTelemetry(
          'training_curation',
          'phase_completed',
          'success',
          'Data Curator completed: Curated 4,800 clean AST pairs. Exact deduplication passed, schema conformity 100%.',
          { checkpointUri: 's3://fgemma/curated-dataset-v3.jsonl' },
          'data_curator'
        );

        // Handoff to Training Designer
        const fromA = AGENT_REGISTRY[3];
        const toA = AGENT_REGISTRY[4];
        setActiveHandoff({
          fromAgent: fromA,
          toAgent: toA,
          artifactTransferred: 's3://fgemma/curated-dataset-v3.jsonl (4,800 items)',
          payloadHash: 'sha256:39f8bc2100',
          timestamp: 'Just now',
        });

        setCurrentStageId('qlora_design');
        setActiveAgentId('training_designer');
        setAgentStatuses((prev) => ({ ...prev, training_designer: 'running' }));

        addTelemetry(
          'qlora_design',
          'phase_started',
          'info',
          'Training Designer Agent running: Configuring QLoRA rank r=64, alpha=128, LR=2e-4, 4-bit bfloat16 adapter topology.',
          {},
          'training_designer'
        );

        stepRef.current = 6;
        scheduleNextStep(1800);
        break;
      }

      // Step 6: Training Designer checks Human Approval Gate!
      case 6: {
        // If human approval is still pending, hold pipeline!
        if (config.approvalStatus === 'PENDING') {
          setAgentStatuses((prev) => ({ ...prev, training_designer: 'blocked' }));
          setBlockedDetails({
            stage: 'Stage 4: Human Approval Required',
            agentName: 'Training Designer Agent',
            reason: 'AWS compute budget of $18.20 requires explicit human authorization before SageMaker cluster provisioning.',
            actionRequired: 'Review the cost envelope in the Run Launcher and click "Authorize Run".',
          });
          addTelemetry(
            'qlora_design',
            'human_approval_required',
            'warning',
            'GATE HOLD: SageMaker training job dispatch paused awaiting human approval.',
            { costUsd: config.estimatedCostUsd },
            'training_designer'
          );
          // Do not advance stepRef until approved!
          return;
        }

        setAgentStatuses((prev) => ({ ...prev, training_designer: 'completed' }));
        setOverallProgress(74);

        addTelemetry(
          'qlora_design',
          'phase_completed',
          'success',
          'Training Designer completed: QLoRA manifest sealed. Trainable parameters: 0.42% (11.2M params).',
          {},
          'training_designer'
        );

        // Handoff to SageMaker Agent
        const fromA = AGENT_REGISTRY[4];
        const toA = AGENT_REGISTRY[5];
        setActiveHandoff({
          fromAgent: fromA,
          toAgent: toA,
          artifactTransferred: 'Training Manifest qlora_config.json',
          payloadHash: 'sha256:b1836c927f',
          timestamp: 'Just now',
        });

        setCurrentStageId('sagemaker_training');
        setActiveAgentId('sagemaker');
        setAgentStatuses((prev) => ({ ...prev, sagemaker: 'running' }));

        addTelemetry(
          'sagemaker_training',
          'sagemaker_job_submitted',
          'info',
          'SageMaker Training Agent: Provisioned on-demand ml.g5.12xlarge cluster. Job ARN created.',
          {
            jobArn: 'arn:aws:sagemaker:us-west-2:891238471:training-job/fgemma-qlora-r64-005',
            instanceType: config.gpuInstanceType,
          },
          'sagemaker'
        );

        stepRef.current = 7;
        scheduleNextStep(2800);
        break;
      }

      // Step 7: SageMaker training finishes -> Handoff to Agent 7 (Evaluation)
      case 7: {
        setAgentStatuses((prev) => ({ ...prev, sagemaker: 'completed' }));
        setOverallProgress(86);

        addTelemetry(
          'sagemaker_training',
          'training_completed',
          'success',
          'SageMaker training job completed: 1,000 steps executed. Final loss: 0.2641. Adapter weights synced to S3.',
          {
            loss: 0.2641,
            step: '1000/1000',
            checkpointUri: 's3://fgemma-registry/checkpoints/run-05/candidate.safetensors',
          },
          'sagemaker'
        );

        // Handoff to Evaluation Agent
        const fromA = AGENT_REGISTRY[5];
        const toA = AGENT_REGISTRY[6];
        setActiveHandoff({
          fromAgent: fromA,
          toAgent: toA,
          artifactTransferred: 's3://fgemma-registry/checkpoints/run-05/candidate.safetensors',
          payloadHash: 'sha256:88194cf210',
          timestamp: 'Just now',
        });

        setCurrentStageId('heldout_eval');
        setActiveAgentId('evaluation');
        setAgentStatuses((prev) => ({ ...prev, evaluation: 'running' }));

        addTelemetry(
          'heldout_eval',
          'eval_started',
          'info',
          'Evaluation Agent running: Testing candidate checkpoint across 420 isolated held-out AST tool scenarios.',
          {},
          'evaluation'
        );

        stepRef.current = 8;
        scheduleNextStep(2600);
        break;
      }

      // Step 8: Evaluation completes -> Deterministic Promotion Agent
      case 8: {
        const isRegressionScenario = scenarioMode === 'regression_reject';
        const candidateScore = isRegressionScenario ? 77.4 : 83.7;
        const baselineScore = 80.4; // Champion from Run 4
        const improvement = candidateScore - baselineScore;

        if (isRegressionScenario) {
          // Evaluation Agent detected regressions
          setAgentStatuses((prev) => ({
            ...prev,
            evaluation: 'completed',
            promotion: 'failed',
          }));
          setOverallProgress(95);

          addTelemetry(
            'heldout_eval',
            'eval_completed',
            'warning',
            'Evaluation Agent finished: AST Pass Rate 77.4% (-3.0%). Detected 3 critical regressions on nested tool arguments.',
            {
              astPassRate: 77.4,
              regressionCount: 3,
            },
            'evaluation'
          );

          // Update Run 5 in comparison
          setRuns((prev) =>
            prev.map((r) =>
              r.runNumber === 5
                ? {
                    ...r,
                    candidateScore: 77.4,
                    improvementPct: -3.0,
                    regressionStatus: 'CRITICAL_REGRESSION',
                    promotionDecision: 'REJECTED',
                    isVerifiable: true,
                    unverifiableReason: 'Deterministic promotion gate rejected candidate due to 3 AST regressions.',
                    completedAt: 'Just now',
                  }
                : r
            )
          );

          addTelemetry(
            'deterministic_promotion',
            'promotion_decision',
            'error',
            'Deterministic Promotion Gate: REJECTED candidate. Zero regressions requirement violated.',
            {
              astPassRate: 77.4,
              improvement: -3.0,
            },
            'promotion'
          );

          setIsRunning(false);
          setActiveAgentId(null);
          return;
        }

        // Normal successful evaluation
        setAgentStatuses((prev) => ({ ...prev, evaluation: 'completed' }));
        setOverallProgress(94);

        addTelemetry(
          'heldout_eval',
          'eval_completed',
          'success',
          `Evaluation Agent completed: Held-out AST Pass Rate achieved 83.7% (+3.3% over Champion). 0 Regressions verified.`,
          {
            candidatePassRate: 83.7,
            baselinePassRate: 80.4,
            improvement: 3.3,
            regressionCount: 0,
          },
          'evaluation'
        );

        // Handoff to Promotion Agent
        const fromA = AGENT_REGISTRY[6];
        const toA = AGENT_REGISTRY[7];
        setActiveHandoff({
          fromAgent: fromA,
          toAgent: toA,
          artifactTransferred: 'Candidate Evaluation Manifest (Verified AST 83.7%)',
          payloadHash: 'sha256:cc9801fa42',
          timestamp: 'Just now',
        });

        setCurrentStageId('deterministic_promotion');
        setActiveAgentId('promotion');
        setAgentStatuses((prev) => ({ ...prev, promotion: 'running' }));

        stepRef.current = 9;
        scheduleNextStep(2000);
        break;
      }

      // Step 9: Champion/Promotion Agent executes deterministic gate
      case 9: {
        setAgentStatuses((prev) => ({ ...prev, promotion: 'completed' }));
        setOverallProgress(100);
        setIsRunning(false);
        setActiveAgentId(null);
        setActiveChampionRunNumber(5);

        // Update Run 5 in comparison table & chart
        setRuns((prev) =>
          prev.map((r) =>
            r.runNumber === 5
              ? {
                  ...r,
                  candidateScore: 83.7,
                  improvementPct: 3.3,
                  regressionStatus: 'ZERO_REGRESSIONS',
                  promotionDecision: 'PROMOTED',
                  manifestHash: 'sha256:cc9801fa42b0981a2f9011de349',
                  checkpointArtifact: 's3://fgemma-registry/checkpoints/run-05/champion.safetensors',
                  runtimeMinutes: 46.5,
                  totalCostUsd: 18.20,
                  isVerifiable: true,
                  completedAt: 'Just now',
                }
              : r
          )
        );

        addTelemetry(
          'deterministic_promotion',
          'promotion_decision',
          'success',
          'Deterministic Promotion Decision: CANDIDATE PROMOTED TO CHAMPION! Meets bar (+3.3% >= +2.5% bar, 0 regressions, cryptographic seal).',
          {
            candidatePassRate: 83.7,
            improvement: 3.3,
            manifestHash: 'sha256:cc9801fa42b0981a2f9011de349',
          },
          'promotion'
        );

        addTelemetry(
          'deterministic_promotion',
          'artifact_retention',
          'success',
          'Artifact Retention: Checkpoint s3://fgemma-registry/checkpoints/run-05/champion.safetensors locked in model registry.',
          { checkpointUri: 's3://fgemma-registry/checkpoints/run-05/champion.safetensors' }
        );

        addTelemetry(
          'stage_launcher',
          'cleanup_completed',
          'info',
          'AWS SageMaker on-demand cluster decommissioned. Zero orphaned compute instances remaining.',
          { costUsd: 18.20 }
        );

        break;
      }

      default:
        setIsRunning(false);
        break;
    }
  };

  // Manual Step Forward
  const handleStepNext = () => {
    executeStep();
  };

  // Trigger specific demo scenarios
  const handleTriggerScenario = (
    mode: 'normal' | 'blocked_quota' | 'blocked_approval' | 'regression_reject'
  ) => {
    setScenarioMode(mode);
    if (mode === 'blocked_quota') {
      setPreflights((prev) =>
        prev.map((p) =>
          p.id === 'quota'
            ? { ...p, status: 'blocked', detail: 'Quota exhausted: 0 active instances available.' }
            : p
        )
      );
    } else {
      setPreflights(INITIAL_PREFLIGHT_CHECKS);
    }

    if (mode === 'blocked_approval') {
      setConfig((prev) => ({ ...prev, approvalStatus: 'PENDING' }));
    } else {
      setConfig((prev) => ({ ...prev, approvalStatus: 'APPROVED' }));
    }
  };

  // Reset demo
  const handleResetDemo = () => {
    clearExecutionTimer();
    setIsRunning(false);
    setIsPaused(false);
    setOverallProgress(0);
    setCurrentStageId('stage_launcher');
    setActiveAgentId(null);
    setActiveHandoff(null);
    setBlockedDetails(null);
    setScenarioMode('normal');
    setPreflights(INITIAL_PREFLIGHT_CHECKS);
    setRuns(INITIAL_RUNS);
    setActiveChampionRunNumber(4);

    const resetStatus: Record<string, AgentStatus> = {};
    AGENT_REGISTRY.forEach((a) => {
      resetStatus[a.id] = 'idle';
    });
    setAgentStatuses(resetStatus);

    setEvents([
      {
        id: 'reset-1',
        timestamp: '23:30:00.000',
        stageId: 'stage_launcher',
        type: 'run_started',
        severity: 'info',
        summary: 'Control room workbench reset. Ready for new autonomous FunctionGemma post-training run.',
        metadata: {},
      },
    ]);
  };

  return (
    <div className="min-h-screen bg-[#060913] text-slate-100 p-4 sm:p-6 lg:p-8 font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Control Room Top Header */}
      <ControlRoomHeader
        reducedMotion={reducedMotion}
        onToggleReducedMotion={() => setReducedMotion((prev) => !prev)}
        onResetDemo={handleResetDemo}
        championScore={
          activeChampionRunNumber === 5 ? 83.7 : runs.find((r) => r.runNumber === activeChampionRunNumber)?.candidateScore || 80.4
        }
        championRunNumber={activeChampionRunNumber}
      />

      {/* Main Control Room Layout Grid */}
      <div className="space-y-6 max-w-[1600px] mx-auto">
        {/* Animated Execution Bench with the 8 AI Agent bots on connected rail */}
        <ExecutionBench
          agents={AGENT_REGISTRY}
          agentStatuses={agentStatuses}
          currentStageId={currentStageId}
          activeAgentId={activeAgentId}
          activeHandoff={activeHandoff}
          overallProgress={overallProgress}
          reducedMotion={reducedMotion}
          onSelectAgent={(agent) => setSelectedAgent(agent)}
          blockedDetails={blockedDetails}
        />

        {/* Row 2: Run Launcher & Live Telemetry Observability Console */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Run Launcher Panel */}
          <div className="lg:col-span-6">
            <RunLauncher
              config={config}
              onChangeConfig={(newCfg) => setConfig((prev) => ({ ...prev, ...newCfg }))}
              preflights={preflights}
              isRunning={isRunning}
              isPaused={isPaused}
              currentStageId={currentStageId}
              onStartRun={handleStartRun}
              onStopCleanup={handleStopCleanup}
              onGrantApproval={handleGrantApproval}
              onRejectApproval={handleRejectApproval}
              onTriggerScenario={handleTriggerScenario}
              executionSpeed={executionSpeed}
              onChangeSpeed={(spd) => setExecutionSpeed(spd)}
              onStepNext={handleStepNext}
            />
          </div>

          {/* Live Telemetry Panel (Metadata-only, zero prompt leakage) */}
          <div className="lg:col-span-6">
            <TelemetryPanel
              events={events}
              onClearEvents={() => setEvents([])}
              isRunning={isRunning}
            />
          </div>
        </div>

        {/* Row 3: Performance Trajectory Chart */}
        <PerformanceChart runs={runs} />

        {/* Row 4: Sequential Run Benchmark Registry Comparison */}
        <RunComparison
          runs={runs}
          activeChampionRunNumber={activeChampionRunNumber}
        />
      </div>

      {/* Agent Detail Inspector Modal */}
      {selectedAgent && (
        <AgentModal
          agent={selectedAgent}
          status={agentStatuses[selectedAgent.id] || 'idle'}
          onClose={() => setSelectedAgent(null)}
          reducedMotion={reducedMotion}
        />
      )}
    </div>
  );
}
