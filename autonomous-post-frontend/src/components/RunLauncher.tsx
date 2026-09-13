import React, { useState } from 'react';
import { RunConfig, PreflightItem, StageId } from '../types';
import { 
  Play, 
  Square, 
  RefreshCw, 
  CheckCircle2, 
  Clock, 
  AlertTriangle, 
  Check, 
  X, 
  Sliders, 
  Coins, 
  ShieldAlert, 
  Cpu, 
  Server,
  Zap,
  FastForward,
  ChevronDown
} from 'lucide-react';

interface RunLauncherProps {
  config: RunConfig;
  onChangeConfig: (newConfig: Partial<RunConfig>) => void;
  preflights: PreflightItem[];
  isRunning: boolean;
  isPaused: boolean;
  currentStageId: StageId;
  onStartRun: () => void;
  onStopCleanup: () => void;
  onGrantApproval: () => void;
  onRejectApproval: () => void;
  onTriggerScenario: (scenario: 'normal' | 'blocked_quota' | 'blocked_approval' | 'regression_reject') => void;
  executionSpeed: number;
  onChangeSpeed: (speed: number) => void;
  onStepNext: () => void;
}

export const RunLauncher: React.FC<RunLauncherProps> = ({
  config,
  onChangeConfig,
  preflights,
  isRunning,
  isPaused,
  currentStageId,
  onStartRun,
  onStopCleanup,
  onGrantApproval,
  onRejectApproval,
  onTriggerScenario,
  executionSpeed,
  onChangeSpeed,
  onStepNext,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Determine active preflight status
  const allPreflightsPassed = preflights.every((p) => p.status === 'passed');
  const hasBlockedPreflight = preflights.some((p) => p.status === 'blocked');

  // Stages that map to the 9 visual start flows
  const flowSteps = [
    { num: 1, label: 'Preflight Checks', active: true, done: allPreflightsPassed },
    { num: 2, label: 'GPU Quota Confirmed', active: true, done: preflights.find(p => p.id === 'quota')?.status === 'passed' },
    { num: 3, label: 'Cost Validated', active: true, done: preflights.find(p => p.id === 'cost')?.status === 'passed' },
    { 
      num: 4, 
      label: 'Human Approval', 
      active: true, 
      done: config.approvalStatus === 'APPROVED',
      blocked: config.approvalStatus === 'PENDING'
    },
    { 
      num: 5, 
      label: 'SageMaker Submission', 
      active: isRunning, 
      done: ['heldout_eval', 'deterministic_promotion'].includes(currentStageId) 
    },
    { 
      num: 6, 
      label: 'Training Progress', 
      active: isRunning, 
      done: ['heldout_eval', 'deterministic_promotion'].includes(currentStageId) 
    },
    { 
      num: 7, 
      label: 'Held-Out Eval', 
      active: isRunning, 
      done: currentStageId === 'deterministic_promotion' 
    },
    { 
      num: 8, 
      label: 'Deterministic Gate', 
      active: isRunning, 
      done: false 
    },
    { 
      num: 9, 
      label: 'Artifact Retention', 
      active: isRunning, 
      done: false 
    },
  ];

  return (
    <div id="run-launcher-panel" className="rounded-2xl bg-[#090e1c] border border-slate-800/80 p-5 shadow-xl flex flex-col justify-between">
      {/* Panel Top Title */}
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-violet-950/80 border border-violet-500/40 flex items-center justify-center text-violet-400">
              <Sliders className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-display text-sm font-bold tracking-wider text-slate-100 uppercase">
                Run Launcher & Controls
              </h3>
              <p className="text-[11px] text-slate-400 font-sans">
                Autonomous post-training parameter initialization & AWS orchestrator
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Speed toggle */}
            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-[11px] font-mono">
              <span className="text-slate-500 px-1.5 flex items-center gap-1">
                <FastForward className="w-3 h-3" />
              </span>
              {[1, 2, 4].map((spd) => (
                <button
                  key={spd}
                  onClick={() => onChangeSpeed(spd)}
                  className={`px-2 py-0.5 rounded text-xs transition-colors ${
                    executionSpeed === spd
                      ? 'bg-cyan-500 text-slate-950 font-bold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {spd}x
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Configuration Matrix */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
          {/* Target Model */}
          <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3">
            <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
              Target Model
            </label>
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold text-cyan-300">
                {config.targetModel}
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/30 text-cyan-400">
                Weights Base
              </span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1 font-sans">
              Google FunctionGemma lightweight tool-calling tuned architecture
            </p>
          </div>

          {/* Reasoning Model */}
          <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3">
            <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
              Reasoning Engine (Research Agent)
            </label>
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold text-violet-300">
                {config.reasoningModel}
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-violet-950/80 border border-violet-500/30 text-violet-400">
                128k ctx
              </span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1 font-sans">
              High-depth curriculum & failure mode synthesis generator
            </p>
          </div>

          {/* Benchmark Suite & Seed */}
          <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3">
            <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
              Benchmark Suite & Seed
            </label>
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs text-slate-200 truncate">
                {config.benchmarkSuite} {config.benchmarkVersion}
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-300">
                seed: {config.seed}
              </span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1 font-sans">
              420 strict held-out multi-tool scenarios with AST validation
            </p>
          </div>

          {/* GPU Instance & Estimated Cost */}
          <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-3">
            <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
              SageMaker GPU Instance & Envelope
            </label>
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold text-emerald-300 flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                {config.gpuInstanceType}
              </span>
              <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-500/30">
                ~${config.estimatedCostUsd.toFixed(2)} USD
              </span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1 font-sans">
              4x NVIDIA A10G (96GB VRAM) • Estimated duration 48 min
            </p>
          </div>
        </div>

        {/* Human Approval Required Gate Notification (Step 4 of Flow) */}
        {config.approvalStatus === 'PENDING' && (
          <div
            id="human-approval-gate"
            className="mt-4 p-3.5 rounded-xl bg-violet-950/40 border border-violet-500/60 shadow-lg animate-pulse"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-violet-900/80 border border-violet-400 flex items-center justify-center text-violet-200">
                  <ShieldAlert className="w-4 h-4" />
                </div>
                <div>
                  <div className="font-mono text-xs font-bold text-violet-200">
                    STAGE 4: HUMAN APPROVAL REQUIRED
                  </div>
                  <div className="text-[11px] text-slate-300 font-sans mt-0.5">
                    SageMaker compute envelope (${config.estimatedCostUsd.toFixed(2)}) requires human sign-off before on-demand job dispatch.
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  id="btn-reject-approval"
                  onClick={onRejectApproval}
                  className="px-3 py-1.5 rounded-lg bg-red-950 hover:bg-red-900 border border-red-500/50 text-red-300 text-xs font-mono font-medium flex items-center gap-1 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                  Reject
                </button>
                <button
                  id="btn-grant-approval"
                  onClick={onGrantApproval}
                  className="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 border border-emerald-400 text-slate-950 text-xs font-mono font-bold flex items-center gap-1 shadow-[0_0_12px_rgba(16,185,129,0.4)] transition-all"
                >
                  <Check className="w-3.5 h-3.5" />
                  Authorize Run
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Start Flow 9-Phase Checklist Mini Matrix */}
        <div className="mt-4 pt-3 border-t border-slate-800/80">
          <div className="flex items-center justify-between mb-2">
            <span className="font-mono text-[10px] text-slate-400 tracking-wider uppercase">
              Start Flow Verification Path (9-Phase Gate)
            </span>
            <span className="font-mono text-[10px] text-cyan-400">
              {config.approvalStatus === 'APPROVED' ? 'Gate Authorized' : 'Human Gate Enforced'}
            </span>
          </div>

          <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-1.5">
            {flowSteps.map((s) => (
              <div
                key={s.num}
                className={`p-1.5 rounded-lg border text-center font-mono text-[10px] transition-all ${
                  s.done
                    ? 'bg-emerald-950/50 border-emerald-500/40 text-emerald-300'
                    : s.blocked
                    ? 'bg-amber-950/60 border-amber-400 text-amber-300 animate-pulse'
                    : 'bg-slate-950/60 border-slate-800 text-slate-500'
                }`}
              >
                <div className="flex items-center justify-center mb-0.5">
                  {s.done ? (
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                  ) : s.blocked ? (
                    <AlertTriangle className="w-3 h-3 text-amber-400" />
                  ) : (
                    <span className="text-[9px] text-slate-600">#{s.num}</span>
                  )}
                </div>
                <div className="truncate leading-tight">{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Interactive Scenario Trigger Buttons for Hackathon Evaluation */}
        <div className="mt-4 p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] font-mono text-slate-400 uppercase">
              Truthful Observer Simulation Bench
            </span>
            <span className="text-[9px] font-mono text-slate-500">
              Simulate real cloud failure & approval limits
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <button
              onClick={() => onTriggerScenario('normal')}
              className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 font-mono text-[10px] transition-colors"
            >
              Default Full Run
            </button>
            <button
              onClick={() => onTriggerScenario('blocked_quota')}
              className="px-2 py-1 rounded bg-amber-950/40 hover:bg-amber-900/60 border border-amber-500/40 text-amber-300 font-mono text-[10px] transition-colors"
            >
              Simulate Quota Blocked
            </button>
            <button
              onClick={() => onTriggerScenario('blocked_approval')}
              className="px-2 py-1 rounded bg-violet-950/40 hover:bg-violet-900/60 border border-violet-500/40 text-violet-300 font-mono text-[10px] transition-colors"
            >
              Require Manual Approval
            </button>
            <button
              onClick={() => onTriggerScenario('regression_reject')}
              className="px-2 py-1 rounded bg-red-950/40 hover:bg-red-900/60 border border-red-500/40 text-red-300 font-mono text-[10px] transition-colors"
            >
              Simulate Regression Reject
            </button>
          </div>
        </div>
      </div>

      {/* Main Execution Action Controls */}
      <div className="mt-5 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {isRunning ? (
            <button
              id="btn-stop-cleanup"
              onClick={onStopCleanup}
              className="px-4 py-2.5 rounded-xl bg-red-950 hover:bg-red-900/90 border border-red-500/60 text-red-200 text-xs font-mono font-bold flex items-center gap-2 shadow-lg transition-all"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              Stop & Cleanup AWS Resources
            </button>
          ) : (
            <button
              id="btn-start-run"
              onClick={onStartRun}
              className="px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-mono font-bold flex items-center gap-2 shadow-[0_0_20px_rgba(6,182,212,0.4)] transition-all hover:scale-102"
            >
              <Play className="w-4 h-4 fill-current" />
              Launch Autonomous Run
            </button>
          )}

          {isRunning && (
            <button
              id="btn-step-next"
              onClick={onStepNext}
              className="px-3 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs font-mono flex items-center gap-1.5 transition-colors"
              title="Manually advance to next agent step"
            >
              <FastForward className="w-3.5 h-3.5 text-cyan-400" />
              Step Forward
            </button>
          )}
        </div>

        <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
          <span>Status:</span>
          {isRunning ? (
            <span className="text-cyan-300 font-semibold flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
              ACTIVE PIPELINE
            </span>
          ) : (
            <span className="text-slate-500">STANDBY / READY</span>
          )}
        </div>
      </div>
    </div>
  );
};
