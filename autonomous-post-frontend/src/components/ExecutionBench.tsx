import React from 'react';
import { AgentDefinition, AgentStatus, StageId, ActiveHandoff } from '../types';
import { AgentAvatar } from './AgentAvatar';
import { 
  ArrowRight, 
  ShieldCheck, 
  AlertCircle, 
  Terminal, 
  Zap, 
  Cpu, 
  ChevronRight,
  Database,
  Lock,
  GitBranch,
  Info
} from 'lucide-react';

interface ExecutionBenchProps {
  agents: AgentDefinition[];
  agentStatuses: Record<string, AgentStatus>;
  currentStageId: StageId;
  activeAgentId: string | null;
  activeHandoff: ActiveHandoff | null;
  overallProgress: number;
  reducedMotion: boolean;
  onSelectAgent: (agent: AgentDefinition) => void;
  blockedDetails?: {
    stage: string;
    agentName: string;
    reason: string;
    actionRequired: string;
  } | null;
}

export const ExecutionBench: React.FC<ExecutionBenchProps> = ({
  agents,
  agentStatuses,
  currentStageId,
  activeAgentId,
  activeHandoff,
  overallProgress,
  reducedMotion,
  onSelectAgent,
  blockedDetails,
}) => {
  // Find current running agent index
  const activeAgent = agents.find((a) => a.id === activeAgentId);
  const activeIndex = activeAgent ? activeAgent.index : 0;

  return (
    <div id="execution-bench-container" className="relative w-full rounded-2xl bg-[#090e1c] border border-slate-800/80 p-5 shadow-2xl overflow-hidden">
      {/* Circuit background accent lines */}
      <div className="absolute inset-0 control-grid-bg opacity-40 pointer-events-none" />
      <div className="absolute top-0 right-0 w-96 h-48 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-96 h-48 bg-purple-500/5 rounded-full blur-3xl pointer-events-none" />

      {/* Header bar of the Bench */}
      <div className="relative z-10 flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800/70">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-sm">
            <Zap className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-display text-base font-bold tracking-wider text-slate-100 uppercase">
                Autonomous Agentic Bench
              </h2>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-cyan-950 text-cyan-300 border border-cyan-500/30">
                8 AGENT ORCHESTRATION
              </span>
            </div>
            <p className="text-xs text-slate-400 font-sans">
              Connected workflow rail with verified cryptographic handoffs & strict deterministic gates
            </p>
          </div>
        </div>

        {/* Bench Progress & Mode Indicator */}
        <div className="flex items-center gap-4">
          <div className="flex flex-col items-end">
            <div className="flex items-center gap-1.5 font-mono text-xs">
              <span className="text-slate-400">PIPELINE EXECUTION:</span>
              <span className="font-bold text-cyan-300">{overallProgress}%</span>
            </div>
            <div className="w-36 h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800 mt-1">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 via-purple-500 to-emerald-500 transition-all duration-500 rounded-full"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
          </div>

          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-[11px] font-mono text-slate-300">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>AWS US-WEST-2</span>
          </div>
        </div>
      </div>

      {/* Warning/Blocked Banner if execution is halted */}
      {blockedDetails && (
        <div
          id="blocked-execution-alert"
          className="relative z-10 mt-4 p-3.5 rounded-xl bg-amber-950/40 border border-amber-500/50 flex items-start gap-3 text-amber-200 shadow-lg"
        >
          <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5 animate-bounce" />
          <div className="flex-1 text-xs">
            <div className="flex items-center gap-2">
              <span className="font-mono font-bold uppercase tracking-wider text-amber-300">
                BLOCKED STATE ENCOUNTERED: {blockedDetails.stage}
              </span>
              <span className="px-1.5 py-0.5 rounded bg-amber-900/60 border border-amber-400/40 font-mono text-[10px] text-amber-200">
                GATE HOLD
              </span>
            </div>
            <p className="mt-1 text-slate-300 font-sans">
              <span className="text-amber-300 font-semibold">{blockedDetails.agentName}:</span> {blockedDetails.reason}
            </p>
            <p className="mt-1 font-mono text-[11px] text-amber-400/90 flex items-center gap-1">
              <span className="font-semibold text-amber-300">Required Resolution:</span> {blockedDetails.actionRequired}
            </p>
          </div>
        </div>
      )}

      {/* Connected Workflow Rail Canvas */}
      <div className="relative z-10 my-8 px-2 overflow-x-auto pb-4 pt-2">
        <div className="min-w-[860px] relative">
          {/* Main Connected Rail Line (Background Track) */}
          <div className="absolute top-[32px] left-[32px] right-[32px] h-[3px] bg-slate-800 -z-0 rounded-full" />

          {/* Active Flow Progress Rail (Laser / Energy Beam) */}
          <div
            className="absolute top-[32px] left-[32px] h-[3px] bg-gradient-to-r from-cyan-500 via-purple-500 to-emerald-400 -z-0 transition-all duration-700 shadow-[0_0_12px_rgba(56,189,248,0.8)]"
            style={{
              width: `${Math.min(
                100,
                Math.max(0, ((Math.max(0, activeIndex - 0.5)) / (agents.length - 1)) * 100)
              )}%`,
            }}
          />

          {/* Animated Energy Particles along rail when active */}
          {!reducedMotion && activeAgentId && (
            <div
              className="absolute top-[30px] -z-0 w-3 h-3 rounded-full bg-cyan-300 shadow-[0_0_10px_#22d3ee] transition-all duration-700"
              style={{
                left: `calc(32px + ${Math.min(
                  100,
                  Math.max(0, ((activeIndex - 1) / (agents.length - 1)) * 100)
                )}% * 0.93)`,
              }}
            />
          )}

          {/* The 8 AI Agent Pods */}
          <div className="relative z-10 grid grid-cols-8 gap-2">
            {agents.map((agent) => {
              const status = agentStatuses[agent.id] || 'idle';
              const isActive = agent.id === activeAgentId;
              const isHandoffTarget = activeHandoff?.toAgent.id === agent.id;

              return (
                <div key={agent.id} className="flex justify-center">
                  <AgentAvatar
                    agent={agent}
                    status={status}
                    isActive={isActive}
                    isHandoffTarget={isHandoffTarget}
                    reducedMotion={reducedMotion}
                    onClick={() => onSelectAgent(agent)}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Live Handoff Telemetry Footer */}
      <div className="relative z-10 pt-3 border-t border-slate-800/80 grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
        {/* Active Handoff Card */}
        <div className="md:col-span-8 bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-wrap sm:flex-nowrap items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-purple-950/60 border border-purple-500/40 flex items-center justify-center text-purple-400 shrink-0">
            <GitBranch className="w-3.5 h-3.5" />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 text-[11px] font-mono">
              <span className="text-slate-400">ACTIVE LIFECYCLE HANDOFF:</span>
              {activeHandoff ? (
                <div className="flex items-center gap-1.5 text-purple-300 font-semibold truncate">
                  <span>{activeHandoff.fromAgent.shortName}</span>
                  <ArrowRight className="w-3 h-3 text-cyan-400 shrink-0" />
                  <span className="text-cyan-300">{activeHandoff.toAgent.shortName}</span>
                </div>
              ) : (
                <span className="text-slate-500">Benchmark Baseline Standby</span>
              )}
            </div>

            <div className="text-xs text-slate-300 font-sans truncate mt-0.5">
              {activeHandoff ? (
                <span className="flex items-center gap-2">
                  <span className="text-slate-400">Artifact:</span>
                  <span className="font-mono text-cyan-200 truncate">{activeHandoff.artifactTransferred}</span>
                </span>
              ) : (
                <span className="text-slate-400">Pipeline ready. Awaiting Run Launcher preflight execution.</span>
              )}
            </div>
          </div>

          {activeHandoff && (
            <div className="hidden lg:flex flex-col items-end text-right font-mono text-[10px] text-slate-400 shrink-0 pl-2 border-l border-slate-800">
              <span className="text-slate-500">PAYLOAD HASH</span>
              <span className="text-cyan-400/90 font-mono truncate max-w-[120px]">
                {activeHandoff.payloadHash}
              </span>
            </div>
          )}
        </div>

        {/* Verification Guard Stamp */}
        <div className="md:col-span-4 bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <div>
              <div className="font-mono text-[10px] text-slate-400">VERIFICATION INTEGRITY</div>
              <div className="text-xs font-semibold text-emerald-300 font-mono">TRUTHFUL OBSERVER</div>
            </div>
          </div>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950/80 text-emerald-300 border border-emerald-500/30">
            SEALED
          </span>
        </div>
      </div>
    </div>
  );
};
