import React from 'react';
import { AgentDefinition, AgentStatus } from '../types';
import { AgentAvatar } from './AgentAvatar';
import { X, ShieldCheck, Database, Cpu, Activity, ArrowRight, CheckCircle2 } from 'lucide-react';

interface AgentModalProps {
  agent: AgentDefinition | null;
  status: AgentStatus;
  onClose: () => void;
  reducedMotion: boolean;
}

export const AgentModal: React.FC<AgentModalProps> = ({
  agent,
  status,
  onClose,
  reducedMotion,
}) => {
  if (!agent) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
      <div
        className="relative w-full max-w-xl rounded-2xl bg-[#0a1022] border border-slate-700/80 p-6 shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top accent banner */}
        <div
          className="absolute top-0 left-0 right-0 h-1.5"
          style={{ backgroundColor: agent.accent.hex }}
        />

        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Header with Avatar and Title */}
        <div className="flex items-start gap-4">
          <AgentAvatar
            agent={agent}
            status={status}
            isActive={status === 'running'}
            reducedMotion={reducedMotion}
          />

          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs text-slate-500">AGENT #{agent.index}</span>
              <span
                className="px-2 py-0.5 rounded text-[10px] font-mono uppercase font-bold"
                style={{
                  backgroundColor: agent.accent.bg,
                  color: agent.accent.hex,
                }}
              >
                {status}
              </span>
            </div>
            <h3 className="font-display text-lg font-bold text-slate-100 mt-0.5">
              {agent.name}
            </h3>
            <p className="text-xs text-slate-400 font-sans mt-0.5">{agent.role}</p>
          </div>
        </div>

        {/* Specialty Description */}
        <div className="mt-5 p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs font-sans text-slate-300 leading-relaxed">
          <span className="font-mono text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
            Specialized Autonomous Mandate
          </span>
          {agent.specialty}
        </div>

        {/* Input/Output Data Handshake */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
          {/* Required Inputs */}
          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <span className="font-mono text-[10px] text-slate-400 uppercase tracking-wider flex items-center gap-1.5 mb-2">
              <Database className="w-3 h-3 text-cyan-400" />
              Required Ingest Artifacts
            </span>
            <ul className="space-y-1.5 text-xs text-slate-300 font-mono">
              {agent.inputsRequired.map((inp, idx) => (
                <li key={idx} className="flex items-start gap-1.5 text-[11px]">
                  <ArrowRight className="w-3 h-3 text-cyan-400 shrink-0 mt-0.5" />
                  <span className="leading-snug">{inp}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Produced Outputs */}
          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <span className="font-mono text-[10px] text-slate-400 uppercase tracking-wider flex items-center gap-1.5 mb-2">
              <ShieldCheck className="w-3 h-3 text-emerald-400" />
              Verified Outputs Produced
            </span>
            <ul className="space-y-1.5 text-xs text-slate-300 font-mono">
              {agent.outputsProduced.map((out, idx) => (
                <li key={idx} className="flex items-start gap-1.5 text-[11px]">
                  <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0 mt-0.5" />
                  <span className="leading-snug text-emerald-200">{out}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Monitored Telemetry Metrics */}
        <div className="mt-4 p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <span className="font-mono text-[10px] text-slate-400 uppercase tracking-wider flex items-center gap-1.5 mb-2">
            <Activity className="w-3 h-3 text-purple-400" />
            Monitored Objective Telemetry
          </span>
          <div className="flex flex-wrap gap-1.5">
            {agent.metricsMonitored.map((m, idx) => (
              <span
                key={idx}
                className="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono text-purple-300"
              >
                {m}
              </span>
            ))}
          </div>
        </div>

        {/* Footer info */}
        <div className="mt-5 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-500">
          <span>Target Platform: AWS SageMaker + FunctionGemma</span>
          <button
            onClick={onClose}
            className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono transition-colors"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
};
