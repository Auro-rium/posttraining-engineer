import React from 'react';
import { AgentDefinition, AgentStatus } from '../types';
import { CheckCircle2, AlertTriangle, XCircle, Loader2, Sparkles } from 'lucide-react';

interface AgentAvatarProps {
  agent: AgentDefinition;
  status: AgentStatus;
  isActive: boolean;
  isHandoffTarget?: boolean;
  reducedMotion?: boolean;
  onClick?: () => void;
}

export const AgentAvatar: React.FC<AgentAvatarProps> = ({
  agent,
  status,
  isActive,
  isHandoffTarget = false,
  reducedMotion = false,
  onClick,
}) => {
  const accent = agent.accent;

  // Visual state styling
  const getContainerAura = () => {
    if (status === 'failed') return 'ring-2 ring-red-500 shadow-[0_0_20px_rgba(239,68,68,0.5)]';
    if (status === 'blocked') return 'ring-2 ring-amber-500 shadow-[0_0_20px_rgba(245,158,11,0.5)]';
    if (isActive || status === 'running') {
      return `ring-2 ${accent.border} shadow-[0_0_24px_${accent.glow}]`;
    }
    if (status === 'completed') return 'ring-1 ring-emerald-500/60 shadow-[0_0_12px_rgba(16,185,129,0.3)]';
    return 'ring-1 ring-slate-800 hover:ring-slate-700';
  };

  const getStatusBadge = () => {
    switch (status) {
      case 'completed':
        return (
          <div className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-emerald-950 border border-emerald-400 flex items-center justify-center text-emerald-400 shadow-md">
            <CheckCircle2 className="w-3.5 h-3.5" />
          </div>
        );
      case 'running':
        return (
          <div className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-slate-950 border border-cyan-400 flex items-center justify-center text-cyan-400 shadow-md">
            <Loader2 className={`w-3 h-3 ${reducedMotion ? '' : 'animate-spin'}`} />
          </div>
        );
      case 'blocked':
        return (
          <div className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-amber-950 border border-amber-400 flex items-center justify-center text-amber-300 shadow-md animate-bounce">
            <AlertTriangle className="w-3 h-3" />
          </div>
        );
      case 'failed':
        return (
          <div className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-red-950 border border-red-500 flex items-center justify-center text-red-400 shadow-md">
            <XCircle className="w-3.5 h-3.5" />
          </div>
        );
      default:
        return (
          <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-slate-900 border border-slate-700 flex items-center justify-center text-[9px] font-mono text-slate-400">
            {agent.index}
          </div>
        );
    }
  };

  // Unique Expressive Bot Face Renderers
  const renderBotFace = () => {
    const isRunning = status === 'running' || isActive;
    const isFailed = status === 'failed';
    const isBlocked = status === 'blocked';
    const isCompleted = status === 'completed';

    const eyeColor = isFailed
      ? '#ef4444'
      : isBlocked
      ? '#f59e0b'
      : isCompleted
      ? '#10b981'
      : isRunning
      ? accent.hex
      : '#64748b';

    switch (agent.avatarStyle) {
      case 'scanner': // Benchmark Agent
        return (
          <svg viewBox="0 0 48 48" className="w-10 h-10">
            {/* Robot Head Body */}
            <rect x="8" y="10" width="32" height="28" rx="6" fill="#0b1329" stroke={eyeColor} strokeWidth="1.5" />
            {/* Top Antenna */}
            <line x1="24" y1="10" x2="24" y2="4" stroke={eyeColor} strokeWidth="1.5" />
            <circle cx="24" cy="4" r="2.5" fill={eyeColor} className={isRunning && !reducedMotion ? 'animate-ping' : ''} />
            {/* Scanner Visor */}
            <rect x="12" y="18" width="24" height="8" rx="3" fill="#030712" stroke={eyeColor} strokeWidth="1" />
            <circle cx={isRunning ? 20 : 24} cy="22" r="2" fill={eyeColor}>
              {isRunning && !reducedMotion && (
                <animate attributeName="cx" values="16;32;16" dur="1.6s" repeatCount="indefinite" />
              )}
            </circle>
            {/* Tool mouth status grid */}
            <line x1="16" y1="31" x2="32" y2="31" stroke={eyeColor} strokeWidth="1" strokeDasharray="2 2" />
          </svg>
        );

      case 'visor': // Failure Analyst
        return (
          <svg viewBox="0 0 48 48" className="w-10 h-10">
            <rect x="8" y="12" width="32" height="26" rx="5" fill="#17120a" stroke={eyeColor} strokeWidth="1.5" />
            {/* Dual mini ears */}
            <rect x="4" y="20" width="4" height="10" rx="1" fill={eyeColor} />
            <rect x="40" y="20" width="4" height="10" rx="1" fill={eyeColor} />
            {/* Wide Holographic Analysis Visor */}
            <path d="M 12 18 L 36 18 L 34 26 L 14 26 Z" fill="#291a05" stroke={eyeColor} strokeWidth="1.2" />
            <line x1="14" y1="22" x2="34" y2="22" stroke={eyeColor} strokeWidth="1.5" strokeDasharray={isRunning ? "4 2" : "none"} />
            {/* Diagnostic bar */}
            <rect x="15" y="30" width="18" height="3" rx="1" fill="#0b0f19" stroke={eyeColor} strokeWidth="0.8" />
            <rect x="16" y="31" width={isRunning ? "12" : "6"} height="1" fill={eyeColor} />
          </svg>
        );

      case 'antenna': // Research Agent
        return (
          <svg viewBox="0 0 48 48" className="w-10 h-10">
            {/* Curved dome head */}
            <path d="M 10 24 C 10 14 38 14 38 24 L 38 36 C 38 39 35 41 32 41 L 16 41 C 13 41 10 39 10 36 Z" fill="#150d24" stroke={eyeColor} strokeWidth="1.5" />
            {/* Triple Reasoning Array */}
            <path d="M 18 14 L 14 6" stroke={eyeColor} strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="14" cy="6" r="2" fill={eyeColor} />
            <path d="M 24 14 L 24 4" stroke={eyeColor} strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="24" cy="4" r="2" fill={eyeColor} />
            <path d="M 30 14 L 34 6" stroke={eyeColor} strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="34" cy="6" r="2" fill={eyeColor} />
            {/* Deep Thinking Optical sensors */}
            <circle cx="18" cy="26" r="3" fill="#030712" stroke={eyeColor} strokeWidth="1.2" />
            <circle cx="18" cy="26" r="1.5" fill={eyeColor} />
            <circle cx="30" cy="26" r="3" fill="#030712" stroke={eyeColor} strokeWidth="1.2" />
            <circle cx="30" cy="26" r="1.5" fill={eyeColor} />
            {/* Neural synch curve */}
            <path d="M 18 34 Q 24 37 30 34" stroke={eyeColor} strokeWidth="1.2" fill="none" strokeLinecap="round" />
          </svg>
        );

      case 'matrix': // Data Curator Agent
        return (
          <svg viewBox="0 0 48 48" className="w-10 h-10">
            <rect x="7" y="11" width="34" height="28" rx="4" fill="#0a1226" stroke={eyeColor} strokeWidth="1.5" />
            {/* Data tape reel / side buffers */}
            <circle cx="14" cy="18" r="2" fill={eyeColor} />
            <circle cx="24" cy="18" r="2" fill={eyeColor} />
            <circle cx="34" cy="18" r="2" fill={eyeColor} />
            <circle cx="14" cy="25" r="2" fill={eyeColor} />
            <circle cx="24" cy="25" r="2" fill={eyeColor} />
            <circle cx="34" cy="25" r="2" fill={eyeColor} />
            {/* Binary data flow mouth */}
            <rect x="12" y="31" width="24" height="4" rx="1" fill="#030712" stroke={eyeColor} strokeWidth="0.8" />
            <line x1="14" y1="33" x2="32" y2="33" stroke={eyeColor} strokeWidth="1" strokeDasharray="3 2" />
          </svg>
        );

      case 'core': // Training Designer Agent
        return (
          <svg viewBox="0 0 48 48" className="w-10 h-10">
            {/* Hexagonal Core casing */}
            <polygon points="24,8 39,16 39,34 24,42 9,34 9,16" fill="#1c0f0a" stroke={eyeColor} strokeWidth="1.5" />
            {/* Hyperparameter Center Reactor */}
            <circle cx="24" cy="25" r="6" fill="#2d150b" stroke={eyeColor} strokeWidth="1.5" />
            <circle cx="24" cy="25" r="3" fill={eyeColor} className={isRunning && !reducedMotion ? 'animate-pulse' : ''} />
            {/* Topology wiring */}
            <line x1="24" y1="8" x2="24" y2="19" stroke={eyeColor} strokeWidth="1" />
            <line x1="9" y1="25" x2="18" y2="25" stroke={eyeColor} strokeWidth="1" />
            <line x1="30" y1="25" x2="39" y2="25" stroke={eyeColor} strokeWidth="1" />
            <line x1="24" y1="31" x2="24" y2="42" stroke={eyeColor} strokeWidth="1" />
          </svg>
        );

      case 'prism': // SageMaker Training Agent
        return (
          <svg viewBox="0 0 48 48" className="w-10 h-10">
            {/* GPU Server chassis frame */}
            <rect x="6" y="10" width="36" height="28" rx="4" fill="#0d1b10" stroke={eyeColor} strokeWidth="1.5" />
            {/* Dual GPU Cooler Fans */}
            <circle cx="16" cy="24" r="6" fill="#051007" stroke={eyeColor} strokeWidth="1.2" />
            <circle cx="16" cy="24" r="2" fill={eyeColor} />
            <circle cx="32" cy="24" r="6" fill="#051007" stroke={eyeColor} strokeWidth="1.2" />
            <circle cx="32" cy="24" r="2" fill={eyeColor} />
            {/* Server LED Array */}
            <circle cx="10" cy="14" r="1" fill={eyeColor} />
            <circle cx="14" cy="14" r="1" fill={eyeColor} />
            <circle cx="18" cy="14" r="1" fill={eyeColor} />
            <line x1="8" y1="34" x2="40" y2="34" stroke={eyeColor} strokeWidth="1" strokeDasharray="4 2" />
          </svg>
        );

      case 'dual-eye': // Evaluation Agent
        return (
          <svg viewBox="0 0 48 48" className="w-10 h-10">
            {/* Precision Scope Shell */}
            <rect x="8" y="11" width="32" height="26" rx="6" fill="#071824" stroke={eyeColor} strokeWidth="1.5" />
            {/* High-accuracy dual crosshair optics */}
            <circle cx="17" cy="23" r="5.5" fill="#020d14" stroke={eyeColor} strokeWidth="1.2" />
            <circle cx="17" cy="23" r="2" fill={eyeColor} />
            <circle cx="31" cy="23" r="5.5" fill="#020d14" stroke={eyeColor} strokeWidth="1.2" />
            <circle cx="31" cy="23" r="2" fill={eyeColor} />
            {/* Calibration reticle tick */}
            <line x1="24" y1="13" x2="24" y2="18" stroke={eyeColor} strokeWidth="1" />
            <line x1="16" y1="31" x2="32" y2="31" stroke={eyeColor} strokeWidth="1.2" />
          </svg>
        );

      case 'halo': // Champion / Promotion Agent
        return (
          <svg viewBox="0 0 48 48" className="w-10 h-10">
            {/* Champion Crown Halo */}
            <path d="M 12 10 L 16 16 L 24 8 L 32 16 L 36 10 L 34 20 L 14 20 Z" fill="#052012" stroke={eyeColor} strokeWidth="1.4" />
            {/* Sovereign Bot Head */}
            <rect x="10" y="20" width="28" height="20" rx="5" fill="#081e13" stroke={eyeColor} strokeWidth="1.5" />
            {/* Triumphant Visor */}
            <rect x="15" y="25" width="18" height="6" rx="2" fill="#020f08" stroke={eyeColor} strokeWidth="1" />
            <circle cx="20" cy="28" r="1.5" fill={eyeColor} />
            <circle cx="28" cy="28" r="1.5" fill={eyeColor} />
            {/* Deterministic Verification Check crest */}
            <path d="M 21 34 L 23 36 L 27 33" stroke={eyeColor} strokeWidth="1.4" fill="none" strokeLinecap="round" />
          </svg>
        );

      default:
        return (
          <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-slate-900 border border-slate-700">
            <Sparkles className="w-5 h-5 text-slate-400" />
          </div>
        );
    }
  };

  return (
    <div
      id={`agent-node-${agent.id}`}
      onClick={onClick}
      className={`relative group flex flex-col items-center cursor-pointer transition-all duration-300 select-none ${
        isActive ? 'scale-105' : 'hover:scale-102 opacity-95 hover:opacity-100'
      }`}
    >
      {/* Bot Housing Pod */}
      <div
        className={`relative w-16 h-16 rounded-xl bg-slate-950/90 flex items-center justify-center p-1.5 transition-all duration-300 ${getContainerAura()} ${
          isHandoffTarget ? 'ring-2 ring-violet-400 animate-pulse' : ''
        }`}
      >
        {/* Expressive Face SVG */}
        <div className="w-full h-full flex items-center justify-center">
          {renderBotFace()}
        </div>

        {/* Status Corner Indicator */}
        {getStatusBadge()}

        {/* Active Ambient Glow behind pod */}
        {(isActive || status === 'running') && (
          <div
            className={`absolute -inset-1 rounded-xl pointer-events-none -z-10 blur-sm opacity-50 ${
              status === 'failed' ? 'bg-red-500' : status === 'blocked' ? 'bg-amber-500' : accent.bg
            }`}
          />
        )}
      </div>

      {/* Label and Role */}
      <div className="mt-2 text-center max-w-[105px]">
        <div className="flex items-center justify-center gap-1">
          <span className="font-mono text-[10px] text-slate-500">#{agent.index}</span>
          <span
            className={`font-mono text-xs font-semibold tracking-wide truncate ${
              isActive || status === 'running'
                ? accent.text
                : status === 'completed'
                ? 'text-emerald-400'
                : status === 'blocked'
                ? 'text-amber-400'
                : status === 'failed'
                ? 'text-red-400'
                : 'text-slate-300'
            }`}
          >
            {agent.shortName}
          </span>
        </div>
        <p className="text-[10px] text-slate-400 line-clamp-1 leading-tight font-sans mt-0.5">
          {agent.role}
        </p>
      </div>

      {/* Stage status pill */}
      <div className="mt-1.5">
        <span
          className={`inline-block px-1.5 py-0.5 text-[9px] font-mono rounded tracking-tight uppercase ${
            status === 'running'
              ? 'bg-cyan-950 text-cyan-300 border border-cyan-500/40 animate-pulse'
              : status === 'completed'
              ? 'bg-emerald-950 text-emerald-300 border border-emerald-500/30'
              : status === 'blocked'
              ? 'bg-amber-950 text-amber-300 border border-amber-500/30 font-bold'
              : status === 'failed'
              ? 'bg-red-950 text-red-300 border border-red-500/30'
              : 'bg-slate-900 text-slate-500 border border-slate-800'
          }`}
        >
          {status}
        </span>
      </div>
    </div>
  );
};
