import React, { useState, useEffect } from 'react';
import { 
  Bot, 
  Cpu, 
  ShieldCheck, 
  Globe, 
  Sparkles, 
  Eye, 
  EyeOff, 
  RotateCcw,
  SlidersHorizontal,
  Flame,
  Zap
} from 'lucide-react';

interface ControlRoomHeaderProps {
  reducedMotion: boolean;
  onToggleReducedMotion: () => void;
  onResetDemo: () => void;
  championScore: number;
  championRunNumber: number;
}

export const ControlRoomHeader: React.FC<ControlRoomHeaderProps> = ({
  reducedMotion,
  onToggleReducedMotion,
  onResetDemo,
  championScore,
  championRunNumber,
}) => {
  const [utcTime, setUtcTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="relative w-full rounded-2xl bg-[#090e1c] border border-slate-800/80 p-4 shadow-2xl mb-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Logo & Main Title */}
        <div className="flex items-center gap-3.5">
          <div className="relative w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-500 via-blue-600 to-violet-700 flex items-center justify-center text-slate-950 shadow-[0_0_20px_rgba(6,182,212,0.4)]">
            <Bot className="w-6 h-6 text-slate-950" />
            <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 border-2 border-[#090e1c] flex items-center justify-center">
              <Zap className="w-2.5 h-2.5 text-slate-950 fill-current" />
            </div>
          </div>

          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="font-display text-lg sm:text-xl font-bold tracking-wider text-slate-100 uppercase">
                Autonomous Post-Training Engineer
              </h1>
              <span className="hidden sm:inline-flex px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950/80 text-cyan-300 border border-cyan-500/40 font-semibold">
                FUNCTIONGEMMA BENCH
              </span>
            </div>
            <p className="text-xs text-slate-400 font-sans mt-0.5">
              Self-directed 8-agent benchmark, curriculum, QLoRA & SageMaker promotion loop
            </p>
          </div>
        </div>

        {/* Status Indicators & Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Champion Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-950/40 border border-emerald-500/40 text-xs font-mono">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <div>
              <span className="text-[10px] text-slate-400 block uppercase">Current Champion</span>
              <span className="font-bold text-emerald-300">
                Run #{championRunNumber} • {championScore.toFixed(1)}% AST
              </span>
            </div>
          </div>

          {/* AWS Region & UTC */}
          <div className="hidden md:flex flex-col items-end px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-400">
            <div className="flex items-center gap-1.5 text-slate-200">
              <Globe className="w-3.5 h-3.5 text-cyan-400" />
              <span>us-west-2 (Oregon)</span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            </div>
            <span className="text-[10px] text-slate-500">{utcTime || 'Syncing UTC clock...'}</span>
          </div>

          {/* Reduced Motion Toggle */}
          <button
            onClick={onToggleReducedMotion}
            className={`px-3 py-1.5 rounded-xl border text-xs font-mono flex items-center gap-1.5 transition-colors ${
              reducedMotion
                ? 'bg-amber-950/60 border-amber-500/40 text-amber-300'
                : 'bg-slate-900 hover:bg-slate-800 border-slate-700 text-slate-300 hover:text-cyan-300'
            }`}
            title="Toggle high-performance / reduced motion"
          >
            {reducedMotion ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            <span>{reducedMotion ? 'Reduced Motion' : 'Fluid Motion'}</span>
          </button>

          {/* Reset Demo */}
          <button
            onClick={onResetDemo}
            className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
            title="Reset Workflow State"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
