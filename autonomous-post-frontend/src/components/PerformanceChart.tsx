import React from 'react';
import { SequentialRunRecord } from '../types';
import { TrendingUp, AlertCircle, ShieldAlert, CheckCircle2 } from 'lucide-react';

interface PerformanceChartProps {
  runs: SequentialRunRecord[];
}

export const PerformanceChart: React.FC<PerformanceChartProps> = ({ runs }) => {
  // Chart dimensions & scaling
  const chartHeight = 220;
  const paddingX = 40;
  const paddingY = 30;

  // Domain: 60% to 90%
  const minScore = 65;
  const maxScore = 90;
  const scoreRange = maxScore - minScore;

  const getY = (score: number) => {
    const clamped = Math.max(minScore, Math.min(maxScore, score));
    const normalized = (clamped - minScore) / scoreRange;
    return chartHeight - paddingY - normalized * (chartHeight - paddingY * 2);
  };

  return (
    <div id="performance-chart-card" className="rounded-2xl bg-[#090e1c] border border-slate-800/80 p-5 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-950/80 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
            <TrendingUp className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-display text-sm font-bold tracking-wider text-slate-100 uppercase">
              Held-Out AST Performance Trajectory
            </h3>
            <p className="text-[11px] text-slate-400 font-sans">
              Strict mathematical baseline vs. candidate pass rate • Rejects unverified metrics
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-slate-500 rounded" />
            <span className="text-slate-400">Zero-Shot Baseline (71.2%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-dashed border-t border-dashed border-amber-400/80" />
            <span className="text-amber-400/90">+2.5% Gate Bar</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_6px_#22d3ee]" />
            <span className="text-cyan-300">Candidate Score</span>
          </div>
        </div>
      </div>

      {/* SVG Chart */}
      <div className="relative mt-4 w-full overflow-x-auto">
        <div className="min-w-[640px]">
          <svg viewBox="0 0 700 240" className="w-full h-56 select-none">
            {/* Background Grid Lines */}
            {[70, 75, 80, 85, 90].map((level) => {
              const y = getY(level);
              return (
                <g key={level}>
                  <line
                    x1="45"
                    y1={y}
                    x2="680"
                    y2={y}
                    stroke="rgba(51, 65, 85, 0.4)"
                    strokeDasharray="3 3"
                  />
                  <text
                    x="12"
                    y={y + 4}
                    fill="#64748b"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    {level}%
                  </text>
                </g>
              );
            })}

            {/* Baseline 71.2% Reference Line */}
            <line
              x1="45"
              y1={getY(71.2)}
              x2="680"
              y2={getY(71.2)}
              stroke="rgba(148, 163, 184, 0.6)"
              strokeWidth="1.5"
            />

            {/* Promotion Bar (+2.5% over 71.2% = 73.7%) */}
            <line
              x1="45"
              y1={getY(73.7)}
              x2="680"
              y2={getY(73.7)}
              stroke="rgba(251, 191, 36, 0.6)"
              strokeWidth="1.5"
              strokeDasharray="4 3"
            />

            {/* Connection Line between valid candidate points */}
            {(() => {
              const validPoints = runs
                .map((r, i) => {
                  const x = 90 + i * 135;
                  if (r.isVerifiable && r.candidateScore !== null) {
                    return { x, y: getY(r.candidateScore), run: r };
                  }
                  return null;
                })
                .filter(Boolean) as { x: number; y: number; run: SequentialRunRecord }[];

              if (validPoints.length < 2) return null;

              const pathD = validPoints.reduce((acc, p, idx) => {
                return idx === 0 ? `M ${p.x} ${p.y}` : `${acc} L ${p.x} ${p.y}`;
              }, '');

              return (
                <path
                  d={pathD}
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="2.5"
                  className="transition-all duration-500"
                />
              );
            })()}

            {/* Run Column Nodes & Markers */}
            {runs.map((r, i) => {
              const x = 90 + i * 135;

              if (!r.isVerifiable || r.candidateScore === null) {
                // Truthful observer rule: reject fake values for unverified or in-progress runs
                return (
                  <g key={r.runId} className="group cursor-help">
                    {/* Pale dashed column indicating unverified / incomplete metric */}
                    <line
                      x1={x}
                      y1="25"
                      x2={x}
                      y2="200"
                      stroke="rgba(100, 116, 139, 0.25)"
                      strokeDasharray="4 4"
                    />

                    {/* Warning placeholder node */}
                    <circle cx={x} cy={getY(75)} r="14" fill="#0f172a" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="2 2" />
                    <text
                      x={x}
                      y={getY(75) + 3}
                      fill="#f59e0b"
                      fontSize="9"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      HOLD
                    </text>

                    {/* Unverified note */}
                    <text
                      x={x}
                      y="218"
                      fill="#94a3b8"
                      fontSize="10"
                      fontFamily="monospace"
                      textAnchor="middle"
                      fontWeight="bold"
                    >
                      Run #{r.runNumber}
                    </text>
                    <text
                      x={x}
                      y="230"
                      fill="#f59e0b"
                      fontSize="9"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      [Unverified / Pending]
                    </text>
                  </g>
                );
              }

              const y = getY(r.candidateScore);
              const isPromoted = r.promotionDecision === 'PROMOTED';
              const isRejected = r.promotionDecision === 'REJECTED';

              return (
                <g key={r.runId} className="group">
                  {/* Stem line from baseline to score */}
                  <line
                    x1={x}
                    y1={getY(r.baselineScore)}
                    x2={x}
                    y2={y}
                    stroke={isPromoted ? '#22c55e' : isRejected ? '#ef4444' : '#06b6d4'}
                    strokeWidth="1.5"
                    strokeDasharray={isRejected ? '3 2' : 'none'}
                  />

                  {/* Candidate Data Point Glow */}
                  <circle
                    cx={x}
                    cy={y}
                    r="8"
                    fill="#020617"
                    stroke={isPromoted ? '#22c55e' : isRejected ? '#ef4444' : '#06b6d4'}
                    strokeWidth="2.5"
                  />
                  <circle
                    cx={x}
                    cy={y}
                    r="3.5"
                    fill={isPromoted ? '#22c55e' : isRejected ? '#ef4444' : '#06b6d4'}
                  />

                  {/* Score Label Bubble */}
                  <rect
                    x={x - 22}
                    y={y - 25}
                    width="44"
                    height="18"
                    rx="4"
                    fill="#0b1329"
                    stroke={isPromoted ? '#22c55e' : '#334155'}
                    strokeWidth="1"
                  />
                  <text
                    x={x}
                    y={y - 13}
                    fill={isPromoted ? '#4ade80' : '#e2e8f0'}
                    fontSize="10"
                    fontFamily="monospace"
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {r.candidateScore.toFixed(1)}%
                  </text>

                  {/* X Axis Label */}
                  <text
                    x={x}
                    y="218"
                    fill="#e2e8f0"
                    fontSize="10"
                    fontFamily="monospace"
                    textAnchor="middle"
                    fontWeight="bold"
                  >
                    Run #{r.runNumber}
                  </text>
                  <text
                    x={x}
                    y="230"
                    fill={isPromoted ? '#4ade80' : isRejected ? '#f87171' : '#94a3b8'}
                    fontSize="9"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    {r.promotionDecision}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Verifiability Footnote Banner */}
      <div className="mt-2 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-slate-400">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
          <span>Metric Truthfulness Constraint: Incomplete or unverified runs are NEVER fabricated.</span>
        </div>
        <div className="text-[11px] text-slate-500">
          Suite: FuncBench-Core (420 AST samples)
        </div>
      </div>
    </div>
  );
};
