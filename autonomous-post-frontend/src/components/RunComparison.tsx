import React, { useState } from 'react';
import { SequentialRunRecord } from '../types';
import { 
  GitCompare, 
  CheckCircle2, 
  XCircle, 
  AlertTriangle, 
  Clock, 
  Copy, 
  Check, 
  Database, 
  Cpu, 
  DollarSign,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  Tag
} from 'lucide-react';

interface RunComparisonProps {
  runs: SequentialRunRecord[];
  activeChampionRunNumber: number;
}

export const RunComparison: React.FC<RunComparisonProps> = ({
  runs,
  activeChampionRunNumber,
}) => {
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1800);
  };

  const getDecisionBadge = (decision: string, verifiable: boolean) => {
    if (!verifiable) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-amber-950/70 text-amber-300 border border-amber-500/40">
          <Clock className="w-3 h-3" />
          AWAITING EVAL
        </span>
      );
    }
    switch (decision) {
      case 'PROMOTED':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-950 text-emerald-300 border border-emerald-500/50 shadow-[0_0_8px_rgba(16,185,129,0.3)]">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            PROMOTED
          </span>
        );
      case 'REJECTED':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-[10px] font-mono font-bold bg-red-950 text-red-300 border border-red-500/40">
            <XCircle className="w-3 h-3 text-red-400" />
            REJECTED
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono text-slate-400 bg-slate-900 border border-slate-800">
            {decision}
          </span>
        );
    }
  };

  const getRegressionBadge = (status: string) => {
    switch (status) {
      case 'ZERO_REGRESSIONS':
        return (
          <span className="font-mono text-xs text-emerald-400 flex items-center gap-1">
            <Check className="w-3 h-3" /> 0 Regressions
          </span>
        );
      case 'CRITICAL_REGRESSION':
        return (
          <span className="font-mono text-xs text-red-400 font-bold flex items-center gap-1">
            <AlertTriangle className="w-3 h-3 text-red-400" /> Critical Regressions
          </span>
        );
      default:
        return <span className="font-mono text-xs text-slate-500">Pending Eval</span>;
    }
  };

  return (
    <div id="run-comparison-panel" className="rounded-2xl bg-[#090e1c] border border-slate-800/80 p-5 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-950/80 border border-blue-500/40 flex items-center justify-center text-blue-400">
            <GitCompare className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-display text-sm font-bold tracking-wider text-slate-100 uppercase">
                Sequential Run Benchmark Registry
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-300">
                5 RUN SEQUENCE
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-sans">
              Cryptographic manifests, AST score deltas, and deterministic promotion logs
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-slate-500">CURRENT PRODUCTION CHAMPION:</span>
          <span className="px-2 py-0.5 rounded bg-emerald-950 border border-emerald-500/40 text-emerald-300 font-bold">
            Run #{activeChampionRunNumber} (80.4%)
          </span>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-xs font-mono border-collapse min-w-[820px]">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase tracking-wider bg-slate-950/50">
              <th className="py-2.5 px-3">Run # / Label</th>
              <th className="py-2.5 px-3">Baseline</th>
              <th className="py-2.5 px-3">Candidate</th>
              <th className="py-2.5 px-3">Delta</th>
              <th className="py-2.5 px-3">Regression Audit</th>
              <th className="py-2.5 px-3">Gate Decision</th>
              <th className="py-2.5 px-3">Cost / Runtime</th>
              <th className="py-2.5 px-3 text-right">Manifest Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {runs.map((r) => {
              const isExpanded = expandedRunId === r.runId;
              const isCurrentChampion = r.runNumber === activeChampionRunNumber;

              return (
                <React.Fragment key={r.runId}>
                  <tr
                    className={`hover:bg-slate-900/40 transition-colors ${
                      isCurrentChampion ? 'bg-emerald-950/10' : ''
                    }`}
                  >
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-200">#{r.runNumber}</span>
                        {isCurrentChampion && (
                          <span className="px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-300 border border-emerald-500/40 text-[9px] font-bold">
                            CHAMPION
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-400 font-sans truncate max-w-[190px]">
                        {r.label}
                      </div>
                    </td>

                    <td className="py-3 px-3 text-slate-300">
                      {r.baselineScore.toFixed(1)}%
                    </td>

                    <td className="py-3 px-3">
                      {r.isVerifiable && r.candidateScore !== null ? (
                        <span className="font-bold text-cyan-300">
                          {r.candidateScore.toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-amber-400/90 italic text-[11px]">
                          [Awaiting Held-Out]
                        </span>
                      )}
                    </td>

                    <td className="py-3 px-3">
                      {r.isVerifiable && r.improvementPct !== null ? (
                        <span
                          className={`font-bold ${
                            r.improvementPct > 0 ? 'text-emerald-400' : 'text-slate-400'
                          }`}
                        >
                          {r.improvementPct > 0 ? '+' : ''}
                          {r.improvementPct.toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>

                    <td className="py-3 px-3">
                      {getRegressionBadge(r.regressionStatus)}
                    </td>

                    <td className="py-3 px-3">
                      {getDecisionBadge(r.promotionDecision, r.isVerifiable)}
                    </td>

                    <td className="py-3 px-3 text-slate-400 text-[11px]">
                      <div>${r.totalCostUsd.toFixed(2)}</div>
                      <div className="text-slate-500 text-[10px]">{r.runtimeMinutes} min</div>
                    </td>

                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => setExpandedRunId(isExpanded ? null : r.runId)}
                        className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 hover:text-cyan-300 text-[10px] inline-flex items-center gap-1 transition-colors"
                      >
                        {isExpanded ? 'Hide Specs' : 'Inspect Manifest'}
                        {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                      </button>
                    </td>
                  </tr>

                  {/* Expanded Inspector Drawer */}
                  {isExpanded && (
                    <tr className="bg-slate-950/90 border-b border-slate-800">
                      <td colSpan={8} className="p-4">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
                          {/* Training & Eval IDs */}
                          <div>
                            <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                              AWS SageMaker Job Identity
                            </span>
                            <div className="space-y-1">
                              <div className="flex items-center justify-between text-slate-300 bg-slate-950 px-2 py-1 rounded border border-slate-800/80">
                                <span className="text-slate-500 text-[10px]">Train:</span>
                                <span className="font-mono text-[11px] truncate max-w-[170px]">
                                  {r.trainingJobId}
                                </span>
                                <button
                                  onClick={() => copyToClipboard(r.trainingJobId, `train-${r.runId}`)}
                                  className="text-slate-500 hover:text-cyan-300"
                                >
                                  {copiedKey === `train-${r.runId}` ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                                </button>
                              </div>
                              <div className="flex items-center justify-between text-slate-300 bg-slate-950 px-2 py-1 rounded border border-slate-800/80">
                                <span className="text-slate-500 text-[10px]">Eval:</span>
                                <span className="font-mono text-[11px] truncate max-w-[170px]">
                                  {r.evalJobId}
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Cryptographic Manifest Hash */}
                          <div>
                            <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                              Manifest SHA256 Integrity
                            </span>
                            <div className="bg-slate-950 p-2 rounded border border-slate-800/80 flex items-center justify-between">
                              <span className="font-mono text-[11px] text-purple-300 truncate max-w-[200px]">
                                {r.manifestHash}
                              </span>
                              <button
                                onClick={() => copyToClipboard(r.manifestHash, `hash-${r.runId}`)}
                                className="text-slate-500 hover:text-cyan-300"
                              >
                                {copiedKey === `hash-${r.runId}` ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                              </button>
                            </div>
                            <div className="text-[10px] text-slate-500 mt-1 font-sans">
                              Completed: {r.completedAt}
                            </div>
                          </div>

                          {/* S3 Artifact Checkpoint URI */}
                          <div>
                            <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                              S3 Checkpoint Artifact Reference
                            </span>
                            <div className="bg-slate-950 p-2 rounded border border-slate-800/80 flex items-center justify-between">
                              <span className="font-mono text-[11px] text-cyan-300 truncate max-w-[200px]">
                                {r.checkpointArtifact}
                              </span>
                              <button
                                onClick={() => copyToClipboard(r.checkpointArtifact, `s3-${r.runId}`)}
                                className="text-slate-500 hover:text-cyan-300"
                              >
                                {copiedKey === `s3-${r.runId}` ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                              </button>
                            </div>
                            {r.unverifiableReason && (
                              <div className="text-[10px] text-amber-400 mt-1 font-sans">
                                Note: {r.unverifiableReason}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
