import React, { useState, useEffect, useRef } from 'react';
import { TelemetryEvent, TelemetryEventType } from '../types';
import { 
  Terminal, 
  ShieldCheck, 
  Download, 
  Trash2, 
  Search, 
  Filter, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Info, 
  Cpu, 
  Activity,
  ArrowDown
} from 'lucide-react';

interface TelemetryPanelProps {
  events: TelemetryEvent[];
  onClearEvents: () => void;
  isRunning: boolean;
}

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({
  events,
  onClearEvents,
  isRunning,
}) => {
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [autoScroll, setAutoScroll] = useState<boolean>(true);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [events, autoScroll]);

  const filteredEvents = events.filter((e) => {
    if (filterSeverity !== 'all' && e.severity !== filterSeverity) return false;
    if (searchQuery.trim() !== '') {
      const q = searchQuery.toLowerCase();
      const matchSummary = e.summary.toLowerCase().includes(q);
      const matchStage = e.stageId.toLowerCase().includes(q);
      const matchMeta = JSON.stringify(e.metadata).toLowerCase().includes(q);
      return matchSummary || matchStage || matchMeta;
    }
    return true;
  });

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'success':
        return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />;
      case 'warning':
        return <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />;
      case 'error':
        return <XCircle className="w-3.5 h-3.5 text-red-400 shrink-0 mt-0.5" />;
      default:
        return <Info className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />;
    }
  };

  const handleExportJson = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(events, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `telemetry-metadata-${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div id="live-telemetry-panel" className="rounded-2xl bg-[#080d1a] border border-slate-800/80 p-5 shadow-xl flex flex-col h-full">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
            <Terminal className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-display text-sm font-bold tracking-wider text-slate-100 uppercase">
                Telemetry & Observability Console
              </h3>
              {isRunning && (
                <span className="flex items-center gap-1 font-mono text-[10px] text-cyan-400 px-1.5 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/30">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                  LIVE
                </span>
              )}
            </div>
            <p className="text-[11px] text-slate-400 font-sans">
              Real-time audit log stream • Zero prompt leakage enforced
            </p>
          </div>
        </div>

        {/* Security & Action controls */}
        <div className="flex items-center gap-2">
          {/* Privacy Seal */}
          <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900 border border-emerald-500/30 text-[10px] font-mono text-emerald-300">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>METADATA ONLY (ZERO LEAK)</span>
          </div>

          <button
            onClick={handleExportJson}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 hover:text-cyan-300 transition-colors"
            title="Export Telemetry JSON"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onClearEvents}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-400 hover:text-red-400 transition-colors"
            title="Clear Stream"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 my-3">
        <div className="relative flex-1 min-w-[160px]">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search metadata, jobs, hashes..."
            className="w-full pl-8 pr-3 py-1.5 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500/50"
          />
        </div>

        <div className="flex items-center gap-1 font-mono text-[11px]">
          {['all', 'info', 'success', 'warning', 'error'].map((sev) => (
            <button
              key={sev}
              onClick={() => setFilterSeverity(sev)}
              className={`px-2 py-1 rounded capitalize transition-colors ${
                filterSeverity === sev
                  ? 'bg-slate-800 text-cyan-300 border border-cyan-500/30'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {sev}
            </button>
          ))}
        </div>
      </div>

      {/* Log Feed Container */}
      <div className="flex-1 min-h-[300px] max-h-[380px] overflow-y-auto rounded-xl bg-slate-950/90 border border-slate-900 p-3 font-mono text-xs space-y-2 select-text">
        {filteredEvents.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-600 py-12">
            <Activity className="w-6 h-6 mb-2 opacity-50" />
            <span>No telemetry events recorded for current filter criteria.</span>
          </div>
        ) : (
          filteredEvents.map((evt) => (
            <div
              key={evt.id}
              className={`p-2 rounded-lg border transition-all ${
                evt.severity === 'error'
                  ? 'bg-red-950/30 border-red-500/40 text-red-200'
                  : evt.severity === 'warning'
                  ? 'bg-amber-950/30 border-amber-500/40 text-amber-200'
                  : evt.severity === 'success'
                  ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-200'
                  : 'bg-slate-900/40 border-slate-800/80 text-slate-300'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-start gap-2">
                  {getSeverityIcon(evt.severity)}
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[10px] text-slate-500">[{evt.timestamp}]</span>
                      <span className="text-[10px] font-bold tracking-wide uppercase text-cyan-400">
                        {evt.type.replace(/_/g, ' ')}
                      </span>
                      {evt.agentId && (
                        <span className="text-[9px] px-1 py-0.2 rounded bg-slate-800 text-slate-400">
                          {evt.agentId.toUpperCase()}
                        </span>
                      )}
                    </div>
                    <p className="mt-0.5 text-xs text-slate-200 font-sans leading-relaxed">
                      {evt.summary}
                    </p>
                  </div>
                </div>

                <span className="text-[9px] text-slate-600 uppercase shrink-0 font-mono">
                  {evt.stageId}
                </span>
              </div>

              {/* Structured Metadata block if present */}
              {Object.keys(evt.metadata).length > 0 && (
                <div className="mt-2 pt-1.5 border-t border-slate-800/60 flex flex-wrap gap-2 text-[10px] font-mono text-slate-400">
                  {evt.metadata.jobArn && (
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800 text-cyan-300 truncate max-w-[280px]">
                      ARN: {evt.metadata.jobArn}
                    </span>
                  )}
                  {evt.metadata.instanceType && (
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800 text-emerald-300">
                      GPU: {evt.metadata.instanceType}
                    </span>
                  )}
                  {evt.metadata.loss !== undefined && (
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800 text-yellow-300">
                      loss: {evt.metadata.loss.toFixed(4)}
                    </span>
                  )}
                  {evt.metadata.step && (
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800 text-slate-300">
                      step: {evt.metadata.step}
                    </span>
                  )}
                  {evt.metadata.astPassRate !== undefined && (
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800 text-sky-300">
                      AST: {evt.metadata.astPassRate.toFixed(1)}%
                    </span>
                  )}
                  {evt.metadata.improvement !== undefined && (
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800 text-emerald-400 font-bold">
                      delta: {evt.metadata.improvement > 0 ? '+' : ''}
                      {evt.metadata.improvement.toFixed(1)}%
                    </span>
                  )}
                  {evt.metadata.manifestHash && (
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800 text-purple-300 truncate max-w-[180px]">
                      hash: {evt.metadata.manifestHash}
                    </span>
                  )}
                  {evt.metadata.blockedReason && (
                    <span className="bg-amber-950/60 px-1.5 py-0.5 rounded border border-amber-500/40 text-amber-300">
                      hold: {evt.metadata.blockedReason}
                    </span>
                  )}
                </div>
              )}
            </div>
          ))
        )}
        <div ref={logEndRef} />
      </div>

      {/* Footer bar with auto-scroll and stream counter */}
      <div className="mt-3 flex items-center justify-between text-[11px] font-mono text-slate-500">
        <label className="flex items-center gap-1.5 cursor-pointer hover:text-slate-300">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="rounded border-slate-700 bg-slate-900 text-cyan-500 focus:ring-0"
          />
          <span>Auto-scroll to latest event</span>
        </label>
        <span>
          Showing {filteredEvents.length} of {events.length} telemetry entries
        </span>
      </div>
    </div>
  );
};
