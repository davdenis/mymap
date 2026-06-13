import React, { useMemo } from 'react';
import { 
  Milestone, 
  Clock, 
  Activity, 
  Layers, 
  CheckCircle2, 
  XCircle,
  HelpCircle,
  Zap,
  Gauge
} from 'lucide-react';

export default function MetricsPanel({
  // Dijkstra Live Stats
  dijkstraStepCount,
  dijkstraVisitedCount,
  dijkstraFrontierCount,
  dijkstraPathFound,
  dijkstraPath,
  
  // A* Live Stats
  astarStepCount,
  astarVisitedCount,
  astarFrontierCount,
  astarPathFound,
  astarPath,

  isFinished,
  graph,
  adj,
  useTimeWeight,
}) {

  // Helper function to calculate path distance, duration, and speed
  const calculateStats = (path) => {
    if (!path || path.length < 2 || !graph) {
      return { distance: 0, duration: 0, avgSpeed: 0 };
    }

    let totalDistance = 0;
    let totalDuration = 0;

    for (let i = 0; i < path.length - 1; i++) {
      const u = path[i];
      const v = path[i + 1];

      const neighbors = adj[u] || [];
      const edge = neighbors.find((n) => n.target === v);

      if (edge) {
        totalDistance += edge.length;
        totalDuration += edge.length / (edge.maxspeed * (1000 / 3600)); // Time in seconds
      } else {
        // Fallback GPS math
        const LON_TO_METERS = 81639.67;
        const LAT_TO_METERS = 111132.95;
        const n1 = graph.nodes[u];
        const n2 = graph.nodes[v];
        if (n1 && n2) {
          const dx = (n2.x - n1.x) * LON_TO_METERS;
          const dy = (n2.y - n1.y) * LAT_TO_METERS;
          const dist = Math.sqrt(dx * dx + dy * dy);
          totalDistance += dist;
          totalDuration += dist / (40 * (1000 / 3600));
        }
      }
    }

    const avgSpeed = totalDistance / 1000 / (totalDuration / 3600);
    return {
      distance: totalDistance,
      duration: totalDuration,
      avgSpeed: isNaN(avgSpeed) ? 0 : avgSpeed,
    };
  };

  const dijkstraStats = useMemo(() => calculateStats(dijkstraPath), [dijkstraPath, graph]);
  const astarStats = useMemo(() => calculateStats(astarPath), [astarPath, graph]);

  const formatDistance = (meters) => {
    if (meters === 0) return '0 m';
    if (meters < 1000) return `${Math.round(meters)} m`;
    return `${(meters / 1000).toFixed(2)} km`;
  };

  const formatDuration = (seconds) => {
    if (seconds === 0) return '0s';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  // Compute live node search reduction percentage
  const reductionPercent = dijkstraVisitedCount > 0
    ? Math.round(((dijkstraVisitedCount - astarVisitedCount) / dijkstraVisitedCount) * 100)
    : 0;

  return (
    <div className="glass-panel p-6 rounded-2xl flex flex-col gap-5 text-slate-200 h-full">
      <div>
        <h3 className="text-lg font-bold text-white tracking-wide flex items-center gap-2">
          <Activity className="text-cyan-400" />
          Live Performance Comparison
        </h3>
        <p className="text-xs text-slate-400 mt-1">Real-time side-by-side search statistics.</p>
      </div>

      {/* Dual Stats Headers */}
      <div className="grid grid-cols-2 gap-4 border-b border-white/5 pb-2 text-center">
        <div className="text-xs font-bold text-slate-400 tracking-wider">DIJKSTRA</div>
        <div className="text-xs font-bold text-cyan-400 tracking-wider">A* SEARCH</div>
      </div>

      {/* Live Iterations / Steps */}
      <div className="flex flex-col gap-3">
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-400 flex items-center gap-1">
            <Layers size={12} /> Steps Count:
          </span>
          <div className="grid grid-cols-2 gap-4 text-center font-mono font-semibold w-1/2">
            <span>{dijkstraStepCount}</span>
            <span className="text-cyan-300">{astarStepCount}</span>
          </div>
        </div>

        {/* Visited Nodes */}
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-400 flex items-center gap-1">
            <CheckCircle2 size={12} className="text-emerald-400" /> Nodes Visited:
          </span>
          <div className="grid grid-cols-2 gap-4 text-center font-mono font-semibold w-1/2">
            <span className="text-slate-200">{dijkstraVisitedCount}</span>
            <span className="text-cyan-300">{astarVisitedCount}</span>
          </div>
        </div>

        {/* Frontier Nodes */}
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-400 flex items-center gap-1">
            <Activity size={12} className="text-amber-400" /> Frontier Nodes:
          </span>
          <div className="grid grid-cols-2 gap-4 text-center font-mono font-semibold w-1/2">
            <span className="text-amber-400">{dijkstraFrontierCount}</span>
            <span className="text-amber-300">{astarFrontierCount}</span>
          </div>
        </div>
      </div>

      {/* Live Efficiency Gauge */}
      {dijkstraVisitedCount > 0 && (
        <div className="p-3 bg-white/5 rounded-xl border border-white/5 flex flex-col gap-2">
          <div className="flex justify-between items-center text-xs">
            <span className="text-slate-400 font-medium flex items-center gap-1">
              <Zap size={12} className="text-amber-400" /> A* Search Space Saved:
            </span>
            <span className="font-mono font-bold text-[#66fcf1]">
              {reductionPercent >= 0 ? `${reductionPercent}%` : '0%'}
            </span>
          </div>
          <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden">
            <div 
              className="bg-gradient-to-r from-cyan-400 to-teal-400 h-full transition-all duration-300"
              style={{ width: `${Math.max(0, Math.min(100, reductionPercent))}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Path Results Comparison (Finished State) */}
      {isFinished && (
        <div className="flex flex-col gap-3 p-4 rounded-xl bg-slate-950/45 border border-white/5 mt-auto">
          <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Final Route Benchmarks
          </h4>

          <div className="grid grid-cols-2 gap-4 text-xs">
            {/* Dijkstra Final stats */}
            <div className="flex flex-col gap-1.5 border-r border-white/5 pr-2">
              <span className="font-bold text-slate-400 text-[10px]">DIJKSTRA ROUTE</span>
              {dijkstraPathFound ? (
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-1 text-slate-300 font-mono">
                    <Milestone size={12} className="text-slate-400" />
                    {formatDistance(dijkstraStats.distance)}
                  </div>
                  <div className="flex items-center gap-1 text-slate-300 font-mono">
                    <Clock size={12} className="text-slate-400" />
                    {formatDuration(dijkstraStats.duration)}
                  </div>
                </div>
              ) : (
                <span className="text-rose-400 italic">No path found</span>
              )}
            </div>

            {/* A* Final stats */}
            <div className="flex flex-col gap-1.5 pl-1">
              <span className="font-bold text-cyan-400 text-[10px]">A* ROUTE</span>
              {astarPathFound ? (
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-1 text-cyan-200 font-mono">
                    <Milestone size={12} className="text-cyan-400" />
                    {formatDistance(astarStats.distance)}
                  </div>
                  <div className="flex items-center gap-1 text-cyan-200 font-mono">
                    <Clock size={12} className="text-cyan-400" />
                    {formatDuration(astarStats.duration)}
                  </div>
                </div>
              ) : (
                <span className="text-rose-400 italic">No path found</span>
              )}
            </div>
          </div>

          {dijkstraPathFound && astarPathFound && (
            <div className="text-[10px] text-slate-500 italic border-t border-white/5 pt-2 mt-1">
              {Math.abs(dijkstraStats.distance - astarStats.distance) < 0.1 ? (
                <span>✓ Both algorithms found the same optimal route of <strong>{formatDistance(astarStats.distance)}</strong>.</span>
              ) : (
                <span>
                  ⚠️ Path distances differ! A* path: <strong>{formatDistance(astarStats.distance)}</strong>. Dijkstra: <strong>{formatDistance(dijkstraStats.distance)}</strong>.
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
