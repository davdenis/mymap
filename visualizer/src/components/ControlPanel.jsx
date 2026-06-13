import React from 'react';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  ChevronRight, 
  Compass, 
  Clock, 
  Activity, 
  Trash2, 
  Shuffle,
  Zap
} from 'lucide-react';

export default function ControlPanel({
  heuristic,
  setHeuristic,
  useTimeWeight,
  setUseTimeWeight,
  speed,
  setSpeed,
  isPlaying,
  isRunning,
  onPlay,
  onPause,
  onStep,
  onReset,
  onClearSelection,
  onRandomPoints,
  startNodeId,
  endNodeId,
}) {
  return (
    <div className="glass-panel p-6 rounded-2xl flex flex-col gap-5 text-slate-200">
      <div>
        <h2 className="text-xl font-bold text-white tracking-wide flex items-center gap-2">
          <Compass className="text-[#66fcf1]" />
          Visualizer Controls
        </h2>
        <p className="text-xs text-slate-400 mt-1">Configure and compare Dijkstra and A* side-by-side.</p>
      </div>

      {/* Origin / Destination Status */}
      <div className="flex flex-col gap-2.5 bg-white/5 p-4 rounded-xl border border-white/5">
        <div className="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase tracking-wider">
          <span>Route Points</span>
          <button
            onClick={onRandomPoints}
            className="flex items-center gap-1 px-2 py-1 bg-cyan-950/50 hover:bg-cyan-900/50 text-[#66fcf1] border border-cyan-800/30 rounded-md transition-colors text-[10px] cursor-pointer"
          >
            <Shuffle size={10} />
            Random Pins
          </button>
        </div>

        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-md shadow-emerald-500/50"></div>
          <div className="flex flex-col flex-1">
            <span className="text-[11px] font-semibold text-slate-300">Origen</span>
            <span className="text-[10px] text-slate-500 font-mono overflow-hidden text-ellipsis whitespace-nowrap max-w-[200px]">
              {startNodeId ? `Node ID: ${startNodeId}` : 'Click on map or get random'}
            </span>
          </div>
        </div>

        <div className="h-px bg-white/5"></div>

        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-rose-500 shadow-md shadow-rose-500/50"></div>
          <div className="flex flex-col flex-1">
            <span className="text-[11px] font-semibold text-slate-300">Destino</span>
            <span className="text-[10px] text-slate-500 font-mono overflow-hidden text-ellipsis whitespace-nowrap max-w-[200px]">
              {endNodeId ? `Node ID: ${endNodeId}` : 'Click on map or get random'}
            </span>
          </div>
        </div>
      </div>

      {/* Heuristic Selector (Applies to A* only) */}
      <div className="flex flex-col gap-1.5">
        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
          <Zap size={10} className="text-cyan-400" />
          A* Heuristic Function
        </label>
        <div className="grid grid-cols-3 gap-1 bg-slate-950/45 p-1 rounded-xl border border-white/5">
          {['haversine', 'euclidean', 'manhattan'].map((h) => (
            <button
              key={h}
              onClick={() => !isRunning && setHeuristic(h)}
              disabled={isRunning}
              className={`py-1.5 rounded-lg text-xs capitalize font-medium transition-colors cursor-pointer ${
                heuristic === h
                  ? 'bg-slate-800 text-white border border-white/10'
                  : 'text-slate-500 hover:text-slate-300'
              } disabled:opacity-50`}
            >
              {h}
            </button>
          ))}
        </div>
      </div>

      {/* Cost Target (Length vs Time) */}
      <div className="flex flex-col gap-1.5">
        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
          <Clock size={10} className="text-cyan-400" />
          Optimization Metric (Real Cost)
        </label>
        <div className="grid grid-cols-2 gap-1.5 bg-slate-950/45 p-1 rounded-xl border border-white/5">
          <button
            onClick={() => !isRunning && setUseTimeWeight(false)}
            disabled={isRunning}
            className={`py-1.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1 transition-all cursor-pointer ${
              !useTimeWeight
                ? 'bg-slate-800 text-white border border-white/10'
                : 'text-slate-500 hover:text-white'
            } disabled:opacity-50`}
          >
            <Compass size={12} />
            Shortest Distance
          </button>
          <button
            onClick={() => !isRunning && setUseTimeWeight(true)}
            disabled={isRunning}
            className={`py-1.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1 transition-all cursor-pointer ${
              useTimeWeight
                ? 'bg-slate-800 text-white border border-white/10'
                : 'text-slate-500 hover:text-white'
            } disabled:opacity-50`}
          >
            <Clock size={12} />
            Fastest Time
          </button>
        </div>
      </div>

      {/* Speed Slider */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between items-center text-[10px] font-bold text-slate-400 uppercase tracking-wider">
          <span>Simulation Speed</span>
          <span className="text-slate-300 normal-case font-mono">
            {speed === 0 ? 'Instant' : `${speed}ms delay`}
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="150"
          step="5"
          value={speed}
          onChange={(e) => setSpeed(parseInt(e.target.value))}
          className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#66fcf1]"
        />
      </div>

      {/* Playback Controls */}
      <div className="flex flex-col gap-2.5 mt-1">
        <div className="flex gap-2">
          {isPlaying ? (
            <button
              onClick={onPause}
              disabled={!startNodeId || !endNodeId}
              className="flex-1 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold flex items-center justify-center gap-1.5 transition-all shadow-lg shadow-amber-500/20 active:scale-95 cursor-pointer disabled:opacity-50"
            >
              <Pause size={16} />
              Pause
            </button>
          ) : (
            <button
              onClick={onPlay}
              disabled={!startNodeId || !endNodeId}
              className="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-[#66fcf1] to-[#45a29e] hover:brightness-110 text-slate-950 font-bold flex items-center justify-center gap-1.5 transition-all shadow-lg shadow-[#66fcf1]/20 active:scale-95 cursor-pointer disabled:opacity-30 disabled:pointer-events-none"
            >
              <Play size={16} />
              {isRunning ? 'Resume' : 'Run Live Parallel'}
            </button>
          )}

          <button
            onClick={onStep}
            disabled={!startNodeId || !endNodeId || isPlaying}
            title="Next Step"
            className="px-3 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-white/5 text-slate-200 flex items-center justify-center transition-all disabled:opacity-50 disabled:pointer-events-none cursor-pointer"
          >
            <ChevronRight size={16} />
          </button>
        </div>

        <div className="flex gap-2">
          <button
            onClick={onReset}
            disabled={!isRunning}
            className="flex-1 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-white/5 text-slate-300 text-xs font-semibold flex items-center justify-center gap-1 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            <RotateCcw size={12} />
            Reset Search
          </button>

          <button
            onClick={onClearSelection}
            disabled={!startNodeId && !endNodeId && !isRunning}
            className="flex-1 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-white/5 text-slate-300 text-xs font-semibold flex items-center justify-center gap-1 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            <Trash2 size={12} />
            Clear All
          </button>
        </div>
      </div>
    </div>
  );
}
