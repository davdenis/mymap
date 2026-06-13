import React, { useState, useEffect, useRef, useMemo } from 'react';
import graphData from './madryn_graph.json';
import { 
  buildAdjacencyList, 
  dijkstraSearch, 
  aStarSearch, 
  reconstructPath 
} from './graph';
import MapVisualizer from './components/MapVisualizer';
import ControlPanel from './components/ControlPanel';
import MetricsPanel from './components/MetricsPanel';
import { Route } from 'lucide-react';

export default function App() {
  // 1. View Mode State
  const [viewMode, setViewMode] = useState('single'); // 'single' | 'compare'

  // 2. Graph State
  const graph = useMemo(() => graphData, []);
  const adj = useMemo(() => buildAdjacencyList(graph.nodes, graph.edges), [graph]);

  // 3. Viewport State shared across both maps
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);

  // 4. Selection State
  const [startNodeId, setStartNodeId] = useState(null);
  const [endNodeId, setEndNodeId] = useState(null);

  // 5. Control Configuration State
  const [algorithm, setAlgorithm] = useState('astar'); // 'astar' | 'dijkstra' (only for single mode)
  const [heuristic, setHeuristic] = useState('haversine'); // 'haversine' | 'euclidean' | 'manhattan'
  const [useTimeWeight, setUseTimeWeight] = useState(false); // false (distance) | true (time)
  const [speed, setSpeed] = useState(25); // delay in ms. 0 = instant

  // 6. Playback / Animation Control
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isFinished, setIsFinished] = useState(false);

  // 7. Dijkstra Live Search State
  const [dijkstraVisited, setDijkstraVisited] = useState(new Set());
  const [dijkstraFrontier, setDijkstraFrontier] = useState(new Set());
  const [dijkstraPrevious, setDijkstraPrevious] = useState({});
  const [dijkstraPath, setDijkstraPath] = useState([]);
  const [dijkstraCurrentNodeId, setDijkstraCurrentNodeId] = useState(null);
  const [dijkstraStepCount, setDijkstraStepCount] = useState(0);
  const [dijkstraPathFound, setDijkstraPathFound] = useState(false);
  const [dijkstraFinished, setDijkstraFinished] = useState(false);

  // 8. A* Live Search State
  const [astarVisited, setAstarVisited] = useState(new Set());
  const [astarFrontier, setAstarFrontier] = useState(new Set());
  const [astarPrevious, setAstarPrevious] = useState({});
  const [astarPath, setAstarPath] = useState([]);
  const [astarCurrentNodeId, setAstarCurrentNodeId] = useState(null);
  const [astarStepCount, setAstarStepCount] = useState(0);
  const [astarPathFound, setAstarPathFound] = useState(false);
  const [astarFinished, setAstarFinished] = useState(false);

  // Generator Refs
  const dijkstraGenRef = useRef(null);
  const astarGenRef = useRef(null);

  // Reset search states when configuration changes
  useEffect(() => {
    handleResetSearch();
  }, [startNodeId, endNodeId, useTimeWeight, viewMode, algorithm]);

  // Clean up search states but keep selection
  const handleResetSearch = () => {
    setIsPlaying(false);
    setIsRunning(false);
    setIsFinished(false);

    // Reset Dijkstra
    setDijkstraVisited(new Set());
    setDijkstraFrontier(new Set());
    setDijkstraPrevious({});
    setDijkstraPath([]);
    setDijkstraCurrentNodeId(null);
    setDijkstraStepCount(0);
    setDijkstraPathFound(false);
    setDijkstraFinished(false);

    // Reset A*
    setAstarVisited(new Set());
    setAstarFrontier(new Set());
    setAstarPrevious({});
    setAstarPath([]);
    setAstarCurrentNodeId(null);
    setAstarStepCount(0);
    setAstarPathFound(false);
    setAstarFinished(false);

    // Reset generators
    dijkstraGenRef.current = null;
    astarGenRef.current = null;
  };

  // Completely clear everything
  const handleClearAll = () => {
    setStartNodeId(null);
    setEndNodeId(null);
    handleResetSearch();
  };

  // Generate random Start and End nodes
  const handleRandomRoute = () => {
    const nodeIds = Object.keys(graph.nodes);
    if (nodeIds.length < 2) return;

    let randStart = nodeIds[Math.floor(Math.random() * nodeIds.length)];
    let randEnd = nodeIds[Math.floor(Math.random() * nodeIds.length)];
    while (randStart === randEnd) {
      randEnd = nodeIds[Math.floor(Math.random() * nodeIds.length)];
    }

    setStartNodeId(randStart);
    setEndNodeId(randEnd);
  };

  // Perform a single step of the parallel or single simulation
  const performSearchStep = () => {
    // 1. Initialize generators
    if (!isRunning) {
      if (viewMode === 'compare') {
        dijkstraGenRef.current = dijkstraSearch(graph.nodes, adj, startNodeId, endNodeId, useTimeWeight);
        astarGenRef.current = aStarSearch(graph.nodes, adj, startNodeId, endNodeId, heuristic, useTimeWeight);
      } else {
        if (algorithm === 'dijkstra') {
          dijkstraGenRef.current = dijkstraSearch(graph.nodes, adj, startNodeId, endNodeId, useTimeWeight);
          setAstarFinished(true);
        } else {
          astarGenRef.current = aStarSearch(graph.nodes, adj, startNodeId, endNodeId, heuristic, useTimeWeight);
          setDijkstraFinished(true);
        }
      }
      setIsRunning(true);
    }

    let isDijkstraDone = dijkstraFinished || (viewMode === 'single' && algorithm === 'astar');
    let isAstarDone = astarFinished || (viewMode === 'single' && algorithm === 'dijkstra');

    // 2. Advance Dijkstra
    if (!isDijkstraDone && dijkstraGenRef.current) {
      const { value, done } = dijkstraGenRef.current.next();
      if (done || !value) {
        setDijkstraFinished(true);
        isDijkstraDone = true;
      } else {
        setDijkstraVisited(value.visited);
        setDijkstraFrontier(value.frontier);
        setDijkstraPrevious(value.previous);
        setDijkstraCurrentNodeId(value.currentNodeId);
        setDijkstraStepCount(value.stepCount);

        if (value.done) {
          const path = reconstructPath(value.previous, endNodeId);
          setDijkstraPath(path);
          setDijkstraPathFound(true);
          setDijkstraFinished(true);
          isDijkstraDone = true;
        }
      }
    }

    // 3. Advance A*
    if (!isAstarDone && astarGenRef.current) {
      const { value, done } = astarGenRef.current.next();
      if (done || !value) {
        setAstarFinished(true);
        isAstarDone = true;
      } else {
        setAstarVisited(value.visited);
        setAstarFrontier(value.frontier);
        setAstarPrevious(value.previous);
        setAstarCurrentNodeId(value.currentNodeId);
        setAstarStepCount(value.stepCount);

        if (value.done) {
          const path = reconstructPath(value.previous, endNodeId);
          setAstarPath(path);
          setAstarPathFound(true);
          setAstarFinished(true);
          isAstarDone = true;
        }
      }
    }

    // 4. Check if both (or active) are finished
    if (isDijkstraDone && isAstarDone) {
      setIsFinished(true);
      setIsPlaying(false);
      return true;
    }

    return false;
  };

  // Play loop effect
  useEffect(() => {
    if (!isPlaying) return;

    // Instant Mode: execute all remaining steps synchronously
    if (speed === 0) {
      let finished = false;
      while (!finished) {
        finished = performSearchStep();
      }
      return;
    }

    // Animated Mode: execute steps with interval delays
    const timer = setInterval(() => {
      const finished = performSearchStep();
      if (finished) {
        clearInterval(timer);
      }
    }, speed);

    return () => clearInterval(timer);
  }, [isPlaying, speed, heuristic, useTimeWeight, dijkstraFinished, astarFinished, isRunning, viewMode, algorithm]);

  return (
    <div className="min-h-screen bg-[#07080c] flex flex-col font-sans selection:bg-[#66fcf1]/30 selection:text-white">
      {/* Top Navbar */}
      <header className="border-b border-white/5 bg-[#090b10]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1440px] mx-auto px-6 py-3.5 flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-[#66fcf1] to-[#45a29e] text-slate-950 shadow-lg shadow-[#66fcf1]/10">
              <Route size={20} className="stroke-[2.5]" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                Pathfinding visualizer
                <span className="text-[9px] uppercase font-mono tracking-widest px-1.5 py-0.5 rounded bg-white/10 text-[#66fcf1] border border-white/5">
                  Puerto Madryn
                </span>
              </h1>
              <p className="text-[10px] text-slate-400 font-medium tracking-wide">
                Interactive street-network routing &bull; Linked coordinates
              </p>
            </div>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center gap-4">
            <div className="flex bg-slate-950 p-1 rounded-xl border border-white/5">
              <button
                onClick={() => setViewMode('single')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  viewMode === 'single'
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                Single Map
              </button>
              <button
                onClick={() => setViewMode('compare')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  viewMode === 'compare'
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                Live Compare
              </button>
            </div>

            <div className="hidden sm:flex items-center gap-4 text-[10px] text-slate-500 border-l border-white/10 pl-4">
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#66fcf1]"></span>
                <span>Visited</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                <span>Frontier</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#10b981]"></span>
                <span>Path</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Grid Content */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Side: Sidebar Controls and Metrics */}
        <div className="flex flex-col gap-6 lg:col-span-1">
          {/* Controls Panel */}
          <ControlPanel
            viewMode={viewMode}
            algorithm={algorithm}
            setAlgorithm={setAlgorithm}
            heuristic={heuristic}
            setHeuristic={setHeuristic}
            useTimeWeight={useTimeWeight}
            setUseTimeWeight={setUseTimeWeight}
            speed={speed}
            setSpeed={setSpeed}
            isPlaying={isPlaying}
            isRunning={isRunning}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onStep={performSearchStep}
            onReset={handleResetSearch}
            onClearSelection={handleClearAll}
            onRandomPoints={handleRandomRoute}
            startNodeId={startNodeId}
            endNodeId={endNodeId}
          />

          {/* Metrics Panel */}
          <MetricsPanel
            viewMode={viewMode}
            algorithm={algorithm}
            // Dijkstra Search
            dijkstraStepCount={dijkstraStepCount}
            dijkstraVisitedCount={dijkstraVisited.size}
            dijkstraFrontierCount={dijkstraFrontier.size}
            dijkstraPathFound={dijkstraPathFound}
            dijkstraPath={dijkstraPath}
            // A* Search
            astarStepCount={astarStepCount}
            astarVisitedCount={astarVisited.size}
            astarFrontierCount={astarFrontier.size}
            astarPathFound={astarPathFound}
            astarPath={astarPath}

            isFinished={isFinished}
            graph={graph}
            adj={adj}
            useTimeWeight={useTimeWeight}
          />
        </div>

        {/* Right Side: Linked Map Visualizer */}
        <div className="lg:col-span-2 h-[550px] lg:h-auto flex flex-col">
          <MapVisualizer
            viewMode={viewMode}
            algorithm={algorithm}
            graph={graph}
            startNodeId={startNodeId}
            endNodeId={endNodeId}
            setStartNodeId={setStartNodeId}
            setEndNodeId={setEndNodeId}
            // Dijkstra Search
            dijkstraVisited={dijkstraVisited}
            dijkstraFrontier={dijkstraFrontier}
            dijkstraPrevious={dijkstraPrevious}
            dijkstraPath={dijkstraPath}
            dijkstraCurrentNodeId={dijkstraCurrentNodeId}
            // A* Search
            astarVisited={astarVisited}
            astarFrontier={astarFrontier}
            astarPrevious={astarPrevious}
            astarPath={astarPath}
            astarCurrentNodeId={astarCurrentNodeId}
            // Synced viewports
            pan={pan}
            setPan={setPan}
            zoom={zoom}
            setZoom={setZoom}
          />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-4 text-center text-xs text-slate-500 bg-[#090b10]/40">
        <p>&copy; 2026 Pathfinding Simulator. Puerto Madryn, Argentina Road Graph Visualization.</p>
      </footer>
    </div>
  );
}
