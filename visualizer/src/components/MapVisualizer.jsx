import React, { useRef, useEffect, useState, useMemo } from 'react';

export default function MapVisualizer({
  viewMode, // 'single' | 'compare'
  algorithm, // 'astar' | 'dijkstra' (only relevant in single mode)
  graph,
  startNodeId,
  endNodeId,
  setStartNodeId,
  setEndNodeId,
  // Dijkstra Search States
  dijkstraVisited,
  dijkstraFrontier,
  dijkstraPrevious,
  dijkstraPath,
  dijkstraCurrentNodeId,
  // A* Search States
  astarVisited,
  astarFrontier,
  astarPrevious,
  astarPath,
  astarCurrentNodeId,
  // Viewport states shared with App.jsx
  pan,
  setPan,
  zoom,
  setZoom,
}) {
  const containerRef = useRef(null);
  const canvasRefDijkstra = useRef(null);
  const canvasRefAStar = useRef(null);

  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredNodeId, setHoveredNodeId] = useState(null);

  // Determine layouts
  const isSideBySide = viewMode === 'compare' && dimensions.width >= 768;
  const canvasWidth = isSideBySide ? Math.floor(dimensions.width / 2 - 4) : dimensions.width;
  const canvasHeight = viewMode === 'compare' && !isSideBySide 
    ? Math.floor(dimensions.height / 2 - 4) 
    : dimensions.height;

  // Compute graph bounds
  const bounds = useMemo(() => {
    if (!graph || !graph.nodes) return null;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (const id in graph.nodes) {
      const node = graph.nodes[id];
      if (node.x < minX) minX = node.x;
      if (node.x > maxX) maxX = node.x;
      if (node.y < minY) minY = node.y;
      if (node.y > maxY) maxY = node.y;
    }

    const LAT_TO_METERS = 111132.95;
    const LON_TO_METERS = 81639.67;

    const spanX = (maxX - minX) * LON_TO_METERS;
    const spanY = (maxY - minY) * LAT_TO_METERS;
    const aspectRatio = spanX / spanY;

    return { minX, maxX, minY, maxY, aspectRatio };
  }, [graph]);

  // Adjust container dimensions on mount/resize
  useEffect(() => {
    const handleResize = () => {
      const container = containerRef.current;
      if (container) {
        setDimensions({
          width: container.clientWidth,
          height: container.clientHeight || 550,
        });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Coordinate projections
  const project = (lon, lat) => {
    const lonScaleFactor = 81639.67 / 111132.95;
    const screenX = lon * lonScaleFactor * zoom + pan.x;
    const screenY = -lat * zoom + pan.y;
    return { x: screenX, y: screenY };
  };

  const unproject = (screenX, screenY) => {
    const lonScaleFactor = 81639.67 / 111132.95;
    const lat = -(screenY - pan.y) / zoom;
    const lon = (screenX - pan.x) / (lonScaleFactor * zoom);
    return { x: lon, y: lat };
  };

  // Center viewport
  const resetViewport = () => {
    if (!bounds || dimensions.width === 0) return;

    const padding = 20;
    const mapWidth = canvasWidth - padding * 2;
    const mapHeight = canvasHeight - padding * 2;

    let scaleX = mapWidth / (bounds.maxX - bounds.minX);
    let scaleY = mapHeight / (bounds.maxY - bounds.minY);
    const lonScaleFactor = 81639.67 / 111132.95;

    const initialZoom = Math.min(scaleX / lonScaleFactor, scaleY);
    setZoom(initialZoom * 0.95);

    const centerLon = (bounds.minX + bounds.maxX) / 2;
    const centerLat = (bounds.minY + bounds.maxY) / 2;

    setPan({
      x: canvasWidth / 2 - centerLon * lonScaleFactor * (initialZoom * 0.95),
      y: canvasHeight / 2 + centerLat * (initialZoom * 0.95),
    });
  };

  // Center on load
  useEffect(() => {
    resetViewport();
  }, [bounds, dimensions.width, viewMode]);

  // Generic draw map function
  const drawMap = (canvas, visited, frontier, previous, path, currentId, titleText) => {
    if (!canvas || !graph || !bounds) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // 1. Draw base road network (edges)
    ctx.lineWidth = Math.max(0.4, zoom * 0.000004);
    ctx.strokeStyle = 'rgba(211, 98, 6, 0.15)'; // Dark orange roads
    ctx.beginPath();
    for (const edge of graph.edges) {
      const src = graph.nodes[edge.source];
      const tgt = graph.nodes[edge.target];
      if (!src || !tgt) continue;

      const pSrc = project(src.x, src.y);
      const pTgt = project(tgt.x, tgt.y);

      ctx.moveTo(pSrc.x, pSrc.y);
      ctx.lineTo(pTgt.x, pTgt.y);
    }
    ctx.stroke();

    // 2. Draw active frontier edges
    if (frontier.size > 0) {
      ctx.lineWidth = Math.max(0.8, zoom * 0.000008);
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.35)'; // Amber
      ctx.beginPath();
      for (const edge of graph.edges) {
        if (frontier.has(edge.target) && previous[edge.target] === edge.source) {
          const src = graph.nodes[edge.source];
          const tgt = graph.nodes[edge.target];
          if (!src || !tgt) continue;
          const pSrc = project(src.x, src.y);
          const pTgt = project(tgt.x, tgt.y);
          ctx.moveTo(pSrc.x, pSrc.y);
          ctx.lineTo(pTgt.x, pTgt.y);
        }
      }
      ctx.stroke();
    }

    // 3. Draw visited edges
    if (visited.size > 0) {
      ctx.lineWidth = Math.max(1.0, zoom * 0.00001);
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.7)'; // Cyan
      ctx.beginPath();
      for (const edge of graph.edges) {
        if (visited.has(edge.target) && previous[edge.target] === edge.source) {
          const src = graph.nodes[edge.source];
          const tgt = graph.nodes[edge.target];
          if (!src || !tgt) continue;
          const pSrc = project(src.x, src.y);
          const pTgt = project(tgt.x, tgt.y);
          ctx.moveTo(pSrc.x, pSrc.y);
          ctx.lineTo(pTgt.x, pTgt.y);
        }
      }
      ctx.stroke();
    }

    // 4. Draw final route path
    if (path && path.length > 1) {
      ctx.lineWidth = Math.max(2.5, zoom * 0.000025);
      ctx.strokeStyle = '#10b981'; // Emerald Green
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();

      const startNode = graph.nodes[path[0]];
      const startPt = project(startNode.x, startNode.y);
      ctx.moveTo(startPt.x, startPt.y);

      for (let i = 1; i < path.length; i++) {
        const node = graph.nodes[path[i]];
        if (!node) continue;
        const pt = project(node.x, node.y);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
    }

    // 5. Draw pins
    if (startNodeId && graph.nodes[startNodeId]) {
      const node = graph.nodes[startNodeId];
      const pt = project(node.x, node.y);
      ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 14, 0, 2*Math.PI); ctx.fill();
      ctx.fillStyle = '#10b981'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 6, 0, 2*Math.PI); ctx.fill(); ctx.stroke();
    }

    if (endNodeId && graph.nodes[endNodeId]) {
      const node = graph.nodes[endNodeId];
      const pt = project(node.x, node.y);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.2)';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 14, 0, 2*Math.PI); ctx.fill();
      ctx.fillStyle = '#f43f5e'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 6, 0, 2*Math.PI); ctx.fill(); ctx.stroke();
    }

    if (currentId && graph.nodes[currentId]) {
      const node = graph.nodes[currentId];
      const pt = project(node.x, node.y);
      ctx.fillStyle = 'rgba(234, 179, 8, 0.4)';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 10, 0, 2*Math.PI); ctx.fill();
      ctx.fillStyle = '#eab308';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 3.5, 0, 2*Math.PI); ctx.fill();
    }

    if (hoveredNodeId && graph.nodes[hoveredNodeId] && hoveredNodeId !== startNodeId && hoveredNodeId !== endNodeId) {
      const node = graph.nodes[hoveredNodeId];
      const pt = project(node.x, node.y);
      ctx.fillStyle = 'rgba(102, 252, 241, 0.3)';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 9, 0, 2*Math.PI); ctx.fill();
      ctx.strokeStyle = '#66fcf1'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 3.5, 0, 2*Math.PI); ctx.stroke();
    }

    // 6. Draw Map Label Overlay
    ctx.fillStyle = 'rgba(9, 11, 16, 0.85)';
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(12, 12, 180, 32, 6);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.fillText(titleText, 102, 32);
  };

  // Draw loop triggers
  useEffect(() => {
    if (viewMode === 'compare' || algorithm === 'dijkstra') {
      drawMap(
        canvasRefDijkstra.current,
        dijkstraVisited,
        dijkstraFrontier,
        dijkstraPrevious,
        dijkstraPath,
        dijkstraCurrentNodeId,
        viewMode === 'compare' ? 'DIJKSTRA (Blind Search)' : 'DIJKSTRA PATHFINDING'
      );
    }
  }, [graph, bounds, dimensions, pan, zoom, startNodeId, endNodeId, dijkstraVisited, dijkstraFrontier, dijkstraPrevious, dijkstraPath, dijkstraCurrentNodeId, hoveredNodeId, viewMode, algorithm]);

  useEffect(() => {
    if (viewMode === 'compare' || algorithm === 'astar') {
      drawMap(
        canvasRefAStar.current,
        astarVisited,
        astarFrontier,
        astarPrevious,
        astarPath,
        astarCurrentNodeId,
        viewMode === 'compare' ? 'A* SEARCH (Heuristics)' : 'A* SEARCH PATHFINDING'
      );
    }
  }, [graph, bounds, dimensions, pan, zoom, startNodeId, endNodeId, astarVisited, astarFrontier, astarPrevious, astarPath, astarCurrentNodeId, hoveredNodeId, viewMode, algorithm]);

  // Viewport Drag/Wheel triggers
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e, targetCanvasRef) => {
    const canvas = targetCanvasRef.current;
    if (!canvas || !graph) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    } else {
      const geoCoords = unproject(mouseX, mouseY);
      let closestNodeId = null;
      let minDistance = 0.002;

      for (const id in graph.nodes) {
        const node = graph.nodes[id];
        const dist = Math.sqrt((node.x - geoCoords.x) ** 2 + (node.y - geoCoords.y) ** 2);
        if (dist < minDistance) {
          minDistance = dist;
          closestNodeId = id;
        }
      }
      setHoveredNodeId(closestNodeId);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e, targetCanvasRef) => {
    e.preventDefault();
    const canvas = targetCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = 1.15;
    const nextZoom = e.deltaY < 0 ? zoom * zoomFactor : zoom / zoomFactor;

    if (nextZoom < 1000 || nextZoom > 1500000) return;

    setPan({
      x: mouseX - (mouseX - pan.x) * (nextZoom / zoom),
      y: mouseY - (mouseY - pan.y) * (nextZoom / zoom),
    });
    setZoom(nextZoom);
  };

  const handleClick = (e, targetCanvasRef) => {
    const canvas = targetCanvasRef.current;
    if (!canvas || !graph || isDragging) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const geoCoords = unproject(mouseX, mouseY);
    let closestNodeId = null;
    let minDistance = 0.002;

    for (const id in graph.nodes) {
      const node = graph.nodes[id];
      const dist = Math.sqrt((node.x - geoCoords.x) ** 2 + (node.y - geoCoords.y) ** 2);
      if (dist < minDistance) {
        minDistance = dist;
        closestNodeId = id;
      }
    }

    if (closestNodeId) {
      if (!startNodeId) {
        setStartNodeId(closestNodeId);
      } else if (!endNodeId && closestNodeId !== startNodeId) {
        setEndNodeId(closestNodeId);
      } else {
        if (closestNodeId === startNodeId) {
          setStartNodeId(null);
        } else if (closestNodeId === endNodeId) {
          setEndNodeId(null);
        } else {
          setStartNodeId(closestNodeId);
          setEndNodeId(null);
        }
      }
    }
  };

  return (
    <div 
      ref={containerRef}
      className="relative w-full h-full min-h-[550px] bg-[#0c0d12] rounded-2xl overflow-hidden border border-white/5 shadow-2xl flex flex-col md:flex-row gap-1 p-1"
    >
      {/* Help Overlay */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2 pointer-events-none">
        <div className="px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur-md border border-white/5 text-[10px] text-slate-400">
          {viewMode === 'compare' 
            ? '🖱️ Pan & zoom synced across both viewports' 
            : '🖱️ Drag to pan &bull; Scroll to zoom'}
        </div>
        <div className="px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur-md border border-white/5 text-[10px] text-slate-400">
          📍 Click on the map to set Origen / Destino
        </div>
      </div>

      <div className="absolute top-4 right-4 z-10">
        <button
          onClick={resetViewport}
          className="px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 border border-white/10 text-[10px] text-slate-200 transition-colors font-medium cursor-pointer"
        >
          Center Viewport
        </button>
      </div>

      {/* Conditionally Render Single Map vs. Dual Maps */}
      {viewMode === 'single' ? (
        <div className="flex-1 h-full relative overflow-hidden rounded-xl border border-white/5">
          <canvas
            ref={algorithm === 'dijkstra' ? canvasRefDijkstra : canvasRefAStar}
            width={canvasWidth}
            height={canvasHeight}
            onMouseDown={handleMouseDown}
            onMouseMove={(e) => handleMouseMove(e, algorithm === 'dijkstra' ? canvasRefDijkstra : canvasRefAStar)}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={(e) => handleClick(e, algorithm === 'dijkstra' ? canvasRefDijkstra : canvasRefAStar)}
            onWheel={(e) => handleWheel(e, algorithm === 'dijkstra' ? canvasRefDijkstra : canvasRefAStar)}
            className="w-full h-full cursor-grab active:cursor-grabbing block"
          />
        </div>
      ) : (
        <>
          {/* Dijkstra Map */}
          <div className="flex-1 h-full relative overflow-hidden rounded-xl border border-white/5">
            <canvas
              ref={canvasRefDijkstra}
              width={canvasWidth}
              height={canvasHeight}
              onMouseDown={handleMouseDown}
              onMouseMove={(e) => handleMouseMove(e, canvasRefDijkstra)}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onClick={(e) => handleClick(e, canvasRefDijkstra)}
              onWheel={(e) => handleWheel(e, canvasRefDijkstra)}
              className="w-full h-full cursor-grab active:cursor-grabbing block"
            />
          </div>

          <div className="hidden md:block w-px bg-white/5 self-stretch"></div>

          {/* A* Map */}
          <div className="flex-1 h-full relative overflow-hidden rounded-xl border border-white/5">
            <canvas
              ref={canvasRefAStar}
              width={canvasWidth}
              height={canvasHeight}
              onMouseDown={handleMouseDown}
              onMouseMove={(e) => handleMouseMove(e, canvasRefAStar)}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onClick={(e) => handleClick(e, canvasRefAStar)}
              onWheel={(e) => handleWheel(e, canvasRefAStar)}
              className="w-full h-full cursor-grab active:cursor-grabbing block"
            />
          </div>
        </>
      )}
    </div>
  );
}
