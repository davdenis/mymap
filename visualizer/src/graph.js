// MinHeap for priority queue operations
class MinHeap {
  constructor() {
    this.heap = [];
  }
  push(val, priority) {
    this.heap.push({ val, priority });
    this.bubbleUp(this.heap.length - 1);
  }
  pop() {
    if (this.heap.length === 0) return null;
    const top = this.heap[0];
    const bottom = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = bottom;
      this.sinkDown(0);
    }
    return top;
  }
  bubbleUp(index) {
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2);
      if (this.heap[parentIndex].priority <= this.heap[index].priority) break;
      [this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]];
      index = parentIndex;
    }
  }
  sinkDown(index) {
    const length = this.heap.length;
    const element = this.heap[index];
    while (true) {
      let leftChildIndex = 2 * index + 1;
      let rightChildIndex = 2 * index + 2;
      let leftChild, rightChild;
      let swap = null;

      if (leftChildIndex < length) {
        leftChild = this.heap[leftChildIndex];
        if (leftChild.priority < element.priority) {
          swap = leftChildIndex;
        }
      }

      if (rightChildIndex < length) {
        rightChild = this.heap[rightChildIndex];
        if (
          (swap === null && rightChild.priority < element.priority) ||
          (swap !== null && rightChild.priority < leftChild.priority)
        ) {
          swap = rightChildIndex;
        }
      }

      if (swap === null) break;
      this.heap[index] = this.heap[swap];
      this.heap[swap] = element;
      index = swap;
    }
  }
  isEmpty() {
    return this.heap.length === 0;
  }
}

// Distance conversion helper variables (Puerto Madryn latitude ~ -42.77)
const LAT_TO_METERS = 111132.95; // meters per degree latitude
const LON_TO_METERS = 81639.67;  // meters per degree longitude at -42.77 deg latitude

export function toRad(deg) {
  return (deg * Math.PI) / 180;
}

// Heuristic functions returning distances in meters
export const heuristics = {
  euclidean: (n1, n2) => {
    const dx = (n2.x - n1.x) * LON_TO_METERS;
    const dy = (n2.y - n1.y) * LAT_TO_METERS;
    return Math.sqrt(dx * dx + dy * dy);
  },
  manhattan: (n1, n2) => {
    const dx = Math.abs(n2.x - n1.x) * LON_TO_METERS;
    const dy = Math.abs(n2.y - n1.y) * LAT_TO_METERS;
    return dx + dy;
  },
  haversine: (n1, n2) => {
    const R = 6371000; // Earth radius in meters
    const dLat = toRad(n2.y - n1.y);
    const dLon = toRad(n2.x - n1.x);
    const lat1 = toRad(n1.y);
    const lat2 = toRad(n2.y);

    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.sin(dLon / 2) * Math.sin(dLon / 2) * Math.cos(lat1) * Math.cos(lat2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  },
};

// Build adjacency list for fast lookup
export function buildAdjacencyList(nodes, edges) {
  const adj = {};
  for (const nodeId in nodes) {
    adj[nodeId] = [];
  }
  for (const edge of edges) {
    // Make sure source and target exist
    if (adj[edge.source] !== undefined && adj[edge.target] !== undefined) {
      adj[edge.source].push({
        target: edge.target,
        length: edge.length,
        maxspeed: edge.maxspeed,
        weight: edge.weight, // time weight (length / speed)
      });
    }
  }
  return adj;
}

/**
 * Generator function for Dijkstra Pathfinding
 * Yields current state on each iteration for step-by-step visualization.
 */
export function* dijkstraSearch(nodes, adj, startId, endId, useTimeWeight = false) {
  const visited = new Set();
  const distances = {};
  const previous = {};
  const inQueue = new Set(); // to visualize frontier nodes

  for (const id in nodes) {
    distances[id] = Infinity;
    previous[id] = null;
  }

  distances[startId] = 0;
  const pq = new MinHeap();
  pq.push(startId, 0);
  inQueue.add(startId);

  let stepCount = 0;

  while (!pq.isEmpty()) {
    const { val: currentNodeId, priority: currentDist } = pq.pop();
    inQueue.delete(currentNodeId);

    if (visited.has(currentNodeId)) continue;
    visited.add(currentNodeId);

    // Yield current state
    stepCount++;
    yield {
      currentNodeId,
      visited: new Set(visited),
      frontier: new Set(inQueue),
      distances: { ...distances },
      previous: { ...previous },
      stepCount,
      done: currentNodeId === endId,
    };

    if (currentNodeId === endId) return;

    const neighbors = adj[currentNodeId] || [];
    for (const neighbor of neighbors) {
      const neighborId = neighbor.target;
      if (visited.has(neighborId)) continue;

      // cost is either length (distance in meters) or time weight
      const cost = useTimeWeight ? neighbor.weight : neighbor.length;
      const newDist = currentDist + cost;

      if (newDist < distances[neighborId]) {
        distances[neighborId] = newDist;
        previous[neighborId] = currentNodeId;
        pq.push(neighborId, newDist);
        inQueue.add(neighborId);
      }
    }
  }

  yield {
    currentNodeId: null,
    visited,
    frontier: inQueue,
    distances,
    previous,
    stepCount,
    done: false, // target not reachable
  };
}

/**
 * Generator function for A* Pathfinding
 * Yields current state on each iteration for step-by-step visualization.
 */
export function* aStarSearch(nodes, adj, startId, endId, heuristicName = 'haversine', useTimeWeight = false) {
  const visited = new Set();
  const gScore = {}; // cost from start to current node
  const fScore = {}; // gScore + heuristic cost to target
  const previous = {};
  const inQueue = new Set();

  const heuristic = heuristics[heuristicName] || heuristics.haversine;
  const targetNode = nodes[endId];

  // Max speed limit helper for admissible time heuristic
  // Maximum speed limit in Puerto Madryn network is 60 km/h.
  const maxSpeedKmh = 60;

  for (const id in nodes) {
    gScore[id] = Infinity;
    fScore[id] = Infinity;
    previous[id] = null;
  }

  gScore[startId] = 0;
  
  // Calculate initial heuristic
  const startNode = nodes[startId];
  let hCost = heuristic(startNode, targetNode);
  if (useTimeWeight) {
    // Time heuristic = distance / max speed (admissible)
    hCost = hCost / maxSpeedKmh;
  }
  
  fScore[startId] = hCost;

  const pq = new MinHeap();
  pq.push(startId, fScore[startId]);
  inQueue.add(startId);

  let stepCount = 0;

  while (!pq.isEmpty()) {
    const { val: currentNodeId } = pq.pop();
    inQueue.delete(currentNodeId);

    if (visited.has(currentNodeId)) continue;
    visited.add(currentNodeId);

    stepCount++;
    yield {
      currentNodeId,
      visited: new Set(visited),
      frontier: new Set(inQueue),
      distances: { ...gScore }, // gScore is the physical/time cost traveled so far
      previous: { ...previous },
      stepCount,
      done: currentNodeId === endId,
    };

    if (currentNodeId === endId) return;

    const neighbors = adj[currentNodeId] || [];
    for (const neighbor of neighbors) {
      const neighborId = neighbor.target;
      if (visited.has(neighborId)) continue;

      const edgeCost = useTimeWeight ? neighbor.weight : neighbor.length;
      const tentativeGScore = gScore[currentNodeId] + edgeCost;

      if (tentativeGScore < gScore[neighborId]) {
        previous[neighborId] = currentNodeId;
        gScore[neighborId] = tentativeGScore;

        const neighborNode = nodes[neighborId];
        let h = heuristic(neighborNode, targetNode);
        if (useTimeWeight) {
          h = h / maxSpeedKmh;
        }

        fScore[neighborId] = tentativeGScore + h;
        pq.push(neighborId, fScore[neighborId]);
        inQueue.add(neighborId);
      }
    }
  }

  yield {
    currentNodeId: null,
    visited,
    frontier: inQueue,
    distances: gScore,
    previous,
    stepCount,
    done: false,
  };
}

/**
 * Reconstructs the shortest path from the previous map
 */
export function reconstructPath(previous, endId) {
  const path = [];
  let curr = endId;
  while (curr !== null && previous[curr] !== undefined) {
    path.push(curr);
    curr = previous[curr];
  }
  path.reverse();
  return path;
}
