import osmnx as ox
import matplotlib.pyplot as plt
import streamlit as st
import math
import time
import heapq

# ------------------ CONFIG ------------------
st.set_page_config(layout="wide")

# ------------------ STATE ------------------
if "last_fig" not in st.session_state:
    st.session_state.last_fig = None

# ------------------ GRAFO ------------------
bbox = [-65.0897, -42.7959, -64.9932, -42.7494]
polygon = ox.utils_geo.bbox_to_poly(bbox)
graph = ox.graph_from_polygon(polygon, network_type="drive")

# ------------------ PESOS ------------------
for edge in graph.edges:
    maxspeed = 40
    if "maxspeed" in graph.edges[edge]:
        val = graph.edges[edge]["maxspeed"]
        if isinstance(val, list):
            maxspeed = min(int(v) for v in val)
        elif isinstance(val, str):
            maxspeed = int(val)
    graph.edges[edge]["maxspeed"] = maxspeed
    graph.edges[edge]["weight"] = graph.edges[edge]["length"] / maxspeed

# ------------------ ESTILOS ------------------
def style_unvisited_edge(edge):
    graph.edges[edge]["color"] = "#d36206"
    graph.edges[edge]["alpha"] = 0.2
    graph.edges[edge]["linewidth"] = 0.5

def style_visited_edge(edge):
    graph.edges[edge]["color"] = "#d36206"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    graph.edges[edge]["color"] = "#e8a900"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    graph.edges[edge]["color"] = "white"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 2

# ------------------ PLOT ------------------
def plot_graph(start=None, end=None):
    fig, ax = ox.plot_graph(
        graph,
        node_size=[graph.nodes[n]["size"] for n in graph.nodes],
        edge_color=[graph.edges[e]["color"] for e in graph.edges],
        edge_alpha=[graph.edges[e]["alpha"] for e in graph.edges],
        edge_linewidth=[graph.edges[e]["linewidth"] for e in graph.edges],
        node_color="white",
        bgcolor="#18080e",
        figsize=(17, 11),  # 👈 grande y panorámico
        show=False,
        close=False
    )

    if start is not None:
        x, y = graph.nodes[start]["x"], graph.nodes[start]["y"]
        ax.scatter(x, y, c="white", s=120, edgecolors="black", zorder=5)

    if end is not None:
        x, y = graph.nodes[end]["x"], graph.nodes[end]["y"]
        ax.scatter(x, y, c="white", s=120, edgecolors="black", zorder=5)

    return fig

# ------------------ DISTANCIAS ------------------
def get_coords(n):
    return graph.nodes[n]["x"], graph.nodes[n]["y"]

def euclidean(a, b):
    x1, y1 = get_coords(a)
    x2, y2 = get_coords(b)
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

def manhattan(a, b):
    x1, y1 = get_coords(a)
    x2, y2 = get_coords(b)
    return abs(x2-x1) + abs(y2-y1)

def haversine(a, b):
    x1, y1 = get_coords(a)
    x2, y2 = get_coords(b)

    lat1, lon1 = math.radians(x1), math.radians(y1)
    lat2, lon2 = math.radians(x2), math.radians(y2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    R = 6371000
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(h), math.sqrt(1-h))

# ------------------ A* ------------------
def a_star(orig, dest, heuristic, on_step=None):
    visited = set()

    for node in graph.nodes:
        graph.nodes[node]["previous"] = None
        graph.nodes[node]["size"] = 0
        graph.nodes[node]["g"] = float("inf")
        graph.nodes[node]["f"] = float("inf")

    for edge in graph.edges:
        style_unvisited_edge(edge)

    graph.nodes[orig]["g"] = 0
    graph.nodes[orig]["f"] = heuristic(orig, dest)

    pq = [(graph.nodes[orig]["f"], orig)]
    step = 0

    while pq:
        _, node = heapq.heappop(pq)
        visited.add(node)

        if on_step:
            on_step(step, node, visited)

        if node == dest:
            return True

        for edge in graph.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]

            g = graph.nodes[node]["g"] + heuristic(node, neighbor)

            if g < graph.nodes[neighbor]["g"]:
                graph.nodes[neighbor]["previous"] = node
                graph.nodes[neighbor]["g"] = g
                graph.nodes[neighbor]["f"] = g + heuristic(neighbor, dest)

                heapq.heappush(pq, (graph.nodes[neighbor]["f"], neighbor))

                for e in graph.out_edges(neighbor):
                    style_active_edge((e[0], e[1], 0))

        step += 1

    return False

# ------------------ PATH ------------------
def reconstruct_path(dest):
    curr = dest
    while graph.nodes[curr]["previous"] is not None:
        prev = graph.nodes[curr]["previous"]
        style_path_edge((prev, curr, 0))
        curr = prev

# ------------------ SIDEBAR ------------------
st.sidebar.title("Controles")

nodes = list(graph.nodes)

start = st.sidebar.selectbox("Origen", nodes)
end = st.sidebar.selectbox("Destino", nodes)

heuristic_name = st.sidebar.selectbox(
    "Heurística",
    ["euclidean", "manhattan", "haversine"]
)

speed = st.sidebar.slider("Velocidad", 1, 50, 10)

run = st.sidebar.button("Ejecutar")

# ------------------ MAIN VIEW ------------------
st.title("A* Pathfinding Visualizer")

placeholder = st.empty()

# mostrar último mapa
if st.session_state.last_fig is not None:
    placeholder.pyplot(st.session_state.last_fig, use_container_width=True)

heuristics = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "haversine": haversine
}

def callback(step, node, visited):
    fig = plot_graph(start, end)
    st.session_state.last_fig = fig
    placeholder.pyplot(fig, use_container_width=True)
    time.sleep(1 / speed)

if run:
    st.session_state.last_fig = None

    found = a_star(start, end, heuristics[heuristic_name], callback)

    if found:
        reconstruct_path(end)
        fig = plot_graph(start, end)
        st.session_state.last_fig = fig
        placeholder.pyplot(fig, use_container_width=True)
        st.success("Camino encontrado")
    else:
        st.error("No se encontró camino")
