import os
import sys
from pathlib import Path

_PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "sumo_patch.py").is_file()), None)
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from sumo_patch import configure_sumo_patch

configure_sumo_patch(__file__)
import csv
import heapq
import numbers
import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
import libsumo
from typing import Optional, Tuple

_ROUTER_GRAPH_CACHE = {}
_EDGE_SUBSCRIPTION_VARS = (libsumo.LAST_STEP_VEHICLE_NUMBER, libsumo.LAST_STEP_MEAN_SPEED)
_DETECTOR_SUBSCRIPTION_VARS = (libsumo.LAST_STEP_VEHICLE_NUMBER,)
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    @njit
    def _heap_less(dist_a, item_a, dist_b, item_b, rank):
        if dist_a < dist_b:
            return True
        if dist_a > dist_b:
            return False
        return rank[item_a] < rank[item_b]

    @njit
    def _heap_push(heap_dist, heap_item, heap_size, dist, item, rank):
        i = heap_size
        heap_dist[i] = dist
        heap_item[i] = item
        heap_size += 1
        while i > 0:
            parent = (i - 1) // 2
            if not _heap_less(heap_dist[i], heap_item[i], heap_dist[parent], heap_item[parent], rank):
                break
            tmp_d = heap_dist[parent]
            tmp_i = heap_item[parent]
            heap_dist[parent] = heap_dist[i]
            heap_item[parent] = heap_item[i]
            heap_dist[i] = tmp_d
            heap_item[i] = tmp_i
            i = parent
        return heap_size

    @njit
    def _heap_pop(heap_dist, heap_item, heap_size, rank):
        out_dist = heap_dist[0]
        out_item = heap_item[0]
        heap_size -= 1
        if heap_size > 0:
            heap_dist[0] = heap_dist[heap_size]
            heap_item[0] = heap_item[heap_size]
            i = 0
            while True:
                left = i * 2 + 1
                if left >= heap_size:
                    break
                right = left + 1
                child = left
                if right < heap_size and _heap_less(
                    heap_dist[right], heap_item[right], heap_dist[left], heap_item[left], rank
                ):
                    child = right
                if not _heap_less(heap_dist[child], heap_item[child], heap_dist[i], heap_item[i], rank):
                    break
                tmp_d = heap_dist[i]
                tmp_i = heap_item[i]
                heap_dist[i] = heap_dist[child]
                heap_item[i] = heap_item[child]
                heap_dist[child] = tmp_d
                heap_item[child] = tmp_i
                i = child
        return heap_size, out_dist, out_item

    @njit
    def _dijkstra_node_indices(start_node, dest_node, adj_indptr, adj_to_node, adj_edge_idx, edge_weight, node_rank):
        num_nodes = node_rank.shape[0]
        if start_node < 0 or dest_node < 0 or start_node >= num_nodes or dest_node >= num_nodes:
            return np.empty(0, dtype=np.int64)
        if start_node == dest_node:
            return np.empty(0, dtype=np.int64)

        dist = np.empty(num_nodes, dtype=np.float64)
        dist.fill(np.inf)
        prev_node = np.empty(num_nodes, dtype=np.int64)
        prev_node.fill(-1)
        prev_edge = np.empty(num_nodes, dtype=np.int64)
        prev_edge.fill(-1)
        visited = np.zeros(num_nodes, dtype=np.bool_)

        capacity = adj_to_node.shape[0] + 1
        heap_dist = np.empty(capacity, dtype=np.float64)
        heap_item = np.empty(capacity, dtype=np.int64)
        heap_size = 0

        dist[start_node] = 0.0
        heap_size = _heap_push(heap_dist, heap_item, heap_size, 0.0, start_node, node_rank)
        found = False

        while heap_size > 0:
            heap_size, d_u, u = _heap_pop(heap_dist, heap_item, heap_size, node_rank)
            if visited[u]:
                continue
            visited[u] = True
            if u == dest_node:
                found = True
                break
            for p in range(adj_indptr[u], adj_indptr[u + 1]):
                v = adj_to_node[p]
                edge_idx = adj_edge_idx[p]
                nd = d_u + edge_weight[edge_idx]
                if nd < dist[v]:
                    dist[v] = nd
                    prev_node[v] = u
                    prev_edge[v] = edge_idx
                    heap_size = _heap_push(heap_dist, heap_item, heap_size, nd, v, node_rank)

        if not found:
            return np.empty(0, dtype=np.int64)

        rev = np.empty(num_nodes, dtype=np.int64)
        n = 0
        cur = dest_node
        while cur != start_node:
            edge_idx = prev_edge[cur]
            if edge_idx < 0:
                return np.empty(0, dtype=np.int64)
            rev[n] = edge_idx
            n += 1
            cur = prev_node[cur]
            if cur < 0:
                return np.empty(0, dtype=np.int64)

        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = rev[n - 1 - i]
        return out

    @njit
    def _dijkstra_edge_indices(start_edges, dest_node, edge_to_node, edge_out_indptr, edge_out_indices, edge_weight, edge_rank):
        num_edges = edge_rank.shape[0]
        if dest_node < 0 or start_edges.shape[0] == 0:
            return np.empty(0, dtype=np.int64)

        dist = np.empty(num_edges, dtype=np.float64)
        dist.fill(np.inf)
        prev_edge = np.empty(num_edges, dtype=np.int64)
        prev_edge.fill(-1)
        visited = np.zeros(num_edges, dtype=np.bool_)

        capacity = edge_out_indices.shape[0] + start_edges.shape[0] + 1
        heap_dist = np.empty(capacity, dtype=np.float64)
        heap_item = np.empty(capacity, dtype=np.int64)
        heap_size = 0

        for i in range(start_edges.shape[0]):
            se = start_edges[i]
            if se < 0 or se >= num_edges:
                continue
            w = edge_weight[se]
            if w < dist[se]:
                dist[se] = w
                prev_edge[se] = -1
                heap_size = _heap_push(heap_dist, heap_item, heap_size, w, se, edge_rank)

        best_end = -1
        while heap_size > 0:
            heap_size, d_u, u = _heap_pop(heap_dist, heap_item, heap_size, edge_rank)
            if visited[u]:
                continue
            visited[u] = True
            if edge_to_node[u] == dest_node:
                best_end = u
                break
            for p in range(edge_out_indptr[u], edge_out_indptr[u + 1]):
                v = edge_out_indices[p]
                nd = d_u + edge_weight[v]
                if nd < dist[v]:
                    dist[v] = nd
                    prev_edge[v] = u
                    heap_size = _heap_push(heap_dist, heap_item, heap_size, nd, v, edge_rank)

        if best_end < 0:
            return np.empty(0, dtype=np.int64)

        rev = np.empty(num_edges, dtype=np.int64)
        n = 0
        cur = best_end
        while cur >= 0:
            rev[n] = cur
            n += 1
            cur = prev_edge[cur]

        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = rev[n - 1 - i]
        return out
else:
    def _dijkstra_node_indices(*args, **kwargs):
        return np.empty(0, dtype=np.int64)

    def _dijkstra_edge_indices(*args, **kwargs):
        return np.empty(0, dtype=np.int64)

def _to_py(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, numbers.Number):
        return float(x)
    return x

def _ensure_sumolib(sumo_binary_path: str):
    sumo_home = os.environ.get("SUMO_HOME", None)
    if not sumo_home:
        sumo_home = os.path.abspath(os.path.join(os.path.dirname(sumo_binary_path), ".."))
    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
    import sumolib
    return sumolib

def get_net_file_from_sumocfg(sumocfg_path: str) -> str:
    root = ET.parse(sumocfg_path).getroot()

    inp = root.find("input")
    net_tag = inp.find("net-file") if inp is not None else root.find("net-file")
    if net_tag is None:
        raise RuntimeError("sumocfg could not find the <net-file> tag.")

    net_val = net_tag.get("value")
    if not net_val or len(net_val.strip()) == 0:
        raise RuntimeError("The value of <net-file> is empty.")

    if not os.path.isabs(net_val):
        net_val = os.path.join(os.path.dirname(sumocfg_path), net_val)

    if not os.path.exists(net_val):
        raise RuntimeError(f"net-file does not exist: {net_val}")

    return net_val

def write_empty_routes_file(out_dir: str, vtype_id: str = "commonType") -> str:
    routes = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    ET.SubElement(routes, "vType", {"id": vtype_id, "lcKeepRight": "0.0"})
    empty_path = os.path.join(out_dir, "empty.rou.xml")
    ET.ElementTree(routes).write(empty_path, encoding="utf-8", xml_declaration=True)
    return empty_path

def build_od_pairs(origin_list, destination_list):
    pairs = []
    for O in origin_list:
        for D in destination_list:
            if O != D:
                pairs.append((O, D))
    return pairs

class TimeDependentShortestPathRouter:
    def __init__(
        self,
        net_file: str,
        sumo_binary_path: str,
        vtype_id="commonType",
        exclude_internal_edges=True,
        speed_floor=1.0,
        fallback_to_maxspeed=True,
    ):
        self.net_file = net_file
        self.vtype_id = vtype_id
        self.exclude_internal_edges = exclude_internal_edges
        self.speed_floor = float(speed_floor)
        self.fallback_to_maxspeed = bool(fallback_to_maxspeed)

        self.adj = {}
        self.edge_len = {}
        self.edge_vmax = {}
        self._travel_time_cache = {}
        self._edge_results = {}

        self._edge_ids = []
        self._edge_index = {}
        self._node_index = {}
        self._node_rank = None
        self._edge_len_arr = None
        self._edge_vmax_arr = None
        self._adj_indptr = None
        self._adj_to_node = None
        self._adj_edge_idx = None
        self._edge_weight = None
        self._edge_weight_valid = False
        self._use_numba = _NUMBA_AVAILABLE

        self.sumolib = _ensure_sumolib(sumo_binary_path)

    def warmup(self):
        cache_key = (os.path.abspath(self.net_file), bool(self.exclude_internal_edges), "node_index_v1")
        cached = _ROUTER_GRAPH_CACHE.get(cache_key)
        if cached is not None:
            (
                self.adj,
                self.edge_len,
                self.edge_vmax,
                self._edge_ids,
                self._edge_index,
                self._node_index,
                self._node_rank,
                self._edge_len_arr,
                self._edge_vmax_arr,
                self._adj_indptr,
                self._adj_to_node,
                self._adj_edge_idx,
            ) = cached
            self._edge_weight = np.empty_like(self._edge_len_arr, dtype=np.float64)
            self._edge_weight_valid = False
            return

        net = self.sumolib.net.readNet(self.net_file)

        self.adj = {}
        self.edge_len = {}
        self.edge_vmax = {}
        edge_ids = []
        node_ids = []
        node_seen = set()

        def add_node(node_id):
            if node_id not in node_seen:
                node_seen.add(node_id)
                node_ids.append(node_id)

        for e in net.getEdges():
            eid = e.getID()
            if self.exclude_internal_edges and eid.startswith(":"):
                continue

            u = e.getFromNode().getID()
            v = e.getToNode().getID()
            add_node(u)
            add_node(v)

            self.adj.setdefault(u, []).append((v, eid))
            edge_ids.append(eid)
            self.edge_len[eid] = float(e.getLength())
            self.edge_vmax[eid] = float(e.getSpeed())

        self._edge_ids = edge_ids
        self._edge_index = {eid: i for i, eid in enumerate(edge_ids)}
        self._node_index = {nid: i for i, nid in enumerate(node_ids)}

        self._node_rank = np.empty(len(node_ids), dtype=np.int64)
        for rank, nid in enumerate(sorted(node_ids)):
            self._node_rank[self._node_index[nid]] = rank

        self._edge_len_arr = np.empty(len(edge_ids), dtype=np.float64)
        self._edge_vmax_arr = np.empty(len(edge_ids), dtype=np.float64)
        for eid, idx in self._edge_index.items():
            self._edge_len_arr[idx] = self.edge_len[eid]
            self._edge_vmax_arr[idx] = self.edge_vmax[eid]

        n_nodes = len(node_ids)
        self._adj_indptr = np.empty(n_nodes + 1, dtype=np.int64)
        cursor = 0
        edge_idx_values = []
        to_node_values = []
        for i, node_id in enumerate(node_ids):
            self._adj_indptr[i] = cursor
            for to_node, eid in self.adj.get(node_id, []):
                to_node_values.append(self._node_index[to_node])
                edge_idx_values.append(self._edge_index[eid])
                cursor += 1
        self._adj_indptr[n_nodes] = cursor
        self._adj_to_node = np.asarray(to_node_values, dtype=np.int64)
        self._adj_edge_idx = np.asarray(edge_idx_values, dtype=np.int64)
        self._edge_weight = np.empty_like(self._edge_len_arr, dtype=np.float64)
        self._edge_weight_valid = False

        _ROUTER_GRAPH_CACHE[cache_key] = (
            self.adj,
            self.edge_len,
            self.edge_vmax,
            self._edge_ids,
            self._edge_index,
            self._node_index,
            self._node_rank,
            self._edge_len_arr,
            self._edge_vmax_arr,
            self._adj_indptr,
            self._adj_to_node,
            self._adj_edge_idx,
        )

    def clear_step_cache(self):
        self._travel_time_cache.clear()
        self._edge_weight_valid = False

    def refresh_edge_results(self):
        self._edge_results = libsumo.edge.getAllSubscriptionResults() or {}
        self._edge_weight_valid = False

    def _speed_to_travel_time(self, eid: str, idx: int, spd) -> float:
        if (not np.isfinite(spd)) or spd <= 0.0:
            if self.fallback_to_maxspeed:
                spd = self._edge_vmax_arr[idx] if self._edge_vmax_arr is not None else self.edge_vmax.get(eid, self.speed_floor)
            if (not np.isfinite(spd)) or spd <= 0.0:
                spd = self.speed_floor
        spd = max(float(spd), self.speed_floor)
        length = self._edge_len_arr[idx] if self._edge_len_arr is not None else self.edge_len.get(eid)
        if length is None:
            raise RuntimeError(f"edge length not found in netfile for edge id: {eid}")
        return float(length) / spd

    def _build_travel_time_array(self):
        if self._edge_weight_valid and self._edge_weight is not None:
            return self._edge_weight
        if self._edge_weight is None:
            self._edge_weight = np.empty_like(self._edge_len_arr, dtype=np.float64)
        get_speed = libsumo.edge.getLastStepMeanSpeed
        edge_results = self._edge_results
        for idx, eid in enumerate(self._edge_ids):
            results = edge_results.get(eid) or {}
            spd = results.get(libsumo.LAST_STEP_MEAN_SPEED)
            if spd is None:
                spd = get_speed(eid)
            self._edge_weight[idx] = self._speed_to_travel_time(eid, idx, spd)
        self._edge_weight_valid = True
        return self._edge_weight

    def _edge_travel_time(self, eid: str) -> float:
        cached = self._travel_time_cache.get(eid)
        if cached is not None:
            return cached
        idx = self._edge_index.get(eid)
        results = self._edge_results.get(eid) or {}
        spd = results.get(libsumo.LAST_STEP_MEAN_SPEED)
        if spd is None:
            spd = libsumo.edge.getLastStepMeanSpeed(eid)
        travel_time = self._speed_to_travel_time(eid, idx, spd) if idx is not None else self.edge_len[eid] / max(float(spd), self.speed_floor)
        self._travel_time_cache[eid] = travel_time
        return travel_time

    def _shortest_time_route_edges_python(self, O: str, D: str):
        if O == D:
            return None
        if O not in self.adj:
            return None

        dist = {O: 0.0}
        prev_node = {}
        prev_edge = {}
        pq = [(0.0, O)]
        visited = set()

        while pq:
            d_u, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == D:
                break

            for v, eid in self.adj.get(u, []):
                w = self._edge_travel_time(eid)
                nd = d_u + w
                if (v not in dist) or (nd < dist[v]):
                    dist[v] = nd
                    prev_node[v] = u
                    prev_edge[v] = eid
                    heapq.heappush(pq, (nd, v))

        if D not in dist:
            return None

        edges = []
        cur = D
        while cur != O:
            if cur not in prev_edge:
                return None
            edges.append(prev_edge[cur])
            cur = prev_node[cur]
        edges.reverse()
        return edges if edges else None

    def shortest_time_route_edges(self, O: str, D: str):
        if O == D:
            return None
        start_node = self._node_index.get(O, -1)
        dest_node = self._node_index.get(D, -1)
        if start_node < 0 or dest_node < 0:
            return None
        if self._use_numba and self._adj_indptr is not None:
            try:
                edge_weight = self._build_travel_time_array()
                path_idx = _dijkstra_node_indices(
                    int(start_node),
                    int(dest_node),
                    self._adj_indptr,
                    self._adj_to_node,
                    self._adj_edge_idx,
                    edge_weight,
                    self._node_rank,
                )
                if path_idx.size == 0:
                    return None
                return [self._edge_ids[int(i)] for i in path_idx]
            except Exception:
                self._use_numba = False
        return self._shortest_time_route_edges_python(O, D)

    def add_vehicle_by_OD(
        self,
        veh_id: str,
        O: str,
        D: str,
        depart="now",
        departLane="free",
        departSpeed="max",
    ) -> bool:
        route_edges = self.shortest_time_route_edges(O, D)
        if not route_edges:
            return False

        route_id = f"r__{veh_id}"
        try:
            libsumo.route.add(route_id, route_edges)
            libsumo.vehicle.add(
                vehID=veh_id,
                routeID=route_id,
                typeID=self.vtype_id,
                depart=depart,
                departLane=departLane,
                departSpeed=departSpeed,
            )
            return True
        except Exception:
            return False

class MySumoEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}
    def __init__(
        self,
        rl_dir: str,
        sumo_binary: str,
        origin_list,
        destination_list,
        input_interval: int,
        detector_interval: int,
        num_OD: int,
        num_det: int,
        state_dim: int,
        answer_dir: str,
        total_step: int,
        app: str,
        seed: Optional[int] = None,
        action_change_coef: float = 0.0,
        action_cos_coef: float = 0.0,
        init_od_prior: Optional[list] = None,
    ):
        super().__init__()

        self.rl_dir = rl_dir
        self.sumo_binary = sumo_binary
        self.origin_list = origin_list
        self.destination_list = destination_list
        self.input_interval = int(input_interval)
        self.detector_interval = int(detector_interval)
        self.interval_steps = self.detector_interval // self.input_interval
        self.num_OD = int(num_OD)
        self.num_det = int(num_det)
        self.state_dim = int(state_dim)
        self.answer_dir = answer_dir
        self.total_step = int(total_step)
        self.action_change_coef = float(action_change_coef)
        self.action_cos_coef = float(action_cos_coef)
        self.init_od_prior = None if init_od_prior is None else np.asarray(init_od_prior, dtype=np.float32)
        self._cur_block_od_sum = np.zeros(self.num_OD, dtype=np.float32)
        self._prev_block_od_sum = np.zeros(self.num_OD, dtype=np.float32)
        self._det_block_sum = np.zeros(self.num_det, dtype=np.float32)
        self._empty_answer = np.array([], dtype=np.float32)
        self._prev_block_valid = False
        self.app = str(app).strip().lower()
        if self.app not in {"extension", "baseline"}:
            raise RuntimeError("APP must be 'extension' or 'baseline'.")
        self._answer_data = self._load_answer_data(self.answer_dir)

        self._ep_r_acc = 0.0
        self._ep_r_action = 0.0

        self.action_space = spaces.MultiBinary(self.num_OD)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.current_step = 0
        self.OD_lst = []
        self.sims_buffer = []
        self._state = None
        self._traj = []

        self._traci_started = False
        self.sim_time = 0
        self.veh_counter = 0

        self.router: Optional[TimeDependentShortestPathRouter] = None
        self.od_pairs = build_od_pairs(self.origin_list, self.destination_list)
        self.det_ids = None
        self.edge_ids = None
        self._last_det_data = None
        self._net_state = None
        self._custom_routing_vehicle_ids = set()

        os.makedirs(os.path.join(self.rl_dir, "dump"), exist_ok=True)

    def _start_sumo(self, seed: Optional[int] = None):
        self._stop_sumo()

        sumocfg_path = os.path.join(self.rl_dir, "run.sumocfg")
        if not os.path.exists(sumocfg_path):
            raise FileNotFoundError(f"run.sumocfg not found: {sumocfg_path}")

        empty_routes = write_empty_routes_file(self.rl_dir, vtype_id="commonType")

        sumo_cmd = [
            self.sumo_binary,
            "-c", sumocfg_path,
            "-r", empty_routes,
            "--junction-taz",
            "--no-warnings",
            "--save-state.precision=4",
        ]
        if seed is not None:
            sumo_cmd.extend(["--seed", str(int(seed))])

        libsumo.start(sumo_cmd)
        self._traci_started = True

        self.det_ids = list(libsumo.inductionloop.getIDList())
        if len(self.det_ids) == 0:
            raise RuntimeError(
                "TraCI could not find the induction loop detector ID."
                "You need to verify that detectors.add.xml is loaded (via additional-files in run.sumocfg)."
            )
        self.edge_ids = list(libsumo.edge.getIDList())
        if len(self.edge_ids) == 0:
            raise RuntimeError("TraCI could not find the edge ID. Please check if the network is loaded.")
        self._net_state = np.empty(len(self.edge_ids) * 2, dtype=np.float32)
        for did in self.det_ids:
            libsumo.inductionloop.subscribe(did, _DETECTOR_SUBSCRIPTION_VARS)
        for edge_id in self.edge_ids:
            libsumo.edge.subscribe(edge_id, _EDGE_SUBSCRIPTION_VARS)

        net_file = get_net_file_from_sumocfg(sumocfg_path)
        if self.router is None or self.router.net_file != net_file:
            self.router = TimeDependentShortestPathRouter(
                net_file=net_file,
                sumo_binary_path=self.sumo_binary,
                vtype_id="commonType",
                exclude_internal_edges=True,
                speed_floor=1.0,
                fallback_to_maxspeed=True,
            )
            self.router.warmup()
        else:
            self.router.clear_step_cache()

    def _stop_sumo(self):
        try:
            if libsumo.isLoaded():
                libsumo.close()
        except Exception:
            pass
        self._traci_started = False

    def _cosine_sim(self, x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))

        if nx < eps and ny < eps:
            return 1.0
        if nx < eps or ny < eps:
            return 0.0
        c = float(np.dot(x, y) / (nx * ny))
        return float(np.clip(c, -1.0, 1.0))

    def _angle_dist(self, x: np.ndarray, y: np.ndarray) -> float:
        c = self._cosine_sim(x, y)
        return float(np.arccos(c))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.OD_lst = []
        self.sims_buffer = []
        self._traj = []

        self.sim_time = 0
        self.veh_counter = 0
        self._last_det_data = None

        self._cur_block_od_sum = np.zeros(self.num_OD, dtype=np.float32)
        self._prev_block_od_sum = np.zeros(self.num_OD, dtype=np.float32)
        self._det_block_sum.fill(0.0)
        self._prev_block_valid = False
        self._custom_routing_vehicle_ids.clear()

        self._ep_r_acc = 0.0
        self._ep_r_action = 0.0

        self._start_sumo(seed=seed)

        time_state = np.array([0], dtype=np.float32)
        net_state = self._collect_network_state()
        det_state = np.zeros(self.num_det, dtype=np.float32)
        act_state = np.concatenate((self._cur_block_od_sum, self._prev_block_od_sum)).astype(np.float32)

        if self.app == "extension":
            self._state = np.concatenate((time_state, det_state, net_state, act_state))
            return self._state, {}
        elif self.app == "baseline":
            self._state = np.concatenate((time_state, det_state, net_state))
            return self._state, {}

    def step(self, action: np.ndarray):
        reward, mape = 0.0, 0.0
        block_od_change, r_action = 0.0, 0.0
        last_det = None

        a = np.asarray(action, dtype=np.float32)
        self._cur_block_od_sum += a

        OD_action = action.tolist()
        self.OD_lst.append(OD_action)

        net_state, det_5s = self._advance_one_step(OD_action)
        self.sims_buffer.append(det_5s)
        self._det_block_sum += det_5s

        time_state = np.array([len(self.OD_lst)], dtype=np.float32)
        det_state = self._det_block_sum.copy()

        self.current_step += 1
        terminated = self.current_step >= self.total_step
        truncated = False

        r_acc, mape = self._compute_reward(self.current_step - 1)

        r_action = 0.0
        block_od_change = 0.0
        cos_sim = 0.0
        theta = 0.0

        if (self.current_step % self.interval_steps) == 0:
            if self._prev_block_valid:

                diff_raw = float(np.sum(np.abs(self._cur_block_od_sum - self._prev_block_od_sum)))
                block_od_change = diff_raw

                cos_sim = self._cosine_sim(self._cur_block_od_sum, self._prev_block_od_sum)
                theta = float(np.arccos(cos_sim))
            else:

                block_od_change = 0.0

                if self.init_od_prior is not None:
                    cos_sim = self._cosine_sim(self._cur_block_od_sum, self.init_od_prior)
                    theta = float(np.arccos(cos_sim))
                else:
                    cos_sim = 0.0
                    theta = 0.0

            r_action = -self.action_change_coef * block_od_change - self.action_cos_coef * theta

            if self.app == "extension":
                reward = float(r_acc) + float(r_action)
            elif self.app == "baseline":
                reward = float(r_acc)

            self._prev_block_od_sum = self._cur_block_od_sum.copy()
            self._cur_block_od_sum[:] = 0.0
            self._prev_block_valid = True

            self._ep_r_acc += float(r_acc)
            self._ep_r_action += float(r_action)

        if (self.current_step % self.interval_steps) == 0:
            self.sims_buffer = []
            self._det_block_sum.fill(0.0)
            last_det = self._last_det_data
            self._last_det_data = None

        act_state = np.concatenate((self._cur_block_od_sum, self._prev_block_od_sum)).astype(np.float32)

        if self.app == "extension":
            self._state = np.concatenate((time_state, det_state, net_state, act_state))
        elif self.app == "baseline":
            self._state = np.concatenate((time_state, det_state, net_state))

        self._traj.append({
            "obs": _to_py(self._state),
            "action": _to_py(action),
            "reward": float(reward),
            "det": _to_py(last_det),
            "r_acc": float(r_acc),
            "r_action": float(r_action),
            "cos_sim": float(cos_sim),
            "theta": float(theta),
        })

        info = {
            "mape": float(mape),
            "block_od_change": float(block_od_change),
            "r_action": float(r_action),
            "cos_sim": float(cos_sim),
            "theta": float(theta),
        }

        if terminated or truncated:
            info["ep_r_acc"] = float(self._ep_r_acc)
            info["ep_r_action"] = float(self._ep_r_action)
            info["trajectory"] = self._traj
            self._stop_sumo()

        return self._state, float(reward), terminated, truncated, info

    def _compute_reward(self, t: int):
        if (t + 1) % self.interval_steps != 0:
            return 0.0, 0.0

        if self._last_det_data is None:
            det = self._det_block_sum
        else:
            det = self._last_det_data

        ans = self._read_answer_data((t + 1) // self.interval_steps - 1)
        if ans.size == 0:
            return 0.0, 0.0

        mse = float(np.mean((det - ans) ** 2))
        non_zero = ans != 0
        mape = float(np.mean(np.abs((det[non_zero] - ans[non_zero]) / ans[non_zero])) * 100) if non_zero.any() else 0.0
        r_acc = -mse
        return r_acc, mape

    def _load_answer_data(self, path: str) -> np.ndarray:
        try:
            with open(path, "r", encoding="utf-8") as file:
                rows = [[float(x) for x in row] for row in csv.reader(file)]
        except FileNotFoundError:
            return np.empty((0, 0), dtype=np.float32)
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _read_answer_data(self, i: int) -> np.ndarray:
        if 0 <= i < self._answer_data.shape[0]:
            return self._answer_data[i]
        return self._empty_answer

    def _collect_network_state(self):
        state = self._net_state
        get_count = libsumo.edge.getLastStepVehicleNumber
        get_speed = libsumo.edge.getLastStepMeanSpeed
        edge_results = libsumo.edge.getAllSubscriptionResults() or {}
        for idx, edge_id in enumerate(self.edge_ids):
            results = edge_results.get(edge_id) or {}
            cnt = results.get(libsumo.LAST_STEP_VEHICLE_NUMBER)
            if cnt is None:
                cnt = get_count(edge_id)
            if cnt > 0:
                spd = results.get(libsumo.LAST_STEP_MEAN_SPEED)
                if spd is None:
                    spd = get_speed(edge_id)
                if (not np.isfinite(spd)) or spd <= 0:
                    spd = 13.89
            else:
                spd = 13.89
            out_idx = idx * 2
            state[out_idx] = float(cnt)
            state[out_idx + 1] = float(spd)
        return state

    def _ensure_custom_routing_mode(self, veh_id: str):
        if veh_id not in self._custom_routing_vehicle_ids:
            libsumo.vehicle.setRoutingMode(veh_id, libsumo.ROUTING_MODE_AGGREGATED_CUSTOM)
            self._custom_routing_vehicle_ids.add(veh_id)

    def _reroute_vehicle(self, veh_id: str):
        self._ensure_custom_routing_mode(veh_id)
        libsumo.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=False)

    def _advance_one_step(self, od_vec):

        if self.router is not None:
            self.router.clear_step_cache()
            self.router.refresh_edge_results()

        for k, demand in enumerate(od_vec):
            demand = int(demand)
            if demand <= 0:
                continue
            O, D = self.od_pairs[k]
            route_edges = self.router.shortest_time_route_edges(O, D)
            if not route_edges:
                continue

            route_id = f"r__{self.current_step}_{k}_{self.veh_counter}"
            try:
                libsumo.route.add(route_id, route_edges)
            except Exception:
                continue

            for m in range(demand):
                veh_id = f"veh_{self.current_step}_{k}_{m}_{self.veh_counter}"
                self.veh_counter += 1
                try:
                    libsumo.vehicle.add(
                        vehID=veh_id,
                        routeID=route_id,
                        typeID="commonType",
                        depart="now",
                        departLane="free",
                        departSpeed="max",
                    )
                    self._ensure_custom_routing_mode(veh_id)
                except Exception:
                    continue

        det_for_state = np.zeros(len(self.det_ids), dtype=np.float32)
        get_all_detector_results = libsumo.inductionloop.getAllSubscriptionResults
        get_step_detector = libsumo.inductionloop.getLastStepVehicleNumber
        get_interval_detector = libsumo.inductionloop.getLastIntervalVehicleNumber
        reroute_vehicle = self._reroute_vehicle
        det_ids = self.det_ids

        for sec in range(self.input_interval):
            libsumo.simulationStep()
            self.sim_time += 1

            if sec == 0:
                for vid in libsumo.vehicle.getIDList():
                    reroute_vehicle(vid)
            for vid in libsumo.simulation.getDepartedIDList():
                reroute_vehicle(vid)

            detector_results = get_all_detector_results() or {}
            for j, did in enumerate(det_ids):
                results = detector_results.get(did) or {}
                count = results.get(libsumo.LAST_STEP_VEHICLE_NUMBER)
                if count is None:
                    count = get_step_detector(did)
                det_for_state[j] += float(count)

            if (self.sim_time % self.detector_interval) == 0:
                self._last_det_data = np.fromiter(
                    (float(get_interval_detector(did)) for did in det_ids),
                    dtype=np.float32,
                    count=len(det_ids),
                )

        net_state = self._collect_network_state()
        return net_state, det_for_state

    def render(self):
        print(f"Step: {self.current_step}, State shape: {None if self._state is None else self._state.shape}")

    def close(self):
        self._stop_sumo()
        print("Environment closed.")
