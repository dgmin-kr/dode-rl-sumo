import os
import sys
from pathlib import Path

_PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "sumo_patch.py").is_file()), None)
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from sumo_patch import configure_sumo_patch, get_sumo_binary

configure_sumo_patch(__file__)
import shutil
import heapq
import pickle
import tempfile
import xml.etree.ElementTree as ET

import libsumo
import pandas as pd
import numpy as np

input_interval = 5
detect_interval = 300
total_duration = 30

main_dir = os.path.dirname(os.path.abspath(__file__))
toy_dir = os.path.dirname(main_dir)
utils_dir = os.path.join(toy_dir, "utils")
answer_dir = os.path.abspath(os.path.join(main_dir, "..", "1__Ground_truth"))

true_OD_path = os.path.join(answer_dir, "true_OD.pkl")

origin_list = ["N1", "N4"]
destination_list = ["N2", "N3"]

num_OD = len(origin_list) * len(destination_list)

sumo_binary = get_sumo_binary(__file__)
sumocfg_dir = os.path.join(utils_dir, "run.sumocfg")

trials = 5
custom_routing_vehicle_ids = set()

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
        raise RuntimeError("Could not find the <net-file> tag in the SUMO configuration.")

    net_val = net_tag.get("value")
    if not net_val or len(net_val.strip()) == 0:
        raise RuntimeError("The <net-file> value is empty.")

    if not os.path.isabs(net_val):
        net_val = os.path.join(os.path.dirname(sumocfg_path), net_val)

    if not os.path.exists(net_val):
        raise RuntimeError(f"net-file does not exist: {net_val}")

    return net_val

class TimeDependentShortestPathRouter:
    def __init__(
        self,
        net_file: str,
        sumo_binary_path: str,
        vtype_id="commonType",
        exclude_internal_edges=True,
        speed_floor=1.0,
        fallback_to_maxspeed=True
    ):
        self.net_file = net_file
        self.vtype_id = vtype_id
        self.exclude_internal_edges = exclude_internal_edges
        self.speed_floor = float(speed_floor)
        self.fallback_to_maxspeed = bool(fallback_to_maxspeed)

        self.adj = {}
        self.edge_len = {}
        self.edge_vmax = {}

        self.sumolib = _ensure_sumolib(sumo_binary_path)

    def warmup(self):
        net = self.sumolib.net.readNet(self.net_file)

        self.adj = {}
        self.edge_len = {}
        self.edge_vmax = {}

        for e in net.getEdges():
            eid = e.getID()
            if self.exclude_internal_edges and eid.startswith(":"):
                continue

            u = e.getFromNode().getID()
            v = e.getToNode().getID()

            self.adj.setdefault(u, []).append((v, eid))
            self.edge_len[eid] = float(e.getLength())
            self.edge_vmax[eid] = float(e.getSpeed())

    def _edge_travel_time(self, eid: str) -> float:
        spd = libsumo.edge.getLastStepMeanSpeed(eid)

        if (not np.isfinite(spd)) or spd <= 0.0:
            if self.fallback_to_maxspeed:
                spd = self.edge_vmax.get(eid, self.speed_floor)
            if (not np.isfinite(spd)) or spd <= 0.0:
                spd = self.speed_floor

        spd = max(float(spd), self.speed_floor)

        length = self.edge_len.get(eid, None)
        if length is None:
            raise RuntimeError(f"edge length not found in netfile for edge id: {eid}")

        return length / spd

    def shortest_time_route_edges(self, O: str, D: str):
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

    def add_vehicle_by_OD(
        self,
        veh_id,
        O,
        D,
        depart="now",
        departLane="free",
        departSpeed="max"
    ):
        route_edges = self.shortest_time_route_edges(O, D)
        if not route_edges:
            return False

        route_id = f"r__{veh_id}"
        libsumo.route.add(route_id, route_edges)

        libsumo.vehicle.add(
            vehID=veh_id,
            routeID=route_id,
            typeID=self.vtype_id,
            depart=depart,
            departLane=departLane,
            departSpeed=departSpeed
        )
        return True

def write_empty_routes_file(out_dir: str, vtype_id="commonType"):
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

def cleanup_detector_outputs():
    for path in {
        os.path.join(main_dir, "det_data.xml"),
        os.path.join(utils_dir, "det_data.xml"),
        os.path.join(toy_dir, "det_data.xml"),
    }:
        if os.path.exists(path):
            os.remove(path)

def build_od_pairs(origin_list, destination_list):

    pairs = []
    for O in origin_list:
        for D in destination_list:
            if O != D:
                pairs.append((O, D))
    return pairs

def get_detector_ids():
    det_ids = libsumo.inductionloop.getIDList()
    if len(det_ids) == 0:
        raise RuntimeError(
            "Could not find induction loop detector IDs in TraCI. "
            "If det_data.xml is not e1Detector output, use the TraCI API matching the detector type."
        )
    return list(det_ids)

def ensure_custom_routing_mode(veh_id: str):
    if veh_id not in custom_routing_vehicle_ids:
        libsumo.vehicle.setRoutingMode(veh_id, libsumo.ROUTING_MODE_AGGREGATED_CUSTOM)
        custom_routing_vehicle_ids.add(veh_id)

def reroute_vehicle(veh_id: str):
    ensure_custom_routing_mode(veh_id)
    libsumo.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=False)

def step_simulation(router, od_pairs, od_vec, det_ids, step_idx, veh_counter_start, sim_time_start):
    veh_counter = veh_counter_start
    sim_time = sim_time_start

    for k, demand in enumerate(od_vec):
        if demand <= 0:
            continue
        O, D = od_pairs[k]
        for m in range(int(demand)):
            veh_id = f"veh_{step_idx}_{k}_{m}_{veh_counter}"
            ok = router.add_vehicle_by_OD(
                veh_id, O, D,
                depart="now",
                departLane="free",
                departSpeed="max"
            )
            veh_counter += 1
            if not ok:
                continue
            try:
                ensure_custom_routing_mode(veh_id)
            except Exception:
                pass

    interval_det_data = None

    for sec in range(input_interval):
        libsumo.simulationStep()
        sim_time += 1

        if sec == 0:
            for vid in libsumo.vehicle.getIDList():
                reroute_vehicle(vid)

        for vid in libsumo.simulation.getDepartedIDList():
            reroute_vehicle(vid)

        if (sim_time % detect_interval) == 0:
            interval_det_data = [
                float(libsumo.inductionloop.getLastIntervalVehicleNumber(did))
                for did in det_ids
            ]

    return interval_det_data, veh_counter, sim_time

def run_trial(inputs: np.ndarray, trial_seed: int):
    custom_routing_vehicle_ids.clear()
    num_step = (total_duration * 60) // input_interval

    det_agg_lst = []
    sim_time = 0
    veh_counter = 0
    traci_started = False
    temp_dir = tempfile.mkdtemp(prefix="dode_org_")

    try:
        cleanup_detector_outputs()
        empty_routes = write_empty_routes_file(temp_dir, vtype_id="commonType")
        net_file = get_net_file_from_sumocfg(sumocfg_dir)

        sumo_cmd = [
            sumo_binary,
            "-c", sumocfg_dir,
            "-r", empty_routes,
            "--junction-taz",
            "--save-state.precision=4",
            "--seed", str(trial_seed)
        ]

        libsumo.start(sumo_cmd)
        traci_started = True

        router = TimeDependentShortestPathRouter(
            net_file=net_file,
            sumo_binary_path=sumo_binary,
            vtype_id="commonType",
            exclude_internal_edges=True,
            speed_floor=1.0,
            fallback_to_maxspeed=True
        )

        router.warmup()

        od_pairs = build_od_pairs(origin_list, destination_list)
        det_ids = get_detector_ids()

        for step_idx in range(num_step):
            od_vec = inputs[step_idx]

            interval_det_data, veh_counter, sim_time = step_simulation(
                router=router,
                od_pairs=od_pairs,
                od_vec=od_vec,
                det_ids=det_ids,
                step_idx=step_idx,
                veh_counter_start=veh_counter,
                sim_time_start=sim_time
            )

            if interval_det_data is not None:
                det_agg_lst.append(interval_det_data)
    finally:
        if traci_started:
            libsumo.close()
        cleanup_detector_outputs()
        shutil.rmtree(temp_dir, ignore_errors=True)

    det_agg_df = pd.DataFrame(det_agg_lst, columns=det_ids)
    return det_agg_df

with open(true_OD_path, "rb") as f:
    inputs = pickle.load(f)
inputs = np.asarray(inputs)

num_step = (total_duration * 60) // input_interval
if inputs.shape[0] != num_step:
    raise RuntimeError(
        f"The number of steps in true_OD.pkl does not match the configuration. "
        f"inputs.shape[0]={inputs.shape[0]} vs num_step={num_step}"
    )

for trial in range(trials):
    det_agg_df = run_trial(inputs=inputs, trial_seed=trial)

    csv_path = os.path.join(main_dir, f"answer_{trial}.csv")
    det_agg_df.to_csv(csv_path, index=False, header=False)
    print(f"[trial={trial}] saved: {csv_path} (rows={len(det_agg_df)})")
