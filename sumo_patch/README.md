# SUMO Runtime Patch Notes

This runtime is a patched SUMO 1.27.0 build packaged for simulation workloads
that use libsumo/TraCI from Python.

## Runtime

- Preferred runtime path: `sumo_patch`
- Python ABI used for `_libsumo.pyd`: CPython 3.11
- This packaged runtime is pruned for Python-driven simulation experiments. It
  keeps the patched SUMO binaries, required DLLs, `tools/libsumo`,
  `tools/sumolib`, `tools/traci`, and `data/xsd`; general SUMO helper tools,
  translations, GUI/OSG DLLs, and PROJ share data are not included.

## Behavior Changes Versus Upstream SUMO

- Lane changing is simplified in selected low-risk cases.
  - `MSLaneChanger` skips discretionary LC2013 speed-gain lane-change checks when the vehicle is already on its route-preferred lane, is not an emergency vehicle, has no opposite-stop case, and has a large clear leader gap.
  - For route-preferred, non-emergency, non-opposite-stop vehicles that do not hit the clear-gap fast path, discretionary lane-change checks are evaluated every 2 simulation seconds by default. Set `DODE_SUMO_LC_DECISION_PERIOD=1` to restore the previous every-step check behavior.
  - This can change microscopic lane choices and therefore downstream dynamics, but the guard avoids obvious safety, emergency, and opposite-lane cases.

- E1 induction-loop XML output is disabled by default.
  - SUMO no longer opens/writes the detector XML file named in detector definitions for induction loops.
  - The detector interval reset still runs, so libsumo/TraCI getters and subscriptions such as `getLastStepVehicleNumber` and `getLastIntervalVehicleNumber` remain usable.
  - Set `DODE_SUMO_WRITE_DETECTOR_XML=1` to restore upstream-style induction-loop XML writing.

- libsumo string lookup caches were added.
  - `libsumo::Edge::getEdge` caches edge ID lookups for the current `MSNet`.
  - `libsumo::InductionLoop::getDetector` and meso detector lookup cache detector ID lookups for the current `MSNet`.
  - These caches are cleared during libsumo cleanup and are intended to be dynamics-neutral.

- Custom raw numeric observation APIs were added to libsumo.
  - `edge.setDODERawEdgeIDs`, `edge.getDODERawLastStepCountsAndSpeeds`, and `edge.getDODERawLastStepMeanSpeeds` return fixed-order numeric arrays for the experiment edge observations.
  - `inductionloop.setDODERawDetectorIDs`, `inductionloop.getDODERawLastStepVehicleNumbers`, and `inductionloop.getDODERawLastIntervalVehicleNumbers` do the same for detector observations.
  - Experiment scripts fall back to standard libsumo APIs when these patched methods are unavailable. Set `DODE_USE_RAW_OBS=0` to disable the raw observation path.

- Repeated `vehicle.rerouteTraveltime` calls are throttled by default.
  - The default period is effectively once per vehicle per episode (`9999` simulation seconds), applied across supported networks.
  - This changes route refresh dynamics and can change rewards, especially in small networks where random seed effects and route choices are more sensitive.
  - Set `DODE_SUMO_REROUTE_PERIOD=0` to restore unthrottled reroute behavior. Set a positive value such as `10` or `30` to allow periodic rerouting at that interval in simulation seconds.

- Step/lane/lane-change/vehicle profiling hooks are present behind `DODE_SUMO_STEP_PROFILE_BUILD`.
  - When compiled with that flag and `DODE_SUMO_STEP_PROFILE` is set, SUMO can print timing sections for `MSNet`, `MSLaneChanger`, `MSLane`, and `MSVehicle`.
  - The default runtime does not enable profiling output.

- Several hot-path refactors in `MSLaneChanger`, `MSLCM_LC2013`, `MSNet`, and `MSVehicle` reduce repeated virtual calls, repeated getter chains, temporary copies, and unnecessary work. These are intended to be behavior-preserving unless covered by the lane-change simplification above.
