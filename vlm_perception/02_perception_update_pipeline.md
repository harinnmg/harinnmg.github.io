# 2. Perception-Update Pipeline

The environment perception (per-POI `description` fields in `pois.json`) is updated by
the VLM through **two paths**. Both run as background threads in
`mission_planner_web.py` so they never block navigation or planning.

A global toggle gates both paths: `VLA_CONTEXT_UPDATE` (env var, default `true`,
switchable at runtime via `POST /api/config` with `{"vla_context_update": bool}`).

---

## Path A — Arrival-triggered single-image update

Runs after the robot completes its actions at **any** POI that did *not* include an
INSPECT action.

**Sequence** (executor: `execute_mission()`; planner: `api_event_arrived()` →
`_bg_update_poi_context()`):

1. Nav2 reaches the POI; the executor runs the stop's actions (WAIT / ROTATE / LOAD / UNLOAD).
2. The executor POSTs `{"poi_id", "name"}` to the planner at
   `POST /api/event/arrived` (`PLANNER_ARRIVAL_EVENT_URL`).
3. The planner validates the POI and spawns a **daemon background thread**; the HTTP
   call returns `202` immediately.
4. The thread fetches the current gripper-camera frame from the executor's
   `GET /camera/snapshot` (JPEG of the latest `/gripper_rgb` frame).
5. The JPEG is base64-encoded and sent to Ollama (`gemma4:12b`) with the prompt:

   > *"At location '\<POI name\>', list the nearby visible objects, landmarks, or
   > obstacles in one concise sentence (max 20 words)."*

6. The VLM's answer becomes the POI's new `description`:
   `POIS_DATA[poi_id]["description"] = description` followed by `save_pois()`
   (persisted to `config/pois.json`).
7. Side effects: a log entry is posted back to the executor
   (`VLA: '<name>' surroundings → <description>`) and a **VLM event** is appended to
   the event store for the dashboard (`GET /api/VLM_events`).

Failure at any step (camera unreachable, Ollama error, empty answer) aborts the update
silently — the previous description is kept.

---

## Path B — Manipulator-driven multi-view inspection

Runs for stops planned with an **INSPECT (action id 7)** action — the richer perception
path added on the `feature/manipulator-inspection` branch. Instead of one opportunistic
frame, the Panda arm deliberately sweeps the camera through **6 poses** and the VLM
fuses all views into one description.

**Sequence** (executor: `_execute_action()` id 7; arm: `panda_execute.py`; planner:
`api_event_inspect_capture()` → `_bg_update_poi_context_multi()`):

1. On INSPECT, the executor launches `panda_execute.py <poi_id> <poi_name>` as a
   subprocess.
2. The script moves the arm through 6 predefined joint-space poses (Pose 1 = home,
   Poses 2–6 = viewpoints), each over ~9.5 s.
3. **After settling at each pose**, it POSTs to the planner:
   `POST /api/event/inspect_capture` with
   `{"poi_id", "name", "pose_index", "total_poses": 6}`.
4. On each capture event, the planner grabs a snapshot from
   `GET /camera/snapshot` and appends the base64 image to an in-memory session buffer
   `INSPECT_SESSIONS[poi_id]`.
5. When `pose_index >= total_poses`, the session is popped and a background thread
   sends **all 6 images in a single Ollama chat message** with the prompt:

   > *"At location '\<POI name\>', analyze these 6 different views from the
   > manipulator. List the nearby visible objects, landmarks, or obstacles in one
   > concise sentence (max 30 words)."*

6. As in Path A, the fused answer overwrites the POI's `description`, is persisted via
   `save_pois()`, logged to the executor (`VLA: '<name>' multi-view → …`), and
   published as a VLM event.
7. Back in the executor's mission loop, `did_inspect = True` **suppresses the Path A
   arrival event** for that stop, so the single-image scan does not overwrite the
   richer multi-view result.

---

## What "updating perception" concretely means

Before an inspection, a POI entry may be pure geometry:

```json
"5": { "name": "rack no 2 with file shelf no2 at the back side", "x": 0.848, "y": 3.306 }
```

After a VLM update it carries learned semantics:

```json
"5": {
    "name": "rack no 2 with file shelf no2 at the back side",
    "x": 0.848,
    "y": 3.306,
    "description": "The area features automatic sliding doors with metal tracks, an overhead digital screen, and an adjacent white-paneled wall."
}
```

Each revisit **replaces** the description with the latest observation, so the stored
perception tracks the current state of the environment (e.g. "garbage bag full with
spillage" vs. "bag absent"). Descriptions survive restarts and even the
`downloaded_path.json → pois.json` auto-conversion, which explicitly preserves
existing `description` fields (matched by POI name).
