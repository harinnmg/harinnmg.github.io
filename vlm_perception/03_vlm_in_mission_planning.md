# 3. How the Updated Perception Is Used in Planning

Updating `pois.json` is only half the story — the value comes from feeding the learned
perception back into the GenAI mission planner. There are three consumption points.

## 3.1 POI SURROUNDINGS CONTEXT in the ReAct system prompt

Every planning request rebuilds the ReAct system prompt via
`get_react_system_prompt()` in `mission_planner_web.py`. It contains:

- **MAP CONTEXT** — static, human-authored notes from
  `gen_AI_integration/map_context.txt` (shelf/material mapping, location types).
- **POI SURROUNDINGS CONTEXT** — built dynamically by
  `_format_poi_context_section()`, which iterates over `POIS_DATA` and emits one line
  per POI that has a VLM-learned description:

  ```
  POI SURROUNDINGS CONTEXT:
  These notes were learned by the VLM when the robot previously arrived at each POI.
  Use them to understand what objects, obstacles, or landmarks are near a POI.
  - POI 5 (rack no 2 with file shelf no2 at the back side): The area features automatic sliding doors...
  ```

The planner rules explicitly instruct the LLM to use it:

- **Rule 9**: *"Use POI SURROUNDINGS CONTEXT when choosing, ordering, or explaining
  stops. Treat it as remembered VLM context about what is near each POI."*
- **Rule 8 / intent mapping — context-aware briefing**: for any mission with 3+ stops
  the planner must call `get_pois`, and if descriptions exist, `send_message` a summary
  to the operator *before* submitting the plan — e.g. *"Garbage point 1 was last seen
  full with spillage. Garbage point 2 bag was absent. Proceeding with patrol."*

So the perception learned on previous patrols directly shapes stop selection, ordering,
and the operator briefing of future missions.

## 3.2 The `visual_query` tool — live perception on demand

The ReAct planner also has a **real-time** perception tool, `tool_visual_query()`:

1. The LLM emits `Action: visual_query` with a free-text prompt
   (e.g. *"Is the path blocked?"*, *"Do you see a red box?"*).
2. The backend fetches the current frame from the executor's `/camera/snapshot`,
   base64-encodes it, and sends prompt + image to the same Gemma 4 VLM.
3. The VLM's answer is returned to the loop as the `Observation:`, which the planner
   reasons over in its next `Thought:`.

This lets a user ask perception questions in the chat ("is there an obstacle in front
of the robot?") and lets the planner verify the scene before committing to a plan —
the same model serving as text-only planner and as VLM.

## 3.3 VLM events on the dashboard

Every successful perception update (single-image or multi-view) appends an entry to an
in-memory event store:

```json
{ "seq": 12, "poi_id": "5", "poi_name": "rack no 2 ...", "description": "..." }
```

The frontend polls `GET /api/VLM_events?since=<cursor>` and surfaces new descriptions
live in the Mission Control dashboard; a matching entry is also pushed into the
executor's mission log (`VLA: '<name>' surroundings → …`). The operator therefore sees
the robot's understanding of the environment evolve in real time as it patrols.

## Static vs. dynamic context — the layered picture

| Layer | Source | Update rate | Example |
|-------|--------|-------------|---------|
| Geometry | Nav2 map + `pois.json` coordinates | Fixed at deployment | POI 5 at (0.848, 3.306) |
| Static semantics | `map_context.txt` (human-written) | Manual | "Shelf D3: Tools" |
| **Dynamic semantics** | **VLM descriptions in `pois.json`** | **Every visit / inspection** | "sliding doors, overhead digital screen, white-paneled wall" |
| Live perception | `visual_query` tool | On demand, per planning step | "Yes, a box is blocking the corridor" |

The VLM owns the bottom two layers — the ones that keep the robot's world model
consistent with a changing environment.
