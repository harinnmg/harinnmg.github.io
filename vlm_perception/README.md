# VLM-Based Environment Perception in bcr_genai

This folder documents how a **Vision-Language Model (VLM)** is used in this project to
build and continuously **update the robot's semantic perception of its environment**,
and how that perception feeds back into GenAI-based mission planning.

## The idea in one paragraph

The robot (bcr_bot, a hospital surveillance/patrol robot simulated with ROS 2 + Nav2)
carries a gripper-mounted RGB camera. Whenever the robot arrives at a Point of Interest
(POI) — or performs a dedicated manipulator-based inspection sweep — camera images are
sent to a locally hosted VLM (**Gemma 4 12B running on Ollama**). The VLM returns a short
natural-language description of what is visible at that location. This description is
**persisted into the POI database (`pois.json`)**, replacing/refreshing the stored
context for that POI. On every subsequent planning request, these learned descriptions
are injected into the LLM mission planner's system prompt as *"POI SURROUNDINGS
CONTEXT"*, so the planner reasons over an up-to-date semantic memory of the world —
not just static map coordinates.

In short: **the map gives geometry, the VLM gives (and keeps refreshing) meaning.**

## Documents

| File | Contents |
|------|----------|
| [01_system_architecture.md](01_system_architecture.md) | Components involved and how they connect (executor node, planner backend, Ollama VLM, manipulator) |
| [02_perception_update_pipeline.md](02_perception_update_pipeline.md) | The two perception-update paths: arrival-triggered single-image updates and manipulator-driven multi-view inspection |
| [03_vlm_in_mission_planning.md](03_vlm_in_mission_planning.md) | How the updated perception is consumed: ReAct planner context injection, the `visual_query` tool, and frontend VLM events |
| [04_configuration_and_endpoints.md](04_configuration_and_endpoints.md) | Environment variables, HTTP endpoints, and runtime toggles |

## Key source files

| File | Role |
|------|------|
| `src/bcr_bot/scripts/mission_planner_web.py` | Flask backend: ReAct planning loop, VLM queries to Ollama, POI context updates, VLM event store |
| `src/bcr_bot/scripts/mission_executor.py` | ROS 2 node: Nav2 navigation, gripper camera capture, `/camera/snapshot` endpoint, arrival event notification |
| `src/bcr_bot/scripts/panda_execute.py` (workspace copy; live copy in `~/arm/src/panda_moveit_config/scripts/`) | Panda arm inspection sequence; posts a capture event to the planner after each of its 6 poses |
| `src/bcr_bot/config/pois.json` | POI database — the persistent store the VLM writes its learned `description` fields into |
| `src/bcr_bot/gen_AI_integration/map_context.txt` | Static, human-authored map context (complements the dynamic VLM-learned context) |
