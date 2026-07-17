# 4. Configuration & Endpoints

## Environment variables

| Variable | Default | Used by | Purpose |
|----------|---------|---------|---------|
| `OLLAMA_URL` | `http://192.168.0.125:11434/api/chat` | planner | Ollama chat API endpoint hosting the VLM |
| `OLLAMA_MODEL` | `gemma4:12b` | planner | The vision-language model (also used for text-only ReAct planning) |
| `MISSION_EXECUTOR_URL` | `http://localhost:5000/mission` | planner | Mission submission endpoint; its base URL is also used to reach `/camera/snapshot` and `/log` |
| `EXECUTOR_BASE_URL` | `http://localhost:5000` | planner | Executor status/cancel proxying |
| `VLA_CONTEXT_UPDATE` | `true` | planner | Master switch for VLM-based POI context updates (both single-image and multi-view) |
| `PLANNER_ARRIVAL_EVENT_URL` | `https://localhost:8080/api/event/arrived` | executor | Where the executor reports POI arrivals |
| `WEB_PORT` | `8080` | planner | Dashboard/API port |

## HTTP endpoints involved in VLM perception

### Mission Executor (port 5000, `mission_executor.py`)

| Endpoint | Method | Role |
|----------|--------|------|
| `/camera/snapshot` | GET | Latest `/gripper_rgb` frame as JPEG (gray placeholder if no feed yet). The single image source for all VLM calls |
| `/log` | POST | Lets the planner push `VLA: ...` entries into the mission log |
| `/mission`, `/status`, `/cancel`, `/pois` | — | Mission lifecycle (not VLM-specific) |

### Mission Planner Web (port 8080, `mission_planner_web.py`)

| Endpoint | Method | Role |
|----------|--------|------|
| `/api/event/arrived` | POST | Arrival notification → spawns single-image VLM context update (Path A) |
| `/api/event/inspect_capture` | POST | Per-pose capture during manipulator inspection; on the last pose spawns the multi-view VLM update (Path B) |
| `/api/VLM_events` | GET | Cursor-based feed of perception updates for the frontend (`?since=<seq>`) |
| `/api/config` | GET/POST | Read or toggle `vla_context_update` at runtime |
| `/camera/snapshot` | GET | Proxy of the executor snapshot for the dashboard |
| `/api/health` | GET | Checks connectivity to both Ollama and the executor |

## VLM request format (Ollama chat API)

All VLM calls use the same payload shape — images are base64 JPEG strings in the
`images` field of a chat message:

```json
{
  "model": "gemma4:12b",
  "messages": [
    {
      "role": "user",
      "content": "At location 'Patient room 5', list the nearby visible objects, landmarks, or obstacles in one concise sentence (max 20 words).",
      "images": ["<base64 jpeg>", "..."]
    }
  ],
  "stream": false
}
```

Timeouts: 120 s for single-image calls (`visual_query`, Path A), 240 s for the 6-image
multi-view call (Path B).

## Operational notes & current limitations

- **Last-write-wins memory**: each update replaces the POI's `description`; there is no
  history, confidence score, or timestamp on observations.
- **Best-effort updates**: any failure (camera offline, Ollama down, empty answer)
  simply skips the update and keeps the old description; missions are never blocked by
  perception.
- **In-memory event store**: `VLM_EVENTS_STORE` and `INSPECT_SESSIONS` are process
  memory — a planner restart clears pending inspection sessions and the event feed
  (the learned descriptions themselves persist in `pois.json`).
- **Snapshot timing**: Path B captures the frame when the planner *receives* the
  per-pose event, so it depends on the arm having settled at the pose (the arm script
  posts after each ~9.5 s trajectory completes).
- **Executor→planner TLS**: the arrival event defaults to `https://localhost:8080` with
  `verify=False`; the inspection script posts to `http://localhost:8080`.
