# Design Decisions — AI-Adaptive Traffic Signal Controller

> All technical infrastructure decisions have been finalized. This document serves as the authoritative reference for implementation.
>
> - ✅ = Decision finalized
> - 🎯 = Key model/UX decision (user-facing)

---

## Part A: Infrastructure Decisions (Resolved)

These decisions are internal — the user should never need to think about them.

---

### 1. SUMO Binary Discovery — ✅ Dual-strategy, fail-fast

- Check `SUMO_HOME` env var first → append `$SUMO_HOME/bin` to `PATH`
- Fallback to `shutil.which('netconvert')`
- Fail with `SumoNotFoundError` including install instructions
- **No** version validation at startup (adds friction, low benefit)

### 2. Netconvert Flags — ✅ Opinionated defaults + optional override

All cleanup flags enabled by default. An `extra_netconvert_flags` parameter is exposed for edge cases but hidden from the demo UX. TLS type: **`static`** — the agent replaces all logic via TraCI.

Full flag set: `--geometry.remove`, `--roundabouts.guess`, `--ramps.guess`, `--junctions.join`, `--tls.guess`, `--tls.join`, `--edges.join`, `--remove-edges.isolated`.

### 3. Phase Counting ($N$) — ✅ GREEN phases only

- $N$ = count of **green/controllable** phases (not yellow, not all-red)
- Yellow phases are **fixed at 4 seconds** (Indian standard; configurable via a constant `YELLOW_DURATION_S = 4`)
- The agent outputs $N$ continuous values → we reconstruct the full cycle by inserting fixed yellow between each green
- `map_processor.py` returns `n_controllable_phases` (the $N$ for PPO) and `phase_details` (full cycle structure for `dynamic_env.py`)

### 4. Junction Selection — ✅ Smart auto-selection

The user picks an intersection from the map. The system handles everything under the hood:

1. `list_traffic_lights()` finds all TLS in the network
2. If exactly 1 TLS → auto-select
3. If multiple → pick the one with the most controlled lanes (busiest junction)
4. If none → re-run netconvert with `--tls.guess` and retry
5. Final fallback → `NoTrafficLightError` with a clear message

The user never sees TLS IDs or XML parsing. They see: *"Junction detected: 4-way intersection, 4 signal phases."*

### 5. Geo Bounding Box — ✅ Caller provides, system validates

`convert_osm_to_net()` accepts an optional `geo_boundary`. If not provided, the full OSM extent is used. No auto-calculation — the upstream map UI handles bounding.

### 6. Error Handling — ✅ Granular exceptions, user-friendly messages

```python
class MapProcessorError(Exception): ...     # Base class
class SumoNotFoundError(MapProcessorError): ...  # SUMO not installed
class NetconvertError(MapProcessorError): ...    # Conversion failed
class NoTrafficLightError(MapProcessorError): ... # No TLS found even after guessing
```

Each exception carries a `.user_message` attribute with a plain-English explanation (no technical jargon) suitable for display in the demo UI.

### 7. Return Values — ✅ Dataclasses, moderate edge metadata

```python
@dataclass
class ConversionResult:
    net_filepath: str
    tls_ids: list[str]
    incoming_edges: dict[str, list[str]]  # For traffic_generator.py

@dataclass
class PhaseInfo:
    tls_id: str
    n_controllable_phases: int   # N for PPO action space
    phase_details: list[dict]    # Full cycle structure
    controlled_lanes: list[str]
    is_symmetrical: bool         # For transfer learning tier selection
```

### 8. File Naming — ✅ Mirror input

`bangalore_koramangala.osm` → `bangalore_koramangala.net.xml` in `data/maps/`.

### 9. Logging — ✅ Python `logging`, no print statements

`INFO` for milestones, `DEBUG` for commands/XML, `WARNING` for fallbacks.

---

## Part B: Model Decisions — 🎯 What the User Cares About

These decisions define how well the system works and how impressive the demo is.

---

### 10. Training Strategy — ✅ Three-Tier Execution

| Tier | When | What Happens | Training Time |
|---|---|---|---|
| **Base Model** | Pre-computed | Train one PPO model per N-way topology (3-way, 4-way, etc.) on randomized dummy networks | Hours (offline) |
| **Zero-Shot** | Standard intersections | Load `base_Nway.zip`, run inference immediately | 0 (instant) |
| **Fine-Tune** | Complex/asymmetric junctions | Load base model, train 10K steps on specific map geometry | ~2–5 min |

**Decision:** `map_processor.py` returns `is_symmetrical` flag. `02_evaluate_map.py` uses this to auto-decide:
- Symmetrical → zero-shot
- Asymmetrical → fine-tune

The user sees: *"This is a complex junction. Fine-tuning for optimal results... (est. 3 min)"*

### 11. State Space Definition — ✅

Gathered at the start of every cycle $t$. Per incoming lane:

| Feature | Type | Source |
|---|---|---|
| Queue length (halted vehicles) | `int` per lane | `traci.lane.getLastStepHaltingNumber()` |
| Incoming volume (vehicles/min) | `float` per lane | `traci.lane.getLastStepVehicleNumber()` × scaling |
| Previous cycle timer values | `float[N]` | Agent's last action |

Total observation size = `(2 × num_incoming_lanes) + N`

### 12. Action Space Definition — ✅

- PPO outputs `float[N]` where each $a_i \in [0, 1]$
- Mapped to seconds: $T_i = T_{min} + a_i \times (T_{max} - T_{min})$
- $T_{min} = 15\text{s}$, $T_{max} = 90\text{s}$
- Yellow phases (4s each) are inserted automatically between greens
- Total cycle time: $\sum T_i + N \times 4\text{s}$ (variable per cycle)

### 13. Reward Function — ✅

$$R = -W_{avg}$$

Where $W_{avg}$ is the mean waiting time of all vehicles during the completed cycle.

> 🎯 **For the demo:** We will track and display $W_{avg}$ over time as a live chart — this is the single most important visualization showing the AI learning to reduce wait times.

---

## Part C: Demo UX Decisions — 🎯 The Presentation Layer

---

### 14. What the User Sees in the Demo

The demo should tell a compelling story: *"Before AI: X seconds average wait. After AI: Y seconds. Z% improvement."*

#### Proposed Demo Flow

```
1. SELECT INTERSECTION
   User picks a point on an embedded map (or enters coords)
   → System auto-crops OSM, converts, detects N phases

2. BASELINE RUN
   "Running static timer simulation..."
   → Fixed 45s/phase, shows average wait time
   → Animation of cars queuing (optional SUMO-GUI or simplified viz)

3. AI RUN
   "Running AI-optimized simulation..."
   → If zero-shot: instant
   → If fine-tuning: progress bar + "Adapting to this junction..."
   → Shows average wait time dropping in real-time

4. RESULTS DASHBOARD
   - Side-by-side: Static vs AI wait times
   - Percentage improvement
   - Timer allocation chart (what durations the AI chose per phase)
   - Learning curve (reward over episodes) for fine-tuned runs
```

### 🟢 Q-UX1: Resolved → Live car animation

SUMO-GUI renders headless in Colab via `pyvirtualdisplay` + `xvfb`. Frames captured with `traci.gui.screenshot()` and compiled into inline animation via `matplotlib.animation` or `IPython.display.HTML`.

### 🟢 Q-UX2: Resolved → Google Colab / Kaggle (GPU runtime)

GPU needed for PPO training. Animation and dashboard rendered inline in the notebook.

### 🟢 Q-UX3: Resolved → User-selectable via interactive map

User searches for any junction on an embedded `ipyleaflet` / `folium` map widget. They can try as many cities/junctions as they want — the system handles everything.

---

## Decision Summary

| Area | Decisions Made | Remaining |
|---|---|---|
| Infrastructure (Part A) | 9/9 ✅ all resolved | None |
| Model (Part B) | 4/4 ✅ all resolved | None |
| Demo UX (Part C) | 3/3 ✅ all resolved | None |

**All decisions finalized.** Full specification captured in [requirements.md](file:///Users/on.ali/Work/python/ai-traffic/requirements.md).

---

*SUMO documentation verified via Context7 MCP server.*
