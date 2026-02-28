# Comprehensive Requirements Document
## AI-Adaptive Traffic Signal Controller

> **Version:** 1.0  
> **Date:** 2026-02-28  
> **Status:** Ready for implementation  
> **Platform:** Google Colab / Kaggle (GPU Runtime)

---

## 1. Project Vision

Build an RL system that **dynamically adjusts traffic signal countdown timers** at any real-world intersection to minimize average vehicle wait times. A user selects an intersection from a map, and the system:

1. Simulates current (static timer) traffic flow
2. Runs an AI-optimized simulation
3. Shows a **live car animation** with a results dashboard proving the improvement

The focus is Indian mega-cities (Mumbai, Bangalore) with notoriously bad traffic.

---

## 2. Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Google Colab Notebook                         │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ Map       │───▸│ Map          │───▸│ Traffic               │  │
│  │ Selection │    │ Processor    │    │ Generator             │  │
│  │ (OSM)     │    │ (.osm→.net)  │    │ (randomTrips.py)      │  │
│  └──────────┘    └──────┬───────┘    └───────────┬───────────┘  │
│                         │ N phases               │ .rou.xml     │
│                         ▼                        ▼              │
│               ┌─────────────────────────────────────┐           │
│               │        Dynamic Gymnasium Env         │           │
│               │  (SUMO simulation via TraCI)         │           │
│               └─────────────────┬───────────────────┘           │
│                                 │                               │
│                    ┌────────────┴────────────┐                  │
│                    ▼                         ▼                  │
│            ┌──────────────┐         ┌──────────────┐            │
│            │ 01_train_base│         │ 02_evaluate  │            │
│            │ (PPO train)  │         │ (inference + │            │
│            │              │         │  fine-tune)  │            │
│            └──────┬───────┘         └──────┬───────┘            │
│                   │                        │                    │
│                   ▼                        ▼                    │
│            ┌──────────────┐    ┌────────────────────┐           │
│            │ Base Models  │    │ Results Dashboard   │           │
│            │ (base_Nway)  │    │ + Live Animation    │           │
│            └──────────────┘    └────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Reinforcement Learning Specification

### 3.1 Algorithm: PPO (Proximal Policy Optimization)

Selected for continuous action space support. Implemented via `stable-baselines3`.

### 3.2 Control Paradigm: Cycle-Based

The agent makes **one decision per traffic cycle** — setting all green durations simultaneously. This produces predictable countdown timers visible to drivers (unlike frame-by-frame reactive switching which was rejected as unsafe).

### 3.3 MDP Definition

#### State Space (Observation)

Collected at the start of each cycle $t$:

| Feature | Dimensionality | Source |
|---|---|---|
| Queue length (halted vehicles) | 1 per incoming lane | `traci.lane.getLastStepHaltingNumber()` |
| Incoming traffic volume | 1 per incoming lane | `traci.lane.getLastStepVehicleNumber()` |
| Previous cycle timer values | $N$ (one per green phase) | Agent's last action |

**Total observation size:** `(2 × L) + N` where `L` = incoming lanes, `N` = controllable phases.

#### Action Space (Continuous)

- Agent outputs: `float[N]` where each $a_i \in [0, 1]$
- Constraint mapping to seconds:

$$T_i = T_{min} + a_i \times (T_{max} - T_{min})$$

| Parameter | Value | Rationale |
|---|---|---|
| $T_{min}$ | 15 seconds | No lane is skipped; minimum safe green |
| $T_{max}$ | 90 seconds | Maximum reasonable wait for opposing traffic |
| Yellow duration | 4 seconds (constant) | Indian standard; auto-inserted between greens |
| Total cycle | $\sum T_i + N \times 4\text{s}$ | Variable per cycle |

#### Reward Function

$$R = -W_{avg}$$

Where $W_{avg}$ = average waiting time of all vehicles during the completed cycle. The agent minimizes waiting by maximizing this (pushing toward zero).

### 3.4 Phase Definition ($N$)

$N$ counts only **green/controllable phases** — not yellow or all-red transitions. Yellow phases are safety constants (4s), automatically inserted between each green. The agent never controls yellow durations.

Example: A standard 4-way intersection with NS-Green → NS-Yellow → EW-Green → EW-Yellow has **$N = 2$** (two green phases the agent controls).

---

## 4. Training Strategy: Three-Tier Execution

### Tier 1: Base Model Training (Offline, One-Time)

- Train one PPO model per topology class: `base_3way.zip`, `base_4way.zip`, `base_5way.zip`
- Trained on **dummy/synthetic networks** with **heavily randomized traffic flows** (domain randomization)
- The agent learns generalized traffic management logic, not specific to any real map
- Training: ~100,000+ timesteps, GPU-accelerated

### Tier 2: Zero-Shot Inference (Instant)

- For standard, symmetrical intersections
- Load `base_Nway.zip` → run immediately on the real map
- No additional training needed

### Tier 3: Fine-Tuning (2–5 minutes)

- For complex, asymmetric Indian junctions (e.g., Silk Board, Saki Naka)
- Load base model → run 10,000 training steps on the specific map geometry
- User sees: *"Adapting AI to this junction... (est. 3 min)"*

**Auto-selection:** `map_processor.py` detects topology symmetry. Symmetric → Tier 2, Asymmetric → Tier 3.

---

## 5. Data Pipeline

### 5.1 Map Ingestion (`map_processor.py`)

| Step | Input | Output | Tool |
|---|---|---|---|
| OSM → SUMO network | `.osm` file | `.net.xml` file | `netconvert` (subprocess) |
| Phase detection | `.net.xml` file | $N$, phase details, TLS info | `xml.etree.ElementTree` |
| Junction auto-select | `.net.xml` file | Target TLS ID | Custom heuristic (busiest junction) |

**Netconvert flags** (opinionated defaults for Indian OSM):
```
--geometry.remove --roundabouts.guess --ramps.guess
--junctions.join --tls.guess --tls.join
--edges.join --remove-edges.isolated
--tls.default-type static
```

**Junction auto-selection logic:**
1. If 1 TLS found → use it
2. If multiple → pick junction with most controlled lanes
3. If none → re-run with `--tls.guess`, retry
4. If still none → error with clear message

### 5.2 Traffic Demand Generation (`traffic_generator.py`)

OSM provides roads, not traffic. We generate synthetic demand:

| Method | Use Case | Tool |
|---|---|---|
| Randomized trips | Generic stress testing | SUMO `randomTrips.py` |
| Weighted demand | Simulate peak-hour bias (e.g., 70% N→S) | Custom flow parameters |

Output: `.rou.xml` route files placed in `data/routes/`.

---

## 6. Directory Structure

```
/ai-traffic
│
├── /data
│   ├── /maps/                 # .osm files + converted .net.xml files
│   └── /routes/               # Generated traffic demand (.rou.xml)
│
├── /models/                   # Saved PPO weights (base_3way.zip, base_4way.zip, etc.)
│
├── /src
│   ├── map_processor.py       # OSM→SUMO conversion + N-phase detection
│   ├── traffic_generator.py   # Wraps randomTrips.py for demand generation
│   ├── dynamic_env.py         # Gymnasium wrapper with dynamic topology support
│   └── traffic_baseline.py    # Static timer execution for benchmarking
│
├── 01_train_base.py           # Script/Notebook: Heavy RL training for Base Models
├── 02_evaluate_map.py         # Script/Notebook: Inference, fine-tuning, and demo
│
├── Readme.md                  # Project overview (TRD)
├── requirements.md            # This document
└── designdecisions.md         # Design rationale and decision log
```

---

## 7. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core scripting |
| RL Algorithm | `stable-baselines3` (PPO) | Policy optimization |
| Simulation | Eclipse SUMO | Microscopic traffic simulation |
| Env Bridge | `gymnasium` + `sumo-rl` | RL environment wrapper |
| SUMO API | TraCI | Real-time simulation control |
| Map Source | OpenStreetMap (OSM) | Real-world intersection geometry |
| Headless Display | `pyvirtualdisplay` + `xvfb` | SUMO rendering in Colab |
| Visualization | `matplotlib` / `ipywidgets` | Charts, dashboards |
| Map Widget | `ipyleaflet` or `folium` | Interactive junction selection |
| GPU Compute | Colab/Kaggle GPU runtime | PPO training acceleration |

---

## 8. Demo UX Specification

### 8.1 User Flow

The demo runs inside a **Google Colab / Kaggle notebook** and follows this sequence:

```
Step 1: SELECT INTERSECTION
├── Interactive map widget (ipyleaflet/folium)
├── User searches for a location (e.g., "Silk Board Junction Bangalore")
├── User clicks to select the junction
└── System: crops OSM data, converts to SUMO net, detects N phases
    └── Display: "4-way intersection detected. 4 signal phases."

Step 2: BASELINE SIMULATION
├── Run static timer simulation (e.g., 45s per phase)
├── Record SUMO animation frames (headless, via xvfb)
└── Display: baseline average wait time + car animation
    └── "Static Timer: Average wait = 72 seconds"

Step 3: AI SIMULATION
├── Auto-select execution tier:
│   ├── Symmetrical → Load base model, run immediately
│   └── Asymmetrical → "Fine-tuning for this junction..." (progress bar)
├── Run AI-controlled simulation
├── Record SUMO animation frames
└── Display: AI wait time + car animation
    └── "AI Optimized: Average wait = 41 seconds"

Step 4: RESULTS DASHBOARD
├── Side-by-side comparison: Static vs AI
├── Percentage improvement: "43% reduction in wait time"
├── Timer allocation chart (what the AI chose per phase)
├── Learning curve (reward over episodes, if fine-tuned)
└── Option: "Try another intersection"
```

### 8.2 Live Car Animation

Since Colab has no native GUI, the animation approach is:

1. **During simulation:** SUMO runs headless with `pyvirtualdisplay` + `xvfb`
2. **Screenshot capture:** SUMO-GUI captures frames at regular intervals via `traci.gui.screenshot()`
3. **Playback:** Frames are compiled into an animation displayed inline in the notebook (using `matplotlib.animation` or `IPython.display.HTML` with embedded video)

The animation shows:
- Cars queuing and moving through the intersection
- Traffic light states changing (red/yellow/green)
- An overlay showing the current timer countdown

### 8.3 Results Dashboard Metrics

| Metric | Visualization | Description |
|---|---|---|
| $W_{avg}$ (Static) | Large number | Baseline average wait |
| $W_{avg}$ (AI) | Large number | AI-optimized average wait |
| Improvement % | Highlighted badge | $(1 - \frac{W_{AI}}{W_{static}}) \times 100$ |
| Timer allocations | Bar chart per phase | What seconds the AI assigned to each direction |
| Wait time over time | Line chart | How $W_{avg}$ evolved across cycles |
| Learning curve | Line chart (fine-tune only) | Reward vs training steps |

---

## 9. Module Specifications

### 9.1 `map_processor.py` — OSM to SUMO Conversion

**Functions:**

| Function | Input | Output | Description |
|---|---|---|---|
| `convert_osm_to_net()` | `.osm` path, output dir, (optional) geo_boundary | `ConversionResult` dataclass | Calls netconvert subprocess with cleanup flags |
| `extract_phase_info()` | `.net.xml` path, TLS ID | `PhaseInfo` dataclass | Parses XML for green phase count ($N$) + details |
| `list_traffic_lights()` | `.net.xml` path | `list[dict]` | Discovers all TLS in the network |

**Return types:**
```python
@dataclass
class ConversionResult:
    net_filepath: str
    tls_ids: list[str]
    incoming_edges: dict[str, list[str]]

@dataclass
class PhaseInfo:
    tls_id: str
    n_controllable_phases: int   # N for PPO
    phase_details: list[dict]
    controlled_lanes: list[str]
    is_symmetrical: bool
```

**Error handling:** Custom exceptions with `.user_message` attribute for UI display:
- `SumoNotFoundError` — SUMO not installed
- `NetconvertError` — Conversion failed
- `NoTrafficLightError` — No TLS even after guessing

### 9.2 `traffic_generator.py` — Synthetic Demand

**Functions:**

| Function | Description |
|---|---|
| `generate_random_trips()` | Wraps SUMO `randomTrips.py` for uniform demand |
| `generate_weighted_demand()` | Biases traffic flow in specific directions (peak-hour simulation) |

**Output:** `.rou.xml` files in `data/routes/`.

### 9.3 `dynamic_env.py` — Gymnasium Environment Wrapper

**Responsibilities:**
- Accept $N$ from `PhaseInfo` at initialization
- Dynamically size `observation_space` and `action_space` based on $N$
- Map PPO outputs $[0,1]$ → $[T_{min}, T_{max}]$ seconds
- Insert fixed 4s yellow phases between greens
- Execute one full cycle per `env.step()` call
- Calculate $R = -W_{avg}$ at cycle end
- Expose SUMO-GUI screenshot capture for animation

### 9.4 `traffic_baseline.py` — Static Timer Benchmark

**Responsibilities:**
- Run SUMO with fixed, equal green durations (e.g., 45s per phase)
- Log $W_{avg}$ for comparison
- Capture animation frames for side-by-side demo

### 9.5 `01_train_base.py` — Base Model Training

**Responsibilities:**
- Create dummy/synthetic SUMO networks for each N-way topology
- Train PPO with domain-randomized traffic flows
- Save `base_Nway.zip` to `/models/`
- Plot and save the learning curve

### 9.6 `02_evaluate_map.py` — Inference & Demo

**Responsibilities:**
- Interactive map widget for junction selection
- Call `map_processor.py` → `traffic_generator.py` → `dynamic_env.py`
- Auto-select execution tier (zero-shot vs fine-tune)
- Run baseline + AI simulations
- Generate car animations and results dashboard
- Display side-by-side comparison

---

## 10. Implementation Phases

| Phase | Modules | Deliverable |
|---|---|---|
| **Phase 1** | `map_processor.py` | OSM → SUMO conversion with phase detection |
| **Phase 2** | `traffic_generator.py` | Synthetic traffic demand generation |
| **Phase 3** | `dynamic_env.py` | Custom Gymnasium environment with cycle-based control |
| **Phase 4** | `traffic_baseline.py` | Static timer benchmark |
| **Phase 5** | `01_train_base.py` | Base model training pipeline |
| **Phase 6** | `02_evaluate_map.py` | Full demo notebook with map selection, animation, dashboard |

---

## 11. Acceptance Criteria

### Functional
- [ ] Processes 3-way, 4-way, and 5-way intersections without code changes
- [ ] Continuous action space correctly bounds timers between 15s–90s
- [ ] Yellow phases are always exactly 4 seconds, never agent-controlled
- [ ] Base models train successfully on dummy networks
- [ ] Zero-shot inference runs instantly on standard intersections
- [ ] Fine-tuning completes within 5 minutes for complex junctions

### Demo UX
- [ ] User can search and select a junction from an interactive map
- [ ] Live car animation plays for both baseline and AI simulations
- [ ] Results dashboard shows clear percentage improvement
- [ ] Timer allocation chart shows what the AI chose
- [ ] Entire demo flow completes in under 10 minutes (excluding training)

### Performance
- [ ] AI achieves measurably lower $W_{avg}$ than static timers on test intersections
- [ ] Training completes within Colab GPU session limits (~12 hours)
- [ ] Simulation runs at faster-than-realtime in headless mode
