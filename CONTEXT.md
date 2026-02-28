# Project Context & Session Notes
## AI-Adaptive Traffic Signal Controller

> **Purpose:** This file captures conversational context, user preferences, and implicit decisions that aren't fully captured in the formal `requirements.md` or `designdecisions.md`. Reference this at the start of any implementation session.

---

## User Working Style

- **Delegate all technical decisions.** The user does not want to be asked about infrastructure, error handling, file naming, XML parsing strategy, logging, etc. Make the best engineering decision and move on.
- **Focus areas the user cares about:** The RL model quality and the final demo UX. Everything else is plumbing.
- **User-friendly first.** Error messages, progress indicators, and output should be in plain English. No TLS IDs, XML paths, or stack traces in the user-facing layer.

---

## How This Project Evolved (Decision History)

The architecture went through 4 iterations before landing on the final design. A future agent **must not** re-propose any of the rejected approaches:

| Iteration | Approach | Why Rejected |
|---|---|---|
| 1 | **DQN + step-based (frame-by-frame) control** | Unsafe — real traffic needs countdown timers visible to drivers. Instant switching is dangerous. |
| 2 | **Cycle-based + discrete actions** (select from timer "buckets" like 40s/40s vs 60s/20s) | Handicaps the AI — if optimal is 43s, buckets can't achieve it. |
| 3 | **Universal brain with zero-padding** (one massive network for all intersection types) | Inefficient training — AI wastes compute learning to ignore "ghost" lanes. |
| **4 (Final)** | **Cycle-based + continuous actions (PPO) + dynamic instantiation** | ✅ Adopted. Precise timer control, efficient per-topology models. |

The full reasoning is in `Readme.md` Section 2 and the original Gemini conversation in `gemini-conversation.md`.

---

## Critical Technical Decisions (Quick Reference)

These decisions were debated and finalized. Do not revisit:

| Decision | Resolution | Rationale |
|---|---|---|
| $N$ = green phases only | Yellow phases are **constant 4s**, never agent-controlled | Agent shouldn't learn to shorten safety transitions |
| TLS default type | `static` | Agent replaces all TLS logic via TraCI; `actuated` would interfere |
| Single intersection (MVP) | One junction at a time, one PPO model per N-way topology | Multi-agent is a future extension, not MVP |
| Junction auto-selection | Pick the junction with the most controlled lanes | User never sees TLS IDs or junction selection UI |
| Transfer learning tiers | Base model → zero-shot (symmetric) or fine-tune (asymmetric) | Avoids training from scratch for every new map |
| Training platform | Google Colab / Kaggle with GPU runtime | GPU needed for PPO; headless SUMO via xvfb |
| Visualization | Live SUMO car animation (headless render → frame capture → inline playback) | User explicitly requested live animation over static charts |
| Map selection | Interactive map widget (ipyleaflet/folium) with search | User searches and clicks; system handles everything |

---

## Key File Map

| File | Role | Status |
|---|---|---|
| `requirements.md` | **Authoritative spec** — architecture, MDP, module specs, UX flow, acceptance criteria | ✅ Complete |
| `designdecisions.md` | Decision log with rationale for every technical choice | ✅ Complete, all 12/12 resolved |
| `Readme.md` | Original TRD + rejected architecture history | Reference only |
| `gemini-conversation.md` | Full brainstorming conversation that produced the TRD | Reference only |
| `CONTEXT.md` | This file — session context for future agents | ✅ This file |

---

## Implementation Order

Follow the phases in `requirements.md` Section 10 strictly:

1. `src/map_processor.py` — OSM → SUMO conversion + phase detection
2. `src/traffic_generator.py` — Synthetic demand generation
3. `src/dynamic_env.py` — Custom Gymnasium environment (cycle-based)
4. `src/traffic_baseline.py` — Static timer benchmark
5. `01_train_base.py` — Base model training pipeline
6. `02_evaluate_map.py` — Full demo notebook (map + animation + dashboard)

---

## Suggested Session Opener

When starting a new implementation session, the user can say:

> *"Implement the AI-Traffic system following `requirements.md` and `CONTEXT.md`. Start with Phase 1. You own all technical decisions."*
