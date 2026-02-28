"""
traffic_baseline.py — Static Timer Benchmark

Runs a SUMO simulation with fixed, equal green durations (e.g., 45s per phase)
to establish a baseline average wait time for comparison with the AI agent.

This module provides the "before" in the "before vs after" demo comparison.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from src.dynamic_env import (
    TrafficEnv,
    T_MIN,
    T_MAX,
    YELLOW_DURATION,
    DEFAULT_SIM_STEPS,
    _import_traci,
    _write_sumo_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STATIC_GREEN = 45  # Default fixed green duration per phase (seconds)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Result of a static-timer baseline simulation."""
    avg_wait: float                    # Overall average wait time (seconds)
    per_cycle_waits: list[float]       # W_avg per cycle
    total_cycles: int                  # Number of completed cycles
    green_duration: float              # Fixed green used (seconds)
    sim_duration: float                # Total simulation time elapsed
    frames_dir: str | None = None      # Directory with animation frames


# ---------------------------------------------------------------------------
# Baseline Runner
# ---------------------------------------------------------------------------

def run_baseline(
    net_file: str,
    route_file: str,
    tls_id: str,
    n_phases: int,
    controlled_lanes: list[str],
    phase_details: list[dict],
    green_duration: float = DEFAULT_STATIC_GREEN,
    sim_duration: int = DEFAULT_SIM_STEPS,
    use_gui: bool = False,
    capture_frames: bool = False,
    frames_dir: str | None = None,
) -> BaselineResult:
    """
    Run a SUMO simulation with static (fixed, equal) green timers.

    Each green phase gets the same duration. Yellow phases are fixed at 4s.
    This provides the baseline W_avg for comparison against the AI agent.

    Args:
        net_file: Path to the .net.xml network file.
        route_file: Path to the .rou.xml route file.
        tls_id: Traffic light system ID.
        n_phases: Number of controllable green phases.
        controlled_lanes: Lanes controlled by this TLS.
        phase_details: Full phase cycle structure.
        green_duration: Fixed green duration for each phase (seconds).
        sim_duration: Maximum simulation duration (seconds).
        use_gui: Launch SUMO-GUI for visualization.
        capture_frames: Capture screenshots for animation.
        frames_dir: Directory for screenshot frames.

    Returns:
        BaselineResult with wait time metrics and frame directory.
    """
    logger.info(
        "Running baseline simulation: %ds green per phase, %d phases, %ds total",
        green_duration, n_phases, sim_duration,
    )

    # Create the environment (reusing TrafficEnv infrastructure)
    env = TrafficEnv(
        net_file=net_file,
        route_file=route_file,
        tls_id=tls_id,
        n_phases=n_phases,
        controlled_lanes=controlled_lanes,
        phase_details=phase_details,
        use_gui=use_gui,
        sim_duration=sim_duration,
        capture_frames=capture_frames,
        frames_dir=frames_dir,
    )

    try:
        obs, info = env.reset()

        # Convert fixed green duration to action space [0, 1]
        # T_i = T_min + a_i * (T_max - T_min)
        # a_i = (T_i - T_min) / (T_max - T_min)
        a = (green_duration - T_MIN) / (T_MAX - T_MIN)
        a = float(np.clip(a, 0.0, 1.0))
        static_action = np.full(n_phases, a, dtype=np.float32)

        terminated = False
        truncated = False
        cycle = 0

        while not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(static_action)
            cycle += 1
            logger.debug(
                "Baseline cycle %d: W_avg=%.1f",
                cycle, info.get("w_avg", 0),
            )

        per_cycle_waits = env.per_cycle_waits
        avg_wait = float(np.mean(per_cycle_waits)) if per_cycle_waits else 0.0

        logger.info(
            "Baseline complete: %d cycles, avg_wait=%.1f seconds",
            cycle, avg_wait,
        )

        return BaselineResult(
            avg_wait=avg_wait,
            per_cycle_waits=per_cycle_waits,
            total_cycles=cycle,
            green_duration=green_duration,
            sim_duration=info.get("sim_time", 0),
            frames_dir=env.frames_dir if capture_frames else None,
        )

    finally:
        env.close()


# ---------------------------------------------------------------------------
# Convenience: Run Baseline from PhaseInfo
# ---------------------------------------------------------------------------

def run_baseline_from_phase_info(
    net_file: str,
    route_file: str,
    phase_info: "PhaseInfo",
    green_duration: float = DEFAULT_STATIC_GREEN,
    sim_duration: int = DEFAULT_SIM_STEPS,
    use_gui: bool = False,
    capture_frames: bool = False,
    frames_dir: str | None = None,
) -> BaselineResult:
    """
    Convenience wrapper to run baseline using a PhaseInfo object directly.

    Args:
        net_file: Path to the .net.xml file.
        route_file: Path to the .rou.xml file.
        phase_info: PhaseInfo from map_processor.
        green_duration: Fixed green per phase (seconds).
        sim_duration: Max simulation duration.
        use_gui: Launch SUMO-GUI.
        capture_frames: Capture animation frames.
        frames_dir: Frame output directory.

    Returns:
        BaselineResult.
    """
    return run_baseline(
        net_file=net_file,
        route_file=route_file,
        tls_id=phase_info.tls_id,
        n_phases=phase_info.n_controllable_phases,
        controlled_lanes=phase_info.controlled_lanes,
        phase_details=phase_info.phase_details,
        green_duration=green_duration,
        sim_duration=sim_duration,
        use_gui=use_gui,
        capture_frames=capture_frames,
        frames_dir=frames_dir,
    )
