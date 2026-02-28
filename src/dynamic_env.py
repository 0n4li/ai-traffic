"""
dynamic_env.py — Custom Gymnasium Environment for RL Traffic Signal Control

A cycle-based Gymnasium environment wrapping SUMO via TraCI.
The agent makes ONE decision per full traffic signal cycle — setting
all green durations simultaneously. This produces predictable countdown
timers visible to drivers.

Key design decisions (from designdecisions.md):
    - N = green phases only (yellow is constant 4s, never agent-controlled)
    - Action space: continuous [0,1]^N → mapped to [T_min, T_max] seconds
    - Reward: R = -W_avg (negative average wait time)
    - One env.step() = one full signal cycle
"""

import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T_MIN = 15       # Minimum green duration (seconds) — no lane is skipped
T_MAX = 90       # Maximum green duration (seconds) — max reasonable wait
YELLOW_DURATION = 4   # Yellow phase duration (seconds) — Indian standard
DEFAULT_SIM_STEPS = 3600  # Default max simulation seconds


# ---------------------------------------------------------------------------
# Helper: Import TraCI
# ---------------------------------------------------------------------------

def _import_traci():
    """Import traci, adding SUMO_HOME/tools to sys.path if needed."""
    try:
        import traci
        return traci
    except ImportError:
        sumo_home = os.environ.get("SUMO_HOME", "")
        if sumo_home:
            tools_dir = os.path.join(sumo_home, "tools")
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
        try:
            import traci
            return traci
        except ImportError:
            raise ImportError(
                "Cannot import traci. Is SUMO installed? "
                "Set SUMO_HOME or install via: apt-get install sumo sumo-tools"
            )


# ---------------------------------------------------------------------------
# SUMO Configuration Writer
# ---------------------------------------------------------------------------

def _write_sumo_config(
    net_file: str,
    route_file: str,
    config_dir: str,
    sim_duration: int = DEFAULT_SIM_STEPS,
    step_length: float = 1.0,
) -> str:
    """Write a minimal SUMO .sumocfg file and return its path."""
    config_path = os.path.join(config_dir, "simulation.sumocfg")

    config_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{os.path.abspath(net_file)}"/>
        <route-files value="{os.path.abspath(route_file)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{sim_duration}"/>
        <step-length value="{step_length}"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>"""

    with open(config_path, "w") as f:
        f.write(config_xml)

    return config_path


# ---------------------------------------------------------------------------
# Traffic Signal Environment
# ---------------------------------------------------------------------------

class TrafficEnv(gym.Env):
    """
    Gymnasium environment for cycle-based traffic signal control.

    The agent outputs N continuous values per step, where each value sets the
    green duration for one phase. A full signal cycle is executed in SUMO,
    and the reward reflects the average vehicle waiting time during that cycle.

    Observation: [queue_lane_1, ..., queue_lane_L, volume_lane_1, ..., volume_lane_L,
                  prev_timer_1, ..., prev_timer_N]
    Action:      [a_1, ..., a_N] where a_i ∈ [0, 1]
    Reward:      R = -W_avg
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        net_file: str,
        route_file: str,
        tls_id: str,
        n_phases: int,
        controlled_lanes: list[str],
        phase_details: list[dict],
        use_gui: bool = False,
        sim_duration: int = DEFAULT_SIM_STEPS,
        capture_frames: bool = False,
        frames_dir: str | None = None,
        render_mode: str | None = None,
    ):
        """
        Initialize the traffic signal environment.

        Args:
            net_file: Path to the SUMO .net.xml network file.
            route_file: Path to the .rou.xml route file.
            tls_id: Traffic light system ID to control.
            n_phases: Number of controllable green phases (N).
            controlled_lanes: List of lane IDs controlled by this TLS.
            phase_details: Full cycle structure from PhaseInfo.
            use_gui: If True, launch SUMO-GUI (requires display).
            sim_duration: Maximum simulation duration in seconds.
            capture_frames: If True, capture screenshots for animation.
            frames_dir: Directory to save captured frames.
            render_mode: Gymnasium render mode.
        """
        super().__init__()

        self.net_file = net_file
        self.route_file = route_file
        self.tls_id = tls_id
        self.n_phases = n_phases
        self.controlled_lanes = controlled_lanes
        self.phase_details = phase_details
        self.use_gui = use_gui
        self.sim_duration = sim_duration
        self.capture_frames = capture_frames
        self.render_mode = render_mode

        # Frame capture setup
        self.frames_dir = frames_dir or tempfile.mkdtemp(prefix="sumo_frames_")
        self.frame_count = 0

        # Incoming lanes for observation (unique, non-internal)
        self.incoming_lanes = self._extract_incoming_lanes()
        self.n_lanes = len(self.incoming_lanes)

        # Green phase indices from phase_details
        self.green_phase_indices = [
            p["index"] for p in phase_details if p["type"] == "green"
        ]

        # Validate N
        if len(self.green_phase_indices) != n_phases:
            logger.warning(
                "Mismatch: n_phases=%d but found %d green phases in phase_details. "
                "Using %d from phase_details.",
                n_phases, len(self.green_phase_indices), len(self.green_phase_indices),
            )
            self.n_phases = len(self.green_phase_indices)

        # ---- Spaces ----
        # Observation: (2 * n_lanes) + n_phases
        #   - [queue_1, ..., queue_L] — halted vehicles per lane
        #   - [volume_1, ..., volume_L] — vehicle count per lane
        #   - [prev_t_1, ..., prev_t_N] — previous cycle timer values (normalized to [0,1])
        obs_size = 2 * self.n_lanes + self.n_phases
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Action: N continuous values in [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_phases,),
            dtype=np.float32,
        )

        logger.info(
            "TrafficEnv initialized: TLS='%s', N=%d phases, %d incoming lanes, "
            "obs_dim=%d, action_dim=%d",
            tls_id, self.n_phases, self.n_lanes, obs_size, self.n_phases,
        )

        # State tracking
        self.traci = None
        self._sumo_running = False
        self._prev_action = np.full(self.n_phases, 0.5, dtype=np.float32)
        self._cycle_count = 0
        self._total_sim_time = 0.0
        self._config_dir = tempfile.mkdtemp(prefix="sumo_config_")
        self._per_cycle_waits: list[float] = []

    def _extract_incoming_lanes(self) -> list[str]:
        """Extract unique incoming lane IDs from controlled_lanes."""
        # controlled_lanes are in format "edgeID_laneIndex"
        unique_lanes = sorted(set(self.controlled_lanes))
        return unique_lanes if unique_lanes else ["dummy_lane_0"]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment: restart SUMO simulation."""
        super().reset(seed=seed)

        # Close existing SUMO if running
        self._close_sumo()

        # Write SUMO config
        config_path = _write_sumo_config(
            self.net_file, self.route_file,
            self._config_dir, self.sim_duration,
        )

        # Start SUMO
        traci = _import_traci()
        self.traci = traci

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", config_path, "--no-warnings"]

        if self.use_gui:
            sumo_cmd += ["--start", "--quit-on-end"]

        # Use a unique label for this connection
        label = f"env_{id(self)}"
        try:
            traci.start(sumo_cmd, label=label)
            self._conn = traci.getConnection(label)
        except Exception:
            # Fallback: try without label for older SUMO versions
            traci.start(sumo_cmd)
            self._conn = traci

        self._sumo_running = True
        self._prev_action = np.full(self.n_phases, 0.5, dtype=np.float32)
        self._cycle_count = 0
        self._total_sim_time = 0.0
        self.frame_count = 0
        self._per_cycle_waits = []

        # Advance a few steps to let vehicles enter the network
        for _ in range(10):
            self._conn.simulationStep()

        obs = self._get_observation()
        info = {"cycle": self._cycle_count}

        logger.debug("Environment reset. Initial obs shape: %s", obs.shape)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one full traffic signal cycle with the given action.

        Args:
            action: Array of N floats in [0, 1], one per green phase.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if not self._sumo_running:
            raise RuntimeError("Environment not running. Call reset() first.")

        # Clip action to valid range
        action = np.clip(action, 0.0, 1.0).astype(np.float32)

        # Map [0,1] → [T_min, T_max] seconds
        green_durations = T_MIN + action * (T_MAX - T_MIN)

        # Execute the full cycle in SUMO
        cycle_wait_times = self._execute_cycle(green_durations)

        # Compute reward: R = -W_avg
        if cycle_wait_times:
            w_avg = np.mean(cycle_wait_times)
        else:
            w_avg = 0.0
        reward = -w_avg
        self._per_cycle_waits.append(w_avg)

        # Update state
        self._prev_action = action.copy()
        self._cycle_count += 1

        # Check termination
        sim_time = self._conn.simulation.getTime()
        self._total_sim_time = sim_time
        min_expected = self._conn.simulation.getMinExpectedNumber()
        terminated = min_expected == 0  # No more vehicles expected
        truncated = sim_time >= self.sim_duration

        # Observation for next step
        obs = self._get_observation()

        info = {
            "cycle": self._cycle_count,
            "w_avg": w_avg,
            "green_durations": green_durations.tolist(),
            "sim_time": sim_time,
            "vehicles_remaining": min_expected,
        }

        logger.debug(
            "Cycle %d: W_avg=%.1f, durations=%s, reward=%.1f",
            self._cycle_count, w_avg,
            [f"{d:.0f}s" for d in green_durations], reward,
        )

        return obs, reward, terminated, truncated, info

    def _execute_cycle(self, green_durations: np.ndarray) -> list[float]:
        """
        Execute one full signal cycle by setting each green phase
        followed by a fixed yellow phase, stepping SUMO through each.

        Returns a list of per-vehicle waiting times during this cycle.
        """
        all_wait_times = []
        conn = self._conn

        for i, green_idx in enumerate(self.green_phase_indices):
            green_dur = int(round(green_durations[i]))

            # ---- Green phase ----
            state = self.phase_details[green_idx]["state"]
            conn.trafficlight.setRedYellowGreenState(self.tls_id, state)
            conn.trafficlight.setPhaseDuration(self.tls_id, float(green_dur))

            for _ in range(green_dur):
                conn.simulationStep()
                all_wait_times.extend(self._collect_waiting_times())

                # Capture frame if enabled
                if self.capture_frames:
                    self._capture_frame()

                # Check if simulation ended
                if conn.simulation.getMinExpectedNumber() == 0:
                    return all_wait_times

            # ---- Yellow phase ----
            yellow_state = self._make_yellow_state(state)
            conn.trafficlight.setRedYellowGreenState(self.tls_id, yellow_state)
            conn.trafficlight.setPhaseDuration(self.tls_id, float(YELLOW_DURATION))

            for _ in range(YELLOW_DURATION):
                conn.simulationStep()
                all_wait_times.extend(self._collect_waiting_times())

                if self.capture_frames:
                    self._capture_frame()

                if conn.simulation.getMinExpectedNumber() == 0:
                    return all_wait_times

        return all_wait_times

    def _make_yellow_state(self, green_state: str) -> str:
        """Convert a green state to yellow: G/g → y, leave r unchanged."""
        return "".join(
            "y" if c in "Gg" else c
            for c in green_state
        )

    def _collect_waiting_times(self) -> list[float]:
        """Collect waiting times from all controlled lanes."""
        waits = []
        conn = self._conn
        for lane_id in self.incoming_lanes:
            try:
                wait = conn.lane.getWaitingTime(lane_id)
                if wait > 0:
                    waits.append(wait)
            except Exception:
                pass  # Lane might not exist yet
        return waits

    def _get_observation(self) -> np.ndarray:
        """
        Build the observation vector:
            [queue_1, ..., queue_L, volume_1, ..., volume_L, prev_t_1, ..., prev_t_N]
        """
        conn = self._conn
        queues = []
        volumes = []

        for lane_id in self.incoming_lanes:
            try:
                q = conn.lane.getLastStepHaltingNumber(lane_id)
                v = conn.lane.getLastStepVehicleNumber(lane_id)
            except Exception:
                q, v = 0, 0
            queues.append(float(q))
            volumes.append(float(v))

        obs = np.array(
            queues + volumes + self._prev_action.tolist(),
            dtype=np.float32,
        )
        return obs

    def _capture_frame(self) -> None:
        """Capture a screenshot from SUMO-GUI for animation."""
        if not self.use_gui:
            return
        try:
            frame_path = os.path.join(
                self.frames_dir, f"frame_{self.frame_count:06d}.png"
            )
            self._conn.gui.screenshot("View #0", frame_path)
            self.frame_count += 1
        except Exception as e:
            logger.debug("Frame capture failed: %s", e)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def per_cycle_waits(self) -> list[float]:
        """Average wait time per cycle (for plotting)."""
        return self._per_cycle_waits.copy()

    @property
    def cycle_count(self) -> int:
        """Number of completed cycles."""
        return self._cycle_count

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _close_sumo(self) -> None:
        """Close the SUMO simulation if running."""
        if self._sumo_running and self.traci:
            try:
                self._conn.close()
            except Exception:
                pass
            self._sumo_running = False

    def close(self) -> None:
        """Close the environment and SUMO."""
        self._close_sumo()
        super().close()

    def __del__(self):
        self._close_sumo()


# ---------------------------------------------------------------------------
# Factory: Create env from PhaseInfo
# ---------------------------------------------------------------------------

def create_traffic_env(
    net_file: str,
    route_file: str,
    phase_info: "PhaseInfo",
    use_gui: bool = False,
    sim_duration: int = DEFAULT_SIM_STEPS,
    capture_frames: bool = False,
    frames_dir: str | None = None,
) -> TrafficEnv:
    """
    Factory function to create a TrafficEnv from a PhaseInfo object.

    This is the recommended way to create the environment — it extracts
    all necessary parameters from the map processor's output.

    Args:
        net_file: Path to SUMO .net.xml file.
        route_file: Path to .rou.xml route file.
        phase_info: PhaseInfo from map_processor.extract_phase_info().
        use_gui: Launch SUMO-GUI.
        sim_duration: Max simulation duration.
        capture_frames: Capture frames for animation.
        frames_dir: Directory for frame screenshots.

    Returns:
        Configured TrafficEnv instance.
    """
    # Import here to avoid circular dependency
    from src.map_processor import PhaseInfo  # noqa: F811

    return TrafficEnv(
        net_file=net_file,
        route_file=route_file,
        tls_id=phase_info.tls_id,
        n_phases=phase_info.n_controllable_phases,
        controlled_lanes=phase_info.controlled_lanes,
        phase_details=phase_info.phase_details,
        use_gui=use_gui,
        sim_duration=sim_duration,
        capture_frames=capture_frames,
        frames_dir=frames_dir,
    )
