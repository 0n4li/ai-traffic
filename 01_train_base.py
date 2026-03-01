"""
01_train_base.py — Base Model Training Pipeline

Trains one PPO model per intersection topology class (3-way, 4-way, 5-way)
on synthetic networks with domain-randomized traffic flows.

These base models learn generalized traffic management logic that can then be:
    - Applied zero-shot to standard, symmetric intersections
    - Fine-tuned for 2-5 minutes on complex, asymmetric junctions

Usage (Colab/Kaggle):
    !python 01_train_base.py --topologies 3 4 5 --timesteps 100000

This is the heavy compute step — run once offline with GPU runtime.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import xml.etree.ElementTree as ET
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.map_processor import find_sumo, PhaseInfo, _clean_env
from src.traffic_generator import generate_random_trips
from src.dynamic_env import TrafficEnv, create_traffic_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = "models"
DEFAULT_TIMESTEPS = 100_000
TRAINING_TOPOLOGIES = [3, 4, 5]  # N-way intersections to train


# ---------------------------------------------------------------------------
# Dummy Network Generation (node/edge XML → netconvert)
# ---------------------------------------------------------------------------

def _get_sumo_binary(name: str) -> str:
    """Get path to a SUMO binary (netconvert, netgenerate, etc.)."""
    sumo_home = find_sumo()
    binary = os.path.join(sumo_home, "bin", name)
    if os.path.isfile(binary):
        return binary
    import shutil
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} binary not found")


def _rebuild_via_netconvert(net_filepath: str) -> bool:
    """
    Re-process a .net.xml through netconvert to rebuild junction logic.

    When netgenerate segfaults on Kaggle, the output has edge/junction
    geometry but is MISSING junction logic (request/response elements,
    internal edges).  Running netconvert on the existing file rebuilds
    these structures and optionally adds TLS via --tls.guess.

    Returns True if the rebuild produced a valid file.
    """
    try:
        netconvert = _get_sumo_binary("netconvert")
    except FileNotFoundError:
        logger.warning("netconvert not found — cannot rebuild network.")
        return False

    tmp_filepath = net_filepath + ".rebuilt.tmp"

    cmd = [
        netconvert,
        "--sumo-net-file", net_filepath,
        "--output-file", tmp_filepath,
        "--tls.join",
        "--tls.default-type", "static",
    ]

    logger.info(
        "Rebuilding network via netconvert (to fix junction logic + add TLS): %s",
        " ".join(cmd),
    )

    try:
        result = subprocess.run(
            cmd, env=_clean_env(), capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        logger.warning("netconvert rebuild timed out.")
        return False

    if result.stderr:
        logger.debug("netconvert rebuild stderr: %s", result.stderr.strip())

    # Accept crash-with-output as success (Kaggle ABI issue)
    if os.path.isfile(tmp_filepath) and os.path.getsize(tmp_filepath) > 100:
        if result.returncode != 0:
            logger.warning(
                "netconvert rebuild exited with code %d but output exists — "
                "treating as success.",
                result.returncode,
            )
        os.replace(tmp_filepath, net_filepath)
        logger.info("Network rebuilt successfully with junction logic + TLS.")
        return True
    else:
        logger.warning(
            "netconvert rebuild failed (exit=%d): %s",
            result.returncode, result.stderr.strip(),
        )
        # Clean up partial file
        if os.path.isfile(tmp_filepath):
            os.unlink(tmp_filepath)
        return False


def _inject_tls_if_missing(net_filepath: str, n_ways: int) -> None:
    """
    Last-resort TLS injection via pure Python XML patching.

    Only used if netconvert rebuild also fails.  Adds <tlLogic>,
    updates connection tl/linkIndex attributes, and adds a <request>
    element to the junction so SUMO can at least load the file.
    """
    tree = ET.parse(net_filepath)
    root = tree.getroot()

    # Check if TLS already exist
    if root.findall(".//tlLogic"):
        logger.info("Network already contains TLS — skipping injection.")
        return

    logger.warning("No TLS in generated network — injecting programmatically (last-resort fallback).")

    # --- 1. Find the centre junction (highest connectivity) ---
    best_junc = None
    best_conn_count = -1
    for junc in root.findall(".//junction"):
        jid = junc.get("id", "")
        if jid.startswith(":"):  # internal junction
            continue
        inc_lanes = junc.get("incLanes", "")
        count = len(inc_lanes.split()) if inc_lanes.strip() else 0
        if count > best_conn_count:
            best_conn_count = count
            best_junc = junc

    if best_junc is None:
        logger.error("No suitable junction found for TLS injection.")
        return

    tls_id = best_junc.get("id")
    logger.info("Injecting TLS at junction '%s' (%d incoming lanes)", tls_id, best_conn_count)

    # Mark junction as traffic_light type
    best_junc.set("type", "traffic_light")

    # --- 2. Collect connections through this junction ---
    connections = []
    for conn in root.findall(".//connection"):
        from_edge = conn.get("from", "")
        to_edge = conn.get("to", "")
        # Skip internal edges
        if from_edge.startswith(":") or to_edge.startswith(":"):
            continue
        connections.append(conn)

    if not connections:
        logger.error("No connections found for junction '%s'.", tls_id)
        return

    # --- 3. Group connections by incoming edge (= arm) ---
    arms: dict[str, list[ET.Element]] = {}
    for conn in connections:
        from_edge = conn.get("from", "")
        arms.setdefault(from_edge, []).append(conn)

    arm_list = sorted(arms.keys())
    n_arms = len(arm_list)
    logger.info("Found %d arms for TLS: %s", n_arms, arm_list)

    if n_arms == 0:
        return

    # --- 4. Build phase states ---
    n_conns = len(connections)
    conn_to_idx: dict[int, int] = {}
    for idx, conn in enumerate(connections):
        conn_to_idx[id(conn)] = idx

    # Create round-robin phases: one green phase per arm
    phases = []
    for arm_idx, arm_edge in enumerate(arm_list):
        arm_conns = arms[arm_edge]
        # Green phase: this arm gets 'G', others get 'r'
        state = ['r'] * n_conns
        for conn in arm_conns:
            cidx = conn_to_idx[id(conn)]
            state[cidx] = 'G'
        phases.append({"state": "".join(state), "duration": "31", "type": "green"})

        # Yellow phase after each green
        y_state = ['r'] * n_conns
        for conn in arm_conns:
            cidx = conn_to_idx[id(conn)]
            y_state[cidx] = 'y'
        phases.append({"state": "".join(y_state), "duration": "4", "type": "yellow"})

    # --- 5. Create <tlLogic> element ---
    tl_logic = ET.SubElement(root, "tlLogic")
    tl_logic.set("id", tls_id)
    tl_logic.set("type", "static")
    tl_logic.set("programID", "0")
    tl_logic.set("offset", "0")

    for phase in phases:
        phase_elem = ET.SubElement(tl_logic, "phase")
        phase_elem.set("duration", phase["duration"])
        phase_elem.set("state", phase["state"])

    # --- 6. Update connections with tl and linkIndex ---
    for idx, conn in enumerate(connections):
        conn.set("tl", tls_id)
        conn.set("linkIndex", str(idx))

    # --- 7. Add <request> element to junction if missing ---
    # This provides the right-of-way logic SUMO needs to load the file.
    existing_request = best_junc.find("request")
    if existing_request is None:
        for i in range(n_conns):
            req = ET.SubElement(best_junc, "request")
            req.set("index", str(i))
            # All-green response: no conflicts (TLS handles conflicts)
            req.set("response", "0" * n_conns)
            req.set("foes", "0" * n_conns)
            req.set("cont", "0")
        logger.info("Added %d <request> elements to junction '%s'.", n_conns, tls_id)

    # --- 8. Write back ---
    tree.write(net_filepath, xml_declaration=True, encoding="UTF-8")
    logger.info(
        "Injected TLS '%s' with %d phases (%d green) for %d connections.",
        tls_id, len(phases), len(phases) // 2, n_conns,
    )


def generate_dummy_network(n_ways: int, output_dir: str) -> tuple[str, PhaseInfo]:
    """
    Generate a synthetic SUMO network for a given N-way intersection.

    Strategy (handles Kaggle segfaults):
        1. Run netgenerate to create raw spider network geometry
        2. Re-process through netconvert to rebuild junction logic + add TLS
        3. If netconvert also fails, fall back to Python TLS injection

    Args:
        n_ways: Number of arms on the intersection (3, 4, or 5).
        output_dir: Directory to save the .net.xml file.

    Returns:
        Tuple of (net_filepath, PhaseInfo).
    """
    os.makedirs(output_dir, exist_ok=True)
    net_filepath = os.path.join(output_dir, f"dummy_{n_ways}way.net.xml")

    # --- Step 1: Generate raw geometry with netgenerate ---
    netgenerate = _get_sumo_binary("netgenerate")
    cmd = [
        netgenerate,
        "--spider",
        "--spider.arm-number", str(n_ways),
        "--spider.circle-number", "1",
        "--spider.space-radius", "150",
        "--output-file", net_filepath,
        "--tls.default-type", "static",
        "--junctions.join",
    ]

    logger.info("Generating dummy %d-way network via netgenerate: %s", n_ways, " ".join(cmd))

    result = subprocess.run(
        cmd, env=_clean_env(), capture_output=True, text=True, timeout=30,
    )

    if result.stderr:
        logger.warning("netgenerate stderr for %d-way: %s", n_ways, result.stderr.strip())

    # On Kaggle, SUMO binaries often segfault (exit -11) during process
    # cleanup AFTER successfully writing the output file — harmless.
    if result.returncode != 0 and os.path.isfile(net_filepath):
        logger.warning(
            "netgenerate exited with code %d for %d-way, but output file exists — "
            "treating as success (likely a cleanup-phase crash).",
            result.returncode, n_ways,
        )
    elif result.returncode != 0:
        diag = (
            f"exit_code={result.returncode}, "
            f"stderr={result.stderr.strip()!r}, "
            f"stdout={result.stdout.strip()!r}"
        )
        raise RuntimeError(
            f"netgenerate failed for {n_ways}-way ({diag}). "
            f"If running on Kaggle, try: !{' '.join(cmd)}"
        )

    if not os.path.isfile(net_filepath):
        raise RuntimeError(f"netgenerate produced no output file: {net_filepath}")

    # --- Step 2: Mandatory Python TLS Injection ---
    # We inject the TLS first to ensure a consistent, N-phase round-robin
    # signal structure across all topologies (guarantees action space size).
    _inject_tls_if_missing(net_filepath, n_ways)

    # --- Step 3: Rebuild via netconvert (fixes junction logic) ---
    # This computes the complex right-of-way rules (request/response matrix)
    # for the TLS we just injected. 
    _rebuild_via_netconvert(net_filepath)

    # Extract phase info from the generated network
    from src.map_processor import auto_select_junction
    phase_info = auto_select_junction(net_filepath)

    logger.info(
        "Dummy %d-way network: TLS='%s', N=%d phases, %d lanes",
        n_ways, phase_info.tls_id,
        phase_info.n_controllable_phases,
        len(phase_info.controlled_lanes),
    )

    return net_filepath, phase_info


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_base_model(
    n_ways: int,
    total_timesteps: int = DEFAULT_TIMESTEPS,
    models_dir: str = MODELS_DIR,
    data_dir: str = "data",
    learning_rate: float = 3e-4,
    n_training_scenarios: int = 3,
    verbose: int = 1,
) -> dict:
    """
    Train a base PPO model for an N-way intersection topology.

    Steps:
        1. Generate a dummy N-way SUMO network
        2. Create multiple traffic scenarios (domain randomization)
        3. Train PPO on each scenario in round-robin fashion
        4. Save model as base_Nway.zip

    Args:
        n_ways: Intersection topology (3, 4, or 5).
        total_timesteps: Total PPO training timesteps.
        models_dir: Directory to save trained models.
        data_dir: Directory for network/route files.
        learning_rate: PPO learning rate.
        n_training_scenarios: Number of traffic scenarios for domain randomization.
        verbose: Verbosity level for PPO training.

    Returns:
        Dict with training metadata (model_path, rewards, etc.).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    logger.info("=" * 60)
    logger.info("Training base model for %d-way intersections", n_ways)
    logger.info("=" * 60)

    maps_dir = os.path.join(data_dir, "maps")
    routes_dir = os.path.join(data_dir, "routes")
    os.makedirs(models_dir, exist_ok=True)

    # Step 1: Generate dummy network
    net_file, phase_info = generate_dummy_network(n_ways, maps_dir)

    # Step 2: Generate training traffic scenarios
    scenarios = []
    for i in range(n_training_scenarios):
        period = max(0.5, 1.0 + i * 0.5)  # Vary traffic density
        demand = generate_random_trips(
            net_file, routes_dir,
            period=period,
            seed=42 + i,
            prefix=f"base_{n_ways}way_s{i}_",
        )
        scenarios.append(demand)
        logger.info("Scenario %d: %d vehicles, period=%.1f", i, demand.num_vehicles, period)

    # Step 3: Train PPO with domain randomization
    # Start with the first scenario
    env = create_traffic_env(
        net_file=net_file,
        route_file=scenarios[0].route_filepath,
        phase_info=phase_info,
        sim_duration=1800,  # 30 min simulation per episode
    )

    # Custom callback to track rewards
    class RewardTracker(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_lengths = []

        def _on_step(self) -> bool:
            if self.locals.get("dones") is not None:
                for i, done in enumerate(self.locals["dones"]):
                    if done:
                        info = self.locals["infos"][i]
                        ep_reward = info.get("episode", {}).get("r", None)
                        if ep_reward is not None:
                            self.episode_rewards.append(ep_reward)
            return True

    reward_tracker = RewardTracker()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=verbose,
        device="auto",  # Use GPU if available
    )

    # Train in rounds across scenarios (domain randomization)
    steps_per_scenario = total_timesteps // n_training_scenarios

    for scenario_idx, scenario in enumerate(scenarios):
        logger.info(
            "Training on scenario %d/%d (%d steps)...",
            scenario_idx + 1, n_training_scenarios, steps_per_scenario,
        )

        # Switch to new scenario's route file
        if scenario_idx > 0:
            env.close()
            env = create_traffic_env(
                net_file=net_file,
                route_file=scenario.route_filepath,
                phase_info=phase_info,
                sim_duration=1800,
            )
            model.set_env(env)

        model.learn(
            total_timesteps=steps_per_scenario,
            callback=reward_tracker,
            reset_num_timesteps=False,
            progress_bar=True,
        )

    # Step 4: Save model
    model_path = os.path.join(models_dir, f"base_{n_ways}way")
    model.save(model_path)
    logger.info("Saved model: %s.zip", model_path)

    env.close()

    # Training metadata
    metadata = {
        "topology": f"{n_ways}-way",
        "model_path": model_path + ".zip",
        "total_timesteps": total_timesteps,
        "n_phases": phase_info.n_controllable_phases,
        "n_lanes": len(phase_info.controlled_lanes),
        "n_scenarios": n_training_scenarios,
        "rewards": reward_tracker.episode_rewards,
    }

    # Save metadata
    meta_path = os.path.join(models_dir, f"base_{n_ways}way_meta.json")
    with open(meta_path, "w") as f:
        json.dump({k: v for k, v in metadata.items() if k != "rewards"}, f, indent=2)

    return metadata


# ---------------------------------------------------------------------------
# Learning Curve Plotting
# ---------------------------------------------------------------------------

def plot_learning_curve(
    rewards: list[float],
    title: str = "Learning Curve",
    save_path: str | None = None,
    window: int = 10,
) -> None:
    """
    Plot the learning curve (reward vs episode).

    Args:
        rewards: List of episode rewards.
        title: Plot title.
        save_path: Path to save the figure (PNG).
        window: Smoothing window size.
    """
    if not rewards:
        logger.warning("No rewards to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Raw")

    # Smoothed curve
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, len(rewards)),
            smoothed,
            color="navy",
            linewidth=2,
            label=f"Smoothed (window={window})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (= -W_avg)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Learning curve saved: %s", save_path)
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """Train base models for all specified topologies."""
    parser = argparse.ArgumentParser(
        description="Train base PPO models for traffic signal control"
    )
    parser.add_argument(
        "--topologies", nargs="+", type=int,
        default=TRAINING_TOPOLOGIES,
        help="Intersection topologies to train (e.g., 3 4 5)",
    )
    parser.add_argument(
        "--timesteps", type=int,
        default=DEFAULT_TIMESTEPS,
        help="Total training timesteps per topology",
    )
    parser.add_argument(
        "--models-dir", type=str,
        default=MODELS_DIR,
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level (0=silent, 1=info, 2=debug)",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        args.verbose, logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("  AI-Adaptive Traffic Signal Controller — Base Training")
    print("=" * 60)
    print(f"  Topologies: {args.topologies}")
    print(f"  Timesteps per topology: {args.timesteps:,}")
    print(f"  Models directory: {args.models_dir}")
    print()

    all_results = {}

    for n_ways in args.topologies:
        print(f"\n{'─' * 40}")
        print(f"  Training {n_ways}-way base model...")
        print(f"{'─' * 40}\n")

        try:
            metadata = train_base_model(
                n_ways=n_ways,
                total_timesteps=args.timesteps,
                models_dir=args.models_dir,
                verbose=args.verbose,
            )
            all_results[n_ways] = metadata

            # Plot learning curve
            if metadata["rewards"]:
                plot_path = os.path.join(args.models_dir, f"learning_curve_{n_ways}way.png")
                plot_learning_curve(
                    metadata["rewards"],
                    title=f"{n_ways}-Way Base Model Learning Curve",
                    save_path=plot_path,
                )
                print(f"  ✅ {n_ways}-way model saved: {metadata['model_path']}")
            else:
                print(f"  ✅ {n_ways}-way model saved (no rewards logged)")

        except Exception as e:
            print(f"  ❌ {n_ways}-way training failed: {e}")
            logger.exception("Training failed for %d-way", n_ways)

    # Summary
    print(f"\n{'=' * 60}")
    print("  Training Summary")
    print(f"{'=' * 60}")
    for n, meta in all_results.items():
        print(f"  {n}-way: {meta['model_path']} ({meta['total_timesteps']:,} steps)")
    print()


if __name__ == "__main__":
    main()
