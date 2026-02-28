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
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.map_processor import find_sumo, PhaseInfo
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


def generate_dummy_network(n_ways: int, output_dir: str) -> tuple[str, PhaseInfo]:
    """
    Generate a synthetic SUMO network for a given N-way intersection.

    Primary approach: generate grid via netgenerate. This avoids netconvert 
    which can segfault on certain platforms (e.g. Kaggle) when parsing XML.

    Args:
        n_ways: Number of arms on the intersection (3, 4, or 5).
        output_dir: Directory to save the .net.xml file.

    Returns:
        Tuple of (net_filepath, PhaseInfo).
    """
    os.makedirs(output_dir, exist_ok=True)
    net_filepath = os.path.join(output_dir, f"dummy_{n_ways}way.net.xml")

    # --- Compile with netgenerate ---
    netgenerate = _get_sumo_binary("netgenerate")
    cmd = [
        netgenerate,
        "--spider",
        "--spider.arm-number", str(n_ways),
        "--spider.circle-number", "1",
        "--spider.space-radius", "150",
        "--output-file", net_filepath,
        "--tls.default-type", "static",
        "--junctions.join"
    ]

    logger.info("Generating dummy %d-way network via netgenerate: %s", n_ways, " ".join(cmd))
    
    # Kaggle's LD_LIBRARY_PATH can mess up SUMO's netconvert (segfault)
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)

    if result.stderr:
        logger.warning("netgenerate stderr for %d-way: %s", n_ways, result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(
            f"netgenerate failed for {n_ways}-way (exit code {result.returncode}): {result.stderr}"
        )

    if not os.path.isfile(net_filepath):
        raise RuntimeError(f"netgenerate produced no output file: {net_filepath}")

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
