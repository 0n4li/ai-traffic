"""
02_evaluate_map.py — Inference, Fine-Tuning, and Demo

Full demo notebook/script for the AI-Adaptive Traffic Signal Controller.

User flow:
    1. SELECT INTERSECTION — Interactive map widget
    2. BASELINE SIMULATION — Static timer benchmark
    3. AI SIMULATION — Zero-shot or fine-tuned PPO inference
    4. RESULTS DASHBOARD — Side-by-side comparison with animation

Usage:
    Run as a Python script or execute cells in Google Colab / Kaggle.

    # Colab setup (run once):
    # !apt-get install -y sumo sumo-tools
    # !pip install stable-baselines3 gymnasium matplotlib folium ipywidgets numpy

    python 02_evaluate_map.py --location "Silk Board Junction Bangalore"
"""

import argparse
import glob
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.map_processor import (
    process_osm_file,
    find_sumo,
    ConversionResult,
    PhaseInfo,
    MapProcessorError,
)
from src.traffic_generator import generate_random_trips, generate_weighted_demand
from src.dynamic_env import create_traffic_env, TrafficEnv
from src.traffic_baseline import run_baseline_from_phase_info, BaselineResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = "models"
DATA_DIR = "data"
FINE_TUNE_STEPS = 10_000     # Training steps for asymmetric junctions
EVAL_SIM_DURATION = 1800     # 30-minute evaluation simulation
DEFAULT_STATIC_GREEN = 45    # Seconds per phase for baseline


# ---------------------------------------------------------------------------
# 1. COLAB ENVIRONMENT SETUP
# ---------------------------------------------------------------------------

def setup_colab_environment():
    """
    Install and configure SUMO for Google Colab / Kaggle.

    Run this cell first in a notebook environment.
    """
    print("🔧 Setting up SUMO environment...")

    # Check if we're in Colab/Kaggle
    in_colab = "google.colab" in sys.modules
    in_kaggle = os.path.exists("/kaggle")

    if in_colab or in_kaggle:
        # Install SUMO
        os.system("apt-get update -qq")
        os.system("apt-get install -y -qq sumo sumo-tools")

        # Set up virtual display for headless rendering
        os.system("apt-get install -y -qq xvfb")
        os.system("pip install -q pyvirtualdisplay")

        # Start virtual display
        try:
            from pyvirtualdisplay import Display
            display = Display(visible=False, size=(1920, 1080))
            display.start()
            print("  ✅ Virtual display started")
        except Exception as e:
            print(f"  ⚠️ Virtual display failed: {e}")
            print("     Animation capture may not work, but simulation will still run.")

    # Set SUMO_HOME
    if not os.environ.get("SUMO_HOME"):
        candidate_paths = [
            "/usr/share/sumo",
            "/usr/local/share/sumo",
            "/opt/homebrew/share/sumo",
        ]
        for path in candidate_paths:
            if os.path.isdir(path):
                os.environ["SUMO_HOME"] = path
                break

    try:
        sumo_home = find_sumo()
        print(f"  ✅ SUMO found: {sumo_home}")
    except Exception as e:
        print(f"  ❌ SUMO not found: {e}")
        return False

    # Install Python dependencies
    os.system("pip install -q stable-baselines3 gymnasium matplotlib folium ipywidgets")
    print("  ✅ Python dependencies installed")
    print("  ✅ Environment ready!\n")
    return True


# ---------------------------------------------------------------------------
# 2. MAP SELECTION
# ---------------------------------------------------------------------------

def download_osm_data(
    lat: float,
    lon: float,
    radius_m: float = 300,
    output_path: str | None = None,
) -> str:
    """
    Download OpenStreetMap data around a given coordinate.

    Args:
        lat: Latitude of the intersection center.
        lon: Longitude of the intersection center.
        radius_m: Radius in meters around the point to download.
        output_path: Path to save the .osm file.

    Returns:
        Path to the downloaded .osm file.
    """
    import urllib.request

    # Convert radius to degrees (approximate)
    delta_lat = radius_m / 111320
    delta_lon = radius_m / (111320 * np.cos(np.radians(lat)))

    west = lon - delta_lon
    south = lat - delta_lat
    east = lon + delta_lon
    north = lat + delta_lat

    if output_path is None:
        output_path = os.path.join(DATA_DIR, "maps", f"map_{lat:.4f}_{lon:.4f}.osm")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Overpass API query
    bbox = f"{south},{west},{north},{east}"
    overpass_url = (
        f"https://overpass-api.de/api/map?bbox={west},{south},{east},{north}"
    )

    print(f"  📥 Downloading OSM data ({radius_m}m radius around {lat:.4f}, {lon:.4f})...")

    try:
        urllib.request.urlretrieve(overpass_url, output_path)
        file_size = os.path.getsize(output_path) / 1024
        print(f"  ✅ Downloaded: {output_path} ({file_size:.0f} KB)")
    except Exception as e:
        raise MapProcessorError(
            f"OSM download failed: {e}",
            user_message=(
                "Could not download map data. Check your internet connection "
                "and try again. If the issue persists, try a different location."
            ),
        )

    return output_path


def create_map_widget():
    """
    Create an interactive map widget for junction selection.

    Returns a folium map that can be displayed in a notebook.
    For non-notebook environments, returns map coordinates from user input.
    """
    try:
        import folium
        from IPython.display import display, HTML

        # Centered on India
        m = folium.Map(
            location=[19.0760, 72.8777],  # Mumbai
            zoom_start=5,
            tiles="OpenStreetMap",
        )

        # Add search functionality
        from folium.plugins import Geocoder
        Geocoder().add_to(m)

        # Add click-to-select functionality via JavaScript
        click_js = """
        <script>
        var lastClick = null;
        document.addEventListener('DOMContentLoaded', function() {
            var mapElement = document.querySelector('.folium-map');
            if (mapElement && mapElement._leaflet_map) {
                mapElement._leaflet_map.on('click', function(e) {
                    lastClick = e.latlng;
                    // Store in a hidden element for retrieval
                    var el = document.getElementById('selected_coords');
                    if (!el) {
                        el = document.createElement('input');
                        el.id = 'selected_coords';
                        el.type = 'hidden';
                        document.body.appendChild(el);
                    }
                    el.value = e.latlng.lat + ',' + e.latlng.lng;
                    console.log('Selected:', e.latlng.lat, e.latlng.lng);
                });
            }
        });
        </script>
        """

        # Add marker for famous junctions
        famous_junctions = {
            "Silk Board Junction, Bangalore": (12.9170, 77.6227),
            "Saki Naka, Mumbai": (19.1021, 72.8871),
            "IFFCO Chowk, Gurgaon": (28.4720, 77.0720),
            "Hebbal Flyover, Bangalore": (13.0358, 77.5970),
            "Dadar TT Circle, Mumbai": (19.0180, 72.8478),
        }

        for name, (lat, lon) in famous_junctions.items():
            folium.Marker(
                [lat, lon],
                popup=f"""
                    <b>{name}</b><br>
                    <button onclick="
                        document.getElementById('selected_coords').value='{lat},{lon}';
                        alert('Selected: {name}');
                    ">Select this junction</button>
                """,
                icon=folium.Icon(color="red", icon="traffic-light", prefix="fa"),
            ).add_to(m)

        print("🗺️  Interactive Map")
        print("   Click on the map or select a pre-marked junction.")
        print("   Famous junctions are marked with red pins.\n")
        return m

    except ImportError:
        print("⚠️  folium not available. Using text input instead.")
        return None


def get_junction_coordinates(location_str: str | None = None) -> tuple[float, float]:
    """
    Get junction coordinates, either from a location string or user input.

    Args:
        location_str: Location description (e.g., "Silk Board Junction Bangalore").

    Returns:
        (latitude, longitude) tuple.
    """
    # Predefined famous junctions for quick demo
    known_junctions = {
        "silk board": (12.9170, 77.6227),
        "saki naka": (19.1021, 72.8871),
        "iffco chowk": (28.4720, 77.0720),
        "hebbal": (13.0358, 77.5970),
        "dadar": (19.0180, 72.8478),
        "andheri": (19.1197, 72.8464),
        "koramangala": (12.9352, 77.6245),
    }

    if location_str:
        location_lower = location_str.lower()
        for key, coords in known_junctions.items():
            if key in location_lower:
                print(f"  📍 Matched known junction: {key.title()} ({coords[0]:.4f}, {coords[1]:.4f})")
                return coords

        # Try geocoding
        try:
            import urllib.request
            import urllib.parse
            query = urllib.parse.quote(location_str)
            url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "AI-Traffic-Controller/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                results = json.loads(resp.read())
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                print(f"  📍 Geocoded: {results[0].get('display_name', location_str)}")
                print(f"     Coordinates: ({lat:.4f}, {lon:.4f})")
                return (lat, lon)
        except Exception as e:
            logger.warning("Geocoding failed: %s", e)

    # Fallback: ask user
    print("\nEnter junction coordinates:")
    lat = float(input("  Latitude: "))
    lon = float(input("  Longitude: "))
    return (lat, lon)


# ---------------------------------------------------------------------------
# 3. PIPELINE EXECUTION
# ---------------------------------------------------------------------------

def process_junction(
    lat: float,
    lon: float,
    radius_m: float = 300,
) -> tuple[ConversionResult, PhaseInfo, str]:
    """
    Full pipeline from coordinates to SUMO-ready simulation files.

    Steps:
        1. Download OSM data
        2. Convert to SUMO network
        3. Auto-select junction
        4. Generate traffic demand

    Returns:
        (ConversionResult, PhaseInfo, route_filepath)
    """
    print("\n" + "=" * 50)
    print("  Step 1: PROCESSING INTERSECTION")
    print("=" * 50 + "\n")

    # Download OSM
    osm_path = download_osm_data(lat, lon, radius_m)

    # Convert and detect phases
    print("  🔄 Converting map to traffic simulation...")
    conversion, phase_info = process_osm_file(osm_path, os.path.join(DATA_DIR, "maps"))

    n_ways = phase_info.n_controllable_phases
    sym_text = "symmetrical" if phase_info.is_symmetrical else "asymmetrical"

    print(f"\n  ✅ Junction detected!")
    print(f"     • Type: {n_ways}-phase intersection ({sym_text})")
    print(f"     • Controlled lanes: {len(phase_info.controlled_lanes)}")
    print(f"     • Signal phases: {n_ways}")

    # Generate traffic demand
    print("\n  🚗 Generating traffic demand...")

    if conversion.incoming_edges.get(phase_info.tls_id):
        demand = generate_weighted_demand(
            conversion.net_filepath,
            conversion.incoming_edges[phase_info.tls_id],
            os.path.join(DATA_DIR, "routes"),
        )
    else:
        demand = generate_random_trips(
            conversion.net_filepath,
            os.path.join(DATA_DIR, "routes"),
        )

    print(f"  ✅ Generated {demand.num_vehicles} vehicles")

    return conversion, phase_info, demand.route_filepath


# ---------------------------------------------------------------------------
# 4. AI INFERENCE / FINE-TUNING
# ---------------------------------------------------------------------------

def select_execution_tier(phase_info: PhaseInfo) -> str:
    """
    Auto-select the execution tier based on junction symmetry.

    Returns:
        "zero_shot" or "fine_tune"
    """
    if phase_info.is_symmetrical:
        print("  🎯 Tier: Zero-Shot Inference (symmetrical junction)")
        print("     Loading pre-trained model — no additional training needed.")
        return "zero_shot"
    else:
        print("  🎯 Tier: Fine-Tuning (asymmetrical junction)")
        print(f"     Adapting AI to this junction... (est. {FINE_TUNE_STEPS // 3000} min)")
        return "fine_tune"


def load_or_train_model(
    phase_info: PhaseInfo,
    net_file: str,
    route_file: str,
    tier: str,
):
    """
    Load a base model or fine-tune for the specific junction.

    Args:
        phase_info: Junction phase information.
        net_file: Path to SUMO network.
        route_file: Path to route file.
        tier: "zero_shot" or "fine_tune".

    Returns:
        Trained/loaded PPO model.
    """
    from stable_baselines3 import PPO

    n = phase_info.n_controllable_phases

    # Try to find a matching base model
    model_path = os.path.join(MODELS_DIR, f"base_{n}way.zip")

    if not os.path.exists(model_path):
        # Try finding any model with matching phase count
        available = glob.glob(os.path.join(MODELS_DIR, "base_*way.zip"))
        print(f"  ⚠️ No base model for {n}-way. Available: {available}")

        if available:
            # Use closest match
            model_path = available[0]
            print(f"  → Using {model_path} as starting point")
            tier = "fine_tune"  # Must fine-tune if topology doesn't match exactly
        else:
            print(f"  → No base model available. Training from scratch...")
            tier = "from_scratch"

    # Create environment for this specific junction
    env = create_traffic_env(
        net_file=net_file,
        route_file=route_file,
        phase_info=phase_info,
        sim_duration=EVAL_SIM_DURATION,
    )

    if tier == "zero_shot":
        print(f"  📦 Loading model: {model_path}")
        model = PPO.load(model_path, env=env)
        print("  ✅ Model loaded — ready for inference")

    elif tier == "fine_tune":
        print(f"  📦 Loading base model: {model_path}")
        model = PPO.load(model_path, env=env)
        print(f"  🔧 Fine-tuning for {FINE_TUNE_STEPS:,} steps...")
        model.learn(
            total_timesteps=FINE_TUNE_STEPS,
            progress_bar=True,
        )
        # Save fine-tuned model
        ft_path = os.path.join(MODELS_DIR, f"finetuned_{phase_info.tls_id}")
        model.save(ft_path)
        print(f"  ✅ Fine-tuning complete. Saved: {ft_path}.zip")

    else:  # from_scratch
        print(f"  🆕 Training new model from scratch ({FINE_TUNE_STEPS * 5:,} steps)...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            verbose=0,
            device="auto",
        )
        model.learn(
            total_timesteps=FINE_TUNE_STEPS * 5,
            progress_bar=True,
        )
        save_path = os.path.join(MODELS_DIR, f"trained_{phase_info.tls_id}")
        model.save(save_path)
        print(f"  ✅ Training complete. Saved: {save_path}.zip")

    env.close()
    return model


def run_ai_simulation(
    model,
    net_file: str,
    route_file: str,
    phase_info: PhaseInfo,
    capture_frames: bool = False,
    frames_dir: str | None = None,
) -> dict:
    """
    Run the AI-controlled simulation and collect results.

    Returns:
        Dict with avg_wait, per_cycle_waits, timer_allocations, frames_dir.
    """
    env = create_traffic_env(
        net_file=net_file,
        route_file=route_file,
        phase_info=phase_info,
        sim_duration=EVAL_SIM_DURATION,
        capture_frames=capture_frames,
        frames_dir=frames_dir,
        use_gui=capture_frames,
    )

    obs, info = env.reset()
    terminated = False
    truncated = False
    all_allocations = []

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        all_allocations.append(info.get("green_durations", []))

    per_cycle_waits = env.per_cycle_waits
    avg_wait = float(np.mean(per_cycle_waits)) if per_cycle_waits else 0.0

    result = {
        "avg_wait": avg_wait,
        "per_cycle_waits": per_cycle_waits,
        "timer_allocations": all_allocations,
        "total_cycles": env.cycle_count,
        "frames_dir": env.frames_dir if capture_frames else None,
    }

    env.close()
    return result


# ---------------------------------------------------------------------------
# 5. ANIMATION
# ---------------------------------------------------------------------------

def create_animation(
    frames_dir: str,
    title: str = "Traffic Simulation",
    fps: int = 10,
    save_path: str | None = None,
):
    """
    Create an animation from captured SUMO-GUI frames.

    Args:
        frames_dir: Directory containing frame_XXXXXX.png files.
        title: Animation title.
        fps: Frames per second.
        save_path: Optional path to save as MP4/GIF.

    Returns:
        matplotlib animation object (for inline display in notebooks).
    """
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))

    if not frame_files:
        print(f"  ⚠️ No frames found in {frames_dir}")
        return None

    print(f"  🎬 Creating animation from {len(frame_files)} frames...")

    # Subsample if too many frames
    max_frames = 200
    if len(frame_files) > max_frames:
        step = len(frame_files) // max_frames
        frame_files = frame_files[::step]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Load first frame
    first_img = plt.imread(frame_files[0])
    im = ax.imshow(first_img)

    def update(frame_idx):
        img = plt.imread(frame_files[frame_idx])
        im.set_data(img)
        return [im]

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(frame_files),
        interval=1000 // fps,
        blit=True,
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
        print(f"  ✅ Animation saved: {save_path}")

    return anim


# ---------------------------------------------------------------------------
# 6. RESULTS DASHBOARD
# ---------------------------------------------------------------------------

def display_dashboard(
    baseline: BaselineResult,
    ai_result: dict,
    phase_info: PhaseInfo,
    learning_curve_path: str | None = None,
) -> None:
    """
    Display the comparison dashboard with all metrics.

    Shows:
        - Side-by-side wait times
        - Percentage improvement
        - Timer allocation chart
        - Wait time evolution over cycles
        - Learning curve (if fine-tuned)
    """
    w_static = baseline.avg_wait
    w_ai = ai_result["avg_wait"]
    improvement = (1 - w_ai / w_static) * 100 if w_static > 0 else 0

    print("\n" + "=" * 60)
    print("  📊 RESULTS DASHBOARD")
    print("=" * 60)

    # Create figure with subplots
    n_plots = 3
    if learning_curve_path and os.path.exists(learning_curve_path):
        n_plots = 4

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "AI-Adaptive Traffic Signal Controller — Results",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ---- Plot 1: Side-by-side comparison ----
    ax1 = axes[0, 0]
    bars = ax1.bar(
        ["Static Timer", "AI Optimized"],
        [w_static, w_ai],
        color=["#e74c3c", "#2ecc71"],
        width=0.5,
        edgecolor="white",
        linewidth=2,
    )
    ax1.set_ylabel("Average Wait Time (seconds)", fontsize=11)
    ax1.set_title("Wait Time Comparison", fontsize=13, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, [w_static, w_ai]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}s",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold",
        )

    # Add improvement badge
    badge_color = "#27ae60" if improvement > 0 else "#e74c3c"
    badge_text = f"{'↓' if improvement > 0 else '↑'} {abs(improvement):.0f}%"
    ax1.text(
        0.5, 0.85, badge_text,
        transform=ax1.transAxes,
        fontsize=20, fontweight="bold",
        ha="center", va="center",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_color, alpha=0.9),
    )

    ax1.grid(axis="y", alpha=0.3)

    # ---- Plot 2: Wait time over cycles ----
    ax2 = axes[0, 1]
    cycles_base = range(len(baseline.per_cycle_waits))
    cycles_ai = range(len(ai_result["per_cycle_waits"]))

    ax2.plot(list(cycles_base), baseline.per_cycle_waits,
             color="#e74c3c", alpha=0.7, label="Static Timer", linewidth=1.5)
    ax2.plot(list(cycles_ai), ai_result["per_cycle_waits"],
             color="#2ecc71", alpha=0.7, label="AI Optimized", linewidth=1.5)

    ax2.set_xlabel("Cycle", fontsize=11)
    ax2.set_ylabel("Average Wait Time (seconds)", fontsize=11)
    ax2.set_title("Wait Time Over Cycles", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ---- Plot 3: Timer allocation chart ----
    ax3 = axes[1, 0]
    if ai_result["timer_allocations"]:
        # Average allocation across all cycles
        allocations = np.array(ai_result["timer_allocations"])
        mean_alloc = allocations.mean(axis=0)
        std_alloc = allocations.std(axis=0)

        phase_labels = [f"Phase {i+1}" for i in range(len(mean_alloc))]
        x = np.arange(len(phase_labels))

        bars = ax3.bar(
            x, mean_alloc,
            yerr=std_alloc,
            color=plt.cm.Set2(np.linspace(0, 1, len(mean_alloc))),
            capsize=5,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels
        for bar, val in zip(bars, mean_alloc):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}s",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
            )

        # Draw static reference line
        ax3.axhline(
            y=DEFAULT_STATIC_GREEN,
            color="#e74c3c",
            linestyle="--",
            alpha=0.7,
            label=f"Static ({DEFAULT_STATIC_GREEN}s)",
        )

        ax3.set_xticks(x)
        ax3.set_xticklabels(phase_labels)
        ax3.set_ylabel("Green Duration (seconds)", fontsize=11)
        ax3.set_title("AI Timer Allocations", fontsize=13, fontweight="bold")
        ax3.legend(fontsize=10)
        ax3.grid(axis="y", alpha=0.3)

    # ---- Plot 4: Learning curve or summary stats ----
    ax4 = axes[1, 1]

    if learning_curve_path and os.path.exists(learning_curve_path):
        img = plt.imread(learning_curve_path)
        ax4.imshow(img)
        ax4.axis("off")
        ax4.set_title("Learning Curve (Fine-Tuning)", fontsize=13, fontweight="bold")
    else:
        # Summary statistics table
        ax4.axis("off")
        summary_data = [
            ["Metric", "Static", "AI", "Change"],
            ["Avg Wait (s)", f"{w_static:.1f}", f"{w_ai:.1f}",
             f"{'↓' if improvement > 0 else '↑'}{abs(improvement):.1f}%"],
            ["Total Cycles", str(baseline.total_cycles), str(ai_result["total_cycles"]), "—"],
            ["Green/Phase", f"{DEFAULT_STATIC_GREEN}s (fixed)", "Dynamic", "—"],
        ]

        table = ax4.table(
            cellText=summary_data[1:],
            colLabels=summary_data[0],
            cellLoc="center",
            loc="center",
            colWidths=[0.25, 0.2, 0.2, 0.2],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor("#34495e")
            table[(0, i)].set_text_props(color="white", fontweight="bold")

        ax4.set_title("Summary", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save dashboard
    dashboard_path = os.path.join(DATA_DIR, "dashboard.png")
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    print(f"  💾 Dashboard saved: {dashboard_path}")

    # Display
    plt.show()
    plt.close(fig)

    # Print text summary
    print(f"\n  {'─' * 40}")
    print(f"  📊 Static Timer:  Average wait = {w_static:.1f} seconds")
    print(f"  🤖 AI Optimized:  Average wait = {w_ai:.1f} seconds")
    print(f"  {'─' * 40}")
    if improvement > 0:
        print(f"  ✅ {improvement:.0f}% REDUCTION in average wait time!")
    else:
        print(f"  ⚠️ AI did not improve on this junction ({improvement:.0f}%)")
    print(f"  {'─' * 40}")


# ---------------------------------------------------------------------------
# 7. FULL DEMO FLOW
# ---------------------------------------------------------------------------

def run_demo(
    location: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_m: float = 300,
    capture_animation: bool = False,
) -> None:
    """
    Run the complete demo: select → baseline → AI → dashboard.

    Args:
        location: Location string (e.g., "Silk Board Junction Bangalore").
        lat: Latitude (if location not provided).
        lon: Longitude (if location not provided).
        radius_m: Map download radius in meters.
        capture_animation: If True, capture SUMO-GUI frames for animation.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  AI-Adaptive Traffic Signal Controller — Live Demo        ║")
    print("╚" + "═" * 58 + "╝\n")

    # ── Step 1: Select intersection ──
    if lat is not None and lon is not None:
        coords = (lat, lon)
    else:
        coords = get_junction_coordinates(location)

    lat, lon = coords

    # ── Step 2: Process junction ──
    try:
        conversion, phase_info, route_file = process_junction(lat, lon, radius_m)
    except MapProcessorError as e:
        print(f"\n  ❌ {e.user_message}")
        return

    # ── Step 3: Baseline simulation ──
    print("\n" + "=" * 50)
    print("  Step 2: BASELINE SIMULATION (Static Timer)")
    print("=" * 50 + "\n")

    print(f"  🔴 Running static timer simulation ({DEFAULT_STATIC_GREEN}s per phase)...")

    baseline = run_baseline_from_phase_info(
        net_file=conversion.net_filepath,
        route_file=route_file,
        phase_info=phase_info,
        green_duration=DEFAULT_STATIC_GREEN,
        sim_duration=EVAL_SIM_DURATION,
        capture_frames=capture_animation,
    )

    print(f"  ✅ Static Timer: Average wait = {baseline.avg_wait:.1f} seconds")

    # ── Step 4: AI simulation ──
    print("\n" + "=" * 50)
    print("  Step 3: AI SIMULATION")
    print("=" * 50 + "\n")

    tier = select_execution_tier(phase_info)

    try:
        model = load_or_train_model(
            phase_info=phase_info,
            net_file=conversion.net_filepath,
            route_file=route_file,
            tier=tier,
        )
    except Exception as e:
        print(f"\n  ❌ Model loading/training failed: {e}")
        print("  → Run 01_train_base.py first to create base models.")
        return

    print(f"\n  🟢 Running AI-optimized simulation...")

    ai_result = run_ai_simulation(
        model=model,
        net_file=conversion.net_filepath,
        route_file=route_file,
        phase_info=phase_info,
        capture_frames=capture_animation,
    )

    print(f"  ✅ AI Optimized: Average wait = {ai_result['avg_wait']:.1f} seconds")

    # ── Step 5: Animations ──
    if capture_animation:
        print("\n" + "=" * 50)
        print("  ANIMATIONS")
        print("=" * 50 + "\n")

        if baseline.frames_dir:
            create_animation(
                baseline.frames_dir,
                title="Baseline: Static Timer",
                save_path=os.path.join(DATA_DIR, "baseline_animation.gif"),
            )

        if ai_result.get("frames_dir"):
            create_animation(
                ai_result["frames_dir"],
                title="AI Optimized",
                save_path=os.path.join(DATA_DIR, "ai_animation.gif"),
            )

    # ── Step 6: Dashboard ──
    print("\n" + "=" * 50)
    print("  Step 4: RESULTS DASHBOARD")
    print("=" * 50)

    display_dashboard(
        baseline=baseline,
        ai_result=ai_result,
        phase_info=phase_info,
    )

    print("\n  🔄 Want to try another intersection? Run this cell again!")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="AI-Adaptive Traffic Signal Controller — Demo"
    )
    parser.add_argument(
        "--location", type=str,
        help="Location to evaluate (e.g., 'Silk Board Junction Bangalore')",
    )
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument(
        "--radius", type=float, default=300,
        help="Map download radius in meters (default: 300)",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Capture SUMO-GUI frames for animation (requires display)",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Run Colab environment setup first",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if args.setup:
        setup_colab_environment()

    run_demo(
        location=args.location,
        lat=args.lat,
        lon=args.lon,
        radius_m=args.radius,
        capture_animation=args.animate,
    )


if __name__ == "__main__":
    main()
