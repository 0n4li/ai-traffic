"""
traffic_generator.py — Synthetic Traffic Demand Generation

Generates vehicle routes (.rou.xml) for SUMO simulations.
OSM provides road geometry but not traffic volumes, so we create synthetic demand
using SUMO's randomTrips.py and custom weighted flow generation.

Two modes:
    1. Random trips — uniform demand for generic stress testing
    2. Weighted demand — directional bias to simulate peak-hour patterns
"""

import logging
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.map_processor import find_sumo, MapProcessorError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRIP_PERIOD = 1.0        # Vehicle insertion period (seconds between vehicles)
DEFAULT_NUM_VEHICLES = 500       # Default number of vehicles to generate
DEFAULT_SIM_DURATION = 3600      # Default simulation duration in seconds (1 hour)
DEFAULT_SEED = 42                # Reproducible randomness


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DemandResult:
    """Result of traffic demand generation."""
    route_filepath: str
    num_vehicles: int
    sim_duration: int
    method: str  # "random" or "weighted"


# ---------------------------------------------------------------------------
# randomTrips.py Discovery
# ---------------------------------------------------------------------------

def _get_random_trips_script() -> str:
    """Locate SUMO's randomTrips.py tool."""
    sumo_home = find_sumo()

    candidates = [
        os.path.join(sumo_home, "tools", "randomTrips.py"),
        os.path.join(sumo_home, "share", "sumo", "tools", "randomTrips.py"),
        # Linux package installs
        "/usr/share/sumo/tools/randomTrips.py",
    ]

    for path in candidates:
        if os.path.isfile(path):
            logger.debug("Found randomTrips.py at: %s", path)
            return path

    raise MapProcessorError(
        "randomTrips.py not found in SUMO tools",
        user_message=(
            "SUMO tools are not properly installed. "
            "The randomTrips.py script is missing. "
            "Please reinstall SUMO with tools: apt-get install sumo sumo-tools"
        ),
    )


# ---------------------------------------------------------------------------
# Random Trip Generation
# ---------------------------------------------------------------------------

def generate_random_trips(
    net_filepath: str,
    output_dir: str = "data/routes",
    period: float = DEFAULT_TRIP_PERIOD,
    sim_duration: int = DEFAULT_SIM_DURATION,
    seed: int = DEFAULT_SEED,
    prefix: str = "",
) -> DemandResult:
    """
    Generate uniform random trips using SUMO's randomTrips.py.

    Creates vehicles with random origin-destination pairs across the network.
    Good for generic stress testing and base model training.

    Args:
        net_filepath: Path to the .net.xml network file.
        output_dir: Directory to save the .rou.xml file.
        period: Seconds between vehicle insertions (lower = more traffic).
        sim_duration: Total simulation duration in seconds.
        seed: Random seed for reproducibility.
        prefix: Optional filename prefix.

    Returns:
        DemandResult with output path and metadata.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net_name = Path(net_filepath).stem.replace(".net", "")
    route_filename = f"{prefix}{net_name}.rou.xml" if prefix else f"{net_name}.rou.xml"
    route_filepath = out_dir / route_filename
    trips_filepath = out_dir / f"{net_name}.trips.xml"

    random_trips = _get_random_trips_script()
    estimated_vehicles = int(sim_duration / period)

    cmd = [
        "python", random_trips,
        "-n", str(net_filepath),
        "-r", str(route_filepath),
        "-o", str(trips_filepath),
        "-e", str(sim_duration),
        "-p", str(period),
        "--seed", str(seed),
        "--validate",
        "--route-file", str(route_filepath),
    ]

    logger.info("Generating random trips: ~%d vehicles over %ds (period=%.1f)",
                estimated_vehicles, sim_duration, period)
    logger.debug("Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise MapProcessorError(
            "randomTrips.py timed out",
            user_message="Traffic generation took too long. Try a smaller network.",
        )

    if result.returncode != 0:
        logger.error("randomTrips.py stderr: %s", result.stderr)
        raise MapProcessorError(
            f"randomTrips.py failed: {result.stderr}",
            user_message="Failed to generate traffic demand. The network may be disconnected.",
        )

    # Count actual vehicles generated
    actual_count = _count_vehicles_in_route_file(str(route_filepath))
    logger.info("Generated %d vehicles → %s", actual_count, route_filepath)

    # Clean up intermediate trips file
    if trips_filepath.exists():
        trips_filepath.unlink()

    return DemandResult(
        route_filepath=str(route_filepath),
        num_vehicles=actual_count,
        sim_duration=sim_duration,
        method="random",
    )


# ---------------------------------------------------------------------------
# Weighted Demand Generation
# ---------------------------------------------------------------------------

def generate_weighted_demand(
    net_filepath: str,
    incoming_edges: list[str],
    output_dir: str = "data/routes",
    sim_duration: int = DEFAULT_SIM_DURATION,
    vehicles_per_hour: int = 1800,
    directional_weights: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    prefix: str = "weighted_",
) -> DemandResult:
    """
    Generate traffic demand with directional bias for peak-hour simulation.

    Creates flow definitions that bias traffic in specific directions,
    simulating real-world patterns like morning commute (70% toward city center).

    Args:
        net_filepath: Path to the .net.xml network file.
        incoming_edges: List of incoming edge IDs (from ConversionResult).
        output_dir: Directory to save the .rou.xml file.
        sim_duration: Total simulation duration in seconds.
        vehicles_per_hour: Total flow rate across all edges.
        directional_weights: Optional dict mapping edge_id → relative weight.
            If None, assigns decreasing weights (first edge gets most traffic).
        seed: Random seed for reproducibility.
        prefix: Filename prefix.

    Returns:
        DemandResult with output path and metadata.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net_name = Path(net_filepath).stem.replace(".net", "")
    route_filepath = out_dir / f"{prefix}{net_name}.rou.xml"

    if not incoming_edges:
        logger.warning("No incoming edges provided, falling back to random trips")
        return generate_random_trips(net_filepath, output_dir, seed=seed)

    # Default weights: bias toward first edges (simulates directional flow)
    if directional_weights is None:
        total = len(incoming_edges)
        directional_weights = {}
        for i, edge in enumerate(incoming_edges):
            # First edge gets highest weight, decreasing proportionally
            directional_weights[edge] = (total - i) / total

    # Normalize weights
    weight_sum = sum(directional_weights.values())
    normalized = {k: v / weight_sum for k, v in directional_weights.items()}

    # Build the route XML with flow elements
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

    # Add a comment
    comment = ET.Comment(f" Weighted demand: {vehicles_per_hour} veh/hr, seed={seed} ")
    root.append(comment)

    # Vehicle type
    vtype = ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5",
                          sigma="0.5", length="5", maxSpeed="15")

    flow_id = 0
    total_vehicles = 0

    for from_edge in incoming_edges:
        weight = normalized.get(from_edge, 1.0 / len(incoming_edges))
        edge_vph = int(vehicles_per_hour * weight)

        if edge_vph == 0:
            continue

        # Calculate period (seconds between vehicles on this edge)
        period = 3600.0 / edge_vph if edge_vph > 0 else 999

        # Create flows from this edge to random destinations
        # Use multiple destination edges for variety
        for to_edge in incoming_edges:
            if to_edge == from_edge:
                continue

            sub_vph = max(1, edge_vph // (len(incoming_edges) - 1))
            sub_period = 3600.0 / sub_vph

            flow = ET.SubElement(root, "flow",
                                 id=f"flow_{flow_id}",
                                 type="car",
                                 begin="0",
                                 end=str(sim_duration),
                                 period=f"{sub_period:.1f}",
                                 **{"from": from_edge, "to": to_edge},
                                 departLane="best",
                                 departSpeed="max")
            flow_id += 1
            total_vehicles += int(sim_duration / sub_period)

    # Write the XML file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(route_filepath), xml_declaration=True, encoding="UTF-8")

    logger.info("Generated weighted demand: ~%d vehicles → %s", total_vehicles, route_filepath)

    return DemandResult(
        route_filepath=str(route_filepath),
        num_vehicles=total_vehicles,
        sim_duration=sim_duration,
        method="weighted",
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _count_vehicles_in_route_file(route_filepath: str) -> int:
    """Count the number of vehicle/trip elements in a route file."""
    try:
        tree = ET.parse(route_filepath)
        root = tree.getroot()
        vehicles = root.findall(".//vehicle") + root.findall(".//trip")
        return len(vehicles)
    except ET.ParseError:
        logger.warning("Could not parse route file to count vehicles")
        return 0


def generate_demand_for_training(
    net_filepath: str,
    incoming_edges: list[str],
    output_dir: str = "data/routes",
    num_scenarios: int = 5,
    base_seed: int = 42,
) -> list[DemandResult]:
    """
    Generate multiple demand scenarios for domain-randomized training.

    Creates a mix of random trips and weighted demand with varying parameters
    to help the base model learn generalizable traffic management.

    Args:
        net_filepath: Path to the .net.xml network file.
        incoming_edges: List of incoming edge IDs.
        output_dir: Output directory for route files.
        num_scenarios: Number of scenarios to generate.
        base_seed: Starting random seed.

    Returns:
        List of DemandResult objects.
    """
    results = []

    for i in range(num_scenarios):
        seed = base_seed + i

        if i % 2 == 0:
            # Random trips with varying density
            period = max(0.5, DEFAULT_TRIP_PERIOD * (0.5 + i * 0.3))
            result = generate_random_trips(
                net_filepath, output_dir,
                period=period, seed=seed,
                prefix=f"train_{i}_",
            )
        else:
            # Weighted demand with varying bias
            vph = 1200 + i * 300
            result = generate_weighted_demand(
                net_filepath, incoming_edges, output_dir,
                vehicles_per_hour=vph, seed=seed,
                prefix=f"train_{i}_",
            )
        results.append(result)

    logger.info("Generated %d training scenarios", len(results))
    return results
