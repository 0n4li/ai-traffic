"""
map_processor.py — OSM to SUMO Conversion + Phase Detection

Converts OpenStreetMap (.osm) files to SUMO network (.net.xml) files,
discovers traffic light systems, and extracts phase information for
the PPO agent's action/observation space sizing.

Handles:
- SUMO binary discovery (dual-strategy: SUMO_HOME → shutil.which)
- netconvert subprocess invocation with opinionated Indian-OSM defaults
- Traffic light discovery and auto-selection (busiest junction)
- Green phase counting (N) and symmetry detection
"""

import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NETCONVERT_DEFAULT_FLAGS = [
    "--geometry.remove",
    "--roundabouts.guess",
    "--ramps.guess",
    "--junctions.join",
    "--tls.guess",
    "--tls.join",
    "--edges.join",
    "--remove-edges.isolated",
    "--tls.default-type", "static",
]

YELLOW_DURATION_S = 4  # Indian standard; never agent-controlled

# Environment keys safe to pass to SUMO subprocesses.
# Kaggle/Colab kernels inject LD_PRELOAD, LD_LIBRARY_PATH, CUDA vars, etc.
# that cause segfaults in native SUMO binaries.
_SAFE_ENV_KEYS = {
    "PATH", "HOME", "USER", "LANG", "TMPDIR", "TEMP", "TMP",
    "SUMO_HOME", "DISPLAY", "XDG_RUNTIME_DIR",
}


def _clean_env() -> dict[str, str]:
    """Build a minimal environment for SUMO binary subprocesses."""
    return {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MapProcessorError(Exception):
    """Base exception for map processing errors."""

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or message


class SumoNotFoundError(MapProcessorError):
    """SUMO is not installed or not found on this system."""

    def __init__(self):
        super().__init__(
            "SUMO binaries not found via SUMO_HOME or PATH.",
            user_message=(
                "SUMO is not installed. Please install it:\n"
                "  • Colab/Linux: !apt-get install -y sumo sumo-tools\n"
                "  • macOS: brew install sumo\n"
                "  • Windows: download from https://sumo.dlr.de/docs/Downloads.php"
            ),
        )


class NetconvertError(MapProcessorError):
    """netconvert failed to convert the OSM file."""

    def __init__(self, stderr: str):
        super().__init__(
            f"netconvert failed: {stderr}",
            user_message=(
                "Failed to convert the map data into a traffic simulation network. "
                "The selected area may not contain enough road data. "
                "Try selecting a larger area or a different intersection."
            ),
        )


class NoTrafficLightError(MapProcessorError):
    """No traffic lights found in the network, even after guessing."""

    def __init__(self):
        super().__init__(
            "No traffic light systems found in the network.",
            user_message=(
                "No traffic signals were found at this location. "
                "This intersection may not have traffic lights in OpenStreetMap. "
                "Try a major intersection on a national highway or city arterial road."
            ),
        )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConversionResult:
    """Result of converting an OSM file to a SUMO network."""
    net_filepath: str
    tls_ids: list[str]
    incoming_edges: dict[str, list[str]]  # tls_id → list of incoming edge IDs


@dataclass
class PhaseInfo:
    """Phase information extracted from a traffic light system."""
    tls_id: str
    n_controllable_phases: int        # N for PPO action space
    phase_details: list[dict]         # Full cycle structure [{state, duration, type}, ...]
    controlled_lanes: list[str]
    is_symmetrical: bool              # For transfer learning tier selection


# ---------------------------------------------------------------------------
# SUMO Discovery
# ---------------------------------------------------------------------------

def find_sumo() -> str:
    """
    Locate the SUMO installation directory.

    Strategy:
        1. Check SUMO_HOME environment variable
        2. Fall back to shutil.which('netconvert') and infer SUMO_HOME

    Returns:
        Path to the SUMO home directory.

    Raises:
        SumoNotFoundError: If SUMO cannot be found.
    """
    # Strategy 1: SUMO_HOME environment variable
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home and os.path.isdir(sumo_home):
        logger.info("Found SUMO via SUMO_HOME: %s", sumo_home)
        # Ensure bin directory is on PATH
        bin_dir = os.path.join(sumo_home, "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            logger.debug("Added %s to PATH", bin_dir)
        return sumo_home

    # Strategy 2: Find netconvert on PATH
    netconvert_path = shutil.which("netconvert")
    if netconvert_path:
        # Infer SUMO_HOME from netconvert location (typically <SUMO_HOME>/bin/netconvert)
        bin_dir = os.path.dirname(os.path.realpath(netconvert_path))
        sumo_home = os.path.dirname(bin_dir)
        os.environ["SUMO_HOME"] = sumo_home
        logger.info("Found SUMO via PATH (netconvert at %s), set SUMO_HOME=%s",
                     netconvert_path, sumo_home)
        return sumo_home

    raise SumoNotFoundError()


def _get_netconvert_binary() -> str:
    """Get the full path to the netconvert binary."""
    sumo_home = find_sumo()
    netconvert = os.path.join(sumo_home, "bin", "netconvert")
    if os.path.isfile(netconvert):
        return netconvert
    # Fallback: it might be directly on PATH
    found = shutil.which("netconvert")
    if found:
        return found
    raise SumoNotFoundError()


def _get_randomtrips_script() -> str:
    """Get the full path to SUMO's randomTrips.py tool."""
    sumo_home = find_sumo()
    # Common locations across SUMO installations
    candidates = [
        os.path.join(sumo_home, "tools", "randomTrips.py"),
        os.path.join(sumo_home, "share", "sumo", "tools", "randomTrips.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise MapProcessorError(
        "randomTrips.py not found",
        user_message="SUMO tools are not properly installed. randomTrips.py is missing.",
    )


# ---------------------------------------------------------------------------
# OSM → SUMO Conversion
# ---------------------------------------------------------------------------

def convert_osm_to_net(
    osm_filepath: str,
    output_dir: str,
    geo_boundary: tuple[float, float, float, float] | None = None,
    extra_netconvert_flags: list[str] | None = None,
) -> ConversionResult:
    """
    Convert an OSM file to a SUMO .net.xml network file.

    Args:
        osm_filepath: Path to the .osm input file.
        output_dir: Directory where the .net.xml will be saved.
        geo_boundary: Optional (west, south, east, north) bounding box to crop the network.
        extra_netconvert_flags: Additional flags to pass to netconvert.

    Returns:
        ConversionResult with the output path, TLS IDs, and incoming edges.

    Raises:
        NetconvertError: If netconvert fails.
    """
    osm_path = Path(osm_filepath)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror input filename: bangalore_koramangala.osm → bangalore_koramangala.net.xml
    net_filename = osm_path.stem + ".net.xml"
    net_filepath = out_dir / net_filename

    netconvert = _get_netconvert_binary()

    cmd = [
        netconvert,
        "--osm-files", str(osm_path),
        "--output-file", str(net_filepath),
    ] + NETCONVERT_DEFAULT_FLAGS

    if geo_boundary:
        west, south, east, north = geo_boundary
        cmd += ["--keep-edges.in-boundary",
                f"{west},{south},{east},{north}"]

    if extra_netconvert_flags:
        cmd += extra_netconvert_flags

    logger.info("Running netconvert: %s", " ".join(cmd))

    try:
        env = _clean_env()

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise NetconvertError("netconvert timed out after 120 seconds")

    if result.returncode != 0 and net_filepath.exists():
        logger.warning(
            "netconvert exited with code %d but output file exists — "
            "treating as success (likely a cleanup-phase crash).",
            result.returncode,
        )
    elif result.returncode != 0:
        logger.error("netconvert stderr: %s", result.stderr)
        raise NetconvertError(result.stderr)

    if not net_filepath.exists():
        raise NetconvertError("Output file was not created")

    logger.info("Network created: %s", net_filepath)

    # Discover TLS and edges from the generated network
    tls_ids = [tls["id"] for tls in list_traffic_lights(str(net_filepath))]
    incoming_edges = _extract_incoming_edges(str(net_filepath), tls_ids)

    return ConversionResult(
        net_filepath=str(net_filepath),
        tls_ids=tls_ids,
        incoming_edges=incoming_edges,
    )


def _extract_incoming_edges(net_filepath: str, tls_ids: list[str]) -> dict[str, list[str]]:
    """Extract incoming edge IDs for each TLS from the .net.xml file."""
    tree = ET.parse(net_filepath)
    root = tree.getroot()
    result: dict[str, list[str]] = {}

    for tls_id in tls_ids:
        incoming = set()
        # Find connections controlled by this TLS
        for conn in root.findall(".//connection"):
            if conn.get("tl") == tls_id:
                edge_from = conn.get("from", "")
                if edge_from and not edge_from.startswith(":"):
                    incoming.add(edge_from)
        result[tls_id] = sorted(incoming)

    return result


# ---------------------------------------------------------------------------
# Traffic Light Discovery
# ---------------------------------------------------------------------------

def list_traffic_lights(net_filepath: str) -> list[dict]:
    """
    Discover all traffic light systems in a SUMO .net.xml file.

    Args:
        net_filepath: Path to the .net.xml file.

    Returns:
        List of dicts with keys: id, type, programID, num_phases.
    """
    tree = ET.parse(net_filepath)
    root = tree.getroot()

    tls_list = []
    for tl in root.findall(".//tlLogic"):
        tls_list.append({
            "id": tl.get("id"),
            "type": tl.get("type", "static"),
            "programID": tl.get("programID", "0"),
            "num_phases": len(tl.findall("phase")),
        })

    logger.info("Found %d traffic light system(s) in %s", len(tls_list), net_filepath)
    return tls_list


# ---------------------------------------------------------------------------
# Phase Extraction
# ---------------------------------------------------------------------------

def extract_phase_info(net_filepath: str, tls_id: str) -> PhaseInfo:
    """
    Extract green phase information from a traffic light system.

    N counts only GREEN/controllable phases — not yellow or all-red transitions.
    Yellow phases are safety constants (4s), automatically inserted between greens.

    Args:
        net_filepath: Path to the .net.xml file.
        tls_id: The traffic light system ID to analyze.

    Returns:
        PhaseInfo with N, phase details, controlled lanes, and symmetry flag.

    Raises:
        NoTrafficLightError: If the specified TLS is not found.
    """
    tree = ET.parse(net_filepath)
    root = tree.getroot()

    # Find the specific TLS
    tl_elem = None
    for tl in root.findall(".//tlLogic"):
        if tl.get("id") == tls_id:
            tl_elem = tl
            break

    if tl_elem is None:
        raise NoTrafficLightError()

    # Parse all phases
    phase_details = []
    green_phase_indices = []

    for i, phase_elem in enumerate(tl_elem.findall("phase")):
        state = phase_elem.get("state", "")
        duration = float(phase_elem.get("duration", "0"))

        # Classify phase type
        phase_type = _classify_phase(state)
        phase_details.append({
            "index": i,
            "state": state,
            "duration": duration,
            "type": phase_type,
        })

        if phase_type == "green":
            green_phase_indices.append(i)

    n_controllable = len(green_phase_indices)
    logger.info("TLS '%s': %d total phases, %d controllable green phases",
                tls_id, len(phase_details), n_controllable)

    # Extract controlled lanes
    controlled_lanes = _get_controlled_lanes(root, tls_id)

    # Detect symmetry
    is_symmetrical = _detect_symmetry(root, tls_id, phase_details, green_phase_indices)

    return PhaseInfo(
        tls_id=tls_id,
        n_controllable_phases=n_controllable,
        phase_details=phase_details,
        controlled_lanes=controlled_lanes,
        is_symmetrical=is_symmetrical,
    )


def _classify_phase(state: str) -> str:
    """
    Classify a phase state string as 'green', 'yellow', or 'all_red'.

    A phase is 'green' if it contains at least one G or g (green/priority green).
    A phase is 'yellow' if it contains y/Y but no G/g.
    Otherwise it's 'all_red' (all r/R).
    """
    has_green = any(c in state for c in "Gg")
    has_yellow = any(c in state for c in "Yy")

    if has_green:
        return "green"
    elif has_yellow:
        return "yellow"
    else:
        return "all_red"


def _get_controlled_lanes(root: ET.Element, tls_id: str) -> list[str]:
    """Get all lane IDs controlled by this TLS from the connection elements."""
    lanes = set()
    for conn in root.findall(".//connection"):
        if conn.get("tl") == tls_id:
            from_lane = conn.get("from", "") + "_" + conn.get("fromLane", "0")
            lanes.add(from_lane)
    return sorted(lanes)


def _detect_symmetry(
    root: ET.Element,
    tls_id: str,
    phase_details: list[dict],
    green_indices: list[int],
) -> bool:
    """
    Detect whether a junction is symmetrical by analyzing phase structure.

    A junction is considered symmetrical if:
    1. Each green phase controls roughly the same number of lanes
    2. The green state patterns show balanced lane allocation

    This determines whether zero-shot inference or fine-tuning is needed.
    """
    if len(green_indices) < 2:
        return True  # Trivially symmetrical

    # Count lanes active (G/g) in each green phase
    lanes_per_phase = []
    for idx in green_indices:
        state = phase_details[idx]["state"]
        active_count = sum(1 for c in state if c in "Gg")
        lanes_per_phase.append(active_count)

    if not lanes_per_phase:
        return True

    # Check if lane counts are balanced (within 30% of each other)
    max_lanes = max(lanes_per_phase)
    min_lanes = min(lanes_per_phase)

    if max_lanes == 0:
        return True

    balance_ratio = min_lanes / max_lanes
    is_sym = balance_ratio >= 0.7

    logger.info("Symmetry analysis for '%s': lanes_per_phase=%s, ratio=%.2f, symmetrical=%s",
                tls_id, lanes_per_phase, balance_ratio, is_sym)
    return is_sym


# ---------------------------------------------------------------------------
# Junction Auto-Selection
# ---------------------------------------------------------------------------

def auto_select_junction(
    net_filepath: str,
    retry_with_tls_guess: bool = True,
) -> PhaseInfo:
    """
    Automatically select the best junction for RL optimization.

    Selection logic:
        1. If exactly 1 TLS found → use it
        2. If multiple → pick the one with the most controlled lanes (busiest)
        3. If none → re-run netconvert with --tls.guess and retry
        4. If still none → raise NoTrafficLightError

    Args:
        net_filepath: Path to the .net.xml file.
        retry_with_tls_guess: Whether to retry with --tls.guess if no TLS found.

    Returns:
        PhaseInfo for the selected junction.
    """
    tls_list = list_traffic_lights(net_filepath)

    if len(tls_list) == 0:
        if retry_with_tls_guess:
            logger.warning("No TLS found. Attempting to re-generate with --tls.guess")
            # Re-run netconvert on the existing net file to add guessed TLS
            _add_tls_guess(net_filepath)
            tls_list = list_traffic_lights(net_filepath)

        if len(tls_list) == 0:
            raise NoTrafficLightError()

    if len(tls_list) == 1:
        selected_id = tls_list[0]["id"]
        logger.info("Single TLS found: %s", selected_id)
    else:
        # Pick junction with the most controlled lanes
        tree = ET.parse(net_filepath)
        root = tree.getroot()

        lane_counts: dict[str, int] = {}
        for tls in tls_list:
            tls_id = tls["id"]
            lanes = _get_controlled_lanes(root, tls_id)
            lane_counts[tls_id] = len(lanes)

        selected_id = max(lane_counts, key=lane_counts.get)  # type: ignore
        logger.info("Multiple TLS found. Selected '%s' with %d controlled lanes (from %s)",
                     selected_id, lane_counts[selected_id], lane_counts)

    phase_info = extract_phase_info(net_filepath, selected_id)
    return phase_info


def _add_tls_guess(net_filepath: str) -> None:
    """Re-process a .net.xml file to add guessed traffic lights."""
    netconvert = _get_netconvert_binary()
    tmp_filepath = net_filepath + ".tmp"

    cmd = [
        netconvert,
        "--sumo-net-file", net_filepath,
        "--output-file", tmp_filepath,
        "--tls.guess",
    ]

    logger.debug("Re-running netconvert with --tls.guess: %s", " ".join(cmd))

    result = subprocess.run(cmd, env=_clean_env(), capture_output=True, text=True, timeout=60)

    # Accept crash-with-output-file as success (Kaggle ABI issue)
    if os.path.exists(tmp_filepath):
        if result.returncode != 0:
            logger.warning(
                "netconvert exited with code %d but output file exists — "
                "treating as success.", result.returncode,
            )
        os.replace(tmp_filepath, net_filepath)
        logger.info("Updated network with guessed TLS")
    else:
        logger.warning("TLS guessing failed: %s", result.stderr)


# ---------------------------------------------------------------------------
# Convenience: Full Pipeline for a Single OSM File
# ---------------------------------------------------------------------------

def process_osm_file(
    osm_filepath: str,
    output_dir: str = "data/maps",
    geo_boundary: tuple[float, float, float, float] | None = None,
) -> tuple[ConversionResult, PhaseInfo]:
    """
    Full pipeline: OSM → SUMO network → auto-select junction → extract phases.

    This is the main entry point for the map processor.

    Args:
        osm_filepath: Path to the .osm file.
        output_dir: Where to save the .net.xml file.
        geo_boundary: Optional bounding box for cropping.

    Returns:
        Tuple of (ConversionResult, PhaseInfo).
    """
    logger.info("Processing OSM file: %s", osm_filepath)

    # Step 1: Convert OSM → SUMO network
    conversion = convert_osm_to_net(osm_filepath, output_dir, geo_boundary)
    logger.info("Conversion complete. %d TLS found.", len(conversion.tls_ids))

    # Step 2: Auto-select junction and extract phase info
    phase_info = auto_select_junction(conversion.net_filepath)
    logger.info(
        "Selected junction '%s': %d-way, %d controllable phases, symmetrical=%s",
        phase_info.tls_id,
        phase_info.n_controllable_phases,
        phase_info.n_controllable_phases,
        phase_info.is_symmetrical,
    )

    return conversion, phase_info
