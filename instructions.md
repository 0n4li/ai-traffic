# Running on Kaggle — Step-by-Step Instructions

## Prerequisites

- A **Kaggle account** (free at [kaggle.com](https://www.kaggle.com))
- Enable **GPU accelerator** for PPO training (free ~30 hrs/week)

---

## Step 1: Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **+ New Notebook**
2. Click **⋮** (three dots, top-right) → **Accelerator** → **GPU T4 x2** (or GPU P100)
3. Set **Persistence** → **Files only** (keeps saved models between sessions)

---

## Step 2: Upload Project Files

**Option A — Upload as a Dataset (Recommended)**

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **+ New Dataset**
2. Name it `ai-traffic-source`
3. Upload the entire project folder:
   ```
   src/map_processor.py
   src/traffic_generator.py
   src/dynamic_env.py
   src/traffic_baseline.py
   src/__init__.py
   01_train_base.py
   02_evaluate_map.py
   ```
4. In the notebook, click **+ Add data** (right sidebar) → search `ai-traffic-source` → **Add**
5. Your files will be at `/kaggle/input/ai-traffic-source/`

**Option B — Paste Directly**

Create cells in the notebook and paste each file using `%%writefile`:
```python
%%writefile src/map_processor.py
# paste contents here...
```

---

## Step 3: Install Dependencies

Run this as the **first cell** in your notebook:

```python
# Cell 1: Environment Setup
import os, sys, subprocess

# Install SUMO
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "sumo", "sumo-tools", "xvfb"], check=True)

# Set SUMO_HOME
os.environ["SUMO_HOME"] = "/usr/share/sumo"
sys.path.insert(0, os.environ["SUMO_HOME"] + "/tools")

# Install Python packages
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "stable-baselines3", "gymnasium", "matplotlib",
                "folium", "ipywidgets", "pyvirtualdisplay"], check=True)

# Start virtual display (needed for SUMO-GUI frame capture)
from pyvirtualdisplay import Display
display = Display(visible=False, size=(1920, 1080))
display.start()

# Verify
import traci
print(f"✅ SUMO_HOME: {os.environ['SUMO_HOME']}")
print(f"✅ traci imported successfully")
print(f"✅ GPU: {subprocess.getoutput('nvidia-smi --query-gpu=name --format=csv,noheader')}")
```

---

## Step 4: Set Up Project Structure

```python
# Cell 2: Copy source files into working directory
import shutil

WORK_DIR = "/kaggle/working/ai-traffic"
INPUT_DIR = "/kaggle/input/ai-traffic-source"  # adjust if different

os.makedirs(f"{WORK_DIR}/src", exist_ok=True)
os.makedirs(f"{WORK_DIR}/data/maps", exist_ok=True)
os.makedirs(f"{WORK_DIR}/data/routes", exist_ok=True)
os.makedirs(f"{WORK_DIR}/models", exist_ok=True)

# Copy source files
for f in ["__init__.py", "map_processor.py", "traffic_generator.py",
          "dynamic_env.py", "traffic_baseline.py"]:
    src = os.path.join(INPUT_DIR, "src", f)
    dst = os.path.join(WORK_DIR, "src", f)
    if os.path.exists(src):
        shutil.copy2(src, dst)

for f in ["01_train_base.py", "02_evaluate_map.py"]:
    src = os.path.join(INPUT_DIR, f)
    dst = os.path.join(WORK_DIR, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

print(f"✅ Working directory: {os.getcwd()}")
print(f"✅ Files: {os.listdir('.')}")
print(f"✅ Src:   {os.listdir('src')}")
```

---

## Step 5: Train Base Models

> ⏱️ **Time:** ~15–30 minutes with GPU for 100K timesteps per topology

```python
# Cell 3: Train base models
from src.map_processor import find_sumo
find_sumo()  # Verify SUMO is found

# Train 3-way, 4-way, and 5-way base models
!python 01_train_base.py --topologies 3 4 5 --timesteps 100000 --verbose 1
```

After training, you'll see `base_3way.zip`, `base_4way.zip`, `base_5way.zip` in `/kaggle/working/ai-traffic/models/`.

> [!TIP]
> To save training time during testing, reduce timesteps: `--timesteps 10000`

---

## Step 6: Run the Demo

```python
# Cell 4: Run evaluation on a real intersection
from src.map_processor import find_sumo
find_sumo()

# Option A: Use a famous junction (no internet needed for geocoding)
!python 02_evaluate_map.py --location "Silk Board Junction Bangalore"

# Option B: Use exact coordinates
# !python 02_evaluate_map.py --lat 12.9170 --lon 77.6227

# Option C: With animation capture (slower, needs virtual display)
# !python 02_evaluate_map.py --location "Silk Board Junction Bangalore" --animate
```

---

## Step 7: View Results

The dashboard image is saved automatically:

```python
# Cell 5: Display results
from IPython.display import Image, display
display(Image("data/dashboard.png"))
```

---

## Quick Reference: All in One Cell

If you want to run everything in a single cell (after setup):

```python
# All-in-one demo
import os, sys
os.chdir("/kaggle/working/ai-traffic")
sys.path.insert(0, ".")

from src.map_processor import process_osm_file, find_sumo
from src.traffic_generator import generate_random_trips
from src.traffic_baseline import run_baseline_from_phase_info
from src.dynamic_env import create_traffic_env

find_sumo()

# Process a junction
from _02_evaluate_map import run_demo
run_demo(location="Silk Board Junction Bangalore")
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `SumoNotFoundError` | Re-run Cell 1 (install step). Verify with `!which netconvert` |
| `No module named 'traci'` | Run `sys.path.insert(0, "/usr/share/sumo/tools")` |
| `No module named 'src'` | Run `os.chdir("/kaggle/working/ai-traffic")` and `sys.path.insert(0, ".")` |
| `ModuleNotFoundError: stable_baselines3` | Run `!pip install stable-baselines3` |
| `No base model found` | Run Cell 3 (training) first before Cell 4 (evaluation) |
| GPU not detected | Notebook settings → Accelerator → GPU T4 x2 |
| OSM download fails | Check internet access. Kaggle notebooks have internet enabled by default |
| `pyvirtualdisplay` fails | Only needed for `--animate` flag. Simulation runs fine without it |

---

## Saving Your Work

- **Models persist** in `/kaggle/working/` between runs (with Persistence = Files)
- To download models: click the **Output** tab → download `models/base_4way.zip`
- To reuse trained models: upload them as a Kaggle dataset and add to your notebook
