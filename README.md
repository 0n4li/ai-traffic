# AI-Adaptive Traffic Signal Controller

A Reinforcement Learning (RL) system engineered to dynamically adjust traffic light countdown timers at real-world intersections to minimize average vehicle wait times.

## 🚀 Overview

This project uses **Proximal Policy Optimization (PPO)** and **Eclipse SUMO** to optimize signal timings. It supports real-world map ingestion via OpenStreetMap (OSM) and features a three-tier execution strategy (Zero-Shot inference, Fine-Tuning, or Training from scratch).

> [!NOTE]
> For a deep dive into the architecture, MDP formulation, and design decisions, see the [**REQUIREMENTS.md**](REQUIREMENTS.md).

---

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

## Step 2: Choose Your Code Upload Method

You only need to do **one** of these options to get the code onto Kaggle.

### Option A: Upload as a Kaggle Dataset (Recommended for stability)

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **+ New Dataset**
2. Name it `ai-traffic-source` and upload your local files.
3. In your notebook, click **+ Add data** (right sidebar) → search `ai-traffic-source` → **Add**
4. Your files will be at `/kaggle/input/ai-traffic-source/`

### Option B: Paste Directly (Quickest for single files)

Create a cell for each file and use `%%writefile`:
```python
%%writefile src/map_processor.py
# paste contents here...
```

### Option C: Git Clone (Best for public repos)

Run this in a cell to pull from GitHub:
```python
!git clone https://github.com/0n4li/ai-traffic.git /kaggle/working/ai-traffic
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

## Step 4: Set Up Workspace

This cell organizes your files into the correct directory structure. Run this after uploading/cloning.

```python
# Cell 2: Setup Workspace
import shutil, os, sys

WORK_DIR = "/kaggle/working/ai-traffic"
INPUT_DIR = "/kaggle/input/ai-traffic-source"

# Path 1: If using Git Clone (Option C), it's already in WORK_DIR
if os.path.exists(WORK_DIR):
    print("✅ Found Git repository")

# Path 2: If using Dataset (Option A), copy files to WORK_DIR
elif os.path.exists(INPUT_DIR):
    print("✅ Copying files from Dataset...")
    os.makedirs(f"{WORK_DIR}/src", exist_ok=True)
    # Copy src files
    for f in os.listdir(f"{INPUT_DIR}/src"):
        shutil.copy2(f"{INPUT_DIR}/src/{f}", f"{WORK_DIR}/src/{f}")
    # Copy root files
    for f in ["01_train_base.py", "02_evaluate_map.py"]:
        if os.path.exists(f"{INPUT_DIR}/{f}"):
            shutil.copy2(f"{INPUT_DIR}/{f}", f"{WORK_DIR}/{f}")

# Path 3: If using %%writefile (Option B) manually
else:
    print("⚠️ Please ensure files were created via %%writefile")
    os.makedirs(f"{WORK_DIR}/src", exist_ok=True)

# Create standard data/model directories
os.makedirs(f"{WORK_DIR}/data/maps", exist_ok=True)
os.makedirs(f"{WORK_DIR}/data/routes", exist_ok=True)
os.makedirs(f"{WORK_DIR}/models", exist_ok=True)

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

print(f"✅ Working directory: {os.getcwd()}")
print(f"✅ Ready to train/evaluate")
```

---

## Step 5: Train Base Models

> ⏱️ **Time:** ~15–30 minutes with GPU for 100K timesteps per topology

```python
# Cell 3: Train base models
!python 01_train_base.py --topologies 3 4 5 --timesteps 100000 --verbose 1
```

---

## Step 6: Run the Demo

```python
# Cell 4: Run evaluation
!python 02_evaluate_map.py --location "Silk Board Junction Bangalore"
```

---

## Handling Code Changes

### Option 1: Update via Git (Easiest)
```python
!cd /kaggle/working/ai-traffic && git pull
```

### Option 2: Update via Dataset
Upload a new version to Kaggle, click "Update Data Sources" in the notebook, and re-run **Cell 2**.

### Option 3: Quick Fixes (`%%writefile`)
```python
%%writefile src/dynamic_env.py
# paste updated contents here...
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `SumoNotFoundError` | Re-run Cell 1 (install step) |
| `No module named 'traci'` | Run `sys.path.insert(0, "/usr/share/sumo/tools")` |
| `No base model found` | Run Cell 3 (training) first |
| `FileNotFound` | Check if you copied the files correctly in Cell 2 |
