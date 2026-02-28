# **Comprehensive Technical Requirements Document: AI-Adaptive Traffic Signal Controller**

## **1. Project Overview & Objectives**

The objective is to engineer a Reinforcement Learning (RL) system capable of dynamically adjusting traffic light countdown timers at real-world intersections to minimize average vehicle wait times. The system will allow the user to select random real-world intersections (specifically targeting high-congestion Indian cities like Mumbai and Bangalore) and run simulations to demonstrate wait-time improvements against traditional static timers.

## **2. Architectural History & Design Decisions**

To understand the current architecture, the development agent must acknowledge the following discarded concepts and the rationale behind the final methodology:

* **Discarded Concept 1: Step-Based (Frame-by-Frame) Control using DQN.**
* *Initial Idea:* The AI makes split-second decisions to switch lights based on immediate pixel/queue data (similar to an Atari game).
* *Rejection Reason:* Real-world traffic requires countdown timers visible to drivers for safety. Instant, unpredictable switching is highly dangerous and impractical.


* **Discarded Concept 2: Cycle-Based Control with Discrete Actions.**
* *Initial Idea:* The AI selects from predefined "buckets" of timer plans (e.g., $40s/40s$ vs. $60s/20s$) at the start of a cycle.
* *Rejection Reason:* Handicaps the AI. If the optimal wait time requires exactly 43 seconds, predefined buckets cannot achieve optimal flow.


* **Discarded Concept 3: The "Universal Brain" (Zero-Padding) for all Intersections.**
* *Initial Idea:* One massive neural network built for an 8-way intersection, where unused lanes for smaller intersections are padded with zeros.
* *Rejection Reason:* Highly inefficient to train; the AI wastes compute learning to ignore "ghost" lanes.


* **Final Approved Architecture:** **Cycle-Based Control + Continuous Actions + Dynamic Instantiation.**
* The AI calculates precise, continuous timer values for an entire cycle in advance. The neural network is dynamically sized exactly to the number of signal phases ($N$) of the target intersection.



## **3. Core Reinforcement Learning Formulation (MDP)**

The Markov Decision Process (MDP) utilizes Proximal Policy Optimization (PPO) to handle continuous outputs.

* **State Space (Observation):** Gathered strictly at the start of every cycle $t$:
1. Queue length (total halted vehicles) per incoming lane.
2. Incoming traffic volume (vehicles/minute approaching).
3. The allocated timer values from the preceding cycle ($t-1$).


* **Action Space (Continuous):**
* The PPO agent outputs an array of floats $a_i \in [0, 1]$ corresponding to each signal phase $i$.
* These percentages are converted to exact seconds using constraint mapping:

$$T_i = T_{min} + a_i \times (T_{max} - T_{min})$$


* *Safety Parameters:* $T_{min}$ (e.g., 15s) ensures no lane is skipped. $T_{max}$ (e.g., 90s) caps the maximum wait limit.


* **Reward Function:**
* Calculated at the end of the executed cycle. The reward $R$ penalizes the AI based on the average wait time ($W_{avg}$) of all vehicles:

$$R = -W_{avg}$$





## **4. Real-World Map Ingestion & Traffic Generation**

To handle random, complex topologies (e.g., asymmetrical 4-way or 5-way junctions in Bangalore or Mumbai), the system requires a robust data pipeline.

* **Map Data:** OpenStreetMap (`.osm`) files will be ingested and converted into SUMO network files (`.net.xml`) using SUMO's `netconvert` utility. The script will automatically parse the file to determine $N$ (the number of signal phases).
* **Simulating Indian Traffic Chaos (Demand Generation):**
* Because `.osm` does not provide live traffic volume, the system must generate synthetic traffic.
* The system will utilize SUMO's `randomTrips.py` utility combined with weighted demand parameters (e.g., heavily biasing flow in specific directions to simulate peak-hour commuting patterns) to accurately stress-test the AI.



## **5. Execution Strategy: Transfer Learning & Fine-Tuning**

Training an RL agent from scratch for every new map click is computationally unviable. The system will employ Domain Randomization and Transfer Learning.

1. **The Base Models (Train Once):** The system will maintain pre-trained base models for standard geometries (a 3-way model, a 4-way model, etc.). These are trained on dummy networks with heavily randomized traffic flows so they learn general traffic management logic.
2. **Zero-Shot Execution:** For standard map intersections, the system loads the correct $N$-way Base Model and runs inference immediately, relying on the agent's generalized knowledge.
3. **Fine-Tuning (The "Mumbai/Bangalore" Protocol):** When a highly chaotic, asymmetrical intersection is imported, the script loads the Base Model but executes a short, localized training loop (e.g., 10,000 steps) specifically on that map's geometry before running the final evaluation.

## **6. Directory Structure & Workspace Division**

The pipeline is strictly divided into two distinct operational environments to separate heavy compute from lightweight inference.

```text
/traffic-rl-controller
│
├── /data
│   ├── /maps                # Raw .osm files and converted .net.xml files
│   └── /routes              # Generated high-volume traffic demand (.rou.xml)
│
├── /models                  # Pre-trained Base Models (e.g., base_3way.zip, base_4way.zip)
│
├── /src
│   ├── map_processor.py     # OSM to SUMO conversion and N-phase detection
│   ├── traffic_generator.py # Wraps randomTrips.py for weighted demand generation
│   ├── dynamic_env.py       # Gymnasium wrapper handling dynamic PPO topologies
│   └── traffic_baseline.py  # Static timer execution for benchmarking
│
├── 01_train_base.py         # Notebook/Script 1: Heavy RL training for Base Models
└── 02_evaluate_map.py       # Notebook/Script 2: Inference, map ingestion, fine-tuning, and testing

```

## **7. Technology Stack**

* **Language:** Python 3.10+
* **RL Algorithm/Brain:** `stable-baselines3` (PPO)
* **Simulation Engine:** Eclipse SUMO (Simulation of Urban MObility)
* **Environment Bridge:** `gymnasium` and `sumo-rl`
* **API Interface:** TraCI (Traffic Control Interface for SUMO)
* **Map Source:** OpenStreetMap (OSM)

