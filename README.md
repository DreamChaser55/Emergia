# Emergia

A fast 2D artificial life simulation written in Python. This project simulates emergent behaviors—such as flocking, clustering, and cellular-like structures—from simple rules of attraction and repulsion between different "species" of particles. 

## Features

- **High Performance:** Utilizes `numba` JIT compilation with parallel processing (`prange`) to distribute physics calculations across multiple CPU cores, simulating thousands of interacting particles at smooth framerates.
- **Spatial Partitioning (Grid System):** Implements a spatial grid to optimize particle interactions, reducing the computational complexity from O(N²) to nearly O(N) by only checking neighboring cells.
- **Real-time Rendering:** Fast batch rendering using `pygame-ce` (Community Edition) via `blits()`, along with Windows DPI-awareness for crisp visuals on high-resolution displays.
- **Emergent Ecosystems:** Randomized interaction matrices ensure every run can produce completely unique patterns and behaviors.
- **Environmental Obstacles (Rocks):** Static rocks of varying sizes provide physical obstacles. Particles dynamically interact with rocks based on unique rules and bounce off them.
- **Toroidal Space:** Particles and rocks wrap around screen edges, creating a continuous infinite environment.

## Installation

Ensure you have Python 3 installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: The project relies on `pygame-ce`, `numpy`, and `numba`.*

## Usage

To start the simulation, run the following command in your terminal:

```bash
python alife.py
```

## Controls

The simulation features an interactive on-screen display and the following keyboard controls:

- **`R`**: Randomize the rules. This generates a new interaction matrix and new colors, effectively spawning a new, unique ecosystem.
- **`E`**: Randomize the environment. This resets the rocks with random positions and sizes, and generates new rock interaction rules for the particles.
- **`SPACE`**: Reset particles. This redistributes the particle positions uniformly across the screen while keeping the current interaction rules intact.
- **`ESC`**: Quit the simulation.

## How It Works

The simulation operates on a few simple physics principles applied to every particle:
1. **Universal Repulsion:** All particles strongly repel each other if they get extremely close, acting as collision avoidance.
2. **Species Interaction:** At medium distances, particles interact based on their "species" (color). These interactions can be attractive or repulsive and are determined by a hidden, randomized interaction matrix.
3. **Rock Interactions & Collisions:** Particles have hard collisions with static rocks. At medium distances, particles also interact with rocks based on randomized attraction/repulsion rules.
4. **Friction & Speed Limits:** A constant friction factor naturally slows particles down over time, and a maximum velocity cap ensures the system remains stable.
5. **Spatial Partitioning:** The environment is divided into a grid where cell sizes match the maximum interaction radius. This allows particles to efficiently ignore anything outside their immediate 9-cell neighborhood, massively speeding up calculations.

## Customization

You can easily tweak the parameters to observe how the simulation behaves under different conditions. Open `alife.py` and modify the constants at the top of the file:

- `TOTAL_PARTICLES`: Increase or decrease the simulation density (default is 3000).
- `NUM_SPECIES`: Change the number of distinct particle types (default is 8).
- `NUM_ROCKS` / `MIN_ROCK_RADIUS` / `MAX_ROCK_RADIUS`: Adjust the count and size constraints of the static environmental rocks.
- `R_INTERACT` & `R_REPEL`: Change how far particles can "see" each other and their collision distance.
- `MAX_VELOCITY` & `FRICTION`: Alter the kinetic properties and movement speed of the particles.
