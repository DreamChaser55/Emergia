import pygame
import numpy as np
from numba import njit, prange
import random
import math
import ctypes

# --- Simulation Parameters ---
WIDTH, HEIGHT = 2560, 1440
FPS = 120

NUM_SPECIES = 8
TOTAL_PARTICLES = 3000

# Rock parameters
NUM_ROCKS = 15
MIN_ROCK_RADIUS = 10.0
MAX_ROCK_RADIUS = 40.0
ROCK_COLOR = (100, 100, 100)

# Physics parameters
FRICTION = 0.85          # Velocity multiplier per frame
R_REPEL = 20.0           # Distance for strong universal repulsion
R_INTERACT = 150.0       # Maximum distance particles can "see" each other
MAX_VELOCITY = 5.0       # Speed cap

BG_COLOR = (0, 0, 0)

# --- Grid Parameters ---
CELL_SIZE = R_INTERACT
GRID_WIDTH = int(math.ceil(WIDTH / CELL_SIZE))
GRID_HEIGHT = int(math.ceil(HEIGHT / CELL_SIZE))
NUM_CELLS = GRID_WIDTH * GRID_HEIGHT

# Precompute grid neighbors and offsets for toriodal wrapping
GRID_NEIGHBORS = np.zeros((GRID_HEIGHT, GRID_WIDTH, 9), dtype=np.int32)
GRID_OFFSETS_X = np.zeros((GRID_HEIGHT, GRID_WIDTH, 9), dtype=np.float32)
GRID_OFFSETS_Y = np.zeros((GRID_HEIGHT, GRID_WIDTH, 9), dtype=np.float32)

for _cy in range(GRID_HEIGHT):
    for _cx in range(GRID_WIDTH):
        _idx = 0
        for _d_cy in range(-1, 2):
            for _d_cx in range(-1, 2):
                _nx_raw = _cx + _d_cx
                _ny_raw = _cy + _d_cy
                
                _offset_x = 0.0
                if _nx_raw < 0:
                    _nx_cell = _nx_raw + GRID_WIDTH
                    _offset_x = -float(WIDTH)
                elif _nx_raw >= GRID_WIDTH:
                    _nx_cell = _nx_raw - GRID_WIDTH
                    _offset_x = float(WIDTH)
                else:
                    _nx_cell = _nx_raw
                    
                _offset_y = 0.0
                if _ny_raw < 0:
                    _ny_cell = _ny_raw + GRID_HEIGHT
                    _offset_y = -float(HEIGHT)
                elif _ny_raw >= GRID_HEIGHT:
                    _ny_cell = _ny_raw - GRID_HEIGHT
                    _offset_y = float(HEIGHT)
                else:
                    _ny_cell = _ny_raw
                    
                _cell_idx = _ny_cell * GRID_WIDTH + _nx_cell
                GRID_NEIGHBORS[_cy, _cx, _idx] = _cell_idx
                GRID_OFFSETS_X[_cy, _cx, _idx] = _offset_x
                GRID_OFFSETS_Y[_cy, _cx, _idx] = _offset_y
                _idx += 1

@njit
def build_grid(positions, head, next_arr, grid_width, grid_height, cell_size, num_particles):
    head[:] = -1
        
    for i in range(num_particles):
        x = positions[i, 0]
        y = positions[i, 1]
        
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        
        if cx < 0: cx = 0
        elif cx >= grid_width: cx = grid_width - 1
        if cy < 0: cy = 0
        elif cy >= grid_height: cy = grid_height - 1
            
        cell_idx = cy * grid_width + cx
        next_arr[i] = head[cell_idx]
        head[cell_idx] = i

@njit(parallel=True, fastmath=True)
def compute_forces_numba(positions, velocities, types, interaction_matrix,
                         rock_positions, rock_radii, rock_interactions, num_rocks, rock_max_dist_sq,
                         num_particles, width, height, r_repel, r_interact, r_interact_sq,
                         friction, max_velocity, head, next_arr, grid_width, grid_height, cell_size,
                         grid_neighbors, grid_offsets_x, grid_offsets_y):
    
    for i in prange(num_particles):
        fx = 0.0
        fy = 0.0
        p1_x = positions[i, 0]
        p1_y = positions[i, 1]
        t1 = types[i]
        
        cx = int(p1_x / cell_size)
        cy = int(p1_y / cell_size)
        if cx < 0: cx = 0
        elif cx >= grid_width: cx = grid_width - 1
        if cy < 0: cy = 0
        elif cy >= grid_height: cy = grid_height - 1
        
        for neighbor_idx in range(9):
            cell_idx = grid_neighbors[cy, cx, neighbor_idx]
            offset_x = grid_offsets_x[cy, cx, neighbor_idx]
            offset_y = grid_offsets_y[cy, cx, neighbor_idx]
            
            j = head[cell_idx]
            
            while j != -1:
                if i == j:
                    j = next_arr[j]
                    continue
                    
                p2_x = positions[j, 0]
                p2_y = positions[j, 1]
                t2 = types[j]
                
                dx = p2_x - p1_x + offset_x
                dy = p2_y - p1_y + offset_y
                    
                dist_sq = dx*dx + dy*dy
                
                if dist_sq > 0.0 and dist_sq < r_interact_sq:
                    dist = math.sqrt(dist_sq)
                    nx = dx / dist
                    ny = dy / dist
                    
                    if dist < r_repel:
                        force = -1.0 * (1.0 - (dist / r_repel))
                    else:
                        base_force = interaction_matrix[t1, t2]
                        mid_point = (r_repel + r_interact) / 2.0
                        force = base_force * (1.0 - abs(dist - mid_point) / (mid_point - r_repel))
                        
                    fx += force * nx
                    fy += force * ny
                
                j = next_arr[j]

        # Rock interactions
        for j in range(num_rocks):
            rx = rock_positions[j, 0]
            ry = rock_positions[j, 1]
            r_radius = rock_radii[j]
            
            dx = rx - p1_x
            dy = ry - p1_y
            
            # Toroidal wrapping for rocks
            if dx > width / 2.0:
                dx -= width
            elif dx < -width / 2.0:
                dx += width
                
            if dy > height / 2.0:
                dy -= height
            elif dy < -height / 2.0:
                dy += height
                
            dist_sq = dx*dx + dy*dy
            
            if dist_sq > 0.0 and dist_sq < rock_max_dist_sq[j]:
                dist = math.sqrt(dist_sq)
                dist_to_surface = dist - r_radius
                
                if dist_to_surface > 0.0 and dist_to_surface < r_interact:
                    nx = dx / dist
                    ny = dy / dist
                    
                    if dist_to_surface < r_repel:
                        force = -1.0 * (1.0 - (dist_to_surface / r_repel))
                    else:
                        base_force = rock_interactions[t1]
                        mid_point = (r_repel + r_interact) / 2.0
                        force = base_force * (1.0 - abs(dist_to_surface - mid_point) / (mid_point - r_repel))
                        
                    fx += force * nx
                    fy += force * ny
                
        vx = (velocities[i, 0] + fx) * friction
        vy = (velocities[i, 1] + fy) * friction
        
        speed = math.hypot(vx, vy)
        if speed > max_velocity:
            vx = (vx / speed) * max_velocity
            vy = (vy / speed) * max_velocity
            
        velocities[i, 0] = vx
        velocities[i, 1] = vy


@njit(parallel=True, fastmath=True)
def update_positions_numba(positions, velocities, rock_positions, rock_radii, num_rocks, num_particles, width, height):
    for i in prange(num_particles):
        px = positions[i, 0] + velocities[i, 0]
        py = positions[i, 1] + velocities[i, 1]
        vx = velocities[i, 0]
        vy = velocities[i, 1]
        
        for j in range(num_rocks):
            rx = rock_positions[j, 0]
            ry = rock_positions[j, 1]
            r_radius = rock_radii[j]
            
            # Vector from rock to particle
            dx = px - rx
            dy = py - ry
            
            # Toroidal wrapping for rocks
            if dx > width / 2.0:
                dx -= width
            elif dx < -width / 2.0:
                dx += width
                
            if dy > height / 2.0:
                dy -= height
            elif dy < -height / 2.0:
                dy += height
                
            dist_sq = dx*dx + dy*dy
            min_dist = r_radius + 2.0 # 2.0 is particle radius
            
            if dist_sq > 0.0 and dist_sq < min_dist * min_dist:
                dist = math.sqrt(dist_sq)
                nx = dx / dist
                ny = dy / dist
                
                # Hard stop - push out
                overlap = min_dist - dist
                px += nx * overlap
                py += ny * overlap
                
                # Reflect velocity
                dot_prod = vx * nx + vy * ny
                if dot_prod < 0:
                    vx -= 2.0 * dot_prod * nx
                    vy -= 2.0 * dot_prod * ny
                    
        px %= width
        py %= height
        
        positions[i, 0] = px
        positions[i, 1] = py
        velocities[i, 0] = vx
        velocities[i, 1] = vy

class Simulation:
    def __init__(self):
        self.randomize_rules()
        self.randomize_rock_rules()
        self.reset_rocks()
        self.reset_particles()

    def randomize_rules(self):
        """Generate random interaction matrix and colors."""
        # Matrix values between -1.0 and 1.0
        self.interaction_matrix = np.random.uniform(-1.0, 1.0, (NUM_SPECIES, NUM_SPECIES)).astype(np.float32)
        
        # Generate random colors for each species
        self.colors = []
        self.species_surfaces = []
        for _ in range(NUM_SPECIES):
            # Keep values above 50 to prevent dark colors
            r = random.randint(50, 255)
            g = random.randint(50, 255)
            b = random.randint(50, 255)
            color = (r, g, b)
            self.colors.append(color)
            
            surf = pygame.Surface((5, 5), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (2, 2), 2)
            self.species_surfaces.append(surf)

    def randomize_rock_rules(self):
        """Generate random rock interactions for each species."""
        self.rock_interactions = np.random.uniform(-1.0, 1.0, NUM_SPECIES).astype(np.float32)

    def reset_particles(self):
        """Initialize particle positions, velocities, and types."""
        self.velocities = np.zeros((TOTAL_PARTICLES, 2), dtype=np.float32)
        
        # Distribute species evenly
        self.types = np.random.randint(0, NUM_SPECIES, TOTAL_PARTICLES)
        
        self.positions = np.zeros((TOTAL_PARTICLES, 2), dtype=np.float32)
        invalid = np.ones(TOTAL_PARTICLES, dtype=bool)
        
        while np.any(invalid):
            num_invalid = np.sum(invalid)
            new_pos = (np.random.rand(num_invalid, 2) * [WIDTH, HEIGHT]).astype(np.float32)
            self.positions[invalid] = new_pos
            
            px = new_pos[:, 0, np.newaxis]
            py = new_pos[:, 1, np.newaxis]
            rx = self.rock_positions[:, 0]
            ry = self.rock_positions[:, 1]
            
            dx = rx - px
            dy = ry - py
            
            # Toroidal wrapping
            dx = dx - float(WIDTH) * np.round(dx / float(WIDTH))
            dy = dy - float(HEIGHT) * np.round(dy / float(HEIGHT))
            
            dist = np.hypot(dx, dy)
            inside = dist < self.rock_radii
            invalid[invalid] = np.any(inside, axis=1)

        self.grid_head = np.full(NUM_CELLS, -1, dtype=np.int32)
        self.grid_next = np.full(TOTAL_PARTICLES, -1, dtype=np.int32)

    def reset_rocks(self):
        """Initialize rocks with random positions and sizes."""
        self.rock_positions = (np.random.rand(NUM_ROCKS, 2) * [WIDTH, HEIGHT]).astype(np.float32)
        self.rock_radii = np.random.uniform(MIN_ROCK_RADIUS, MAX_ROCK_RADIUS, NUM_ROCKS).astype(np.float32)
        self.rock_max_dist_sq = ((R_INTERACT + self.rock_radii)**2).astype(np.float32)

    def apply_forces(self):
        """Computation of particle forces using numba."""
        build_grid(self.positions, self.grid_head, self.grid_next, GRID_WIDTH, GRID_HEIGHT, np.float32(CELL_SIZE), TOTAL_PARTICLES)
        compute_forces_numba(
            self.positions, self.velocities, self.types, self.interaction_matrix,
            self.rock_positions, self.rock_radii, self.rock_interactions, NUM_ROCKS, self.rock_max_dist_sq,
            TOTAL_PARTICLES, np.float32(WIDTH), np.float32(HEIGHT), np.float32(R_REPEL), np.float32(R_INTERACT), np.float32(R_INTERACT * R_INTERACT),
            np.float32(FRICTION), np.float32(MAX_VELOCITY), self.grid_head, self.grid_next, GRID_WIDTH, GRID_HEIGHT, np.float32(CELL_SIZE),
            GRID_NEIGHBORS, GRID_OFFSETS_X, GRID_OFFSETS_Y
        )

    def update_positions(self):
        """Move particles and apply wrapping boundaries."""
        update_positions_numba(
            self.positions, self.velocities, 
            self.rock_positions, self.rock_radii, 
            NUM_ROCKS, TOTAL_PARTICLES, 
            np.float32(WIDTH), np.float32(HEIGHT)
        )

    def draw(self, screen):
        """Render the particles and rocks."""
        screen.fill(BG_COLOR)
        
        # Draw rocks
        for i in range(NUM_ROCKS):
            x, y = int(self.rock_positions[i, 0]), int(self.rock_positions[i, 1])
            r = int(self.rock_radii[i])
            
            # Base rock
            pygame.draw.circle(screen, ROCK_COLOR, (x, y), r)
            
            # Edge wrapping bounding box checks
            wrap_x = []
            if x - r < 0: wrap_x.append(WIDTH)
            elif x + r >= WIDTH: wrap_x.append(-WIDTH)
            
            wrap_y = []
            if y - r < 0: wrap_y.append(HEIGHT)
            elif y + r >= HEIGHT: wrap_y.append(-HEIGHT)
            
            for dx in wrap_x:
                pygame.draw.circle(screen, ROCK_COLOR, (x + dx, y), r)
            for dy in wrap_y:
                pygame.draw.circle(screen, ROCK_COLOR, (x, y + dy), r)
            for dx in wrap_x:
                for dy in wrap_y:
                    pygame.draw.circle(screen, ROCK_COLOR, (x + dx, y + dy), r)
            
        # Fast rendering using blits
        int_positions = self.positions.astype(np.int32) - 2
        blits = [(self.species_surfaces[self.types[i]], (int_positions[i, 0], int_positions[i, 1])) for i in range(TOTAL_PARTICLES)]
        screen.blits(blits)

def main():
    try:
        ctypes.windll.user32.SetProcessDPIAware() # type: ignore
    except AttributeError:
        pass
    pygame.init()
    screen = pygame.display.set_mode(size=(WIDTH, HEIGHT))
    pygame.display.set_caption(f"Particle Life - {TOTAL_PARTICLES} Particles, {NUM_SPECIES} Species")
    clock = pygame.time.Clock()
    
    sim = Simulation()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    sim.reset_particles()
                elif event.key == pygame.K_r:
                    sim.randomize_rules()
                elif event.key == pygame.K_e:
                    sim.reset_rocks()
                    sim.randomize_rock_rules()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        sim.apply_forces()
        sim.update_positions()
        sim.draw(screen)

        # UI text
        font = pygame.font.SysFont(None, 48)
        text = font.render(f"SPACE: Reset positions | R: Randomize rules | E: Randomize environment | FPS: {int(clock.get_fps())}", True, (200, 200, 200))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
