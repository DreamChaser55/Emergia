# GPU Migration Plan: NVIDIA Warp & Zero-Copy OpenGL

This document outlines the architecture and step-by-step plan to migrate the Emergia simulation from CPU-bound Numba to **NVIDIA Warp** for physics calculations, and a zero-copy OpenGL pipeline for rendering.

## Why NVIDIA Warp + OpenGL?
- **NVIDIA Warp** compiles Python syntax directly to CUDA, offering massive parallel performance for physics simulations. It natively includes spatial data structures like `wp.HashGrid`, saving us from writing complex GPU spatial partitioning from scratch.
- **Zero-Copy OpenGL** uses CUDA-OpenGL interop. Instead of copying particle positions from the GPU to the CPU just to draw them, we allocate OpenGL Vertex Buffer Objects (VBOs) directly in GPU VRAM, map them to CUDA, and let Warp write physics results *directly* into the rendering buffers.

---

## Phase 1: Environment & Dependencies

We will move away from `pygame-ce` and `numba`, transitioning to a modern GPU tech stack.

**New Dependencies:**
1. `warp-lang`: For GPU physics computation.
2. `cuda-python`: For the low-level CUDA-OpenGL memory sharing API (`cudaGraphicsGLRegisterBuffer`).
3. `PyOpenGL`: For OpenGL rendering commands.
4. `glfw`: For creating the window and handling keyboard inputs (replaces Pygame's windowing).

---

## Phase 2: Physics Engine Migration (NVIDIA Warp)

We will port the core logic into Warp, taking advantage of its built-in features.

1. **Data Structures:** 
   Particle positions, velocities, and types will be defined as Warp arrays (`wp.array(dtype=wp.vec2)`).
2. **Spatial Partitioning:** 
   We will completely delete the manual linked-list grid code. Warp provides `wp.HashGrid(dim_x, dim_y, dim_z)`. Every frame, we simply call `grid.build(positions, radius=R_INTERACT)` on the GPU.
3. **Physics Kernels:**
   The `compute_forces` and `update_positions` Numba loops will become `@wp.kernel` functions. Inside the forces kernel, neighbor search becomes a simple built-in iterator:
   ```python
   for neighbor_id in wp.hash_grid_query(grid, particle_pos, R_INTERACT):
       # apply attraction/repulsion
   ```
4. **Toroidal Space & Rocks:**
   Modulo math for screen wrapping and static rock collisions will be ported as inline `@wp.func` math routines.

---

## Phase 3: Zero-Copy Memory Interop

This is the bridge between physics and rendering. We must trick OpenGL and CUDA into sharing the exact same VRAM addresses.

1. **Create OpenGL VBOs:**
   Initialize empty buffers in OpenGL for `positions` (vec2) and `colors` (vec3) sized for `TOTAL_PARTICLES`.
2. **CUDA Registration:**
   Use `cuda-python` to register these VBOs with CUDA (`cudaGraphicsGLRegisterBuffer`).
3. **Wrap for Warp:**
   Map the resources to get raw CUDA memory pointers. We then tell Warp to treat these pointers as native arrays using `wp.array(ptr=mapped_pointer, dtype=wp.vec2, shape=TOTAL_PARTICLES)`.
4. **Result:**
   When Warp updates `positions[i]` in the physics kernel, the OpenGL VBO is instantly updated because they are the *exact same memory block*. No data copies are performed.

---

## Phase 4: Rendering Pipeline

We will write a minimal shader pipeline to replace Pygame's `surface.blits()`.

1. **Shaders:**
   * **Vertex Shader:** Reads the mapped VBOs and transforms the 2D pixel coordinates (e.g., 2560x1440) into OpenGL's Normalized Device Coordinates (-1.0 to 1.0).
   * **Fragment Shader:** Draws the particles using `GL_POINTS`. We will use a mathematical distance check from the center of the point in the fragment shader to discard corners and render perfect, anti-aliased circles.
2. **Static Environment:**
   Rocks will be drawn using a secondary static VBO or simple uniform arrays since they do not change frame-to-frame.
3. **Main Loop:**
   ```python
   while not glfw.window_should_close(window):
       # 1. Handle GLFW Keyboard Inputs (Space, R, E)
       
       # 2. Map VBOs for CUDA writing
       cudaGraphicsMapResources(...)
       
       # 3. Physics Step (Warp)
       grid.build(...)
       wp.launch(compute_forces, ...)
       wp.launch(update_positions, ...)
       
       # 4. Unmap VBOs (Return control to OpenGL)
       cudaGraphicsUnmapResources(...)
       
       # 5. Render Step (OpenGL)
       glClear(...)
       glDrawArrays(GL_POINTS, 0, TOTAL_PARTICLES)
       glfwSwapBuffers(window)
   ```