# MuJoCo or Isaac Sim?

## 1. Hard gate (run `check_platform.py`)

| Fact | Consequence |
|---|---|
| macOS, or no NVIDIA GPU, or non-RTX NVIDIA | Isaac Sim cannot run here. MuJoCo is the only locally testable backend. |
| Linux/Windows + RTX GPU + Isaac Sim installed | Both are possible. Decide on the merits below. |
| RTX present, Isaac not installed | Isaac possible after install (`pip install isaacsim[all]` / Omniverse); ask before installing ~15 GB. |

## 2. Merits

| Need | Prefer | Why |
|---|---|---|
| Contact-rich manipulation (grasping, in-hand, insertion), legged locomotion research | MuJoCo | Accurate, well-studied soft-contact model; fast; deterministic |
| Fast iteration on a laptop / CPU / macOS / CI | MuJoCo | pip install, starts in ms, renders offscreen anywhere |
| Model-based control, MPC, derivatives, system ID | MuJoCo | Analytical derivatives, MJX (JAX), clean state access |
| Massive GPU-parallel RL (thousands of envs) | Isaac Lab, or MuJoCo MJX / Playground / Warp | Both scale; Isaac Lab if the team already uses it |
| Photoreal camera data, synthetic datasets, domain randomisation of materials/lighting | Isaac Sim | RTX path tracing, Replicator, annotators (segmentation, depth, bboxes) |
| Realistic RTX lidar / radar / camera sensor models | Isaac Sim | Physically based sensor simulation |
| Huge scenes (factories, warehouses, cities) authored in USD / Omniverse | Isaac Sim | USD composition, instancing, streaming |
| Cloth, soft bodies, cables | Either (MuJoCo flex/composite; PhysX deformables/particles) | MuJoCo is easier to test here |
| Fluids with surface interaction (floating boats, waves) | Neither natively; MuJoCo + per-step buoyancy forces is simplest | See robots.md, marine |
| Self-driving with camera-only perception at scale | Isaac for photorealism; MuJoCo for dynamics/RL prototyping | Ask which matters more |

## 3. Decide

- Default to **MuJoCo** unless a row above clearly favours Isaac AND the user can run it.
- If the request names a backend, use it, and mention the trade-off briefly if it is a poor fit.
- Isaac preferred but not runnable here -> recommend in the brief: "build + fully test in MuJoCo
  now; export USD (+ Isaac smoke script) for your GPU machine". Let the user choose.
- Write the decision and the one-line reason into SPEC.md.
