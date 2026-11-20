"""simkit: helpers for building and smoke-testing MuJoCo environments (see the sim-env-builder skill).

Modules: assets (GLB -> MuJoCo), procedural (fallback props, terrain), scene (MJCF builder),
render (views, contact sheets, filmstrips), physics_exam (physics examiner), layout (placement
checks), smoke (runs everything), usd_export (Isaac Sim stage from a MuJoCo model), world
(per-step environment dynamics), flood (water on land: depth field, wading physics, manhole
fountains, storm look), reference (keyframes from the user's photos/videos, side-by-side compare).
"""
__version__ = "0.2.0"
