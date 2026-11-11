"""Hardware/software facts that decide MuJoCo vs Isaac Sim. Prints JSON.

    python check_platform.py
"""
import json
import os
import platform
import shutil
import subprocess


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


info = {"os": platform.system(), "arch": platform.machine(), "python": platform.python_version()}
if info["os"] == "Darwin":
    info["cpu"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem = sh(["sysctl", "-n", "hw.memsize"])
    info["ram_gb"] = round(int(mem) / 1e9, 1) if mem.isdigit() else None
else:
    info["cpu"] = platform.processor()
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        info["ram_gb"] = None
gpus = []
if shutil.which("nvidia-smi"):
    out = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    gpus = [g.strip() for g in out.splitlines() if g.strip()]
info["nvidia_gpus"] = gpus
rtx = any("RTX" in g or "A100" in g or "H100" in g or "L40" in g or "A6000" in g for g in gpus)
isaac_paths = [p for p in (os.environ.get("ISAAC_SIM_PATH", ""), os.path.expanduser("~/isaacsim"),
                           os.path.expanduser("~/.local/share/ov/pkg")) if p and os.path.exists(p)]
try:
    import importlib.util
    has_isaac_pkg = importlib.util.find_spec("isaacsim") is not None
except Exception:  # noqa: BLE001
    has_isaac_pkg = False
info["isaac_sim_found"] = bool(isaac_paths or has_isaac_pkg)
info["isaac_sim_paths"] = isaac_paths
try:
    import mujoco
    info["mujoco"] = mujoco.__version__
except Exception:  # noqa: BLE001
    info["mujoco"] = None
info["isaac_runnable_here"] = bool(info["os"] in ("Linux", "Windows") and rtx)
info["notes"] = []
if info["os"] == "Darwin":
    info["notes"].append("macOS: Isaac Sim cannot run here (needs an NVIDIA RTX GPU on Linux/Windows). MuJoCo runs natively.")
elif not gpus:
    info["notes"].append("No NVIDIA GPU detected: Isaac Sim cannot run here; MuJoCo (CPU) can.")
elif not rtx:
    info["notes"].append("NVIDIA GPU found but not an RTX/datacenter card: Isaac Sim needs RTX.")
if info["isaac_runnable_here"] and not info["isaac_sim_found"]:
    info["notes"].append("RTX GPU present but Isaac Sim not found: install via pip (isaacsim) or Omniverse launcher.")
print(json.dumps(info, indent=1))
