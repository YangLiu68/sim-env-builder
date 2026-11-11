#!/usr/bin/env bash
# Print the path of a Python that has the simulation stack; create (or repair) a shared venv on first use.
#   PY=$(bash <skill>/scripts/setup_env.sh)
# Order: $SIMENV_PYTHON, shared venv ($SIMENV_HOME or ~/.cache/sim-env-builder/venv), created on demand.
# Missing packages in an existing venv are installed in place (no rebuild).
set -e
HOME_DIR="${SIMENV_HOME:-$HOME/.cache/sim-env-builder}"
VENV="$HOME_DIR/venv"
PKGS="mujoco>=3.2 trimesh numpy pillow scipy networkx imageio imageio-ffmpeg usd-core gymnasium"
CHECK="import mujoco, trimesh, PIL, imageio, numpy, scipy, networkx, gymnasium"
if [ -n "${SIMENV_PYTHON:-}" ] && "$SIMENV_PYTHON" -c "$CHECK" 2>/dev/null; then echo "$SIMENV_PYTHON"; exit 0; fi
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "$CHECK" 2>/dev/null; then echo "$VENV/bin/python"; exit 0; fi
mkdir -p "$HOME_DIR"
exec 9>"$HOME_DIR/.setup.lock"
if command -v flock >/dev/null 2>&1; then flock 9; fi
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "$CHECK" 2>/dev/null; then echo "$VENV/bin/python"; exit 0; fi
install() {
  if command -v uv >/dev/null 2>&1; then uv pip install --python "$VENV/bin/python" -q $PKGS >&2
  else "$VENV/bin/pip" install -q $PKGS >&2; fi
}
if [ -x "$VENV/bin/python" ]; then
  echo "adding missing packages to $VENV ..." >&2
else
  echo "creating $VENV (one-time, ~1 min)..." >&2
  if command -v uv >/dev/null 2>&1; then uv venv "$VENV" --python 3.12 -q >&2
  else python3 -m venv "$VENV" >&2 && "$VENV/bin/pip" install -q --upgrade pip >&2; fi
fi
install
"$VENV/bin/python" -c "$CHECK" >&2
echo "$VENV/bin/python"
