"""Showcase video for any environment dir (wrapper around simkit.showcase from the live plugin).

    .venv/bin/python harness/showcase.py <env_dir> [--out showcase.mp4] [--seconds 10] [--controller controller.py] [--name TITLE]
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugin/skills/sim-env-builder/scripts"))
runpy.run_module("simkit.showcase", run_name="__main__")
