"""Quick geometry facts about a GLB so an agent can size and orient it without guessing."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def _r(v, n=3):
    return [round(float(x), n) for x in v]


def inspect_glb(path_or_bytes: str | Path | bytes) -> dict:
    """Bounds, size, triangle/material counts, textures, animation and skin flags of a GLB.

    Sizes are reported in the file's own units (glTF says metres, but many models use other
    scales) in both frames: glTF (+Y up) and Z-up (MuJoCo / Isaac default: x'=x, y'=-z, z'=y).
    """
    import trimesh

    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene", process=False)
    meshes = []
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        g = scene.geometry.get(gname)
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            meshes.append((g, tf))
    if not meshes:
        return {"ok": False, "error": "no triangle meshes in file"}
    pts = []
    tris = 0
    materials, textured, vertex_colours = set(), False, False
    for g, tf in meshes:
        v = np.asarray(g.vertices, np.float64)
        pts.append(v @ tf[:3, :3].T + tf[:3, 3])
        tris += len(g.faces)
        vis = g.visual
        if getattr(vis, "kind", None) == "vertex":
            vertex_colours = True
        mat = getattr(vis, "material", None)
        if mat is not None:
            materials.add(getattr(mat, "name", None) or id(mat))
            if getattr(mat, "baseColorTexture", None) is not None or getattr(mat, "image", None) is not None:
                textured = True
    p = np.concatenate(pts)
    lo, hi = p.min(axis=0), p.max(axis=0)
    ext = hi - lo
    ext_zup = np.array([ext[0], ext[2], ext[1]])
    # Which way is "front"? glTF convention is +Z, but the longest horizontal axis is a better
    # hint for vehicles/furniture; report both and let renders decide.
    longest = "x" if ext_zup[0] >= ext_zup[1] else "y"
    gltf = _gltf_json(data)
    anims = [a.get("name", f"anim{i}") for i, a in enumerate(gltf.get("animations", []))]
    return {
        "ok": True,
        "triangles": int(tris),
        "mesh_count": len(meshes),
        "material_count": len(materials),
        "textured": textured,
        "vertex_colours": vertex_colours,
        "skinned": bool(gltf.get("skins")),
        "animations": anims,
        "bounds_gltf_yup": {"min": _r(lo), "max": _r(hi), "size_xyz": _r(ext)},
        "size_zup_xyz": _r(ext_zup),
        "longest_horizontal_axis_zup": longest,
        "height_zup": round(float(ext_zup[2]), 3),
        "notes": (
            "Sizes are in file units: rescale to real-world size (e.g. car length 4.5 m, "
            "person 1.75 m, chair seat 0.45 m). glTF is Y-up; rotate +90 deg about X for Z-up. "
            "Facing direction is not standardised on Poly Pizza: render to confirm."
        ),
    }


def _gltf_json(data: bytes) -> dict:
    import json

    if data[:4] != b"glTF":
        return {}
    pos = 12
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "little")
        ctype = data[pos + 4 : pos + 8]
        if ctype == b"JSON":
            return json.loads(data[pos + 8 : pos + 8 + length])
        pos += 8 + length
    return {}
