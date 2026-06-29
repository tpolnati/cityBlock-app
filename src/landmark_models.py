"""Detailed landmark 3D-model resolver + stitching for LuminaMaps.

OSM extrusion (even with ``building:part``) only gives blocky massing. For a
landmark to be *recognisable* in the print, we fetch a genuinely detailed model
and stitch it into the city tile in place of the boxy extrusion.

Tiered resolver (best available wins, always degrades gracefully):

1. **Wikidata ``P4896`` -> Wikimedia Commons STL.** OSM tags landmarks with
   ``wikidata=Q…``; that entity often links a CC-licensed, print-ready STL on
   Commons (Eiffel Tower, Statue of Liberty, Parthenon…). Free, no API key.
2. **Sketchfab.** Far broader coverage (incl. skyscrapers like the Burj
   Khalifa). Requires a free user API token; we only download CC-licensed,
   downloadable models. glTF/glb is merged to a single printable geometry.
3. *(no model found)* -> caller falls back to procedural OSM geometry.

A fetched model is scaled to the landmark's printable target height, oriented
Z-up, optionally rotated, and translated to the landmark's position on the tile.
Everything network-bound is cached so re-exports don't re-download.
"""

from __future__ import annotations

import io

import numpy as np
import streamlit as st
import trimesh

# Wikimedia asks for a descriptive User-Agent on API/file requests.
USER_AGENT = "LuminaMaps/1.0 (3D map manufacturing tool; contact: operator)"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
COMMONS_FILEPATH_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/{fname}"
SKETCHFAB_SEARCH_URL = "https://api.sketchfab.com/v3/search"
SKETCHFAB_DOWNLOAD_URL = "https://api.sketchfab.com/v3/models/{uid}/download"

# Sketchfab license slugs we consider acceptable for commercial product use.
# (CC-BY / CC-BY-SA require attribution; CC0 is public domain. SA forbids
# closed redistribution of the *model file* but not selling a physical print.)
SKETCHFAB_OK_LICENSES = {"cc0", "cc-by", "cc-by-sa"}

_REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Wikidata / Commons
# ---------------------------------------------------------------------------
def _session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


@st.cache_data(show_spinner=False)
def wikidata_model_filename(qid: str) -> str | None:
    """Return the Commons filename in a Wikidata entity's P4896 claim, or None."""
    if not qid:
        return None
    try:
        resp = _session().get(WIKIDATA_ENTITY_URL.format(qid=qid), timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        entity = resp.json()["entities"][qid]
        claims = entity.get("claims", {}).get("P4896", [])
        for claim in claims:
            value = claim["mainsnak"].get("datavalue", {}).get("value")
            if value:
                return value  # a Commons filename string, e.g. "EiffelTower fixed.stl"
    except Exception:  # noqa: BLE001 - any failure -> no model
        return None
    return None


@st.cache_data(show_spinner=False)
def download_commons_stl(fname: str) -> bytes | None:
    """Download a Commons STL file by name, returning raw bytes (or None)."""
    if not fname:
        return None
    try:
        url = COMMONS_FILEPATH_URL.format(fname=fname.replace(" ", "_"))
        resp = _session().get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Sketchfab
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def sketchfab_find_downloadable(name: str, token: str) -> str | None:
    """Search Sketchfab for a downloadable, CC-licensed model; return its uid."""
    if not name or not token:
        return None
    try:
        params = {
            "type": "models", "q": name, "downloadable": "true",
            "archives_flavours": "false", "count": 10, "sort_by": "-relevance",
        }
        resp = _session().get(
            SKETCHFAB_SEARCH_URL, params=params,
            headers={"Authorization": f"Token {token}"}, timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        for item in resp.json().get("results", []):
            slug = (item.get("license") or {}).get("slug", "")
            if slug in SKETCHFAB_OK_LICENSES:
                return item.get("uid")
    except Exception:  # noqa: BLE001
        return None
    return None


@st.cache_data(show_spinner=False)
def sketchfab_download_glb(uid: str, token: str) -> bytes | None:
    """Download a Sketchfab model's glb via the download endpoint (temp URL)."""
    if not uid or not token:
        return None
    try:
        meta = _session().get(
            SKETCHFAB_DOWNLOAD_URL.format(uid=uid),
            headers={"Authorization": f"Token {token}"}, timeout=_REQUEST_TIMEOUT,
        )
        meta.raise_for_status()
        gltf = meta.json().get("gltf") or {}
        url = gltf.get("url")
        if not url:
            return None
        blob = _session().get(url, timeout=_REQUEST_TIMEOUT)
        blob.raise_for_status()
        return blob.content
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Bytes -> single printable trimesh
# ---------------------------------------------------------------------------
def _scene_to_mesh(loaded) -> trimesh.Trimesh | None:
    """Collapse a loaded Scene/Geometry into one watertight-ish Trimesh."""
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        # Apply the scene graph transforms then merge to one body.
        mesh = trimesh.util.concatenate(geoms)
    else:
        return None
    if mesh.is_empty or len(mesh.faces) == 0:
        return None
    return mesh


def load_mesh_bytes(data: bytes, file_type: str) -> trimesh.Trimesh | None:
    """Parse downloaded model bytes (stl/glb/gltf) into a single Trimesh."""
    if not data:
        return None
    try:
        loaded = trimesh.load(io.BytesIO(data), file_type=file_type, force="scene")
        return _scene_to_mesh(loaded)
    except Exception:  # noqa: BLE001
        try:
            loaded = trimesh.load(io.BytesIO(data), file_type=file_type)
            return _scene_to_mesh(loaded)
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Resolver: identity -> detailed mesh (cached as a resource)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def fetch_landmark_mesh(qid: str | None, name: str | None, sketchfab_token: str | None):
    """Best-effort detailed mesh for a landmark, or None. Tries Wikidata then Sketchfab."""
    # 1) Wikidata P4896 -> Commons STL
    if qid:
        fname = wikidata_model_filename(qid)
        if fname and fname.lower().endswith(".stl"):
            mesh = load_mesh_bytes(download_commons_stl(fname), "stl")
            if mesh is not None:
                return mesh, f"Wikidata/Commons ({fname})"

    # 2) Sketchfab (needs a token)
    if name and sketchfab_token:
        uid = sketchfab_find_downloadable(name, sketchfab_token)
        if uid:
            mesh = load_mesh_bytes(sketchfab_download_glb(uid, sketchfab_token), "glb")
            if mesh is not None:
                return mesh, f"Sketchfab ({uid})"

    return None, None


# ---------------------------------------------------------------------------
# Stitching: fit a fetched model onto the tile
# ---------------------------------------------------------------------------
def fit_model(
    model: trimesh.Trimesh,
    *,
    centroid_mm: tuple[float, float],
    target_height_mm: float,
    base_top_mm: float,
    embed_mm: float = 0.2,
    up_axis: str = "z",
    rotate_deg: float = 0.0,
    max_footprint_mm: float | None = None,
) -> trimesh.Trimesh | None:
    """Scale/orient/place a downloaded model so it sits on the tile as a landmark.

    The model is oriented Z-up, uniformly scaled so its height equals
    ``target_height_mm`` (preserving its proportions), optionally rotated about Z,
    then dropped onto the base top at ``centroid_mm`` with a slight embed so it
    welds to the plate. If scaling to height would make the footprint wildly
    larger than the real one, it is clamped down to ``max_footprint_mm``.
    """
    if model is None or model.is_empty:
        return None
    m = model.copy()

    # Orient so the building's vertical axis is +Z.
    if up_axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif up_axis == "x":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    ext = m.extents
    if ext[2] <= 0:
        return None

    # Uniform scale to the requested print height.
    scale = target_height_mm / ext[2]
    # Don't let a tall-but-skinny model balloon past the real footprint.
    if max_footprint_mm and max_footprint_mm > 0:
        footprint_after = max(ext[0], ext[1]) * scale
        if footprint_after > max_footprint_mm:
            scale = max_footprint_mm / max(ext[0], ext[1])
    m.apply_scale(scale)

    if rotate_deg:
        m.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(rotate_deg), [0, 0, 1], m.centroid)
        )

    # Centre in XY, set the model's base to z=0, then place on the tile.
    minb, maxb = m.bounds
    cx = (minb[0] + maxb[0]) / 2.0
    cy = (minb[1] + maxb[1]) / 2.0
    m.apply_translation([-cx, -cy, -minb[2]])
    m.apply_translation([centroid_mm[0], centroid_mm[1], base_top_mm - embed_mm])

    # Best-effort cleanup; downloaded models are usually print-ready already.
    try:
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.fix_normals()
    except Exception:  # noqa: BLE001
        pass
    return m
