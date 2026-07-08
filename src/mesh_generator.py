"""3D mesh generation + STL export for LuminaMaps (Step 2).

All ``trimesh`` work lives here. Consumes millimetre-space ``TileGeom`` objects
from ``geometry_processor`` and produces watertight, manifold-friendly STL
meshes, one per printed tile.

Construction per tile
----------------------
* Solid rectangular **base plate**: bottom flush at Z=0, top at Z=base_mm.
* **Buildings**: extruded prisms, slightly embedded into the base for a solid
  weld, rising from the base top.
* **Parks**: a thin raised pad on the base top (subtle land texture).
* **Water / roads**: carved as recessed channels into the base top via boolean
  difference (the "negative space"), with a graceful fallback if the boolean
  engine struggles on pathological geometry.
* Everything is centred at X=0, Y=0 and the bottom sits flush at Z=0.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

from src import landmark_models, roofs
from src.geometry_processor import LandmarkRecord, TileGeom
from src.terrain import TileTerrain

# Depth (mm) of carved water/road channels, capped relative to base thickness.
CARVE_MAX_MM = 1.2
# How far (mm) buildings/parks sink into the base top to guarantee a solid weld.
EMBED_MM = 0.2
# Thin raised park layer thickness (mm).
PARK_MAX_MM = 0.8
# Raised-road ribbon thickness (mm).
ROAD_RAISE_MM = 0.6
# Accurate bridge: deck clearance above what it crosses, deck thickness, spacing
# between support piers along the span, and pier column footprint (mm).
BRIDGE_CLEARANCE_MM = 2.2
BRIDGE_DECK_MM = 0.9
BRIDGE_PIER_SPACING_MM = 16.0
BRIDGE_PIER_SIZE_MM = 1.6
# Integrated print supports: no downward-facing deck span is left unsupported by
# more than this, filled with thin snippable posts (small cubes) to the surface.
SUPPORT_MAX_SPAN_MM = 8.0
SUPPORT_POST_MM = 0.9
# Bridge superstructure (rendered by OSM bridge:structure) — towers, cables, arch.
BRIDGE_TOWER_RISE_MM = 9.0       # tower height above the deck
BRIDGE_TOWER_SIZE_MM = 2.0       # tower footprint (along the deck)
BRIDGE_CABLE_R_MM = 0.5          # main suspension cable radius
BRIDGE_HANGER_R_MM = 0.3         # vertical hanger / spandrel radius
BRIDGE_ARCH_R_MM = 0.9           # arch rib radius
SUSPENSION_STRUCTURES = {"suspension", "cable_stayed", "cable-stayed", "cablestayed"}
ARCH_STRUCTURES = {"arch", "tied_arch", "tied-arch", "through_arch", "viaduct_arch"}


@dataclass
class TileMesh:
    """A finished tile mesh plus a few stats for the UI."""

    name: str
    mesh: trimesh.Trimesh
    # Human-readable note per landmark that got a detailed downloaded model.
    model_sources: list[str] = field(default_factory=list)

    @property
    def triangles(self) -> int:
        return len(self.mesh.faces)

    @property
    def watertight(self) -> bool:
        return bool(self.mesh.is_watertight)

    @property
    def dims_mm(self) -> tuple[float, float, float]:
        ext = self.mesh.extents
        return (float(ext[0]), float(ext[1]), float(ext[2]))

    def to_stl_bytes(self) -> bytes:
        return self.mesh.export(file_type="stl")


# ---------------------------------------------------------------------------
# Polygon -> mesh helpers
# ---------------------------------------------------------------------------
def _iter_polys(geom):
    """Yield individual shapely Polygons from any (Multi)Polygon / collection."""
    if geom is None or geom.is_empty:
        return
    gtype = geom.geom_type
    if gtype == "Polygon":
        yield geom
    elif gtype in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            yield from _iter_polys(g)


def _extrude(geom, height: float, z0: float = 0.0):
    """Extrude a (Multi)Polygon to a given height, optionally lifted to ``z0``.

    Returns a single concatenated ``Trimesh`` or ``None``. Individual polygon
    failures are skipped so one bad ring can't sink the tile.
    """
    if height <= 0:
        return None
    meshes = []
    for poly in _iter_polys(geom):
        if poly.area <= 0:
            continue
        poly = poly.buffer(0)            # heal self-touching rings
        for clean in _iter_polys(poly):
            if clean.area <= 0:
                continue
            try:
                m = trimesh.creation.extrude_polygon(clean, height)
            except Exception:            # noqa: BLE001 - degenerate triangulation
                continue
            if z0:
                m.apply_translation([0.0, 0.0, z0])
            meshes.append(m)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _base_plate(width_mm: float, height_mm: float, thickness_mm: float) -> trimesh.Trimesh:
    """Solid centred base box: bottom at Z=0, top at Z=thickness."""
    plate = trimesh.creation.box(extents=[width_mm, height_mm, thickness_mm])
    plate.apply_translation([0.0, 0.0, thickness_mm / 2.0])
    return plate


def _carve(base: trimesh.Trimesh, cutter_geom, depth: float, top_z: float):
    """Subtract recessed channels (cutter extruded down from ``top_z``) from base.

    Returns (mesh, carved?). On any boolean failure, returns the untouched base
    so the tile still exports.
    """
    if cutter_geom is None or depth <= 0:
        return base, False
    cutter = _extrude(cutter_geom, depth)
    if cutter is None:
        return base, False
    cutter.apply_translation([0.0, 0.0, top_z - depth])
    try:
        result = trimesh.boolean.difference([base, cutter])
        if result is None or result.is_empty or len(result.faces) == 0:
            return base, False
        return result, True
    except Exception:                    # noqa: BLE001 - boolean engine hiccup
        return base, False


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
def _terrain_solid(tt: TileTerrain) -> trimesh.Trimesh | None:
    """Build a watertight terrain slab: heightmap top, flat bottom at Z=0."""
    gx, gy, z = tt.gx_mm, tt.gy_mm, tt.z_mm
    ny, nx = z.shape
    xx, yy = np.meshgrid(gx, gy)

    top = np.column_stack([xx.ravel(), yy.ravel(), z.ravel()])
    bot = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    verts = np.vstack([top, bot])
    n = nx * ny

    def ti(j, i):        # top vertex index
        return j * nx + i

    def bi(j, i):        # bottom vertex index
        return n + j * nx + i

    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = ti(j, i), ti(j, i + 1), ti(j + 1, i + 1), ti(j + 1, i)
            faces += [[a, b, c], [a, c, d]]                       # top (up)
            a2, b2, c2, d2 = bi(j, i), bi(j, i + 1), bi(j + 1, i + 1), bi(j + 1, i)
            faces += [[a2, c2, b2], [a2, d2, c2]]                 # bottom (down)
    for i in range(nx - 1):                                       # south / north skirts
        faces += [[ti(0, i), bi(0, i), bi(0, i + 1)], [ti(0, i), bi(0, i + 1), ti(0, i + 1)]]
        faces += [[ti(ny - 1, i + 1), bi(ny - 1, i + 1), bi(ny - 1, i)],
                  [ti(ny - 1, i + 1), bi(ny - 1, i), ti(ny - 1, i)]]
    for j in range(ny - 1):                                       # west / east skirts
        faces += [[ti(j + 1, 0), bi(j + 1, 0), bi(j, 0)], [ti(j + 1, 0), bi(j, 0), ti(j, 0)]]
        faces += [[ti(j, nx - 1), bi(j, nx - 1), bi(j + 1, nx - 1)],
                  [ti(j, nx - 1), bi(j + 1, nx - 1), ti(j + 1, nx - 1)]]

    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
        mesh.merge_vertices()
        mesh.fix_normals()
        return mesh if not mesh.is_empty else None
    except Exception:  # noqa: BLE001
        return None


def _boundary_edges(faces: np.ndarray):
    """Edges that belong to exactly one triangle (the polygon boundary)."""
    from collections import defaultdict

    count = defaultdict(int)
    order = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (a, b) if a < b else (b, a)
            count[key] += 1
            order.setdefault(key, (a, b))
    return [order[k] for k, c in count.items() if c == 1]


def _drape_layer(geom, sampler, z_offset: float, thickness: float) -> trimesh.Trimesh | None:
    """A thin slab that follows the terrain surface (for water / roads / parks)."""
    meshes = []
    for poly in _iter_polys(geom):
        if poly.area <= 0:
            continue
        poly = poly.buffer(0)
        for clean in _iter_polys(poly):
            try:
                v2d, tris = trimesh.creation.triangulate_polygon(clean, engine="earcut")
            except Exception:  # noqa: BLE001
                continue
            if len(tris) == 0 or len(v2d) == 0:
                continue
            base_z = np.array([sampler(x, y) + z_offset for x, y in v2d])
            top = np.column_stack([v2d[:, 0], v2d[:, 1], base_z + thickness])
            bot = np.column_stack([v2d[:, 0], v2d[:, 1], base_z])
            m = len(v2d)
            verts = np.vstack([top, bot])
            faces = [[a, b, c] for a, b, c in tris]                       # top
            faces += [[c + m, b + m, a + m] for a, b, c in tris]          # bottom
            for a, b in _boundary_edges(np.asarray(tris)):               # sides
                faces += [[a, b, b + m], [a, b + m, a + m]]
            try:
                mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
                if not mesh.is_empty and len(mesh.faces):
                    meshes.append(mesh)
            except Exception:  # noqa: BLE001
                continue
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _bridge_column(p, ang, gs, deck_bottom, along, cross):
    """A vertical box from local ground up into the deck (pier / abutment / post)."""
    top = deck_bottom + EMBED_MM
    bottom = gs - EMBED_MM
    h = top - bottom
    if h <= 0.3:
        return None
    col = trimesh.creation.box(extents=[along, cross, h])
    col.apply_transform(trimesh.transformations.rotation_matrix(ang, [0, 0, 1]))
    col.apply_translation([p.x, p.y, bottom + h / 2.0])
    return col


def _cyl(p1, p2, r):
    """A thin cylinder between two 3-D points (hanger / spandrel)."""
    try:
        return trimesh.creation.cylinder(radius=r, segment=[p1, p2], sections=8)
    except Exception:  # noqa: BLE001 - degenerate/zero-length segment
        return None


def _tube(points, r, sections=8):
    """A single watertight tube swept through a polyline (cable / arch rib).

    Built as one closed body so chained segments don't leave non-manifold joints
    the way many separate cylinders would after vertex merging.
    """
    pts = np.asarray([p for p in points], dtype=float)
    if len(pts) < 2:
        return None
    seg = np.diff(pts, axis=0)
    tang = np.zeros_like(pts)
    tang[1:-1] = seg[:-1] + seg[1:]
    tang[0], tang[-1] = seg[0], seg[-1]
    up = np.array([0.0, 0.0, 1.0])

    ring = []
    for P, T in zip(pts, tang):
        tn = np.linalg.norm(T)
        T = T / tn if tn > 1e-9 else np.array([1.0, 0.0, 0.0])
        n1 = np.cross(T, up)
        if np.linalg.norm(n1) < 1e-6:
            n1 = np.cross(T, np.array([1.0, 0.0, 0.0]))
        n1 /= (np.linalg.norm(n1) or 1.0)
        n2 = np.cross(T, n1)
        n2 /= (np.linalg.norm(n2) or 1.0)
        for k in range(sections):
            a = 2 * math.pi * k / sections
            ring.append(P + r * (math.cos(a) * n1 + math.sin(a) * n2))
    verts = list(ring)
    faces = []
    S = sections
    for i in range(len(pts) - 1):
        for k in range(S):
            a, b = i * S + k, i * S + (k + 1) % S
            c, d = (i + 1) * S + k, (i + 1) * S + (k + 1) % S
            faces += [[a, b, d], [a, d, c]]
    c0 = len(verts); verts.append(pts[0])
    for k in range(S):
        faces.append([c0, (k + 1) % S, k])
    cN = len(verts); verts.append(pts[-1]); base = (len(pts) - 1) * S
    for k in range(S):
        faces.append([cN, base + k, base + (k + 1) % S])
    try:
        m = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=True)
        m.merge_vertices(); m.fix_normals()
        return m if not m.is_empty else None
    except Exception:  # noqa: BLE001
        return None


def _perp(ang):
    """Unit vector perpendicular to a heading (across the deck)."""
    return (-math.sin(ang), math.cos(ang))


def _deck_and_supports(line_mm, width_mm, ground_fn, add_supports, deck_bottom, deck_top, ang,
                       min_feature_mm=0.1):
    """The shared substructure: deck slab + piers/abutments + print supports."""
    post_sz = max(SUPPORT_POST_MM, min_feature_mm)
    length = line_mm.length
    meshes = []
    ribbon = line_mm.buffer(width_mm / 2.0, cap_style=2, join_style=1)
    deck = _extrude(ribbon, deck_top - deck_bottom, z0=deck_bottom)
    if deck is not None:
        meshes.append(deck)

    inset = BRIDGE_PIER_SIZE_MM / 2.0
    usable = max(0.0, length - 2.0 * inset)
    n = max(1, int(round(usable / BRIDGE_PIER_SPACING_MM)))
    pier_ds = [inset + usable * k / n for k in range(n + 1)]
    for k, d in enumerate(pier_ds):
        p = line_mm.interpolate(d)
        cross = width_mm if (k == 0 or k == n) else BRIDGE_PIER_SIZE_MM
        col = _bridge_column(p, ang, ground_fn(p.x, p.y), deck_bottom, BRIDGE_PIER_SIZE_MM, cross)
        if col is not None:
            meshes.append(col)

    if add_supports:
        for a, b in zip(pier_ds, pier_ds[1:]):
            gap = b - a
            sub = max(1, int(math.ceil(gap / SUPPORT_MAX_SPAN_MM)))
            for j in range(1, sub):
                p = line_mm.interpolate(a + gap * j / sub)
                col = _bridge_column(p, ang, ground_fn(p.x, p.y), deck_bottom, post_sz, post_sz)
                if col is not None:
                    meshes.append(col)
    return meshes


def _suspension_superstructure(line_mm, width_mm, ground_fn, deck_top, ang, min_feature_mm=0.1):
    """Two towers + draped main cables + vertical hangers (e.g. the Golden Gate)."""
    cable_r = max(BRIDGE_CABLE_R_MM, min_feature_mm / 2.0)
    hanger_r = max(BRIDGE_HANGER_R_MM, min_feature_mm / 2.0)
    length = line_mm.length
    px, py = _perp(ang)
    half = width_mm / 2.0
    f1, f2 = 0.22, 0.78                       # tower positions along the span
    tower_top = deck_top + BRIDGE_TOWER_RISE_MM
    main_sag = BRIDGE_TOWER_RISE_MM * 0.62    # cable dip between towers
    meshes = []

    # Towers straddling the deck at f1 / f2.
    for f in (f1, f2):
        p = line_mm.interpolate(length * f)
        gs = ground_fn(p.x, p.y)
        col = _bridge_column(p, ang, gs, tower_top, BRIDGE_TOWER_SIZE_MM, width_mm)
        if col is not None:
            meshes.append(col)

    def cable_z(s):
        if s <= f1:
            return deck_top + (tower_top - deck_top) * (s / f1)
        if s >= f2:
            return deck_top + (tower_top - deck_top) * ((1 - s) / (1 - f2))
        t = (s - f1) / (f2 - f1)
        return tower_top - main_sag * (1 - (2 * t - 1) ** 2)

    steps = max(24, int(length / 3))
    for side in (+1, -1):
        pts = []
        for k in range(steps + 1):
            s = k / steps
            p = line_mm.interpolate(length * s)
            pts.append([p.x + px * side * half, p.y + py * side * half, cable_z(s)])
        cable = _tube(pts, cable_r)
        if cable is not None:
            meshes.append(cable)
        # hangers from the cable down to the deck edge, between the towers.
        for k in range(steps + 1):
            s = k / steps
            if f1 < s < f2 and k % 2 == 0 and pts[k][2] - deck_top > 0.6:
                h = _cyl(pts[k], [pts[k][0], pts[k][1], deck_top], hanger_r)
                if h is not None:
                    meshes.append(h)
    return meshes


def _arch_superstructure(line_mm, width_mm, ground_fn, deck_bottom, deck_top, ang, min_feature_mm=0.1):
    """A curved arch rib below the deck, springing from the banks, plus spandrels."""
    arch_r = max(BRIDGE_ARCH_R_MM, min_feature_mm / 2.0)
    hanger_r = max(BRIDGE_HANGER_R_MM, min_feature_mm / 2.0)
    length = line_mm.length
    px, py = _perp(ang)
    half = width_mm / 2.0 * 0.8
    end_g = max(ground_fn(*line_mm.coords[0]), ground_fn(*line_mm.coords[-1]))
    crown = deck_bottom - 0.2                 # arch peak just under the deck
    rise = max(0.0, crown - end_g)
    if rise < 1.0:
        return []
    meshes = []
    steps = max(20, int(length / 3))
    for side in (+1, -1):
        pts = []
        for k in range(steps + 1):
            s = k / steps
            p = line_mm.interpolate(length * s)
            z = end_g + rise * math.sin(math.pi * s)
            pts.append([p.x + px * side * half, p.y + py * side * half, z])
        rib = _tube(pts, arch_r)
        if rib is not None:
            meshes.append(rib)
        # spandrel columns from the arch up to the deck.
        for k in range(steps + 1):
            if 0 < k < steps and k % 3 == 0 and deck_bottom - pts[k][2] > 0.6:
                col = _cyl(pts[k], [pts[k][0], pts[k][1], deck_bottom + EMBED_MM], hanger_r)
                if col is not None:
                    meshes.append(col)
    return meshes


def _build_bridge(line_mm, width_mm: float, ground_fn, add_supports: bool = True,
                  structure=None, min_feature_mm: float = 0.1):
    """Build a bridge from its centreline, rendered by ``bridge:structure``.

    All types share a flat deck held at a clearance above what it crosses (set to
    the higher abutment end so it meets the approach roads), with piers/abutments
    and print supports. On top of that:
      * suspension / cable-stayed -> two towers, draped main cables, hangers;
      * arch -> a curved arch rib under the deck with spandrel columns;
      * otherwise -> a beam/trestle bridge (deck on piers).

    A deck narrower than ``min_feature_mm`` is skipped (unprintable), and thin
    elements (posts, cables, hangers, ribs) are floored to that thickness.
    """
    coords = list(line_mm.coords)
    if len(coords) < 2 or width_mm < min_feature_mm:
        return []

    ang = float(np.arctan2(coords[-1][1] - coords[0][1], coords[-1][0] - coords[0][0]))
    end_ground = max(ground_fn(*coords[0]), ground_fn(*coords[-1]))
    deck_bottom = end_ground + BRIDGE_CLEARANCE_MM
    deck_top = deck_bottom + BRIDGE_DECK_MM

    meshes = _deck_and_supports(line_mm, width_mm, ground_fn, add_supports,
                                deck_bottom, deck_top, ang, min_feature_mm)
    s = (structure or "").strip().lower()
    try:
        if s in SUSPENSION_STRUCTURES:
            meshes += _suspension_superstructure(line_mm, width_mm, ground_fn, deck_top, ang, min_feature_mm)
        elif s in ARCH_STRUCTURES:
            meshes += _arch_superstructure(line_mm, width_mm, ground_fn, deck_bottom, deck_top, ang, min_feature_mm)
    except Exception:  # noqa: BLE001 - superstructure is best-effort; keep the deck
        pass
    return meshes


# ---------------------------------------------------------------------------
# Tile assembly
# ---------------------------------------------------------------------------
def _build_landmark(lm: LandmarkRecord, ground: float, *,
                    fetch_models: bool, sketchfab_token: str | None):
    """Return (meshes, source_note). Tries a detailed downloaded model, else
    falls back to extruded walls plus an optional procedural roof."""
    # 1) Detailed model injection (Wikidata/Commons -> Sketchfab).
    if fetch_models and (lm.qid or (lm.name and sketchfab_token)):
        try:
            model, source = landmark_models.fetch_landmark_mesh(lm.qid, lm.name, sketchfab_token)
        except Exception:                # noqa: BLE001
            model, source = None, None
        if model is not None:
            placed = landmark_models.fit_model(
                model,
                centroid_mm=lm.centroid_mm,
                target_height_mm=lm.target_height_mm,
                base_top_mm=ground,
                embed_mm=EMBED_MM,
                max_footprint_mm=lm.max_footprint_mm * 1.6,
            )
            if placed is not None and len(placed.faces) > 0:
                return [placed], f"{lm.name or lm.qid}: {source}"

    # 2) Procedural fallback: extruded walls (+ roof if tagged).
    meshes = []
    for z_bottom, z_top, geom in lm.wall_pieces:
        height = (z_top - z_bottom) + EMBED_MM
        if height <= 0:
            continue
        wm = _extrude(geom, height, z0=ground + z_bottom - EMBED_MM)
        if wm is not None:
            meshes.append(wm)
    if lm.roof_shape and lm.roof_height_mm > 0:
        roof = roofs.build_roof(
            lm.footprint_mm, ground + lm.roof_base_mm - EMBED_MM,
            lm.roof_shape, lm.roof_height_mm + EMBED_MM, lm.roof_direction,
        )
        if roof is not None:
            meshes.append(roof)
    return meshes, None


def build_tile(tile: TileGeom, *, carve_water: bool = True, engrave_roads: bool = False,
               add_supports: bool = True, min_feature_mm: float = 0.1, weld: bool = False,
               fetch_models: bool = False, sketchfab_token: str | None = None) -> TileMesh:
    """Assemble a single printable tile mesh from its prepared geometry."""
    base_t = tile.base_mm
    terrain: TileTerrain | None = tile.terrain
    parts: list[trimesh.Trimesh] = []
    sources: list[str] = []

    # Ground height under a point: the terrain surface when terrain is on,
    # otherwise the flat base-plate top. Features are draped onto this.
    def ground(x: float, y: float) -> float:
        return terrain.sample(x, y) if terrain is not None else base_t

    def centroid_ground(geom) -> float:
        c = geom.centroid
        return ground(float(c.x), float(c.y))

    def raised(geom, thickness: float):
        """A raised feature (road/bridge/park) sitting on the surface."""
        if geom is None:
            return None
        if terrain is not None:
            return _drape_layer(geom, terrain.sample, 0.0, thickness)
        return _extrude(geom, thickness + EMBED_MM, z0=base_t - EMBED_MM)

    # --- Base: terrain solid, or a flat plate (water/roads optionally carved) ---
    base = None
    if terrain is not None:
        base = _terrain_solid(terrain)
    if base is None:
        terrain = None                           # terrain failed -> flat fallback
        base = _base_plate(tile.size_w_mm, tile.size_h_mm, base_t)
        cutters = []
        if carve_water and tile.water is not None:
            cutters.append(tile.water)
        if engrave_roads and tile.roads is not None:
            cutters.append(tile.roads)
        if cutters:
            from shapely.ops import unary_union

            depth = min(CARVE_MAX_MM, base_t * 0.6)
            base, _ = _carve(base, unary_union(cutters), depth, base_t)
    parts.append(base)

    # --- Water (drape on terrain; carved into a flat base above) ---------------
    if terrain is not None and tile.water is not None:
        dm = _drape_layer(tile.water, terrain.sample, 0.0, 0.4)
        if dm is not None:
            parts.append(dm)

    # --- Parks: thin raised pad -----------------------------------------------
    pm = raised(tile.parks, min(PARK_MAX_MM, base_t * 0.4 if terrain is None else PARK_MAX_MM))
    if pm is not None:
        parts.append(pm)

    # --- Roads: raised ribbons (unless engraved into a flat base) --------------
    if not (engrave_roads and terrain is None):
        rm = raised(tile.roads, ROAD_RAISE_MM)
        if rm is not None:
            parts.append(rm)

    # --- Bridges: rendered by real-world structure (suspension/arch/beam) ------
    for line_mm, width_mm, structure in tile.bridges:
        parts.extend(_build_bridge(line_mm, width_mm, ground, add_supports=add_supports,
                                   structure=structure, min_feature_mm=min_feature_mm))

    # --- Bulk buildings (draped onto the ground) ------------------------------
    for height_mm, geom in tile.building_bins:
        g0 = centroid_ground(geom)
        bm = _extrude(geom, height_mm + EMBED_MM, z0=g0 - EMBED_MM)
        if bm is not None:
            parts.append(bm)

    # --- Generic 3D parts (not named landmarks): stacked z_bottom..z_top -------
    for z_bottom, z_top, geom in tile.detail_buildings:
        height = (z_top - z_bottom) + EMBED_MM
        if height <= 0:
            continue
        g0 = centroid_ground(geom)
        dm = _extrude(geom, height, z0=g0 + z_bottom - EMBED_MM)
        if dm is not None:
            parts.append(dm)

    # --- Named landmarks: detailed model injection or procedural fallback ------
    for lm in tile.landmarks:
        g0 = ground(lm.centroid_mm[0], lm.centroid_mm[1])
        meshes, source = _build_landmark(
            lm, g0, fetch_models=fetch_models, sketchfab_token=sketchfab_token)
        parts.extend(meshes)
        if source:
            sources.append(source)

    # --- Combine --------------------------------------------------------------
    if weld and len(parts) > 1:
        try:
            mesh = trimesh.boolean.union(parts)
            if mesh is None or mesh.is_empty:
                mesh = trimesh.util.concatenate(parts)
        except Exception:                # noqa: BLE001
            mesh = trimesh.util.concatenate(parts)
    else:
        mesh = trimesh.util.concatenate(parts)

    _finalize(mesh)
    return TileMesh(name=tile.name, mesh=mesh, model_sources=sources)


def _finalize(mesh: trimesh.Trimesh) -> None:
    """Clean up + guarantee the mesh is centred in XY and flush at Z=0."""
    mesh.merge_vertices()
    # trimesh 4.x: filter faces via boolean/index masks rather than removed helpers.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.fix_normals()
    # Centre X/Y on the origin; drop the bottom to exactly Z=0.
    minx, miny, minz = mesh.bounds[0]
    maxx, maxy, _ = mesh.bounds[1]
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    mesh.apply_translation([-cx, -cy, -minz])


def build_all(tiles: list[TileGeom], *, carve_water: bool = True, engrave_roads: bool = False,
              add_supports: bool = True, min_feature_mm: float = 0.1, weld: bool = False,
              fetch_models: bool = False, sketchfab_token: str | None = None) -> list[TileMesh]:
    """Build every tile in a preset (1 tile normally, 4 for the Tier-3 mega map)."""
    return [
        build_tile(t, carve_water=carve_water, engrave_roads=engrave_roads,
                   add_supports=add_supports, min_feature_mm=min_feature_mm, weld=weld,
                   fetch_models=fetch_models, sketchfab_token=sketchfab_token)
        for t in tiles
    ]
