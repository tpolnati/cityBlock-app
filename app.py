"""LuminaMaps - Internal 3D Map Manufacturing Tool.

Streamlit front-end for generating 3D-printable city-map STLs from real-world
OpenStreetMap data, formatted for specific printers / shadowbox frame sizes.

Scope
-----
* Step 1: UI, session-state management, preset system, dynamic UTM projection,
  cached data fetch + validation, 2D preview.
* Step 2: full 2D -> 3D pipeline (geometry_processor + mesh_generator) — height
  resolution, bbox cropping, footprint unioning, line buffering, extrusion,
  water/road carving, tiling, centering, watertight STL export with downloads.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import streamlit as st

from src import geometry_processor, mesh_generator, osm_fetcher, terrain

# ---------------------------------------------------------------------------
# Preset system
# ---------------------------------------------------------------------------
# size_mm  : (width, height) of a single printed tile, in millimetres.
# base_mm  : thickness of the solid base plate.
# z_mult   : multiplier applied to real building heights for dramatic relief.
# tiles    : how the physical map is split. 1 = single tile. The Tier 3 "Mega"
#            preset slices a 500x500mm map into a 2x2 grid of flush 250x250mm
#            STLs (handled during Step 2 export).
PRESETS: dict[str, dict] = {
    "The Desk Block (Tier 1)": {
        "tier": 1,
        "size_mm": (125.0, 175.0),
        "base_mm": 8.0,
        "z_mult": 2.5,
        "tiles": (1, 1),
        "tile_size_mm": (125.0, 175.0),
        "features": ["buildings", "waterways", "roads", "parks"],
        "blurb": "Compact desktop piece. 125 × 175 mm, 8 mm base, 2.5× height.",
    },
    "10x10 Standard Backlit (Tier 2)": {
        "tier": 2,
        "size_mm": (250.0, 250.0),
        "base_mm": 2.5,
        "z_mult": 3.5,
        "tiles": (1, 1),
        "tile_size_mm": (250.0, 250.0),
        "features": ["buildings", "waterways", "roads", "parks"],
        "blurb": "Flagship backlit framed art. 250 × 250 mm, 2.5 mm base, 3.5× height.",
    },
    "20x20 Mega Map Tile (Tier 3)": {
        "tier": 3,
        "size_mm": (500.0, 500.0),          # full map footprint
        "base_mm": 2.5,
        "z_mult": 3.0,
        "tiles": (2, 2),                     # sliced into a 2x2 grid
        "tile_size_mm": (250.0, 250.0),      # each STL tile
        "features": ["buildings", "waterways", "roads", "parks"],
        "blurb": "Massive 500 × 500 mm map sliced into four flush 250 × 250 mm STLs.",
    },
}


# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------
def _init_state() -> None:
    """Seed the keys we persist across Streamlit re-runs."""
    defaults = {
        "map_data": None,          # osm_fetcher.MapData | None
        "last_query": "",          # the location string that produced map_data
        "last_radius": None,       # the radius that produced map_data
        "center_latlon": None,     # (lat, lon)
        "fetch_error": None,       # str | None
        "generated": None,         # list[dict] of exported tiles, or None
        "gen_caption": "",         # human-readable description of last generation
        "model_sources": [],       # notes on landmarks rendered from downloaded models
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------
def _render_preview(md: osm_fetcher.MapData) -> None:
    """Draw a quick 2D matplotlib preview of the projected layers (UTM metres)."""
    fig, ax = plt.subplots(figsize=(7, 7))

    if not md.parks.empty:
        md.parks.plot(ax=ax, color="#7bb274", alpha=0.5, linewidth=0)
    if not md.water.empty:
        md.water.plot(ax=ax, color="#4a90d9", alpha=0.7, linewidth=0)
    if not md.roads.empty:
        md.roads.plot(ax=ax, color="#555555", linewidth=0.5)
    if not md.buildings.empty:
        md.buildings.plot(ax=ax, color="#2b2b2b", linewidth=0)

    ax.set_aspect("equal")
    ax.set_title("Projected layers (local UTM, metres)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_polys(ax, geom, **kwargs):
    """Plot a shapely (Multi)Polygon onto a matplotlib axis."""
    if geom is None or geom.is_empty:
        return
    geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in geoms:
        if poly.geom_type != "Polygon":
            continue
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, **kwargs)


def _render_tile_preview(tile: geometry_processor.TileGeom) -> None:
    """Top-down preview of a single prepared tile in millimetre space."""
    fig, ax = plt.subplots(figsize=(5, 5))
    half_w, half_h = tile.size_w_mm / 2.0, tile.size_h_mm / 2.0
    ax.add_patch(plt.Rectangle((-half_w, -half_h), tile.size_w_mm, tile.size_h_mm,
                               fill=False, edgecolor="#cccccc", linewidth=1))
    _plot_polys(ax, tile.parks, color="#7bb274", alpha=0.5, linewidth=0)
    _plot_polys(ax, tile.water, color="#4a90d9", alpha=0.7, linewidth=0)
    _plot_polys(ax, tile.roads, color="#888888", alpha=0.8, linewidth=0)
    for _h, geom in tile.building_bins:
        _plot_polys(ax, geom, color="#2b2b2b", linewidth=0)
    # Generic 3D parts in a mid tone.
    for _zb, _zt, geom in tile.detail_buildings:
        _plot_polys(ax, geom, color="#6a6a6a", alpha=0.9, linewidth=0)
    # Named landmarks highlighted in gold so attractions stand out.
    for lm in tile.landmarks:
        _plot_polys(ax, lm.footprint_mm, color="#d4a017", alpha=0.97, linewidth=0)
    ax.set_aspect("equal")
    ax.set_xlim(-half_w * 1.05, half_w * 1.05)
    ax.set_ylim(-half_h * 1.05, half_h * 1.05)
    ax.set_title(f"{tile.name} — {tile.size_w_mm:g} × {tile.size_h_mm:g} mm")
    ax.set_xlabel("mm")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _slugify(text: str) -> str:
    """Turn a location string into a safe filename stem."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "luminamap").strip().lower())
    return slug.strip("-") or "luminamap"


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="LuminaMaps Studio", page_icon="🗺️", layout="wide")
    _init_state()

    st.title("🗺️ LuminaMaps — 3D City Map Studio")
    st.caption(
        "Internal manufacturing tool — fetch real OSM geometry, validate it, "
        "and (Step 2) extrude printable STLs for shadowbox wall art."
    )

    # ---- Sidebar: inputs -------------------------------------------------
    with st.sidebar:
        st.header("1 · Location")
        location = st.text_input(
            "City, neighbourhood, or address",
            value=st.session_state["last_query"] or "Brooklyn, New York",
            help="Anything geocodable, e.g. 'Trastevere, Rome' or '1600 Amphitheatre Pkwy'.",
        )
        radius_m = st.number_input(
            "Radius (metres)",
            min_value=100,
            max_value=5000,
            value=500,
            step=50,
            help="Geographic radius pulled around the location centre.",
        )

        st.header("2 · Map Type")
        preset_name = st.selectbox("Preset", list(PRESETS.keys()))
        preset = PRESETS[preset_name]
        st.info(preset["blurb"])

        st.header("3 · Fetch")
        fetch_clicked = st.button("📡 Fetch Map Data", type="primary", use_container_width=True)

    # ---- Preset details panel -------------------------------------------
    cols = st.columns(4)
    w, h = preset["size_mm"]
    tx, ty = preset["tiles"]
    cols[0].metric("Tile / map size", f"{w:g} × {h:g} mm")
    cols[1].metric("Base plate", f"{preset['base_mm']:g} mm")
    cols[2].metric("Z multiplier", f"{preset['z_mult']:g}×")
    cols[3].metric("STL tiles", f"{tx * ty}" + (f"  ({tx}×{ty})" if tx * ty > 1 else ""))

    # ---- Fetch action ----------------------------------------------------
    if fetch_clicked:
        st.session_state["fetch_error"] = None
        try:
            with st.spinner(f"Geocoding '{location}'…"):
                lat, lon = osm_fetcher.geocode_location(location)
            st.session_state["center_latlon"] = (lat, lon)

            with st.spinner(
                f"Querying OpenStreetMap around ({lat:.5f}, {lon:.5f}) within "
                f"{radius_m} m… this can take a moment."
            ):
                md = osm_fetcher.fetch_map_data(lat, lon, float(radius_m))

            # Hard validation: no buildings -> halt, per the brief.
            if not md.has_buildings:
                st.session_state["map_data"] = None
                st.session_state["fetch_error"] = (
                    "No building footprints were returned for this area. This is "
                    "common in rural locations. Increase the radius or pick a "
                    "denser, more urban location, then fetch again."
                )
            else:
                st.session_state["map_data"] = md
                st.session_state["last_query"] = location
                st.session_state["last_radius"] = float(radius_m)
        except ValueError as exc:
            st.session_state["map_data"] = None
            st.session_state["fetch_error"] = str(exc)

    # ---- Results / validation output ------------------------------------
    if st.session_state["fetch_error"]:
        st.error(st.session_state["fetch_error"])

    md = st.session_state["map_data"]
    if md is None:
        st.markdown(
            "👈 Enter a location, choose a preset, and click **Fetch Map Data** "
            "to pull and validate OSM geometry."
        )
        return

    # Success — summarise the fetched, projected data.
    st.success(
        f"Fetched **{md.building_count}** buildings around "
        f"{st.session_state['last_query']} "
        f"({md.radius_m:g} m radius) — projected to **{md.utm_crs}**."
    )

    m = st.columns(4)
    m[0].metric("Buildings", md.building_count)
    m[1].metric("Water features", md.water_count)
    m[2].metric("Roads", md.road_count)
    m[3].metric("Parks / green", md.park_count)

    for note in md.notes:
        st.warning(note)

    left, right = st.columns([3, 2])
    with left:
        _render_preview(md)
    with right:
        st.subheader("Fetch summary")
        lat, lon = md.center_latlon
        st.write(
            {
                "center_lat": round(lat, 6),
                "center_lon": round(lon, 6),
                "radius_m": md.radius_m,
                "utm_crs": md.utm_crs,
            }
        )
        bounds = md.bounds_m
        if bounds:
            minx, miny, maxx, maxy = bounds
            st.write(
                {
                    "extent_x_m": round(maxx - minx, 1),
                    "extent_y_m": round(maxy - miny, 1),
                }
            )
        st.caption(
            "Verify the layers above look correct, then generate the 3D model below."
        )

    # ---- 4 · 3D generation & export -------------------------------------
    st.divider()
    st.header("4 · 3D Generation & Export")

    g1, g2 = st.columns([2, 3])
    with g1:
        z_mult = st.slider(
            "Z-axis multiplier (height exaggeration)",
            min_value=1.0, max_value=6.0,
            value=float(preset["z_mult"]), step=0.1,
            help="Adjusting this recomputes ONLY the 3D mesh — the OSM data is "
                 "cached and is not re-downloaded.",
        )

        detail_options = list(geometry_processor.DETAIL_LEVELS.keys())
        detail_level = st.select_slider(
            "Detail level",
            options=detail_options,
            value=geometry_processor.DEFAULT_DETAIL_LEVEL,
            help="Higher levels keep landmarks razor-sharp and reconstruct "
                 "skyscraper setbacks from OSM building:part data.",
        )
        st.caption(geometry_processor.DETAIL_LEVELS[detail_level]["blurb"])
        landmark_emphasis = st.slider(
            "Landmark emphasis ×",
            min_value=1.0, max_value=3.0, value=1.0, step=0.1,
            help="Extra height multiplier applied to landmarks & 3D parts so "
                 "major attractions (e.g. Burj Khalifa) tower over the city.",
            disabled=not geometry_processor.DETAIL_LEVELS[detail_level]["landmarks"],
        )
        max_height_mm = st.slider(
            "Max relief height (mm)",
            min_value=10.0, max_value=150.0,
            value=float(geometry_processor.DEFAULT_MAX_HEIGHT_MM), step=5.0,
            help="Printable ceiling. The tallest features smoothly approach this "
                 "height (so an 800 m tower won't print a metre tall) while small "
                 "buildings stay near true-to-scale.",
        )
        st.markdown("**Terrain (elevation)**")
        terrain_on = st.checkbox(
            "Add real terrain / elevation (online)", value=False,
            help="Downloads a Digital Elevation Model and builds a topographic "
                 "base with buildings draped onto the hills. Great for hilly "
                 "cities; off by default (a flat base backlights more evenly).",
        )
        terrain_exagg = st.slider(
            "Terrain exaggeration ×", min_value=0.5, max_value=5.0, value=2.0, step=0.5,
            help="Real relief is subtle at map scale — exaggerate it to make hills read.",
            disabled=not terrain_on,
        )
        terrain_cap = st.slider(
            "Max terrain relief (mm)", min_value=5.0, max_value=60.0, value=25.0, step=5.0,
            help="Caps how tall the hills print above the base plate.",
            disabled=not terrain_on,
        )

        carve = st.checkbox("Carve water/roads into base", value=True,
                            help="Ignored when terrain is on (features drape on the surface).")
        weld = st.checkbox(
            "Weld into single manifold (boolean union — slower)", value=False,
            help="Off: fast concatenation (buildings embedded into the base, "
                 "slicer-safe). On: a true boolean union for a perfectly manifold file.",
        )
        feat = st.columns(3)
        inc_water = feat[0].checkbox("Water", value=True)
        inc_roads = feat[1].checkbox("Roads", value=True)
        inc_parks = feat[2].checkbox("Parks", value=True)

        st.markdown("**Detailed landmark models**")
        fetch_models = st.checkbox(
            "Fetch real 3D models for landmarks (online, slower)", value=False,
            help="Downloads detailed, recognisable models for famous landmarks and "
                 "stitches them in: Wikidata/Wikimedia Commons first (free), then "
                 "Sketchfab if a token is provided. Falls back to procedural roofs "
                 "and massing when no model is found.",
        )
        sketchfab_token = st.text_input(
            "Sketchfab API token (optional)", value="", type="password",
            help="Free token from sketchfab.com/settings/password → API. Greatly "
                 "widens coverage (e.g. skyscrapers like the Burj Khalifa). Only "
                 "CC-licensed, downloadable models are used.",
            disabled=not fetch_models,
        ).strip() or None
        generate = st.button("🧱 Generate STL(s)", type="primary", use_container_width=True)

    if generate:
        try:
            dem = None
            if terrain_on:
                with st.spinner("Fetching elevation data (DEM)…"):
                    dem = terrain.fetch_terrain(md.center_latlon, md.radius_m, md.utm_crs)
                if dem is None:
                    st.warning(
                        "Couldn't fetch elevation data (network/API). Building a "
                        "flat base instead."
                    )
            with st.spinner("Processing 2D geometry (crop, union, buffer, simplify)…"):
                tiles = geometry_processor.prepare_tiles(
                    md, preset, z_mult,
                    detail_level=detail_level,
                    landmark_emphasis=landmark_emphasis,
                    max_height_mm=max_height_mm,
                    terrain=dem,
                    terrain_exaggeration=terrain_exagg,
                    terrain_cap_mm=terrain_cap,
                    include_water=inc_water,
                    include_roads=inc_roads,
                    include_parks=inc_parks,
                )
            spin = "Extruding {n} tile(s), fetching landmark models, exporting STL…" if fetch_models \
                else "Extruding {n} tile(s) and exporting STL…"
            with st.spinner(spin.format(n=len(tiles))):
                meshes = mesh_generator.build_all(
                    tiles, carve_features=carve, weld=weld,
                    fetch_models=fetch_models, sketchfab_token=sketchfab_token,
                )

            stem = _slugify(st.session_state["last_query"])
            generated = []
            all_sources = []
            for tg, tm in zip(tiles, meshes):
                dims = tm.dims_mm
                all_sources.extend(tm.model_sources)
                generated.append({
                    "name": tm.name,
                    "filename": f"luminamap_{stem}_{tm.name}.stl",
                    "bytes": tm.to_stl_bytes(),
                    "triangles": tm.triangles,
                    "watertight": tm.watertight,
                    "landmarks": tg.landmark_count,
                    "model_sources": tm.model_sources,
                    "dims": (round(dims[0], 1), round(dims[1], 1), round(dims[2], 1)),
                    "preview": tg,
                })
            st.session_state["generated"] = generated
            st.session_state["model_sources"] = all_sources
            emph = f" · landmarks ×{landmark_emphasis:g}" if landmark_emphasis != 1.0 else ""
            terr = f" · terrain ×{terrain_exagg:g}" if (terrain_on and dem is not None) else ""
            st.session_state["gen_caption"] = (
                f"{preset_name} · Z×{z_mult:g} · {detail_level} detail{emph}{terr} · "
                f"{'welded' if weld else 'concatenated'}"
                f"{'' if (terrain_on and dem is not None) else (' · carved' if carve else '')}"
            )
        except Exception as exc:  # noqa: BLE001 - surface a clean error in the UI
            st.session_state["generated"] = None
            st.error(f"3D generation failed: {exc}")

    # ---- Generated results (persist across reruns / downloads) -----------
    generated = st.session_state["generated"]
    if generated:
        st.success(
            f"Generated **{len(generated)}** STL tile(s) — {st.session_state['gen_caption']}."
        )
        with g2:
            _render_tile_preview(generated[0]["preview"])
            if len(generated) > 1:
                st.caption(f"Preview shows {generated[0]['name']} of {len(generated)} tiles.")

        sources = st.session_state["model_sources"] or []
        if sources:
            st.markdown("**Detailed models stitched in:**")
            for src in sources:
                st.markdown(f"- 🏛️ {src}")

        st.subheader("Export")
        cols = st.columns(min(len(generated), 4))
        for idx, item in enumerate(generated):
            col = cols[idx % len(cols)]
            with col:
                wt = "✅ watertight" if item["watertight"] else "⚠️ not watertight"
                w, h, t = item["dims"]
                n_models = len(item.get("model_sources") or [])
                lm = ""
                if item.get("landmarks"):
                    lm = f"  \n⭐ {item['landmarks']} landmark(s)"
                    if n_models:
                        lm += f", {n_models} detailed model(s)"
                st.markdown(
                    f"**{item['name']}**  \n"
                    f"{w} × {h} × {t} mm  \n"
                    f"{item['triangles']:,} triangles  \n{wt}{lm}"
                )
                st.download_button(
                    "⬇️ Download STL",
                    data=item["bytes"],
                    file_name=item["filename"],
                    mime="model/stl",
                    use_container_width=True,
                    key=f"dl_{idx}",
                )


if __name__ == "__main__":
    main()
