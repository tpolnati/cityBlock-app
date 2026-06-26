"""LuminaMaps - Internal 3D Map Manufacturing Tool.

Streamlit front-end for generating 3D-printable city-map STLs from real-world
OpenStreetMap data, formatted for specific printers / shadowbox frame sizes.

STEP 1 SCOPE
------------
This file currently implements the UI, Streamlit session-state management,
the Preset system, dynamic UTM projection, and the cached data fetch +
validation. The actual 3D mesh generation (geometry_processor / mesh_generator)
is wired in during Step 2 — for now we stop after confirming the map data
fetches correctly and preview it visually.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from src import osm_fetcher

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
            "Verify the layers above look correct. 3D extrusion + STL export "
            "arrives in Step 2."
        )


if __name__ == "__main__":
    main()
