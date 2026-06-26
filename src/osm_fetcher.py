"""OSM data retrieval, validation, and projection for LuminaMaps.

This module is the *only* place that talks to the internet (via osmnx /
Overpass). Everything here is wrapped in Streamlit caching so that touching a
UI slider in app.py never re-triggers a network download — the heavy fetch is
keyed purely on (latitude, longitude, radius).

Design notes
------------
* osmnx 2.x API is assumed (``ox.features_from_point`` / ``ox.geocode`` /
  ``ox.projection``).
* All geometry is projected into a *single* dynamically-chosen local UTM CRS so
  that buildings, water, roads and parks stay perfectly aligned and are in
  metres (the unit we later scale down to millimetres for printing).
* Failures for any one feature layer are non-fatal: an empty GeoDataFrame is
  returned for that layer so the rest of the map can still be built. Only the
  *buildings* layer being empty is treated as a hard error (handled in app.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import osmnx as ox
import streamlit as st

# ---------------------------------------------------------------------------
# osmnx global configuration
# ---------------------------------------------------------------------------
# Use osmnx's on-disk HTTP cache as a second line of defence on top of
# Streamlit's in-memory cache. Keep the console quiet inside Streamlit.
ox.settings.use_cache = True
ox.settings.log_console = False
# Be a little patient: dense urban queries can take a while on Overpass.
ox.settings.requests_timeout = 180


# OSM tag filters for each feature layer. Kept deliberately *broad* — per the
# project brief we must NOT cull small/unique footprints, so we pull every
# building and a wide net of land features.
BUILDING_TAGS = {"building": True}

WATER_TAGS = {
    "natural": ["water", "bay", "strait"],
    "waterway": ["river", "stream", "canal", "riverbank", "dock"],
    "landuse": ["reservoir", "basin"],
}

ROAD_TAGS = {
    "highway": [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "unclassified", "service", "living_street",
        "pedestrian", "motorway_link", "primary_link", "secondary_link",
    ],
}

PARK_TAGS = {
    "leisure": ["park", "garden", "recreation_ground", "pitch", "playground"],
    "landuse": ["grass", "recreation_ground", "village_green", "meadow", "forest"],
    "natural": ["wood", "scrub", "grassland"],
}


# ---------------------------------------------------------------------------
# Container for a fully-fetched, projected map
# ---------------------------------------------------------------------------
@dataclass
class MapData:
    """Everything app.py needs from a single fetch, already in local UTM metres."""

    center_latlon: tuple[float, float]          # (lat, lon) used for the query
    radius_m: float                             # requested radius in metres
    utm_crs: str                                # e.g. "EPSG:32633"
    buildings: gpd.GeoDataFrame
    water: gpd.GeoDataFrame
    roads: gpd.GeoDataFrame
    parks: gpd.GeoDataFrame
    notes: list[str] = field(default_factory=list)  # non-fatal fetch warnings

    # --- convenience accessors -------------------------------------------------
    @property
    def building_count(self) -> int:
        return 0 if self.buildings is None else len(self.buildings)

    @property
    def water_count(self) -> int:
        return 0 if self.water is None else len(self.water)

    @property
    def road_count(self) -> int:
        return 0 if self.roads is None else len(self.roads)

    @property
    def park_count(self) -> int:
        return 0 if self.parks is None else len(self.parks)

    @property
    def has_buildings(self) -> bool:
        return self.building_count > 0

    @property
    def bounds_m(self) -> Optional[tuple[float, float, float, float]]:
        """(minx, miny, maxx, maxy) in UTM metres across all non-empty layers."""
        frames = [
            g for g in (self.buildings, self.water, self.roads, self.parks)
            if g is not None and not g.empty
        ]
        if not frames:
            return None
        bounds = [f.total_bounds for f in frames]
        minx = min(b[0] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxx = max(b[2] for b in bounds)
        maxy = max(b[3] for b in bounds)
        return (minx, miny, maxx, maxy)


# ---------------------------------------------------------------------------
# UTM helpers
# ---------------------------------------------------------------------------
def utm_crs_from_latlon(lat: float, lon: float) -> str:
    """Return the EPSG code for the WGS84 / UTM zone containing (lat, lon).

    Northern hemisphere -> 326xx, southern -> 327xx, where xx is the zone.
    Computing this ourselves (rather than letting each layer pick its own zone
    via ``project_gdf``) guarantees every layer shares one identical CRS.
    """
    zone = int((lon + 180.0) / 6.0) + 1
    zone = max(1, min(60, zone))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


# ---------------------------------------------------------------------------
# Geocoding (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def geocode_location(query: str) -> tuple[float, float]:
    """Geocode a free-text location (city / neighbourhood / address) to (lat, lon).

    Raises
    ------
    ValueError
        If the location cannot be geocoded.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Please enter a location.")
    try:
        lat, lon = ox.geocode(query)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the UI
        raise ValueError(
            f"Could not find '{query}'. Try a more specific address or city name."
        ) from exc
    return float(lat), float(lon)


# ---------------------------------------------------------------------------
# Low-level single-layer fetch (not cached on its own; called by fetch_map_data)
# ---------------------------------------------------------------------------
def _fetch_features(lat: float, lon: float, radius_m: float, tags: dict) -> gpd.GeoDataFrame:
    """Fetch one feature layer as a GeoDataFrame in EPSG:4326.

    Returns an empty GeoDataFrame (never raises) so a single missing layer can't
    sink the whole fetch.
    """
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
    except Exception:  # noqa: BLE001 - e.g. osmnx InsufficientResponseError in rural areas
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Drop rows with no geometry; keep the index for traceability.
    gdf = gdf[~gdf.geometry.isna()].copy()
    return gdf


def _project(gdf: gpd.GeoDataFrame, utm_crs: str) -> gpd.GeoDataFrame:
    """Project a (possibly empty) GeoDataFrame into the shared UTM CRS."""
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=utm_crs)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(utm_crs)


# ---------------------------------------------------------------------------
# Top-level cached fetch
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_map_data(lat: float, lon: float, radius_m: float) -> MapData:
    """Fetch + project all map layers for a point. Cached on (lat, lon, radius).

    This is the expensive, network-bound call. Because it is wrapped in
    ``st.cache_data`` and keyed only on the geographic inputs, adjusting any
    downstream UI control (Z-multiplier, preset, etc.) will reuse this result
    instead of hitting Overpass again.
    """
    utm_crs = utm_crs_from_latlon(lat, lon)
    notes: list[str] = []

    buildings_ll = _fetch_features(lat, lon, radius_m, BUILDING_TAGS)
    water_ll = _fetch_features(lat, lon, radius_m, WATER_TAGS)
    roads_ll = _fetch_features(lat, lon, radius_m, ROAD_TAGS)
    parks_ll = _fetch_features(lat, lon, radius_m, PARK_TAGS)

    if water_ll.empty:
        notes.append("No waterways/water bodies found in this area.")
    if roads_ll.empty:
        notes.append("No major roads found in this area.")
    if parks_ll.empty:
        notes.append("No parks/green spaces found in this area.")

    return MapData(
        center_latlon=(lat, lon),
        radius_m=radius_m,
        utm_crs=utm_crs,
        buildings=_project(buildings_ll, utm_crs),
        water=_project(water_ll, utm_crs),
        roads=_project(roads_ll, utm_crs),
        parks=_project(parks_ll, utm_crs),
        notes=notes,
    )
