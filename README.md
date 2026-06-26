# 🗺️ LuminaMaps Studio

Internal manufacturing tool for **LuminaMaps** — premium 3D-printed city maps for
framed, backlit shadowbox wall art. It pulls real-world OpenStreetMap geometry,
processes it into clean, printer-ready solids, and exports watertight `.stl`
files sized exactly for your printers and frame tiers.

---

## What it does

A Streamlit app that takes a **location + radius + preset** and produces
3D-printable STL tiles:

1. **Geocode** a city, neighbourhood, or address.
2. **Fetch** buildings, waterways, roads, and parks from OSM (cached).
3. **Project** into a dynamically chosen local UTM zone (metres).
4. **Validate** — halts with a clear error if no buildings are found.
5. **Process heights** from `building:levels` / `height` tags (1 level = 3 m;
   missing buildings get a random 3–9 m height), scaled by the preset's
   Z-multiplier.
6. **Build geometry** — crop to the frame, union overlapping footprints, buffer
   roads/waterways into polygons, simplify redundant nodes.
7. **Generate mesh** — solid base plate, extruded buildings, raised park pads,
   carved water/road channels.
8. **Export** centered, flush, watertight STL(s) — one per tile.

---

## Quick start

```bash
# 1. install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 2. launch the app
streamlit run app.py
```

Then in the browser:

1. Type a **location** (e.g. `Trastevere, Rome` or `1600 Amphitheatre Pkwy`).
2. Set a **radius** in metres (denser/urban areas work best).
3. Choose a **preset** (see below).
4. Click **📡 Fetch Map Data** and confirm the 2D preview looks right.
5. Adjust the **Z-axis multiplier** and toggles, then click **🧱 Generate STL(s)**.
6. **Download** each tile.

> ℹ️ Adjusting the Z-multiplier (or any 3D option) recomputes **only the mesh** —
> the OSM data is cached and is never re-downloaded.

---

## Presets

| Preset | Tier | Tile size | Base | Z-mult | Output |
|---|---|---|---|---|---|
| **The Desk Block** | 1 | 125 × 175 mm | 8.0 mm | 2.5× | 1 STL |
| **10×10 Standard Backlit** | 2 | 250 × 250 mm | 2.5 mm | 3.5× | 1 STL |
| **20×20 Mega Map Tile** | 3 | 250 × 250 mm | 2.5 mm | 3.0× | 4 STLs (500 × 500 mm map sliced into a flush 2 × 2 grid) |

All presets include **buildings + waterways + roads + parks**. Tiers 2 and 3 use
a thin base for LED backlighting; Tier 1 uses a thick solid base for a desk piece.

---

## Landmark detail

Major attractions need to be **recognisable** in the print — a flat box where the
Burj Khalifa should be won't do. The app handles this with three mechanisms:

- **`building:part` (Simple 3D Buildings).** OSM stores skyscraper massing as
  stacked part-polygons, each with its own `height` and `min_height`. At **High**
  / **Maximum** detail these are extruded individually, reproducing setbacks and
  the tapering spire that make a tower identifiable.
- **Landmark detection.** Buildings tagged `tourism`/`historic`/`man_made=tower`,
  carrying a `wikidata`/`wikipedia` reference, having a notable `building` value
  (cathedral, stadium, tower…), or simply very tall are flagged as landmarks.
  They're kept at full footprint fidelity (little/no simplification) and never
  merged into their neighbours. They're highlighted **gold** in the 2D preview.
- **Printable height with preserved silhouette.** A landmark's parts are scaled
  *proportionally to that building's own height*, and the overall top is bounded
  by **Max relief height** via a smooth `tanh` knee. So an 828 m tower tops out
  at (say) 50 mm instead of printing a metre tall — while the 200 / 450 / 650 m
  setbacks keep their exact ratios and the spire still reads as the spire. Small
  buildings nearby stay near true-to-scale.

### Detail levels

| Level | building:part | Landmarks | Simplification | Use it for |
|---|---|---|---|---|
| **Standard** | off | off | coarse | Fast drafts; flat-topped prisms |
| **High** (default) | on | on | light on landmarks | Most maps — sharp landmarks, fast bulk |
| **Maximum** | on | on | none | Hero pieces; every node preserved |

The **Landmark emphasis ×** slider adds extra height to landmarks/parts so they
tower further above the city; **Max relief height** caps the tallest feature so
prints stay sensible. Changing any of these recomputes the mesh only — no refetch.

## Project structure

```
app.py                       Streamlit UI, session state, preset logic, fetch + export flow
src/
  osm_fetcher.py             OSM retrieval, validation, dynamic UTM projection (cached)
  geometry_processor.py      shapely/geopandas: crop, union, buffer, simplify, tile, scale → mm
  mesh_generator.py          trimesh: base plate, extrusion, carving, weld, centering, STL export
requirements.txt
```

The fetch layer is the **only** code that touches the network and is wrapped in
`@st.cache_data`, keyed purely on `(lat, lon, radius)`.

---

## 3D printing notes

- **Orientation:** print flat, as exported — the bottom is flush at `Z = 0` and
  the model is centered at `X = 0, Y = 0`.
- **Watertight / manifold:** by default tiles are assembled by fast
  concatenation with buildings embedded `0.2 mm` into the base (slicer-safe).
  Enable **"Weld into single manifold"** for a true boolean union if your slicer
  is strict — it's slower but yields a single closed solid.
- **Carved features:** water and roads are recessed into the base top as
  negative space (depth capped at `1.2 mm` or 60% of base thickness). Toggle
  off if you prefer a flat base.
- **Backlighting (Tiers 2 & 3):** the thin `2.5 mm` base is designed to diffuse
  integrated LED light; print the base layer in a translucent filament.
- **Detail fidelity:** small sheds and unique footprints are intentionally
  **not** culled. Simplification only removes redundant colinear vertices, so
  architectural micro-detail is preserved.

### Tunable defaults

| Parameter | Location | Default |
|---|---|---|
| Level height | `geometry_processor.LEVEL_HEIGHT_M` | 3.0 m |
| Random height fallback | `geometry_processor.DEFAULT_MIN_M / MAX_M` | 3–9 m |
| Detail-level simplify tolerances | `geometry_processor.DETAIL_LEVELS` | 0.0–0.30 m |
| Max relief height (cap) | `geometry_processor.DEFAULT_MAX_HEIGHT_MM` | 50 mm |
| Landmark height thresholds | `geometry_processor.LANDMARK_*_HEIGHT_M` | 60 / 120 m |
| Height bin size | `geometry_processor.HEIGHT_BIN_MM` | 0.5 mm |
| Road / waterway widths | `geometry_processor.ROAD_WIDTH_M / WATERWAY_WIDTH_M` | by OSM class |
| Carve depth | `mesh_generator.CARVE_MAX_MM` | 1.2 mm |
| Building base embed | `mesh_generator.EMBED_MM` | 0.2 mm |
| Park pad thickness | `mesh_generator.PARK_MAX_MM` | 0.8 mm |

---

## Tech stack

Streamlit · osmnx · geopandas · shapely · pyproj · trimesh · manifold3d ·
mapbox-earcut · matplotlib
