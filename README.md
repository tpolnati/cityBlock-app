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

### Detailed landmark models (recognisable icons)

OSM extrusion alone can't make the Burj Khalifa *look* like the Burj Khalifa. Turn
on **"Fetch real 3D models for landmarks"** and the app downloads a genuinely
detailed model for each detected landmark and stitches it into the city at the
right place and a printable height. Sources are tried best-first:

1. **Wikidata `P4896` → Wikimedia Commons STL.** OSM tags landmarks with
   `wikidata=Q…`; that entity often links a free, CC-licensed, print-ready STL
   (Eiffel Tower, Statue of Liberty, Parthenon…). No key needed.
2. **Sketchfab** *(optional — paste a free API token)*. Far broader coverage,
   including skyscrapers like the Burj Khalifa. Only **downloadable, CC-licensed**
   models are used. Get a token at `sketchfab.com/settings/password` → API.
3. **Procedural fallback** — if no model is found, the landmark still renders from
   its OSM `building:part` massing plus a reconstructed roof (dome/spire/…), which
   is already far more recognisable than a flat box.

A fetched model is oriented Z-up, uniformly scaled to the landmark's printable
height, and placed at its footprint centroid. Downloads are cached, so re-exports
don't re-fetch. The export panel lists which landmarks got a detailed model and
from where.

> **Coverage & expectations.** No free dataset has detailed printable models for
> *every* landmark on Earth. Wikidata/Commons is exact but sparse; Sketchfab is
> broad but needs a token and its models vary in quality/scale. Where neither has
> a model, the procedural roof/massing fallback keeps the building recognisable.
> Model fetching is **opt-in** because it hits external services and is slower —
> but there's no time limit on export, so leave it on for hero pieces.

### Roads & bridges

Roads render as **raised ridges** on the base (or draped on terrain), sized by
OSM class. A **Road detail** selector controls how much of the network appears:

| Road detail | Includes |
|---|---|
| Major roads only | motorway → tertiary |
| Major + residential *(default)* | + residential / unclassified |
| All roads (incl. service) | + service / living-street / pedestrian |

Tiny ways — footways, tracks, cycleways, steps, alleys — are **always dropped**.
**Bridges** (`bridge=yes`) are pulled out of the road network and printed as
**taller raised decks** so crossings stand out; the flat road beneath a deck is
removed to avoid a doubled ribbon. (Optional: *Engrave roads* cuts them into a
flat base instead of raising them.)

### Terrain / elevation

By default the ground is flat (which backlights most evenly). Turn on **"Add
real terrain / elevation"** and the app downloads a Digital Elevation Model,
builds a **topographic base**, and **drapes the city onto the hills** — so
places like San Francisco, Rome or Lisbon actually read as hilly.

- **Data:** sampled on a grid from a public elevation API (OpenTopoData SRTM
  30 m, then Open-Elevation as fallback) — no key, cached.
- **Draping:** each building/landmark sits on the ground height at its footprint;
  water, roads and parks are laid as thin layers that follow the surface.
- **Controls:** **Terrain exaggeration ×** (real relief is subtle at map scale)
  and **Max terrain relief (mm)** to keep prints sensible. The lowest ground
  still keeps a solid base thickness, and the result stays watertight.
- **Tiling:** the Tier-3 mega map samples terrain at exact tile edges, so the
  four tiles line up seamlessly.
- **Note:** with terrain on, water/roads drape on the surface instead of being
  carved into a flat base, and the base is uneven — worth considering for the
  backlit tiers. It's **opt-in** and off by default.

## Project structure

```
app.py                       Streamlit UI, session state, preset logic, fetch + export flow
src/
  osm_fetcher.py             OSM retrieval, validation, dynamic UTM projection (cached)
  terrain.py                 DEM fetch (elevation API) + per-tile heightmap (cached)
  geometry_processor.py      shapely/geopandas: crop, union, buffer, simplify, landmark
                             grouping, tiling, terrain draping, scale → mm
  landmark_models.py         Wikidata/Commons + Sketchfab model resolver + stitching (cached)
  roofs.py                   procedural roof solids (dome, spire, pyramidal, gabled, hipped…)
  mesh_generator.py          trimesh: base/terrain solid, extrusion, carving, draping,
                             model injection, weld, centering, STL export
requirements.txt
```

The OSM fetch layer is wrapped in `@st.cache_data` keyed on `(lat, lon, radius)`;
landmark-model downloads are cached separately by Wikidata id / name. Toggling
detail, height or model options recomputes the mesh only — never the OSM data.

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
| Terrain exaggeration / cap | `geometry_processor.DEFAULT_TERRAIN_EXAGG / _CAP_MM` | 2.0× / 25 mm |
| Terrain grid / tile resolution | `terrain.fetch_terrain grid_n`, `TERRAIN_TILE_RES` | 56 / 48 |
| Height bin size | `geometry_processor.HEIGHT_BIN_MM` | 0.5 mm |
| Road / waterway widths | `geometry_processor.ROAD_WIDTH_M / WATERWAY_WIDTH_M` | by OSM class |
| Raised road / bridge height | `mesh_generator.ROAD_RAISE_MM / BRIDGE_RAISE_MM` | 0.6 / 2.0 mm |
| Carve depth | `mesh_generator.CARVE_MAX_MM` | 1.2 mm |
| Building base embed | `mesh_generator.EMBED_MM` | 0.2 mm |
| Park pad thickness | `mesh_generator.PARK_MAX_MM` | 0.8 mm |

---

## Tech stack

Streamlit · osmnx · geopandas · shapely · pyproj · trimesh · manifold3d ·
mapbox-earcut · requests (Wikidata/Commons/Sketchfab) · matplotlib
