"""2D geometry processing for LuminaMaps (Step 2).

All pure-``shapely`` / ``geopandas`` operations live here: bounding-box
cropping, ``unary_union`` of overlapping footprints, LineString -> Polygon
buffering for roads/waterways, building-height resolution, and microscopic
simplification.

NOTE: Intentionally left as a stub. Per the Step 1 instructions we stop after
the data fetch is verified and do NOT implement 3D/geometry processing yet.
"""

from __future__ import annotations

# Implemented in Step 2:
#   - crop_to_bbox(gdf, bbox)               -> shapely boolean intersection
#   - union_footprints(buildings_gdf)       -> unary_union to kill internal faces
#   - buffer_lines(lines_gdf, width)        -> LineString to Polygon
#   - resolve_heights(buildings_gdf, z_mult)-> building:levels / height tags
#   - simplify(gdf, tolerance)              -> remove colinear nodes only
