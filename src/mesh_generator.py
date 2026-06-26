"""3D mesh generation + STL export for LuminaMaps (Step 2).

All ``trimesh`` / ``numpy-stl`` work lives here: extruding 2D polygons,
generating the solid base plate, carving negative space for water/roads,
scaling to the physical bed size, centering to the origin (X=0, Y=0), ensuring
a flat watertight bottom at Z=0, and merging everything into a single STL.

NOTE: Intentionally left as a stub. Per the Step 1 instructions we stop after
the data fetch is verified and do NOT implement 3D generation yet.
"""

from __future__ import annotations

# Implemented in Step 2:
#   - build_base_plate(size_mm, base_mm)    -> solid rectangular plate
#   - extrude_buildings(polys, heights)     -> trimesh extrusions
#   - subtract_features(mesh, cutters)      -> boolean carve water/roads
#   - scale_to_bed(mesh, size_mm)           -> fit physical bed
#   - center_to_origin(mesh)                -> X=0, Y=0, flush Z=0
#   - export_stl(mesh, path)                -> watertight, manifold output
