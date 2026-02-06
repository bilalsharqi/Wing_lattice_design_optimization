import Part, Draft, importDXF

# =====================
# USER PARAMETERS
# =====================
dxf_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\hud\V3\100x100_hyperunifrom.dxf"
output_step = r"C:\Users\silber\Downloads\output.step"
extrude_thickness = 3.0   # mm
# cube_size = 94.53+14        # mm
# cube_length = 84.91+4
cube_size = 110       # mm
cube_length = 100
exclude_faces = [0]       # exclude outer contours
# cube_offset = FreeCAD.Vector(4.04, -5+0.26, 0)  # for hexagonal
cube_offset = FreeCAD.Vector(-50, -55, 0)  # for hexagonal

# =====================

doc = FreeCAD.activeDocument() or FreeCAD.newDocument("AutoNetwork")
importDXF.insert(dxf_path, doc.Name) # --- 1. Import DXF ---
objs = list(doc.Objects)

# Downgrade all imported polylines into edges
downgraded_objs = []
for obj in objs:
    try:
        res = Draft.downgrade([obj], delete=True)
        for r in res:
            if isinstance(r, list):
                downgraded_objs.extend(r)
            else:
                downgraded_objs.append(r)
    except Exception as e:
        print("Downgrade failed for:", obj.Label, e)
# doc.recompute()

# Upgrade twice to form faces
up1, _ = Draft.upgrade(downgraded_objs, delete=True)
up2, _ = Draft.upgrade(up1, delete=True)
faces = up2
doc.recompute()

# --- 3. Extrude faces ---
extrudes = []
for i, f in enumerate(faces):
    if i in exclude_faces:
        continue
    ex = doc.addObject("Part::Extrusion", f"Extrude_{i}")
    ex.Base = f
    ex.Dir = FreeCAD.Vector(0, 0, extrude_thickness)
    extrudes.append(ex)

# --- 4. Union all extrusions ---
if len(extrudes) > 1:
    union = doc.addObject("Part::MultiFuse", "Union")
    union.Shapes = extrudes
else:
    union = extrudes[0]

# --- 5. Add cube ---
doc.recompute()

cube = doc.addObject("Part::Box", "Cube")
cube.Length = cube_length
cube.Width = cube_size
cube.Height = extrude_thickness
cube.Placement.Base = cube_offset

# --- 6. Cut extrusions from cube ---
cut = doc.addObject("Part::Cut", "Network")
cut.Base = cube
cut.Tool = union
# --- 7. Export STEP ---
doc.recompute()
Part.export([cut], output_step)
print(f"Done! Exported network")
