import bpy
import numpy as np
from pathlib import Path

ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
WDIR = ROOT / "生成模型" / "组装" / "v010R11_MMD_三渲二舞台_PRIVATE"
bpy.ops.wm.open_mainfile(filepath=str(WDIR / "RadDoll_V10R18_FullSong6265_KorolevskiyTheater_HSR_MMD_Master_PRIVATE.blend"))

arm = bpy.data.objects["ARM-FullBody_V10"]
arm.data.pose_position = 'REST'
bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()

skin = bpy.data.objects["GEO-Sotai_Skin"].evaluated_get(deps)
uv_lay = skin.data.uv_layers[0].data

# Collect triangulated RD_skin (material_index == 0) faces in rest world space
skin.data.calc_loop_triangles()
tri_uv = []
tri_xyz = []
tri_nrm = []
for lt in skin.data.loop_triangles:
    if lt.material_index != 0:
        continue
    uvs = [uv_lay[li].uv[:] for li in lt.loops]
    xyzs = [(skin.matrix_world @ skin.data.vertices[vi].co)[:] for vi in lt.vertices]
    nrms = [(skin.matrix_world.to_3x3() @ skin.data.vertices[vi].normal).normalized()[:] for vi in lt.vertices]
    tri_uv.append(uvs)
    tri_xyz.append(xyzs)
    tri_nrm.append(nrms)

# Collect front hair fringe / bang vertices in rest world space for Bang Drop-Shadow calculation
bang_pts = []
for o in bpy.data.objects:
    if o.type == 'MESH' and (o.name.startswith("GEO-Hair-Fringe") or o.name.startswith("GEO-Hair-Temple")):
        oe = o.evaluated_get(deps)
        for v in oe.data.vertices:
            w = oe.matrix_world @ v.co
            if w.y < -0.05:
                bang_pts.append(w[:])

# Collect eye & mouth 3D landmarks from RadDoll_FacialParts in rest pose
fp = bpy.data.objects["RadDoll_FacialParts"].evaluated_get(deps)
fp.data.calc_loop_triangles()
fp_uv = fp.data.uv_layers[0].data

eye_xyz = [(fp.matrix_world @ fp.data.vertices[vi].co)[:] for p in fp.data.polygons if p.material_index == 0 for vi in p.vertices]
lash_xyz = [(fp.matrix_world @ fp.data.vertices[vi].co)[:] for p in fp.data.polygons if p.material_index == 1 for vi in p.vertices]
mouth_xyz = [(fp.matrix_world @ fp.data.vertices[vi].co)[:] for p in fp.data.polygons if p.material_index == 4 for vi in p.vertices]

# Also export FP triangles per slot for exact UV atlas verification
fp_slot_tris = {}
for s_i in range(6):
    fp_slot_tris[f"fp_uv_{s_i}"] = np.array(
        [[fp_uv[li].uv[:] for li in lt.loops] for lt in fp.data.loop_triangles if lt.material_index == s_i],
        dtype=np.float32
    )

out_npz = Path(r"C:\Temp\xm_face_uv_geom.npz")
np.savez_compressed(
    str(out_npz),
    tri_uv=np.array(tri_uv, dtype=np.float32),
    tri_xyz=np.array(tri_xyz, dtype=np.float32),
    tri_nrm=np.array(tri_nrm, dtype=np.float32),
    bang_pts=np.array(bang_pts, dtype=np.float32),
    eye_xyz=np.array(eye_xyz, dtype=np.float32),
    lash_xyz=np.array(lash_xyz, dtype=np.float32),
    mouth_xyz=np.array(mouth_xyz, dtype=np.float32),
    **fp_slot_tris
)
print("EXPORTED_GEOM_NPZ:", out_npz.stat().st_size, "skin_tris=", len(tri_uv), "bang_pts=", len(bang_pts))
