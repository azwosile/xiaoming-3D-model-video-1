import json, math, shutil, time
from pathlib import Path
import bpy
from mathutils import Vector

t0 = time.time()
WDIR = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
BLEND_R19 = WDIR / "RadDoll_V10R19_FullSong6265_KorolevskiyTheater_HSR_Genshin_Master_PRIVATE.blend"
CUTS_JSON = WDIR / "camera_hard_cuts_6265.json"
SHOWCASE_RAW = WDIR / "showcase_v10r19_raw720p_f0425_0604"
SEG1_RAW     = WDIR / "mv_v10r19_seg1_raw720p"
SEG1_RAW.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=str(BLEND_R19))
scn = bpy.context.scene

arm      = bpy.data.objects["ARM-FullBody_V10"]
skin_obj = bpy.data.objects["GEO-Sotai_Skin"]
cam_obj  = bpy.data.objects["MMD_MV_CAMERA"]
em_orig  = bpy.data.objects["Head_SDF_Origin"]
em_fwd   = bpy.data.objects["Head_SDF_Fwd"]
em_right = bpy.data.objects["Head_SDF_Right"]

# Remove old step=3 action on GEO-Sotai_Skin so we bake exact 1-frame CONSTANT keys across 1..6265
if skin_obj.animation_data and skin_obj.animation_data.action:
    bpy.data.actions.remove(skin_obj.animation_data.action)

# Include camera hierarchy in keep_eval so cam_obj.matrix_world updates accurately while heavy meshes are muted
keep_eval = {arm, cam_obj, em_orig, em_fwd, em_right}
p = cam_obj.parent
while p is not None:
    keep_eval.add(p)
    p = p.parent

saved_vp = []
for ob in bpy.data.objects:
    if ob not in keep_eval and not ob.hide_viewport:
        saved_vp.append(ob)
        ob.hide_viewport = True

stage_key = Vector((0.55, -0.65, 0.52)).normalized()
cam_locs = {}
cam_quats = {}
cam_lenses = {}
sample_dbg = {}

for f in range(1, 6266):
    scn.frame_set(f)
    h0 = em_orig.matrix_world.translation
    hf = (em_fwd.matrix_world.translation - h0).normalized()
    hr = (em_right.matrix_world.translation - h0).normalized()
    cmw = cam_obj.matrix_world
    c_pos = cmw.translation.copy()
    c_quat = cmw.to_quaternion()
    cam_locs[f] = c_pos
    cam_quats[f] = c_quat
    cam_lenses[f] = float(cam_obj.data.lens)

    to_cam = (c_pos - h0).normalized()
    cam_right = cmw.to_3x3().col[0].normalized()
    cam_up    = cmw.to_3x3().col[1].normalized()
    cam_34 = (to_cam + 0.58 * cam_right + 0.36 * cam_up).normalized()
    l_key  = (0.62 * cam_34 + 0.38 * stage_key).normalized()

    lf = hf.dot(l_key)
    lr = hr.dot(l_key)
    h_len = math.sqrt(lf * lf + lr * lr + 1e-6)
    cos_f = lf / h_len
    sin_r = lr / h_len
    u_light = max(0.76, min(0.96, 0.5 * cos_f + 0.5))
    side_01 = max(0.0, min(1.0, 0.5 * sin_r + 0.5))
    if abs(sin_r) < 0.26:
        side_01 = 0.68 if sin_r >= -0.05 else 0.32

    skin_obj.color = (u_light, side_01, 1.0, 1.0)
    skin_obj.keyframe_insert(data_path="color", frame=f)
    if f in (1, 112, 446, 447, 451, 1084, 1270, 2400):
        sample_dbg[f] = (round(u_light, 3), round(side_01, 3))

# Set CONSTANT interpolation on GEO-Sotai_Skin.color FCurves
if skin_obj.animation_data and skin_obj.animation_data.action:
    act = skin_obj.animation_data.action
    fcurves = getattr(act, "fcurves", None)
    if fcurves is None and hasattr(act, "layers"):
        for lay in act.layers:
            for st in lay.strips:
                for cb in getattr(st, "channelbags", []):
                    for fc in cb.fcurves:
                        for kp in fc.keyframe_points:
                            kp.interpolation = 'CONSTANT'
    elif fcurves is not None:
        for fc in fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'

for ob in saved_vp:
    ob.hide_viewport = False
bpy.context.view_layer.update()

# Detect 1-frame camera hard cuts across 1..6265
cuts_all = []
for f in range(1, 6265):
    dl = (cam_locs[f + 1] - cam_locs[f]).length
    qd = min(1.0, abs(cam_quats[f].dot(cam_quats[f + 1])))
    da = math.degrees(2.0 * math.acos(qd))
    df = abs(cam_lenses[f + 1] - cam_lenses[f])
    if dl > 0.45 or da > 12.0 or df > 4.0:
        cuts_all.append([f, f + 1, round(dl, 3), round(da, 2), round(df, 2)])

cuts_650 = [(c[0], c[1]) for c in cuts_all if c[1] <= 650]
expected_650 = [
    (113, 114), (213, 214), (257, 258), (268, 269), (313, 314), (358, 359), (402, 403),
    (446, 447), (457, 458), (473, 474), (540, 541), (624, 625), (631, 632), (641, 642),
]
print("CUTS_1_650:", cuts_650, flush=True)
print("MATCH_EXPECTED_650:", cuts_650 == expected_650, flush=True)

cuts_seg1 = [[c[0], c[1]] for c in cuts_all if c[1] <= 2400]
cuts_seg2 = [[c[0], c[1]] for c in cuts_all if 2401 <= c[0] and c[1] <= 4320]
cuts_seg3 = [[c[0], c[1]] for c in cuts_all if 4321 <= c[0]]

CUTS_JSON.write_text(json.dumps({
    "total_frames": 6265,
    "cuts_all_detail": cuts_all,
    "cuts_seg1_1_2400": cuts_seg1,
    "cuts_seg2_2401_4320": cuts_seg2,
    "cuts_seg3_4321_6265": cuts_seg3,
}, indent=2), encoding="utf-8")

# Set persistent EEVEE 720p lossless PNG defaults for fast batch rendering
scn.camera = cam_obj
scn.render.engine = 'BLENDER_EEVEE'
scn.eevee.taa_render_samples = 48
scn.render.use_persistent_data = True
scn.render.resolution_x = 1280
scn.render.resolution_y = 720
scn.render.resolution_percentage = 100
scn.render.fps = 30
scn.render.image_settings.file_format = 'PNG'
scn.render.image_settings.color_mode = 'RGB'
scn.render.image_settings.compression = 15

bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_R19))

# Seed mv_v10r19_seg1_raw720p with existing clean showcase frames (excluding +/-2 around hard cuts inside 425..604)
cut_frames_425_604 = set()
for a, b in [(446, 447), (457, 458), (473, 474), (540, 541)]:
    for k in range(a - 2, b + 3):
        cut_frames_425_604.add(k)

seeded = 0
if SHOWCASE_RAW.exists():
    for idx in range(180):
        f = 425 + idx
        if f in cut_frames_425_604:
            continue
        src_p = SHOWCASE_RAW / f"frame_{idx:04d}.png"
        dst_p = SEG1_RAW / f"frame_{f:04d}.png"
        if src_p.exists() and src_p.stat().st_size > 20000 and not dst_p.exists():
            shutil.copy2(src_p, dst_p)
            seeded += 1

print(f"PREFLIGHT_OK in {time.time()-t0:.1f}s | cuts_seg1={len(cuts_seg1)} cuts_all={len(cuts_all)} | seeded_720p={seeded} | samples={sample_dbg}", flush=True)
