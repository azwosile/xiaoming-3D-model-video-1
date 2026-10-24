import time
import numpy as np
from pathlib import Path
import bpy, addon_utils

t0 = time.time()
WDIR = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
CDIR = WDIR / "mmd_cache_PRIVATE"
F_START, F_END = 1, 6265

bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("bl_ext.blender_org.mmd_tools", default_set=False)

miku_pmx = str(next((CDIR / "yyb_miku").rglob("*.pmx")))
vmds = sorted((CDIR / "world_motion").glob("*.vmd"), key=lambda p: p.stat().st_size, reverse=True)
dance_vmd, cam_vmd = str(vmds[0]), str(vmds[1])

bpy.ops.mmd_tools.import_model(filepath=miku_pmx, scale=0.08 * 1.13)
miku_arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
jp2bl = {pb.mmd_bone.name_j: pb.name for pb in miku_arm.pose.bones if hasattr(pb, "mmd_bone")}

bpy.context.view_layer.update()
Mw0 = miku_arm.matrix_world.copy()
TRACKED_JP = [
    "下半身", "上半身", "上半身2", "首", "頭", "左目", "右目",
    "左肩", "左腕", "左腕捩", "左ひじ", "左手捩", "左手首",
    "右肩", "右腕", "右腕捩", "右ひじ", "右手捩", "右手首",
    "左足", "左ひざ", "左足首", "左つま先",
    "右足", "右ひざ", "右足首", "右つま先",
]
for s_jp in ("左", "右"):
    for fn_jp in ("親指０", "親指１", "親指２", "人指１", "人指２", "人指３", "中指１", "中指２", "中指３", "薬指１", "薬指２", "薬指３", "小指１", "小指２", "小指３"):
        if (s_jp + fn_jp) in jp2bl:
            TRACKED_JP.append(s_jp + fn_jp)

# 还需保存辅助骨骼静息位置（用于计算腕/手捩静息方向）
AUX_HEAD_JP = list(set(TRACKED_JP + ["左腕", "左ひじ", "左手首", "左中指１", "右腕", "右ひじ", "右手首", "右中指１"]))
rest_head = {jpn: np.array(Mw0 @ miku_arm.data.bones[jp2bl[jpn]].head_local, dtype=np.float64) for jpn in AUX_HEAD_JP if jpn in jp2bl}
rest_tail = {jpn: np.array(Mw0 @ miku_arm.data.bones[jp2bl[jpn]].tail_local, dtype=np.float64) for jpn in AUX_HEAD_JP if jpn in jp2bl}
rest_R    = np.stack([np.array((Mw0.to_3x3() @ miku_arm.data.bones[jp2bl[jpn]].matrix_local.to_3x3()).normalized(), dtype=np.float64) for jpn in TRACKED_JP], axis=0)

bpy.ops.object.select_all(action='DESELECT')
miku_arm.select_set(True)
bpy.context.view_layer.objects.active = miku_arm
bpy.ops.mmd_tools.import_vmd(filepath=dance_vmd, scale=0.08 * 1.13)

cam_d_tmp = bpy.data.cameras.new("TmpMMDCam")
cam_o_tmp = bpy.data.objects.new("TmpMMDCam", cam_d_tmp)
bpy.context.scene.collection.objects.link(cam_o_tmp)
bpy.ops.object.select_all(action='DESELECT')
cam_o_tmp.select_set(True)
bpy.context.view_layer.objects.active = cam_o_tmp
bpy.ops.mmd_tools.import_vmd(filepath=cam_vmd, scale=0.08 * 1.13)
mmd_cam_real = next(o for o in bpy.data.objects if o.type == 'CAMERA')

N_F = F_END - F_START + 1
N_B = len(TRACKED_JP)
bl_names = [jp2bl[jpn] for jpn in TRACKED_JP]
b_hips = jp2bl["下半身"]
b_head = jp2bl["頭"]
b_ankL = jp2bl["左足首"]
b_ankR = jp2bl["右足首"]

hips_pos = np.zeros((N_F, 3), dtype=np.float32)
head_z   = np.zeros(N_F, dtype=np.float32)
ank_z    = np.zeros((N_F, 2), dtype=np.float32)
pose_R   = np.zeros((N_F, N_B, 3, 3), dtype=np.float32)

cam_loc  = np.zeros((N_F, 3), dtype=np.float32)
cam_quat = np.zeros((N_F, 4), dtype=np.float32)
cam_lens = np.zeros(N_F, dtype=np.float32)
cam_sw   = np.zeros(N_F, dtype=np.float32)
cam_sh   = np.zeros(N_F, dtype=np.float32)

sc = bpy.context.scene
for i, f in enumerate(range(F_START, F_END + 1)):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    arm_ev = miku_arm.evaluated_get(dg)
    Mw = arm_ev.matrix_world
    R_mw = Mw.to_3x3()
    pbs_ev = arm_ev.pose.bones

    hips_pos[i] = Mw @ pbs_ev[b_hips].head
    head_z[i]   = (Mw @ pbs_ev[b_head].head).z
    ank_z[i, 0] = (Mw @ pbs_ev[b_ankL].head).z
    ank_z[i, 1] = (Mw @ pbs_ev[b_ankR].head).z
    for bi, bln in enumerate(bl_names):
        pose_R[i, bi] = (R_mw @ pbs_ev[bln].matrix.to_3x3()).normalized()

    cam_ev = mmd_cam_real.evaluated_get(dg)
    cmw = cam_ev.matrix_world
    cam_loc[i]  = cmw.translation
    cam_quat[i] = cmw.to_quaternion()
    cam_lens[i] = cam_ev.data.lens
    cam_sw[i]   = cam_ev.data.sensor_width
    cam_sh[i]   = cam_ev.data.sensor_height

# Check if Miku ever enters chair zone (|x| < 0.36, 1.18 < y < 1.68) after f480!
after_stand = np.arange(N_F) >= 480
in_chair_xy = after_stand & (np.abs(hips_pos[:, 0]) < 0.36) & (hips_pos[:, 1] > 1.18) & (hips_pos[:, 1] < 1.68)
print(f"POST-f480 CHAIR ZONE HITS: count={int(in_chair_xy.sum())}", flush=True)
if in_chair_xy.any():
    f_hits = np.where(in_chair_xy)[0] + 1
    print(f"  Hit frame range: f{f_hits.min()}..f{f_hits.max()}, hips_z=[{hips_pos[in_chair_xy,2].min():.3f}, {hips_pos[in_chair_xy,2].max():.3f}]", flush=True)

np.savez_compressed(
    WDIR / "oracle_full_6265.npz",
    tracked_jp=np.array(TRACKED_JP),
    aux_jp=np.array(list(rest_head.keys())),
    rest_head=np.stack([rest_head[k] for k in rest_head.keys()], axis=0),
    rest_tail=np.stack([rest_tail[k] for k in rest_head.keys()], axis=0),
    rest_R=rest_R,
    hips_pos=hips_pos,
    head_z=head_z,
    ank_z=ank_z,
    pose_R=pose_R,
    cam_loc=cam_loc,
    cam_quat=cam_quat,
    cam_lens=cam_lens,
    cam_sw=cam_sw,
    cam_sh=cam_sh,
    cam_fit=str(mmd_cam_real.data.sensor_fit),
)
print(f"ORACLE_6265_SAVED in {time.time()-t0:.1f}s!", flush=True)
