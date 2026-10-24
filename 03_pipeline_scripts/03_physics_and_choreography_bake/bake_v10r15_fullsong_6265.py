import json, math, random, time
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector, Matrix, Quaternion, Euler

t0 = time.time()
WDIR = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
SRC_BLEND  = WDIR / "RadDoll_V10R15_MMD_Heart舞台三渲二纯净平涂精修_PRIVATE.blend"
OUT_BLEND  = WDIR / "RadDoll_V10R15_FullSong6265_MMD_Heart舞台三渲二全曲母带_PRIVATE.blend"
ORACLE_NPZ = WDIR / "oracle_full_6265.npz"
AUDIO_JSON = WDIR / "audio_envelope_6265f.json"

F_START = 1
F_SEAM  = 660
F_END   = 6265
N_TOTAL = F_END - F_START + 1   # 6265
N_NEW   = F_END - F_SEAM        # 5605 (f661..f6265)
DT      = 1.0 / 30.0

bpy.ops.wm.open_mainfile(filepath=str(SRC_BLEND))
sc        = bpy.context.scene
arm_ob    = bpy.data.objects["ARM-FullBody_V10"]
skin_ob   = bpy.data.objects["GEO-Sotai_Skin"]
facial_ob = bpy.data.objects["RadDoll_FacialParts"]
shoeL_ob  = bpy.data.objects["GEO-Shoe_L"]
shoeR_ob  = bpy.data.objects["GEO-Shoe_R"]
cam_obj   = bpy.data.objects["MMD_MV_CAMERA"]

ora = np.load(ORACLE_NPZ, allow_pickle=True)
tracked_jp     = [str(x) for x in ora["tracked_jp"]]
jp_idx         = {jpn: i for i, jpn in enumerate(tracked_jp)}
aux_jp         = [str(x) for x in ora["aux_jp"]]
INV_S113 = 1.0 / 1.13
miku_rest_head = {jpn: Vector((ora["rest_head"][i] * INV_S113).tolist()) for i, jpn in enumerate(aux_jp)}
miku_rest_tail = {jpn: Vector((ora["rest_tail"][i] * INV_S113).tolist()) for i, jpn in enumerate(aux_jp)}
miku_rest_R    = {jpn: Matrix(ora["rest_R"][i].tolist()) for i, jpn in enumerate(tracked_jp)}
miku_hips_pos  = ora["hips_pos"] * INV_S113   # exact scale=0.08 world coords!
miku_ank_z     = ora["ank_z"]    * INV_S113
miku_pose_R    = ora["pose_R"]
cam_loc_arr    = ora["cam_loc"]  * INV_S113   # exact scale=0.08 MMD camera coords!
cam_quat_arr   = ora["cam_quat"]   # (6265, 4)
cam_lens_arr   = ora["cam_lens"]   # (6265,)

aud        = json.loads(AUDIO_JSON.read_text(encoding="utf-8"))
env_norm   = aud["env_norm"]
env_bright = aud["env_bright"]

print(f"[1/6] Loaded V10R15 + oracle ({len(tracked_jp)} bones, {F_END}f) in {time.time()-t0:.1f}s", flush=True)

# -------------------------------------------------------------------------
# Helper: Blender 5.2 Slotted Action FCurve batch extension (preserving <=660!)
# -------------------------------------------------------------------------
def get_action_fcurves(id_data):
    ad = id_data.animation_data
    if not ad or not ad.action:
        raise RuntimeError(f"No action on {id_data.name}")
    act = ad.action
    if hasattr(act, "layers") and len(act.layers) > 0:
        for layer in act.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbags"):
                    for cb in strip.channelbags:
                        if len(cb.fcurves) > 0:
                            return cb.fcurves
    if hasattr(act, "fcurves"):
        return act.fcurves
    raise RuntimeError(f"Cannot locate fcurves on action {act.name}")

def ensure_fcurve(fcurves, data_path, index=0):
    for fc in fcurves:
        if fc.data_path == data_path and fc.array_index == index:
            return fc
    return fcurves.new(data_path=data_path, index=index)

def extend_fcurve_after_seam(fc, frames_new, values_new, constant_cut_indices=None):
    pts_keep = []
    for kp in fc.keyframe_points:
        if kp.co.x <= F_SEAM + 0.5:
            pts_keep.append((float(kp.co.x), float(kp.co.y), kp.interpolation))
    fc.keyframe_points.clear()
    total_n = len(pts_keep) + len(frames_new)
    fc.keyframe_points.add(total_n)
    co_flat = np.empty(total_n * 2, dtype=np.float32)
    idx_k = len(pts_keep)
    for i, (fx, fy, _) in enumerate(pts_keep):
        co_flat[2 * i]     = fx
        co_flat[2 * i + 1] = fy
    co_flat[2 * idx_k :: 2]     = frames_new
    co_flat[2 * idx_k + 1 :: 2] = values_new
    fc.keyframe_points.foreach_set("co", co_flat)
    for i, (_, _, interp) in enumerate(pts_keep):
        fc.keyframe_points[i].interpolation = interp
    for j in range(len(frames_new)):
        fc.keyframe_points[idx_k + j].interpolation = 'LINEAR'
    if constant_cut_indices:
        for c_i in constant_cut_indices:
            prev_idx = idx_k + c_i - 1
            if 0 <= prev_idx < total_n:
                fc.keyframe_points[prev_idx].interpolation = 'CONSTANT'
    fc.update()

# -------------------------------------------------------------------------
# Snapshot existing V10R15 f660 pose & camera for 0.00mm seam verification!
# -------------------------------------------------------------------------
sc.frame_set(F_SEAM)
bpy.context.view_layer.update()
pbs = arm_ob.pose.bones
exist_hips_loc_660 = pbs["Hips"].location.copy()
exist_quat_660     = {pb.name: pb.rotation_quaternion.copy() for pb in pbs}
exist_cam_loc_660  = cam_obj.location.copy()
exist_cam_quat_660 = cam_obj.rotation_quaternion.copy()

# -------------------------------------------------------------------------
# Armature Rest Geometry + Exact Shoe-Sole LBS Precomputation
# -------------------------------------------------------------------------
Mw_xm     = arm_ob.matrix_world.copy()
Mw_xm_inv = Mw_xm.inverted()
R_arm_w     = Mw_xm.to_3x3().normalized()
R_arm_w_inv = R_arm_w.inverted()
arm_world_scale = Mw_xm.to_scale().z  # 1.130

xm_rest_R, xm_rest_u, xm_rest_head = {}, {}, {}
bone_local_R   = {}
bone_head_arm  = {}
bone_tail_arm  = {}
bone_parent_nm = {}
for b in arm_ob.data.bones:
    Rw = (R_arm_w @ b.matrix_local.to_3x3()).normalized()
    hw = Mw_xm @ b.head_local
    tw = Mw_xm @ b.tail_local
    xm_rest_R[b.name]    = Rw
    xm_rest_head[b.name] = hw
    dv = tw - hw
    xm_rest_u[b.name]    = dv.normalized() if dv.length > 1e-6 else Vector((0, 0, 1))
    bone_local_R[b.name]  = b.matrix_local.to_3x3().normalized()
    bone_head_arm[b.name] = b.head_local.copy()
    bone_tail_arm[b.name] = b.tail_local.copy()
    bone_parent_nm[b.name] = b.parent.name if b.parent else None

bone_rest_rel_R = {}
bone_rest_rel_p = {}
bone_rest_tail_local = {}
for b in arm_ob.data.bones:
    bn = b.name
    if b.parent:
        pn = b.parent.name
        bone_rest_rel_R[bn] = (bone_local_R[pn].inverted() @ bone_local_R[bn]).normalized()
        bone_rest_rel_p[bn] = bone_local_R[pn].inverted() @ (bone_head_arm[bn] - bone_head_arm[pn])
    else:
        bone_rest_rel_R[bn] = bone_local_R[bn]
        bone_rest_rel_p[bn] = bone_head_arm[bn].copy()
    bone_rest_tail_local[bn] = bone_local_R[bn].inverted() @ (bone_tail_arm[bn] - bone_head_arm[bn])

# Shoe LBS sole helper (verified against evaluated_get.to_mesh)
R_mw_np = np.array(Mw_xm.to_3x3(), dtype=np.float64)
t_mw_np = np.array(Mw_xm.translation, dtype=np.float64)

def build_shoe_sole_lbs(sh_ob, foot_bn, toe_bn):
    me = sh_ob.data
    vg_map = {vg.name: vg.index for vg in sh_ob.vertex_groups}
    if_f = vg_map.get(foot_bn, -1)
    if_t = vg_map.get(toe_bn, -1)
    # Rest mesh coords directly in Armature local space (at rest!)
    M_ob_to_arm = Mw_xm_inv @ sh_ob.matrix_world
    pts_arm, wf_l, wt_l = [], [], []
    p_all = [M_ob_to_arm @ v.co for v in me.vertices]
    wz_all = [(Mw_xm @ p).z for p in p_all]
    z_min0 = min(wz_all)
    for v, p_a, wz in zip(me.vertices, p_all, wz_all):
        if wz <= z_min0 + 0.035:
            wf, wt = 0.0, 0.0
            for g in v.groups:
                if g.group == if_f: wf = g.weight
                elif g.group == if_t: wt = g.weight
            ws = wf + wt
            if ws > 1e-4:
                wf /= ws; wt /= ws
            else:
                wf = 1.0; wt = 0.0
            pts_arm.append([p_a.x, p_a.y, p_a.z])
            wf_l.append(wf); wt_l.append(wt)
    pts_np = np.array(pts_arm, dtype=np.float64)
    Rf0 = np.array(bone_local_R[foot_bn], dtype=np.float64)
    Rt0 = np.array(bone_local_R[toe_bn], dtype=np.float64)
    pf0 = np.array(bone_head_arm[foot_bn], dtype=np.float64)
    pt0 = np.array(bone_head_arm[toe_bn], dtype=np.float64)
    return (
        (pts_np - pf0) @ Rf0,
        (pts_np - pt0) @ Rt0,
        np.array(wf_l, dtype=np.float64)[:, None],
        np.array(wt_l, dtype=np.float64)[:, None],
    )

sole_L = build_shoe_sole_lbs(shoeL_ob, "foot.L", "toe.L")
sole_R = build_shoe_sole_lbs(shoeR_ob, "foot.R", "toe.R")

def eval_sole_min_z(R_arm_map, p_arm_map):
    zm = 999.0
    for (fbn, tbn), (df, dt_v, wf_a, wt_a) in [(("foot.L", "toe.L"), sole_L), (("foot.R", "toe.R"), sole_R)]:
        Rf = np.array(R_arm_map[fbn], dtype=np.float64)
        Rt = np.array(R_arm_map[tbn], dtype=np.float64)
        pf = np.array(p_arm_map[fbn], dtype=np.float64)
        pt = np.array(p_arm_map[tbn], dtype=np.float64)
        p_posed_arm = wf_a * (pf + df @ Rf.T) + wt_a * (pt + dt_v @ Rt.T)
        wz = (p_posed_arm @ R_mw_np[2, :]) + t_mw_np[2]
        zm = min(zm, float(wz.min()))
    return zm - 0.00148

# Verify Sole LBS at f=500 & f=660!
for f_chk in (500, 660):
    sc.frame_set(f_chk)
    dg_c = bpy.context.evaluated_depsgraph_get()
    Rm_c = {bn: pbs[bn].matrix.to_3x3().normalized() for bn in ("foot.L", "toe.L", "foot.R", "toe.R")}
    pm_c = {bn: pbs[bn].head.copy() for bn in ("foot.L", "toe.L", "foot.R", "toe.R")}
    z_lbs = eval_sole_min_z(Rm_c, pm_c)
    z_msh = 999.0
    for sh_ob in (shoeL_ob, shoeR_ob):
        eo = sh_ob.evaluated_get(dg_c); me = eo.to_mesh()
        co = np.empty(len(me.vertices) * 3, dtype=np.float32); me.vertices.foreach_get("co", co)
        Ms = np.array(eo.matrix_world, dtype=np.float32); eo.to_mesh_clear()
        z_msh = min(z_msh, float(((co.reshape(-1, 3) @ Ms[:3, 2]) + Ms[2, 3]).min()))
    print(f"  [Sole LBS verify f={f_chk}] z_lbs={z_lbs:.5f} vs z_mesh={z_msh:.5f} (err={abs(z_lbs-z_msh)*1000:.3f}mm)", flush=True)

# -------------------------------------------------------------------------
# Verbatim V10R12 BONE_MAP + R_xm_apose_base + Pure-Mathutils FK!
# -------------------------------------------------------------------------
z_miku_hips0 = float(miku_rest_head["下半身"].z)
z_xm_hips0   = float(xm_rest_head["Hips"].z)
leg_ratio    = z_xm_hips0 / max(z_miku_hips0, 0.5)
miku_hips_rest_pos = miku_rest_head["下半身"]
xm_hips_rest_pos   = xm_rest_head["Hips"]
R_hips_rest_w      = xm_rest_R["Hips"]
R_hips_rest_w_inv  = R_hips_rest_w.inverted()

BONE_MAP = [
    ("Hips",     "下半身",  "direct"),
    ("Spine",    "上半身",  "direct"),
    ("Chest",    "上半身2", "direct"),
    ("Neck",     "首",     "direct"),
    ("Head",     "頭",     "direct"),
    ("LeftEye",  "左目",   "direct"),
    ("RightEye", "右目",   "direct"),
]
for side, s_jp in [("L", "左"), ("R", "右")]:
    BONE_MAP.extend([
        (f"shoulder.{side}",            f"{s_jp}肩",     "swing"),
        (f"upper_arm.{side}",           f"{s_jp}腕捩",   "arm_upper"),
        (f"lower_arm.{side}",           f"{s_jp}ひじ",   "swing"),
        (f"lower_arm_Twist.{side}",     f"{s_jp}手捩",   "swing"),
        (f"hand.{side}",                f"{s_jp}手首",   "swing"),
        (f"upper_leg.{side}",           f"{s_jp}足",     "swing"),
        (f"lower_leg.{side}",           f"{s_jp}ひざ",   "swing"),
        (f"foot.{side}",                f"{s_jp}足首",   "direct"),
        (f"toe.{side}",                 f"{s_jp}つま先", "direct"),
        (f"Thumb Proximal.{side}",      f"{s_jp}親指０", "swing"),
        (f"Thumb Intermediate.{side}",  f"{s_jp}親指１", "swing"),
        (f"Thumb Distal.{side}",        f"{s_jp}親指２", "swing"),
        (f"Index Proximal.{side}",      f"{s_jp}人指１", "swing"),
        (f"Index Intermediate.{side}",  f"{s_jp}人指２", "swing"),
        (f"Index Distal.{side}",        f"{s_jp}人指３", "swing"),
        (f"Middle Proximal.{side}",     f"{s_jp}中指１", "swing"),
        (f"Middle Intermediate.{side}", f"{s_jp}中指２", "swing"),
        (f"Middle Distal.{side}",       f"{s_jp}中指３", "swing"),
        (f"Ring Proximal.{side}",       f"{s_jp}薬指１", "swing"),
        (f"Ring Intermediate.{side}",   f"{s_jp}薬指２", "swing"),
        (f"Ring Distal.{side}",         f"{s_jp}薬指３", "swing"),
        (f"Little Proximal.{side}",     f"{s_jp}小指１", "swing"),
        (f"Little Intermediate.{side}", f"{s_jp}小指２", "swing"),
        (f"Little Distal.{side}",       f"{s_jp}小指３", "swing"),
    ])

R_xm_apose_base = {}
for xm_bn, jp_bn, mode in BONE_MAP:
    if xm_bn not in xm_rest_R or jp_bn not in miku_rest_R:
        continue
    if mode == "direct":
        R_xm_apose_base[xm_bn] = xm_rest_R[xm_bn]
    else:
        if jp_bn.endswith("腕捩"):
            s_jp = jp_bn[0]
            v_m = (miku_rest_head[f"{s_jp}ひじ"] - miku_rest_head[f"{s_jp}腕"]).normalized()
        elif jp_bn.endswith("手捩"):
            s_jp = jp_bn[0]
            v_m = (miku_rest_head[f"{s_jp}手首"] - miku_rest_head[f"{s_jp}ひじ"]).normalized()
        elif jp_bn.endswith("手首"):
            s_jp = jp_bn[0]
            v_m = (miku_rest_head[f"{s_jp}中指１"] - miku_rest_head[f"{s_jp}手首"]).normalized()
        else:
            dv_m = miku_rest_tail[jp_bn] - miku_rest_head[jp_bn]
            v_m = dv_m.normalized() if dv_m.length > 1e-4 else (miku_rest_head[f"{jp_bn[0]}中指１"] - miku_rest_head[f"{jp_bn[0]}手首"]).normalized()
        q_sw = xm_rest_u[xm_bn].rotation_difference(v_m)
        R_xm_apose_base[xm_bn] = (q_sw.to_matrix() @ xm_rest_R[xm_bn]).normalized()

VALID_BONE_MAP = [(xb, jb, md) for (xb, jb, md) in BONE_MAP if xb in R_xm_apose_base and jb in jp_idx]
RETARGET_NAMES = [xb for xb, _, _ in VALID_BONE_MAP]

# Pure-dict FK helpers (zero Blender depsgraph calls!)
cur_q_local = {}
cur_R_arm   = {}
cur_p_arm   = {}
hips_loc_cur = Vector((0.0, 0.0, 0.0))

def reset_fk():
    cur_R_arm.clear()
    cur_p_arm.clear()

def get_arm_R(bn):
    if bn in cur_R_arm:
        return cur_R_arm[bn]
    pn = bone_parent_nm[bn]
    q_b = cur_q_local.get(bn, Quaternion((1.0, 0.0, 0.0, 0.0)))
    if pn:
        R_par = get_arm_R(pn)
        R_c = (R_par @ bone_rest_rel_R[bn] @ q_b.to_matrix()).normalized()
    else:
        R_c = (bone_rest_rel_R[bn] @ q_b.to_matrix()).normalized()
    cur_R_arm[bn] = R_c
    return R_c

def get_arm_p(bn):
    if bn in cur_p_arm:
        return cur_p_arm[bn]
    pn = bone_parent_nm[bn]
    if pn:
        p_c = get_arm_p(pn) + get_arm_R(pn) @ bone_rest_rel_p[bn]
    else:
        p_c = bone_rest_rel_p[bn].copy()
    if bn == "Hips":
        p_c = p_c + bone_local_R["Hips"] @ hips_loc_cur
    cur_p_arm[bn] = p_c
    return p_c

def get_arm_tail(bn):
    return get_arm_p(bn) + get_arm_R(bn) @ bone_rest_tail_local[bn]

def set_world_rot(bn, R_target_world):
    R_target_arm = (R_arm_w_inv @ R_target_world).normalized()
    pn = bone_parent_nm[bn]
    if pn:
        R_par = get_arm_R(pn)
        R_basis = (R_par @ bone_rest_rel_R[bn]).inverted() @ R_target_arm
    else:
        R_basis = bone_rest_rel_R[bn].inverted() @ R_target_arm
    q_b = R_basis.to_quaternion()
    q_b.normalize()
    cur_q_local[bn] = q_b
    cur_R_arm[bn]   = R_target_arm

# -------------------------------------------------------------------------
# Phase 2: Full-Song Kinematic Solve f1..f6265 (exact warm-start for physics + zero-seam f660!)
# -------------------------------------------------------------------------
t_ret = time.time()
hips_loc_all    = np.zeros((N_TOTAL, 3), dtype=np.float64)
retarget_q_all  = {bn: np.zeros((N_TOTAL, 4), dtype=np.float64) for bn in RETARGET_NAMES}
hips_pos_seq    = np.zeros((N_TOTAL, 3), dtype=np.float64)
hips_yaw_seq    = np.zeros(N_TOTAL, dtype=np.float64)
thighL_lift_seq = np.zeros(N_TOTAL, dtype=np.float64)
thighR_lift_seq = np.zeros(N_TOTAL, dtype=np.float64)
wristL_pos_seq  = np.zeros((N_TOTAL, 3), dtype=np.float64)
wristR_pos_seq  = np.zeros((N_TOTAL, 3), dtype=np.float64)
chest_z_seq     = np.zeros(N_TOTAL, dtype=np.float64)
head_pos_seq    = np.zeros((N_TOTAL, 3), dtype=np.float64)
headup_lidded_seq = np.zeros(N_TOTAL, dtype=np.float64)
up_pitch_clamped_seq = np.zeros(N_TOTAL, dtype=np.float64)

b_head_miku = jp_idx["頭"]

for idx in range(N_TOTAL):
    f = F_START + idx
    cur_q_local.clear()
    reset_fk()

    mp = Vector(miku_hips_pos[idx].tolist())
    dz_hips = (mp.z - miku_hips_rest_pos.z) * leg_ratio
    target_hips_w = Vector((
        mp.x * 0.98,
        mp.y,
        xm_hips_rest_pos.z + dz_hips,
    ))

    # Seat support for opening chair segment (f <= 462) AND ending return-to-chair (f >= 6248)!
    if f <= 462 or f >= 6248:
        if f <= 462:
            w_sit_clamp = 1.0 if f <= 445 else max(0.0, (462 - f) / 17.0)
        else:
            w_sit_clamp = min(1.0, (f - 6248) / 12.0)
        min_hips_sit_z = 0.652
        if target_hips_w.z < min_hips_sit_z:
            target_hips_w.z = target_hips_w.z * (1.0 - w_sit_clamp) + (min_hips_sit_z + 0.15 * (target_hips_w.z - min_hips_sit_z)) * w_sit_clamp

    d_hips_w = target_hips_w - xm_hips_rest_pos
    hips_loc_cur = (R_hips_rest_w_inv @ d_hips_w) / arm_world_scale

    # Head-up pitch inspection (verbatim V10R12!)
    R_m_head = Matrix(miku_pose_R[idx, b_head_miku].tolist())
    R_head_raw_w = (R_m_head @ miku_rest_R["頭"].inverted() @ R_xm_apose_base["Head"]).normalized()
    M_head_delta_raw = R_head_raw_w @ xm_rest_R["Head"].inverted()
    v_face_fwd_raw = (M_head_delta_raw @ Vector((0.0, -1.0, 0.0))).normalized()
    up_pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, v_face_fwd_raw.z))))

    extra_chest_arch_rad = 0.0
    lidded_comp = 0.0
    if up_pitch_deg > 8.5:
        excess_deg = up_pitch_deg - 8.5
        extra_chest_arch_rad = math.radians(min(8.5, excess_deg * 0.22))
        lidded_comp = min(0.18, excess_deg * 0.0048)
    headup_lidded_seq[idx] = lidded_comp

    for xm_bn, jp_bn, mode in VALID_BONE_MAP:
        R_m_pose  = Matrix(miku_pose_R[idx, jp_idx[jp_bn]].tolist())
        R_m_rest  = miku_rest_R[jp_bn]
        R_delta_w = R_m_pose @ R_m_rest.inverted()
        R_tgt_w   = (R_delta_w @ R_xm_apose_base[xm_bn]).normalized()

        if xm_bn == "Chest" and extra_chest_arch_rad > 1e-4:
            q_arch = Quaternion(Vector((-1.0, 0.0, 0.0)), extra_chest_arch_rad).to_matrix()
            R_tgt_w = (q_arch @ R_tgt_w).normalized()

        elif xm_bn in ("Neck", "Head") and up_pitch_deg > 8.5:
            M_hd = R_tgt_w @ xm_rest_R[xm_bn].inverted()
            v_fwd = (M_hd @ Vector((0.0, -1.0, 0.0))).normalized()
            cur_el = math.degrees(math.asin(max(-1.0, min(1.0, v_fwd.z))))
            cap_start = 6.0 if xm_bn == "Neck" else 8.5
            if cur_el > cap_start:
                new_el = cap_start + 0.19 * (cur_el - cap_start)
                horiz = Vector((v_fwd.x, v_fwd.y, 0.0))
                if horiz.length > 1e-5:
                    horiz.normalize()
                    v_fwd_clamped = (horiz * math.cos(math.radians(new_el)) + Vector((0.0, 0.0, math.sin(math.radians(new_el))))).normalized()
                    q_fix = v_fwd.rotation_difference(v_fwd_clamped).to_matrix()
                    R_tgt_w = (q_fix @ R_tgt_w).normalized()

        elif xm_bn in ("LeftEye", "RightEye") and up_pitch_deg > 14.0:
            look_down_rad = math.radians(min(8.5, (up_pitch_deg - 14.0) * 0.30))
            q_eye = Quaternion(Vector((+1.0, 0.0, 0.0)), look_down_rad).to_matrix()
            R_tgt_w = (q_eye @ R_tgt_w).normalized()

        elif xm_bn.startswith("shoulder."):
            R_base_sh = R_xm_apose_base[xm_bn]
            q_diff_sh = (R_tgt_w @ R_base_sh.inverted()).to_quaternion()
            if q_diff_sh.angle > math.radians(5.5):
                q_clamped = Quaternion(q_diff_sh.axis, math.radians(5.5) + 0.20 * (q_diff_sh.angle - math.radians(5.5)))
                R_tgt_w = (q_clamped.to_matrix() @ R_base_sh).normalized()

        elif mode == "arm_upper":
            sgn_x = +1.0 if xm_bn.endswith(".L") else -1.0
            u_cur = ((R_tgt_w @ xm_rest_R[xm_bn].inverted()) @ xm_rest_u[xm_bn]).normalized()
            u_mod = Vector(u_cur)
            if u_mod.z > 0.10:
                u_mod.z = 0.10 + 0.30 * (u_mod.z - 0.10)
            lat = sgn_x * u_mod.x
            if lat < 0.18:
                u_mod.x = sgn_x * (0.18 + 0.25 * (lat - 0.18))
            u_mod.normalize()
            R_tgt_w = (u_cur.rotation_difference(u_mod).to_matrix() @ R_tgt_w).normalized()

        elif xm_bn.startswith("lower_arm.") and not xm_bn.startswith("lower_arm_Twist"):
            side_s = xm_bn.split(".")[1]
            R_ua_w = R_arm_w @ get_arm_R(f"upper_arm.{side_s}")
            u_ua_w = ((R_ua_w @ xm_rest_R[f"upper_arm.{side_s}"].inverted()) @ xm_rest_u[f"upper_arm.{side_s}"]).normalized()
            u_fa_w = ((R_tgt_w @ xm_rest_R[xm_bn].inverted()) @ xm_rest_u[xm_bn]).normalized()
            dot_uf = max(-1.0, min(1.0, float(u_ua_w.dot(u_fa_w))))
            bend_deg = math.degrees(math.acos(dot_uf))
            if bend_deg > 100.0:
                new_bend = 100.0 + 0.25 * (bend_deg - 100.0)
                ax_b = u_ua_w.cross(u_fa_w)
                if ax_b.length > 1e-5:
                    ax_b.normalize()
                    u_fa_safe = (Quaternion(ax_b, math.radians(new_bend)) @ u_ua_w).normalized()
                    R_tgt_w = (u_fa_w.rotation_difference(u_fa_safe).to_matrix() @ R_tgt_w).normalized()

        set_world_rot(xm_bn, R_tgt_w)

    # Record clamped head pitch
    R_hd_fin = R_arm_w @ get_arm_R("Head")
    v_f_fin  = ((R_hd_fin @ xm_rest_R["Head"].inverted()) @ Vector((0.0, -1.0, 0.0))).normalized()
    up_pitch_clamped_seq[idx] = math.degrees(math.asin(max(-1.0, min(1.0, v_f_fin.z))))

    # Wide-cuff 0.242m wrist cushion (verbatim V10R12!)
    wL_now = Mw_xm @ get_arm_p("hand.L")
    wR_now = Mw_xm @ get_arm_p("hand.R")
    dist_wr = (wL_now - wR_now).length
    if dist_wr < 0.242:
        push_rad = min(0.16, (0.242 - dist_wr) * 1.05)
        for side_c, sgn_c in [("L", +1.0), ("R", -1.0)]:
            fa_bn = f"lower_arm.{side_c}"
            R_fa_cur = R_arm_w @ get_arm_R(fa_bn)
            q_sep = Euler((-push_rad * 0.45, 0.0, sgn_c * push_rad * 0.85), 'XYZ').to_matrix()
            # Clear cached downstream arm transforms on this side before updating lower_arm!
            cur_R_arm.clear()
            cur_p_arm.clear()
            set_world_rot(fa_bn, (q_sep @ R_fa_cur).normalized())
        cur_R_arm.clear()
        cur_p_arm.clear()

    # Standing dance shoe-sole grounding Z=0.002m (verbatim V10R12!)
    if f >= 480 and f < 6248:
        Rm_s = {bn: get_arm_R(bn) for bn in ("foot.L", "toe.L", "foot.R", "toe.R")}
        pm_s = {bn: get_arm_p(bn) for bn in ("foot.L", "toe.L", "foot.R", "toe.R")}
        z_sole_min = eval_sole_min_z(Rm_s, pm_s)
        ankL_z = float(miku_ank_z[idx, 0])
        ankR_z = float(miku_ank_z[idx, 1])
        if z_sole_min < 0.001 or (ankL_z < 0.115 or ankR_z < 0.115):
            if z_sole_min < 0.028:
                corr_z = 0.002 - z_sole_min
                d_hips_w.z += corr_z
                hips_loc_cur = (R_hips_rest_w_inv @ d_hips_w) / arm_world_scale
                cur_p_arm.clear()

    hips_w_cur = Mw_xm @ get_arm_p("Hips")
    hips_fwd_w = (R_arm_w @ get_arm_R("Hips") @ Vector((0, 0, -1))).normalized()
    hips_pos_seq[idx] = [hips_w_cur.x, hips_w_cur.y, hips_w_cur.z]
    yaw_val = math.atan2(hips_fwd_w.x, -hips_fwd_w.y)
    hips_yaw_seq[idx] = yaw_val

    for side, seq_arr in [("L", thighL_lift_seq), ("R", thighR_lift_seq)]:
        ul_bn = f"upper_leg.{side}"
        ul_vec = ((Mw_xm @ get_arm_tail(ul_bn)) - (Mw_xm @ get_arm_p(ul_bn))).normalized()
        fwd_amt = -ul_vec.y * math.cos(yaw_val) + ul_vec.x * math.sin(yaw_val)
        seq_arr[idx] = math.atan2(max(0.0, fwd_amt), max(0.05, -ul_vec.z))

    wL_f = Mw_xm @ get_arm_p("hand.L")
    wR_f = Mw_xm @ get_arm_p("hand.R")
    wristL_pos_seq[idx] = [wL_f.x, wL_f.y, wL_f.z]
    wristR_pos_seq[idx] = [wR_f.x, wR_f.y, wR_f.z]
    chest_z_seq[idx]    = float((Mw_xm @ get_arm_p("Chest")).z)
    hd_w = Mw_xm @ get_arm_p("Head")
    head_pos_seq[idx]   = [hd_w.x, hd_w.y, hd_w.z]

    hips_loc_all[idx] = [hips_loc_cur.x, hips_loc_cur.y, hips_loc_cur.z]
    for bn in RETARGET_NAMES:
        q = cur_q_local[bn]
        retarget_q_all[bn][idx] = [q.w, q.x, q.y, q.z]

# Check exact match at f=660 (idx=659) against V10R15!
idx_seam = F_SEAM - 1  # 659
hips_diff_660 = np.array(exist_hips_loc_660, dtype=np.float64) - hips_loc_all[idx_seam]
max_rot_diff_660 = 0.0
for bn in RETARGET_NAMES:
    q_ex = exist_quat_660[bn]
    q_nw = Quaternion(retarget_q_all[bn][idx_seam].tolist())
    if q_ex.dot(q_nw) < 0: q_nw = -q_nw
    deg_d = math.degrees(2.0 * math.acos(min(1.0, abs(q_ex.dot(q_nw)))))
    if deg_d > max_rot_diff_660: max_rot_diff_660 = deg_d

print(f"[2/6] Kinematic solve {N_TOTAL}f in {time.time()-t_ret:.2f}s | f660 match: hips_err={np.linalg.norm(hips_diff_660)*1000:.4f}mm, rot_err={max_rot_diff_660:.5f}deg, max_clamped_head_up={up_pitch_clamped_seq.max():.2f}deg", flush=True)

# C1 blend any tiny sub-mm residual across f661..f668 & enforce quaternion sign continuity from exist_quat_660!
for k in range(1, min(9, N_NEW + 1)):
    w_s = 0.5 * (1.0 + math.cos(math.pi * k / 8.0))
    hips_loc_all[idx_seam + k] += hips_diff_660 * w_s

for bn in RETARGET_NAMES:
    q_prev = np.array([exist_quat_660[bn].w, exist_quat_660[bn].x, exist_quat_660[bn].y, exist_quat_660[bn].z], dtype=np.float64)
    q0 = retarget_q_all[bn][idx_seam].copy()
    if np.dot(q_prev, q0) < 0: q0 = -q0
    dq_seam = Quaternion(q_prev.tolist()) @ Quaternion(q0.tolist()).inverted()
    for k in range(1, N_NEW + 1):
        qk = retarget_q_all[bn][idx_seam + k]
        if k <= 8:
            w_s = 0.5 * (1.0 + math.cos(math.pi * k / 8.0))
            qb = Quaternion((1, 0, 0, 0)).slerp(dq_seam, w_s) @ Quaternion(qk.tolist())
            qb.normalize()
            qk = np.array([qb.w, qb.x, qb.y, qb.z], dtype=np.float64)
        if np.dot(q_prev, qk) < 0: qk = -qk
        retarget_q_all[bn][idx_seam + k] = qk
        q_prev = qk

# -------------------------------------------------------------------------
# Phase 3: Verbatim V10R12 68-Bone Physics Integrator across f1..f6265!
# -------------------------------------------------------------------------
t_phy = time.time()
def central_acc(arr):
    vel = np.gradient(arr, DT, axis=0)
    acc = np.gradient(vel, DT, axis=0)
    return vel, np.clip(acc, -18.0, 18.0)

hips_vel, hips_acc = central_acc(hips_pos_seq)
head_vel, head_acc = central_acc(head_pos_seq)
_, wrL_acc         = central_acc(wristL_pos_seq)
_, wrR_acc         = central_acc(wristR_pos_seq)
_, chZ_acc         = central_acc(chest_z_seq)

COAT_SECTORS = [("FL", -0.115, -0.095), ("FR", +0.115, -0.095), ("SL", -0.185, +0.010), ("SR", +0.185, +0.010), ("BK", 0.000, +0.145)]
skirt_out_axes = {}
for sec, _, _ in COAT_SECTORS:
    h01 = xm_rest_head[f"PHY_Coat_{sec}_01"]
    hips0 = xm_rest_head["Hips"]
    rad_xy = Vector((h01.x - hips0.x, h01.y - hips0.y, 0.0))
    if rad_xy.length < 1e-4: rad_xy = Vector((0.0, 1.0, 0.0))
    rad_xy.normalize()
    tangent_flare_ax = Vector((rad_xy.y, -rad_xy.x, 0.0)).normalized()
    skirt_out_axes[sec] = (rad_xy, tangent_flare_ax)

q_sk  = np.zeros((5, 2), dtype=np.float64); dq_sk = np.zeros((5, 2), dtype=np.float64)
q_aux = np.zeros(4, dtype=np.float64);      dq_aux = np.zeros(4, dtype=np.float64)
q_hc  = np.zeros((12, 2), dtype=np.float64); dq_hc = np.zeros((12, 2), dtype=np.float64)
q_ah  = np.zeros(2, dtype=np.float64);      dq_ah = np.zeros(2, dtype=np.float64)
hc_az = np.array([math.radians(i * 30.0) for i in range(12)], dtype=np.float64)

def step_phys(fi, sub_dt):
    global q_sk, dq_sk, q_aux, dq_aux, q_hc, dq_hc, q_ah, dq_ah
    ah = hips_acc[fi]; hd_a = head_acc[fi]; hv = head_vel[fi]
    thL = thighL_lift_seq[fi]; thR = thighR_lift_seq[fi]
    hz  = hips_pos_seq[fi, 2]
    sit_w = 0.0
    if fi < 468:
        sit_w = max(0.0, min(1.0, (0.86 - hz) / 0.16)) * (1.0 if fi < 448 else (468 - fi) / 20.0)
    elif fi >= 6245:
        sit_w = max(0.0, min(1.0, (0.86 - hz) / 0.16)) * min(1.0, (fi - 6245) / 14.0)

    for si, (sec, _, _) in enumerate(COAT_SECTORS):
        rad_v, _ = skirt_out_axes[sec]
        inert = -(ah[0] * rad_v.x + ah[1] * rad_v.y) * 0.032 + max(0.0, -ah[2]) * 0.018
        min_q0, min_q1 = -0.10, -0.12
        if sec == "FL":
            min_q0 = max(min_q0, thL * 0.78 + 0.34 * sit_w)
            min_q1 = max(min_q1, thL * 0.36)
        elif sec == "FR":
            min_q0 = max(min_q0, thR * 0.78 + 0.34 * sit_w)
            min_q1 = max(min_q1, thR * 0.36)
        elif sec in ("SL", "SR"):
            th_s = thL if sec == "SL" else thR
            min_q0 = max(min_q0, 0.54 * sit_w + th_s * 0.28)
            min_q1 = max(min_q1, 0.32 * sit_w)
        elif sec == "BK":
            min_q0 = max(min_q0, 0.62 * sit_w)
            min_q1 = max(min_q1, 0.48 * sit_w)

        k0, c0 = 62.0, 9.2
        k1, c1 = 46.0, 7.6
        f0 = -k0 * (q_sk[si, 0] - min_q0) - c0 * dq_sk[si, 0] + inert * 14.0
        if q_sk[si, 0] < min_q0:
            f0 += 280.0 * (min_q0 - q_sk[si, 0])
        f1 = -k1 * (q_sk[si, 1] - max(min_q1, q_sk[si, 0] * 0.55)) - c1 * dq_sk[si, 1] + inert * 19.0
        dq_sk[si, 0] += f0 * sub_dt; q_sk[si, 0] = np.clip(q_sk[si, 0] + dq_sk[si, 0] * sub_dt, min_q0 - 0.02, 1.25)
        dq_sk[si, 1] += f1 * sub_dt; q_sk[si, 1] = np.clip(q_sk[si, 1] + dq_sk[si, 1] * sub_dt, min_q1 - 0.02, 1.30)

    drv_aux = [
        float(np.clip(-wrL_acc[fi, 2] * 0.015 + wrL_acc[fi, 1] * 0.010, -0.45, 0.45)),
        float(np.clip(-wrR_acc[fi, 2] * 0.015 + wrR_acc[fi, 1] * 0.010, -0.45, 0.45)),
        float(np.clip(-chZ_acc[fi] * 0.016, -0.35, 0.35)),
        float(np.clip(-chZ_acc[fi] * 0.016, -0.35, 0.35)),
    ]
    for ai in range(4):
        fa = -75.0 * (q_aux[ai] - drv_aux[ai]) - 10.5 * dq_aux[ai]
        dq_aux[ai] += fa * sub_dt
        q_aux[ai] = float(np.clip(q_aux[ai] + dq_aux[ai] * sub_dt, -0.42, 0.48))

    for hi in range(12):
        az = hc_az[hi]
        rx, ry = math.sin(az), math.cos(az)
        in_h = -(hd_a[0] * rx + hd_a[1] * ry) * 0.018 - (hv[0] * rx + hv[1] * ry) * 0.05 + max(0.0, -hd_a[2]) * 0.010
        fh0 = -88.0 * (q_hc[hi, 0] - 0.03) - 11.5 * dq_hc[hi, 0] + in_h * 18.0
        fh1 = -68.0 * (q_hc[hi, 1] - q_hc[hi, 0] * 0.65) - 9.5 * dq_hc[hi, 1] + in_h * 24.0
        dq_hc[hi, 0] += fh0 * sub_dt; q_hc[hi, 0] = np.clip(q_hc[hi, 0] + dq_hc[hi, 0] * sub_dt, -0.05, 0.38)
        dq_hc[hi, 1] += fh1 * sub_dt; q_hc[hi, 1] = np.clip(q_hc[hi, 1] + dq_hc[hi, 1] * sub_dt, -0.06, 0.46)

    in_ah = -hd_a[1] * 0.022 - hd_a[2] * 0.016
    fa0 = -95.0 * q_ah[0] - 9.0 * dq_ah[0] + in_ah * 22.0
    fa1 = -72.0 * (q_ah[1] - q_ah[0] * 0.7) - 7.5 * dq_ah[1] + in_ah * 28.0
    dq_ah[0] += fa0 * sub_dt; q_ah[0] = float(np.clip(q_ah[0] + dq_ah[0] * sub_dt, -0.35, 0.38))
    dq_ah[1] += fa1 * sub_dt; q_ah[1] = float(np.clip(q_ah[1] + dq_ah[1] * sub_dt, -0.42, 0.45))

for _ in range(60):
    for _ in range(4): step_phys(0, DT / 4.0)

phys_traj_sk  = np.zeros((N_TOTAL, 5, 2)); phys_traj_aux = np.zeros((N_TOTAL, 4))
phys_traj_hc  = np.zeros((N_TOTAL, 12, 2)); phys_traj_ah  = np.zeros((N_TOTAL, 2))
for fi in range(N_TOTAL):
    for _ in range(4): step_phys(fi, DT / 4.0)
    phys_traj_sk[fi] = q_sk; phys_traj_aux[fi] = q_aux
    phys_traj_hc[fi] = q_hc; phys_traj_ah[fi] = q_ah

# Convert physics trajectories into local bone quaternions for f660..f6265!
PHYS_BONE_NAMES = []
for sec, _, _ in COAT_SECTORS:
    for sfx in ("01", "02"):
        bn = f"PHY_Coat_{sec}_{sfx}"
        if bn in pbs: PHYS_BONE_NAMES.append(bn)
for bn in ["PHY_Cuff_L", "PHY_Cuff_R", "PHY_Bow_L", "PHY_Bow_R"]:
    if bn in pbs: PHYS_BONE_NAMES.append(bn)
for hi in range(12):
    for sj in (1, 2, 3, 4):
        bn = f"Hc_{hi}_{sj}"
        if bn in pbs: PHYS_BONE_NAMES.append(bn)
for sj in (1, 2, 3, 4, 5):
    bn = f"Hair_Ahoge_{sj:02d}"
    if bn in pbs: PHYS_BONE_NAMES.append(bn)

phys_q_new = {bn: np.zeros((N_NEW + 1, 4), dtype=np.float64) for bn in PHYS_BONE_NAMES}

for k in range(N_NEW + 1):
    fi = idx_seam + k  # 659..6264 (f=660..6265)
    cur_q_local.clear()
    reset_fk()
    for bn in RETARGET_NAMES:
        qw, qx, qy, qz = retarget_q_all[bn][fi]
        cur_q_local[bn] = Quaternion((qw, qx, qy, qz))

    yaw_q = Quaternion(Vector((0, 0, 1)), float(hips_yaw_seq[fi])).to_matrix()
    for si, (sec, _, _) in enumerate(COAT_SECTORS):
        _, ax_rest = skirt_out_axes[sec]
        ax_w = (yaw_q @ ax_rest).normalized()
        for cj, sfx in [(0, "01"), (1, "02")]:
            bn = f"PHY_Coat_{sec}_{sfx}"
            if bn in pbs:
                ang = float(phys_traj_sk[fi, si, cj])
                pn = bone_parent_nm[bn]
                R_par_cur = R_arm_w @ get_arm_R(pn)
                R_par_rst = R_arm_w @ bone_local_R[pn]
                R_base_w  = (R_par_cur @ R_par_rst.inverted() @ xm_rest_R[bn]).normalized()
                R_new_w   = (Quaternion(ax_w, ang).to_matrix() @ R_base_w).normalized()
                set_world_rot(bn, R_new_w)
                q = cur_q_local[bn]
                phys_q_new[bn][k] = [q.w, q.x, q.y, q.z]

    for ai, bn in enumerate(["PHY_Cuff_L", "PHY_Cuff_R", "PHY_Bow_L", "PHY_Bow_R"]):
        if bn in pbs:
            q = Quaternion((1, 0, 0), float(phys_traj_aux[fi, ai]))
            phys_q_new[bn][k] = [q.w, q.x, q.y, q.z]

    for hi in range(12):
        for seg_idx, vals in [((1, 2), phys_traj_hc[fi, hi, 0] * 0.5), ((3, 4), phys_traj_hc[fi, hi, 1] * 0.5)]:
            for sj in seg_idx:
                bn = f"Hc_{hi}_{sj}"
                if bn in pbs:
                    q = Quaternion((1, 0, 0), float(vals))
                    phys_q_new[bn][k] = [q.w, q.x, q.y, q.z]

    for aj, segs in [(0, [1, 2]), (1, [3, 4, 5])]:
        for sj in segs:
            bn = f"Hair_Ahoge_{sj:02d}"
            if bn in pbs:
                q = Quaternion((1, 0, 0), float(phys_traj_ah[fi, aj] * 0.45))
                phys_q_new[bn][k] = [q.w, q.x, q.y, q.z]

phys_seam_max_deg = 0.0
for bn in PHYS_BONE_NAMES:
    q_ex = exist_quat_660[bn]
    q0 = Quaternion(phys_q_new[bn][0].tolist())
    if q_ex.dot(q0) < 0: q0 = -q0
    deg_d = math.degrees(2.0 * math.acos(min(1.0, abs(q_ex.dot(q0)))))
    if deg_d > phys_seam_max_deg: phys_seam_max_deg = deg_d
    dq_s = q_ex @ q0.inverted()
    q_prev = np.array([q_ex.w, q_ex.x, q_ex.y, q_ex.z], dtype=np.float64)
    for k in range(1, N_NEW + 1):
        qk = phys_q_new[bn][k]
        if k <= 8:
            w_s = 0.5 * (1.0 + math.cos(math.pi * k / 8.0))
            qb = Quaternion((1, 0, 0, 0)).slerp(dq_s, w_s) @ Quaternion(qk.tolist())
            qb.normalize()
            qk = np.array([qb.w, qb.x, qb.y, qb.z], dtype=np.float64)
        if np.dot(q_prev, qk) < 0: qk = -qk
        phys_q_new[bn][k] = qk
        q_prev = qk

print(f"[3/6] 67-Bone Physics solved ({N_TOTAL}f) in {time.time()-t_phy:.2f}s | f660 match max_err={phys_seam_max_deg:.4f}deg", flush=True)

# -------------------------------------------------------------------------
# Phase 4: Batch-Extend Armature + Dual Shape-Keys + MMD_MV_CAMERA (f661..f6265)
# -------------------------------------------------------------------------
t_fc = time.time()
frames_new_f32 = np.arange(F_SEAM + 1, F_END + 1, dtype=np.float32)  # 661..6265
arm_fcs = get_action_fcurves(arm_ob)

for ax_i in range(3):
    fc = ensure_fcurve(arm_fcs, 'pose.bones["Hips"].location', ax_i)
    extend_fcurve_after_seam(fc, frames_new_f32, hips_loc_all[F_SEAM:, ax_i].astype(np.float32))

for bn in RETARGET_NAMES:
    dp = f'pose.bones["{bn}"].rotation_quaternion'
    for qi in range(4):
        fc = ensure_fcurve(arm_fcs, dp, qi)
        extend_fcurve_after_seam(fc, frames_new_f32, retarget_q_all[bn][F_SEAM:, qi].astype(np.float32))

for bn in PHYS_BONE_NAMES:
    dp = f'pose.bones["{bn}"].rotation_quaternion'
    for qi in range(4):
        fc = ensure_fcurve(arm_fcs, dp, qi)
        extend_fcurve_after_seam(fc, frames_new_f32, phys_q_new[bn][1:, qi].astype(np.float32))

# Dual-mesh Lip-sync + Blinks + Head-Up Lidded Gaze (verbatim V10R12!)
BLINK_CENTERS = [58, 168, 265, 378, 468, 566, 642]
rng_bl = random.Random(20261023)
bc_cur = 738
while bc_cur < F_END - 10:
    BLINK_CENTERS.append(bc_cur)
    bc_cur += rng_bl.randint(84, 114)

def blink_weight(f):
    w = 0.0
    for bc in BLINK_CENTERS:
        d = f - bc
        if -2 <= d <= 2:
            if d == -2: w = max(w, 0.35)
            elif d == -1: w = max(w, 0.88)
            elif d == 0: w = max(w, 0.96)
            elif d == 1: w = max(w, 0.72)
            elif d == 2: w = max(w, 0.25)
    return w

env_arr    = np.array(env_norm[:F_END + 5], dtype=np.float64)
env_smooth = np.convolve(env_arr, np.array([0.2, 0.6, 0.2]), mode='same')

sk_arrays = {kn: np.zeros(N_NEW, dtype=np.float32) for kn in ("Eye_Close_Blink", "あ", "い", "う", "お", "Mouth_Smile_1_Large", "Eyebrow_Up", "Eyebrow_Down")}

for k in range(N_NEW):
    fi = F_SEAM + k
    f  = fi + 1
    voc = float(env_smooth[fi])
    bri = float(env_bright[fi])
    bw  = max(blink_weight(f), float(headup_lidded_seq[fi]))

    if voc < 0.08:
        w_a = w_i = w_u = w_o = 0.0
        w_smile = 0.28 + 0.10 * math.sin(f * 0.05)
    else:
        amp = min(0.62, voc * 0.68)
        if bri > 0.56:
            w_i = amp * 0.72; w_a = amp * 0.28; w_o = 0.0; w_u = 0.0
        elif bri < 0.36:
            w_o = amp * 0.58; w_u = amp * 0.25; w_a = amp * 0.17; w_i = 0.0
        else:
            w_a = amp * 0.76; w_o = amp * 0.24; w_i = 0.0; w_u = 0.0
        w_smile = 0.14

    w_brow_up   = 0.22 * max(0.0, math.sin(f * 0.045)) + (0.18 if f > 475 else 0.0)
    w_brow_down = 0.0

    sk_arrays["Eye_Close_Blink"][k]     = bw
    sk_arrays["あ"][k]                  = w_a
    sk_arrays["い"][k]                  = w_i
    sk_arrays["う"][k]                  = w_u
    sk_arrays["お"][k]                  = w_o
    sk_arrays["Mouth_Smile_1_Large"][k] = w_smile
    sk_arrays["Eyebrow_Up"][k]          = w_brow_up
    sk_arrays["Eyebrow_Down"][k]        = w_brow_down

for mesh_ob in (skin_ob, facial_ob):
    mesh_sk = mesh_ob.data.shape_keys
    if mesh_sk and mesh_sk.animation_data:
        sk_fcs = get_action_fcurves(mesh_sk)
        for kn, arr_v in sk_arrays.items():
            if kn in mesh_sk.key_blocks:
                fc = ensure_fcurve(sk_fcs, f'key_blocks["{kn}"].value', 0)
                extend_fcurve_after_seam(fc, frames_new_f32, arr_v)

# MMD_MV_CAMERA extension (preserving exact f660 Z offset & CONSTANT camera cuts!)
z_off_660 = float(exist_cam_loc_660.z - cam_loc_arr[F_SEAM - 1, 2])
print(f"[4/6] Camera f660 Z offset verified: exist_z - mmd_z = {z_off_660:.5f}m", flush=True)

cam_loc_new  = np.zeros((N_NEW, 3), dtype=np.float32)
cam_quat_new = np.zeros((N_NEW, 4), dtype=np.float32)
cam_lens_new = cam_lens_arr[F_SEAM : F_END].astype(np.float32)

prev_cloc = np.array(exist_cam_loc_660, dtype=np.float64)
prev_cq   = np.array([exist_cam_quat_660.w, exist_cam_quat_660.x, exist_cam_quat_660.y, exist_cam_quat_660.z], dtype=np.float64)
cut_indices = []

for k in range(N_NEW):
    oi = F_SEAM + k
    cloc = cam_loc_arr[oi].astype(np.float64).copy()
    cloc[2] += z_off_660
    cq = cam_quat_arr[oi].astype(np.float64).copy()
    jump = float(np.linalg.norm(cloc - prev_cloc))
    if np.dot(prev_cq, cq) < 0: cq = -cq
    if jump > 0.45:
        cut_indices.append(k)
    cam_loc_new[k]  = cloc
    cam_quat_new[k] = cq
    prev_cloc = cloc
    prev_cq   = cq

cam_fcs  = get_action_fcurves(cam_obj)
camd_fcs = get_action_fcurves(cam_obj.data)
for ax_i in range(3):
    fc = ensure_fcurve(cam_fcs, "location", ax_i)
    extend_fcurve_after_seam(fc, frames_new_f32, cam_loc_new[:, ax_i], constant_cut_indices=cut_indices)
for qi in range(4):
    fc = ensure_fcurve(cam_fcs, "rotation_quaternion", qi)
    extend_fcurve_after_seam(fc, frames_new_f32, cam_quat_new[:, qi], constant_cut_indices=cut_indices)
fc_lens = ensure_fcurve(camd_fcs, "lens", 0)
extend_fcurve_after_seam(fc_lens, frames_new_f32, cam_lens_new, constant_cut_indices=cut_indices)

sc.frame_start = 1
sc.frame_end   = F_END
print(f"[5/6] FCurves extended ({N_NEW} new frames, {len(cut_indices)} camera cuts in f661..f6265) in {time.time()-t_fc:.2f}s", flush=True)

# -------------------------------------------------------------------------
# Real Depsgraph Seam Audit across f659 -> f660 -> f661 -> f662!
# -------------------------------------------------------------------------
def snap_heads(frame_no):
    sc.frame_set(frame_no)
    bpy.context.view_layer.update()
    pts = {pb.name: (Mw_xm @ pb.head).copy() for pb in pbs}
    return pts, cam_obj.location.copy()

p659, c659 = snap_heads(659)
p660, c660 = snap_heads(660)
p661, c661 = snap_heads(661)
p662, c662 = snap_heads(662)

s59_60 = max((p660[k] - p659[k]).length for k in p660)
s60_61 = max((p661[k] - p660[k]).length for k in p660)
s61_62 = max((p662[k] - p661[k]).length for k in p660)
print(f"  [SEAM AUDIT] max bone step: f659->660={s59_60*1000:.2f}mm | f660->661={s60_61*1000:.2f}mm | f661->662={s61_62*1000:.2f}mm", flush=True)
print(f"  [CAM  AUDIT] cam step:      f659->660={(c660-c659).length*1000:.2f}mm | f660->661={(c661-c660).length*1000:.2f}mm | f661->662={(c662-c661).length*1000:.2f}mm", flush=True)

sc.frame_set(1200)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND), compress=True)
print(f"[6/6] SAVED FULL-SONG MASTER: {OUT_BLEND.name} ({OUT_BLEND.stat().st_size/1048576:.1f} MB) in {time.time()-t0:.1f}s!", flush=True)
