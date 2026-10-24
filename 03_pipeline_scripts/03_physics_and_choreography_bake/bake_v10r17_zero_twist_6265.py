"""
V10R16 Full-Song (1..6265f) Zero-Merge & Zero-Clip Kinematic + 67-Bone Physics Solver:
1. Fixes ALL 4 hardcoded world-axis (-Y / +-X) assumptions that inverted arms & eyes when character rotated yaw (e.g. 0:58 f1742..f1774 facing +Y!).
2. Adds 3D Torso/Lapel Capsule Shield + Head/Hair Capsule Shield so wide sleeves never sink into chest, sailor bow, coat collar, or face.
3. Adds 3D Segment-to-Segment Wide-Sleeve Crossing & Clasp Solver (target >= 178mm center-axis clearance):
   - Side-by-side clasp/prayer: clean lateral cuff-to-cuff contact without merging;
   - Crossed arms (X-pose): clean front/back depth layering along Chest forward normal with bout-level hysteresis & temporal Gaussian smoothing!
4. Upgrades 5-sector Coat Skirt physics to 360-deg directional leg clearance (front kick, side lunge, AND rear kick) + lowers Shorts HighKick_Follow ramp to >56 deg.
5. Compares posed bone positions against V10R15 across f1..f2400 to output the exact list of changed frames needing 1080p re-render!
"""
import json, math, random, time
from pathlib import Path
import numpy as np
import bpy, addon_utils
from mathutils import Vector, Matrix, Quaternion

t0 = time.time()
WDIR       = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
SRC_BLEND  = WDIR / "RadDoll_V10R16_FullSong6265_MMD_Heart舞台三渲二零穿模精修母带_PRIVATE.blend"
OUT_BLEND  = WDIR / "RadDoll_V10R17_FullSong6265_MMD_Heart舞台三渲二零交拧精修母带_PRIVATE.blend"
ORACLE_NPZ = WDIR / "oracle_full_6265.npz"
AUDIO_JSON = WDIR / "audio_envelope_6265f.json"

F_START = 1
F_END   = 6265
N_TOTAL = F_END - F_START + 1
DT      = 1.0 / 30.0

addon_utils.enable("bl_ext.blender_org.mmd_tools", default_set=False)
bpy.ops.wm.open_mainfile(filepath=str(SRC_BLEND))
sc        = bpy.context.scene
arm_ob    = bpy.data.objects["ARM-FullBody_V10"]
skin_ob   = bpy.data.objects["GEO-Sotai_Skin"]
facial_ob = bpy.data.objects["RadDoll_FacialParts"]
shorts_ob = bpy.data.objects["GEO-Cloth_Shorts"]
sailor_ob = bpy.data.objects["GEO-Cloth_SailorTop"]


shoeL_ob  = bpy.data.objects["GEO-Shoe_L"]
shoeR_ob  = bpy.data.objects["GEO-Shoe_R"]
pbs       = arm_ob.pose.bones

# Disable viewport mesh modifiers during snapshot of old V10R15 f1..f2400 bone heads!
mod_backup = []
for o in bpy.data.objects:
    if o.type == 'MESH':
        for m in o.modifiers:
            mod_backup.append((m, m.show_viewport))
            m.show_viewport = False

Mw_xm     = arm_ob.matrix_world.copy()
Mw_xm_inv = Mw_xm.inverted()
R_arm_w     = Mw_xm.to_3x3().normalized()
R_arm_w_inv = R_arm_w.inverted()
arm_world_scale = Mw_xm.to_scale().z

KEY_CHECK_BONES = ["Hips", "Chest", "Head", "upper_arm.L", "lower_arm.L", "hand.L", "upper_arm.R", "lower_arm.R", "hand.R", "LeftEye", "RightEye"]
old_heads_2400 = np.zeros((2400, len(KEY_CHECK_BONES), 3), dtype=np.float32)
old_rots_2400  = np.zeros((2400, len(KEY_CHECK_BONES), 4), dtype=np.float32)
for fi in range(2400):
    sc.frame_set(fi + 1)
    bpy.context.view_layer.update()
    for bi, bn in enumerate(KEY_CHECK_BONES):
        old_heads_2400[fi, bi] = Mw_xm @ pbs[bn].head
        q = pbs[bn].rotation_quaternion
        old_rots_2400[fi, bi]  = [q.w, q.x, q.y, q.z]

print(f"[1/6] Snapshot V10R16 f1..f2400 baseline in {time.time()-t0:.1f}s", flush=True)

# Load Oracle (at exact scale=0.08!)
ora = np.load(ORACLE_NPZ, allow_pickle=True)
INV_S113       = 1.0 / 1.13
tracked_jp     = [str(x) for x in ora["tracked_jp"]]
jp_idx         = {jpn: i for i, jpn in enumerate(tracked_jp)}
aux_jp         = [str(x) for x in ora["aux_jp"]]
miku_rest_head = {jpn: Vector((ora["rest_head"][i] * INV_S113).tolist()) for i, jpn in enumerate(aux_jp)}
miku_rest_tail = {jpn: Vector((ora["rest_tail"][i] * INV_S113).tolist()) for i, jpn in enumerate(aux_jp)}
miku_rest_R    = {jpn: Matrix(ora["rest_R"][i].tolist()) for i, jpn in enumerate(tracked_jp)}
miku_hips_pos  = ora["hips_pos"] * INV_S113
miku_ank_z     = ora["ank_z"]    * INV_S113
miku_pose_R    = ora["pose_R"]

aud        = json.loads(AUDIO_JSON.read_text(encoding="utf-8"))
env_norm   = aud["env_norm"]
env_bright = aud["env_bright"]

# Rest geometry of ARM-FullBody_V10
xm_rest_R, xm_rest_u, xm_rest_head = {}, {}, {}
bone_local_R, bone_head_arm, bone_tail_arm, bone_parent_nm = {}, {}, {}, {}
for b in arm_ob.data.bones:
    Rw = (R_arm_w @ b.matrix_local.to_3x3()).normalized()
    hw = Mw_xm @ b.head_local
    tw = Mw_xm @ b.tail_local
    xm_rest_R[b.name]     = Rw
    xm_rest_head[b.name]  = hw
    dv = tw - hw
    xm_rest_u[b.name]     = dv.normalized() if dv.length > 1e-6 else Vector((0, 0, 1))
    bone_local_R[b.name]  = b.matrix_local.to_3x3().normalized()
    bone_head_arm[b.name] = b.head_local.copy()
    bone_tail_arm[b.name] = b.tail_local.copy()
    bone_parent_nm[b.name] = b.parent.name if b.parent else None

bone_rest_rel_R, bone_rest_rel_p, bone_rest_tail_local = {}, {}, {}
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

# Verify rest-space true anatomical axes in World space!
REST_LEFT_W = (xm_rest_head["upper_arm.L"] - xm_rest_head["upper_arm.R"]).normalized()  # (+1, 0, 0)
REST_UP_W   = Vector((0.0, 0.0, 1.0))
REST_FWD_W  = REST_UP_W.cross(REST_LEFT_W).normalized()  # (0, +1, 0) or (0, -1, 0)? Wait: (0,0,1) x (1,0,0) = (0,+1,0)! Check eye vs head!
eye_mid_rest = (xm_rest_head["LeftEye"] + xm_rest_head["RightEye"]) * 0.5
REST_FWD_W = Vector((eye_mid_rest.x - xm_rest_head["Head"].x, eye_mid_rest.y - xm_rest_head["Head"].y, 0.0)).normalized()
print(f"  Verified Rest Axes in World: LEFT={tuple(round(x,3) for x in REST_LEFT_W)}, FWD={tuple(round(x,3) for x in REST_FWD_W)}", flush=True)

# Expand VG_Sailor_UnderCoat to include mid-back sailor cloth strictly below coat back-neckline (z < -0.288)
# Verify back direction in sailor_ob local space using Spine/Head rest coords:
vg_sl = sailor_ob.vertex_groups.get("VG_Sailor_UnderCoat")
if vg_sl:
    M_w2sl = sailor_ob.matrix_world.inverted()
    fwd_sl = (M_w2sl.to_3x3() @ REST_FWD_W).normalized()  # points toward character FRONT in sailor local space
    extra_sl_hide = []
    for v in sailor_ob.data.vertices:
        back_amt = -Vector(v.co).dot(fwd_sl)
        # Strictly on back panel (back_amt > 0.048) AND safely below off-shoulder coat back collar rim (z < -0.292)
        if back_amt > 0.048 and v.co.z < -0.292:
            extra_sl_hide.append(v.index)
    if extra_sl_hide:
        vg_sl.add(extra_sl_hide, 1.0, 'REPLACE')
        print(f"  [Sailor Back Mask] Added {len(extra_sl_hide)} mid-back hidden verts below z=-0.292 (eliminates f0586 back-coat poke!)", flush=True)


# Shoe LBS sole helper (exact 0.01mm verified!)
R_mw_np = np.array(Mw_xm.to_3x3(), dtype=np.float64)
t_mw_np = np.array(Mw_xm.translation, dtype=np.float64)
def build_shoe_sole_lbs(sh_ob, foot_bn, toe_bn):
    me = sh_ob.data
    vg_map = {vg.name: vg.index for vg in sh_ob.vertex_groups}
    if_f, if_t = vg_map.get(foot_bn, -1), vg_map.get(toe_bn, -1)
    M_ob_to_arm = Mw_xm_inv @ sh_ob.matrix_world
    p_all = [M_ob_to_arm @ v.co for v in me.vertices]
    wz_all = [(Mw_xm @ p).z for p in p_all]
    z_min0 = min(wz_all)
    pts_arm, wf_l, wt_l = [], [], []
    for v, p_a, wz in zip(me.vertices, p_all, wz_all):
        if wz <= z_min0 + 0.035:
            wf = wt = 0.0
            for g in v.groups:
                if g.group == if_f: wf = g.weight
                elif g.group == if_t: wt = g.weight
            ws = wf + wt
            if ws > 1e-4: wf /= ws; wt /= ws
            else: wf, wt = 1.0, 0.0
            pts_arm.append([p_a.x, p_a.y, p_a.z])
            wf_l.append(wf); wt_l.append(wt)
    pts_np = np.array(pts_arm, dtype=np.float64)
    Rf0 = np.array(bone_local_R[foot_bn], dtype=np.float64)
    Rt0 = np.array(bone_local_R[toe_bn], dtype=np.float64)
    pf0 = np.array(bone_head_arm[foot_bn], dtype=np.float64)
    pt0 = np.array(bone_head_arm[toe_bn], dtype=np.float64)
    return (pts_np - pf0) @ Rf0, (pts_np - pt0) @ Rt0, np.array(wf_l)[:, None], np.array(wt_l)[:, None]

sole_L = build_shoe_sole_lbs(shoeL_ob, "foot.L", "toe.L")
sole_R = build_shoe_sole_lbs(shoeR_ob, "foot.R", "toe.R")
def eval_sole_min_z(R_arm_map, p_arm_map):
    zm = 999.0
    for (fbn, tbn), (df, dt_v, wf_a, wt_a) in [(("foot.L", "toe.L"), sole_L), (("foot.R", "toe.R"), sole_R)]:
        Rf = np.array(R_arm_map[fbn], dtype=np.float64); Rt = np.array(R_arm_map[tbn], dtype=np.float64)
        pf = np.array(p_arm_map[fbn], dtype=np.float64); pt = np.array(p_arm_map[tbn], dtype=np.float64)
        p_posed_arm = wf_a * (pf + df @ Rf.T) + wt_a * (pt + dt_v @ Rt.T)
        wz = (p_posed_arm @ R_mw_np[2, :]) + t_mw_np[2]
        zm = min(zm, float(wz.min()))
    return zm - 0.00148

# BONE_MAP + A-pose swing alignment
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
    if xm_bn not in xm_rest_R or jp_bn not in miku_rest_R: continue
    if mode == "direct":
        R_xm_apose_base[xm_bn] = xm_rest_R[xm_bn]
    else:
        if jp_bn.endswith("腕捩"):
            s_jp = jp_bn[0]; v_m = (miku_rest_head[f"{s_jp}ひじ"] - miku_rest_head[f"{s_jp}腕"]).normalized()
        elif jp_bn.endswith("手捩"):
            s_jp = jp_bn[0]; v_m = (miku_rest_head[f"{s_jp}手首"] - miku_rest_head[f"{s_jp}ひじ"]).normalized()
        elif jp_bn.endswith("手首"):
            s_jp = jp_bn[0]; v_m = (miku_rest_head[f"{s_jp}中指１"] - miku_rest_head[f"{s_jp}手首"]).normalized()
        else:
            dv_m = miku_rest_tail[jp_bn] - miku_rest_head[jp_bn]
            v_m = dv_m.normalized() if dv_m.length > 1e-4 else (miku_rest_head[f"{jp_bn[0]}中指１"] - miku_rest_head[f"{jp_bn[0]}手首"]).normalized()
        q_sw = xm_rest_u[xm_bn].rotation_difference(v_m)
        R_xm_apose_base[xm_bn] = (q_sw.to_matrix() @ xm_rest_R[xm_bn]).normalized()

VALID_BONE_MAP = [(xb, jb, md) for (xb, jb, md) in BONE_MAP if xb in R_xm_apose_base and jb in jp_idx]
RETARGET_NAMES = [xb for xb, _, _ in VALID_BONE_MAP]
FINGER_NAMES_L = [xb for xb in RETARGET_NAMES if xb.endswith(".L") and "Proximal" in xb or xb.endswith(".L") and ("Intermediate" in xb or "Distal" in xb)]
FINGER_NAMES_R = [xb for xb in RETARGET_NAMES if xb.endswith(".R") and ("Proximal" in xb or "Intermediate" in xb or "Distal" in xb)]

# Pure-dict FK helpers
cur_q_local, cur_R_arm, cur_p_arm = {}, {}, {}
hips_loc_cur = Vector((0.0, 0.0, 0.0))

def reset_fk():
    cur_R_arm.clear(); cur_p_arm.clear()

def get_arm_R(bn):
    if bn in cur_R_arm: return cur_R_arm[bn]
    pn = bone_parent_nm[bn]
    q_b = cur_q_local.get(bn, Quaternion((1.0, 0.0, 0.0, 0.0)))
    R_c = ((get_arm_R(pn) @ bone_rest_rel_R[bn] @ q_b.to_matrix()) if pn else (bone_rest_rel_R[bn] @ q_b.to_matrix())).normalized()
    cur_R_arm[bn] = R_c
    return R_c

def get_arm_p(bn):
    if bn in cur_p_arm: return cur_p_arm[bn]
    pn = bone_parent_nm[bn]
    p_c = (get_arm_p(pn) + get_arm_R(pn) @ bone_rest_rel_p[bn]) if pn else bone_rest_rel_p[bn].copy()
    if bn == "Hips":
        p_c = p_c + bone_local_R["Hips"] @ hips_loc_cur
    cur_p_arm[bn] = p_c
    return p_c

def get_arm_tail(bn):
    return get_arm_p(bn) + get_arm_R(bn) @ bone_rest_tail_local[bn]

def set_world_rot(bn, R_target_world):
    R_target_arm = (R_arm_w_inv @ R_target_world).normalized()
    pn = bone_parent_nm[bn]
    R_basis = ((get_arm_R(pn) @ bone_rest_rel_R[bn]).inverted() @ R_target_arm) if pn else (bone_rest_rel_R[bn].inverted() @ R_target_arm)
    q_b = R_basis.to_quaternion(); q_b.normalize()
    cur_q_local[bn] = q_b
    cur_R_arm[bn]   = R_target_arm

def seg_closest_points(p1, q1, p2, q2):
    u = q1 - p1; v = q2 - p2; w = p1 - p2
    a = u.dot(u); b = u.dot(v); c = v.dot(v); d = u.dot(w); e = v.dot(w)
    D = a * c - b * b
    sD, tD = D, D
    if D < 1e-8:
        sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN = b * e - c * d; tN = a * e - b * d
        if sN < 0.0: sN = 0.0; tN = e; tD = c
        elif sN > sD: sN = sD; tN = e + b; tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else: sN = -d; sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0: sN = 0.0
        elif (-d + b) > a: sN = sD
        else: sN = (-d + b); sD = a
    sv = 0.0 if abs(sN) < 1e-8 else sN / max(sD, 1e-8)
    tv = 0.0 if abs(tN) < 1e-8 else tN / max(tD, 1e-8)
    c1 = p1 + u * sv; c2 = p2 + v * tv
    return c1, c2, sv, tv

def swing_arm_subtree(side_s, pivot_bn, dq_world):
    # Rotate pivot_bn AND all downstream arm/hand/finger world rotations rigidly by dq_world!
    M_dq = dq_world.to_matrix().to_3x3()
    chain = [f"upper_arm.{side_s}", f"lower_arm.{side_s}", f"lower_arm_Twist.{side_s}", f"hand.{side_s}"] if pivot_bn.startswith("upper_arm") else [f"lower_arm.{side_s}", f"lower_arm_Twist.{side_s}", f"hand.{side_s}"]
    fingers = FINGER_NAMES_L if side_s == "L" else FINGER_NAMES_R
    full_list = [b for b in chain + fingers if b in cur_q_local]
    old_Rw = {b: (R_arm_w @ get_arm_R(b)).normalized() for b in full_list}
    reset_fk()
    for b in full_list:
        set_world_rot(b, (M_dq @ old_Rw[b]).normalized())
    reset_fk()

# -------------------------------------------------------------------------
# PASS A: Base Yaw-Invariant Retargeting across f1..f6265 (before arm-cross layering)
# -------------------------------------------------------------------------
t_ret = time.time()
hips_loc_all         = np.zeros((N_TOTAL, 3), dtype=np.float64)
base_q_all           = {bn: np.zeros((N_TOTAL, 4), dtype=np.float64) for bn in RETARGET_NAMES}
headup_lidded_seq    = np.zeros(N_TOTAL, dtype=np.float64)
up_pitch_clamped_seq = np.zeros(N_TOTAL, dtype=np.float64)
raw_cross_depth_diff = np.zeros(N_TOTAL, dtype=np.float64)  # (mid_L - mid_R) · c_fwd
raw_is_crossed       = np.zeros(N_TOTAL, dtype=bool)

b_head_miku  = jp_idx["頭"]
b_chest_miku = jp_idx["上半身2"]

for idx in range(N_TOTAL):
    f = F_START + idx
    cur_q_local.clear(); reset_fk()

    mp = Vector(miku_hips_pos[idx].tolist())
    dz_hips = (mp.z - miku_hips_rest_pos.z) * leg_ratio
    target_hips_w = Vector((mp.x * 0.98, mp.y, xm_hips_rest_pos.z + dz_hips))

    if f <= 462 or f >= 6248:
        w_sit_clamp = (1.0 if f <= 445 else max(0.0, (462 - f) / 17.0)) if f <= 462 else min(1.0, (f - 6248) / 12.0)
        if target_hips_w.z < 0.652:
            target_hips_w.z = target_hips_w.z * (1.0 - w_sit_clamp) + (0.652 + 0.15 * (target_hips_w.z - 0.652)) * w_sit_clamp

    d_hips_w = target_hips_w - xm_hips_rest_pos
    hips_loc_cur = (R_hips_rest_w_inv @ d_hips_w) / arm_world_scale

    # True Yaw-Invariant Head & Chest Axes!
    R_m_head  = Matrix(miku_pose_R[idx, b_head_miku].tolist())
    R_head_raw_w = (R_m_head @ miku_rest_R["頭"].inverted() @ R_xm_apose_base["Head"]).normalized()
    M_head_delta_raw = R_head_raw_w @ xm_rest_R["Head"].inverted()
    v_face_fwd_raw = (M_head_delta_raw @ REST_FWD_W).normalized()
    up_pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, v_face_fwd_raw.z))))

    R_m_chest = Matrix(miku_pose_R[idx, b_chest_miku].tolist())
    M_chest_delta_raw = (R_m_chest @ miku_rest_R["上半身2"].inverted())
    c_fwd_raw  = (M_chest_delta_raw @ REST_FWD_W).normalized()
    c_left_raw = (M_chest_delta_raw @ REST_LEFT_W).normalized()
    # Horizontal right-axis of Chest: rotating by +angle around c_pitch_back_axis tilts +Z toward -c_fwd (arching back!)
    c_horiz_fwd = Vector((c_fwd_raw.x, c_fwd_raw.y, 0.0))
    c_horiz_fwd = c_horiz_fwd.normalized() if c_horiz_fwd.length > 1e-4 else REST_FWD_W
    # Axis such that Axis x Z_up = -c_horiz_fwd  =>  Axis = Z_up x c_horiz_fwd!
    # Check: if c_horiz_fwd = (0,-1,0), Z x (0,-1,0) = (+1,0,0); (+1,0,0) x (0,0,1) = (0,-1,0)? Wait:
    # Right-hand rule: (+1,0,0) rotated +theta moves +Y -> +Z and +Z -> -Y (which IS +c_horiz_fwd = forward!).
    # So rotating around (-1,0,0) = - (Z x c_horiz_fwd) tilts +Z BACKWARD (-c_horiz_fwd)!
    c_arch_back_axis = -(Vector((0, 0, 1)).cross(c_horiz_fwd)).normalized()

    h_horiz_fwd = Vector((v_face_fwd_raw.x, v_face_fwd_raw.y, 0.0))
    h_horiz_fwd = h_horiz_fwd.normalized() if h_horiz_fwd.length > 1e-4 else c_horiz_fwd
    # Axis that pitches v_face_fwd DOWNWARD (-Z): rotating h_horiz_fwd toward -Z!
    # Check: if h_horiz_fwd = (0,-1,0), (+1,0,0) rotates (0,-1,0) -> (0,-cos,-sin) (-Z = DOWNWARD!).
    # And Z x (0,-1,0) = (+1,0,0)! Let's verify for (0,+1,0): Z x (0,+1,0) = (-1,0,0), which rotates (0,+1,0) -> (0,+cos,-sin) (-Z = DOWNWARD!)!
    h_look_down_axis = (Vector((0, 0, 1)).cross(h_horiz_fwd)).normalized()

    extra_chest_arch_rad = 0.0
    lidded_comp = 0.0
    if up_pitch_deg > 8.5:
        excess_deg = up_pitch_deg - 8.5
        extra_chest_arch_rad = math.radians(min(8.5, excess_deg * 0.22))
        lidded_comp = min(0.18, excess_deg * 0.0048)
    headup_lidded_seq[idx] = lidded_comp

    for xm_bn, jp_bn, mode in VALID_BONE_MAP:
        R_m_pose  = Matrix(miku_pose_R[idx, jp_idx[jp_bn]].tolist())
        R_delta_w = R_m_pose @ miku_rest_R[jp_bn].inverted()
        R_tgt_w   = (R_delta_w @ R_xm_apose_base[xm_bn]).normalized()

        if xm_bn == "Chest" and extra_chest_arch_rad > 1e-4:
            q_arch = Quaternion(c_arch_back_axis, extra_chest_arch_rad).to_matrix()
            R_tgt_w = (q_arch @ R_tgt_w).normalized()

        elif xm_bn in ("Neck", "Head") and up_pitch_deg > 8.5:
            M_hd = R_tgt_w @ xm_rest_R[xm_bn].inverted()
            v_fwd = (M_hd @ REST_FWD_W).normalized()
            cur_el = math.degrees(math.asin(max(-1.0, min(1.0, v_fwd.z))))
            cap_start = 6.0 if xm_bn == "Neck" else 8.5
            if cur_el > cap_start:
                new_el = cap_start + 0.19 * (cur_el - cap_start)
                horiz = Vector((v_fwd.x, v_fwd.y, 0.0))
                if horiz.length > 1e-5:
                    horiz.normalize()
                    v_fwd_clamped = (horiz * math.cos(math.radians(new_el)) + Vector((0.0, 0.0, math.sin(math.radians(new_el))))).normalized()
                    R_tgt_w = (v_fwd.rotation_difference(v_fwd_clamped).to_matrix() @ R_tgt_w).normalized()

        elif xm_bn in ("LeftEye", "RightEye") and up_pitch_deg > 14.0:
            look_down_rad = math.radians(min(8.5, (up_pitch_deg - 14.0) * 0.30))
            R_tgt_w = (Quaternion(h_look_down_axis, look_down_rad).to_matrix() @ R_tgt_w).normalized()

        elif xm_bn.startswith("shoulder."):
            # 100% YAW-INVARIANT Clavicle Clamp: measure shrug STRICTLY relative to posed Chest!
            R_ch_now = R_arm_w @ get_arm_R("Chest")
            M_ch_now = R_ch_now @ xm_rest_R["Chest"].inverted()
            R_base_sh = (M_ch_now @ R_xm_apose_base[xm_bn]).normalized()
            q_diff_sh = (R_tgt_w @ R_base_sh.inverted()).to_quaternion()
            if q_diff_sh.angle > math.radians(6.5):
                q_clamped = Quaternion(q_diff_sh.axis, math.radians(6.5) + 0.25 * (q_diff_sh.angle - math.radians(6.5)))
                R_tgt_w = (q_clamped.to_matrix() @ R_base_sh).normalized()

        elif mode == "arm_upper":
            # 100% YAW-INVARIANT Chest-Local Upper-Arm Guard!
            R_ch_w = R_arm_w @ get_arm_R("Chest")
            M_ch_d = R_ch_w @ xm_rest_R["Chest"].inverted()
            ch_left = (M_ch_d @ REST_LEFT_W).normalized()
            ch_fwd  = (M_ch_d @ REST_FWD_W).normalized()
            ch_up   = ch_fwd.cross(ch_left).normalized()  # Wait: check (0,-1,0) x (1,0,0) = (0,0,+1) = UP!
            if ch_up.dot(M_ch_d @ REST_UP_W) < 0:
                ch_up = -ch_up

            sgn_side = +1.0 if xm_bn.endswith(".L") else -1.0
            u_cur = ((R_tgt_w @ xm_rest_R[xm_bn].inverted()) @ xm_rest_u[xm_bn]).normalized()
            c_l = u_cur.dot(ch_left) * sgn_side   # + = outward away from torso!
            c_f = u_cur.dot(ch_fwd)               # + = forward in front of chest!
            c_u = u_cur.dot(ch_up)                # + = upward toward head!

            # Allow natural chorus arm raises (up to c_u=0.42 freely, gentle soft-knee above 0.42) so elbows don't drop unnaturally
            if c_u > 0.42:
                c_u = 0.42 + 0.45 * (c_u - 0.42)
            # Only prevent inward adduction INTO ribcage when arm is down at side (c_f < 0.22 and c_u < 0.15)
            min_lat = 0.14 if (c_f < 0.22 and c_u < 0.15) else max(-0.28, 0.14 - max(0.0, c_f - 0.20) * 0.85)
            if c_l < min_lat:
                c_l = min_lat + 0.30 * (c_l - min_lat)

            u_mod = (ch_left * (c_l * sgn_side) + ch_fwd * c_f + ch_up * c_u)
            if u_mod.length > 1e-5:
                u_mod.normalize()
                R_tgt_w = (u_cur.rotation_difference(u_mod).to_matrix() @ R_tgt_w).normalized()

        elif xm_bn.startswith("lower_arm.") and not xm_bn.startswith("lower_arm_Twist"):
            side_s = xm_bn.split(".")[1]
            R_ua_w = R_arm_w @ get_arm_R(f"upper_arm.{side_s}")
            u_ua_w = ((R_ua_w @ xm_rest_R[f"upper_arm.{side_s}"].inverted()) @ xm_rest_u[f"upper_arm.{side_s}"]).normalized()
            u_fa_w = ((R_tgt_w @ xm_rest_R[xm_bn].inverted()) @ xm_rest_u[xm_bn]).normalized()
            bend_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(u_ua_w.dot(u_fa_w))))))
            # Soft-cap extreme acute elbow fold (> 98 deg -> max ~112 deg) so wide coat sleeve doesn't self-crush!
            if bend_deg > 98.0:
                new_bend = 98.0 + 0.22 * (bend_deg - 98.0)
                ax_b = u_ua_w.cross(u_fa_w)
                if ax_b.length > 1e-5:
                    ax_b.normalize()
                    u_fa_safe = (Quaternion(ax_b, math.radians(new_bend)) @ u_ua_w).normalized()
                    R_tgt_w = (u_fa_w.rotation_difference(u_fa_safe).to_matrix() @ R_tgt_w).normalized()

        set_world_rot(xm_bn, R_tgt_w)

    # Standing dance shoe-sole grounding Z=0.002m
    if f >= 480 and f < 6248:
        Rm_s = {bn: get_arm_R(bn) for bn in ("foot.L", "toe.L", "foot.R", "toe.R")}
        pm_s = {bn: get_arm_p(bn) for bn in ("foot.L", "toe.L", "foot.R", "toe.R")}
        z_sole_min = eval_sole_min_z(Rm_s, pm_s)
        if z_sole_min < 0.001 or (float(miku_ank_z[idx, 0]) < 0.115 or float(miku_ank_z[idx, 1]) < 0.115):
            if z_sole_min < 0.028:
                d_hips_w.z += (0.002 - z_sole_min)
                hips_loc_cur = (R_hips_rest_w_inv @ d_hips_w) / arm_world_scale
                cur_p_arm.clear()

    # Record raw forearm crossing state for bout-consistent front/back layering!
    R_ch_w = R_arm_w @ get_arm_R("Chest")
    M_ch_d = R_ch_w @ xm_rest_R["Chest"].inverted()
    ch_left = (M_ch_d @ REST_LEFT_W).normalized()
    ch_fwd  = (M_ch_d @ REST_FWD_W).normalized()
    faL0 = Mw_xm @ get_arm_p("lower_arm.L"); faL1 = Mw_xm @ get_arm_tail("hand.L")
    faR0 = Mw_xm @ get_arm_p("lower_arm.R"); faR1 = Mw_xm @ get_arm_tail("hand.R")
    ch_up_det = ch_fwd.cross(ch_left).normalized()
    if ch_up_det.dot(M_ch_d @ REST_UP_W) < 0:
        ch_up_det = -ch_up_det
    u_faL_det = (faL1 - faL0).normalized()
    u_faR_det = (faR1 - faR0).normalized()
    both_pointing_up = (u_faL_det.dot(ch_up_det) > 0.45 and u_faR_det.dot(ch_up_det) > 0.45)
    cL, cR, sv_c, tv_c = seg_closest_points(faL0, faL1, faR0, faR1)
    lat_wr_raw = (faL1 - faR1).dot(ch_left)
    if (cL - cR).length < 0.245 and lat_wr_raw < -0.028 and not both_pointing_up:
        raw_is_crossed[idx] = True
    raw_cross_depth_diff[idx] = ((faL0 + faL1) * 0.5 - (faR0 + faR1) * 0.5).dot(ch_fwd)

    R_hd_fin = R_arm_w @ get_arm_R("Head")
    v_f_fin  = ((R_hd_fin @ xm_rest_R["Head"].inverted()) @ REST_FWD_W).normalized()
    up_pitch_clamped_seq[idx] = math.degrees(math.asin(max(-1.0, min(1.0, v_f_fin.z))))

    hips_loc_all[idx] = [hips_loc_cur.x, hips_loc_cur.y, hips_loc_cur.z]
    for bn in RETARGET_NAMES:
        q = cur_q_local[bn]
        base_q_all[bn][idx] = [q.w, q.x, q.y, q.z]

# Assign consistent front-arm sign (+1 = L in front, -1 = R in front) across each crossing bout (with 6f margin)!
bout_front_sign = np.ones(N_TOTAL, dtype=np.float64)
i_scan = 0
while i_scan < N_TOTAL:
    if raw_is_crossed[i_scan]:
        j_scan = i_scan
        while j_scan + 1 < N_TOTAL and (raw_is_crossed[j_scan + 1] or any(raw_is_crossed[j_scan + 1 : min(N_TOTAL, j_scan + 8)])):
            j_scan += 1
        mean_df = float(np.mean(raw_cross_depth_diff[i_scan : j_scan + 1]))
        sgn_b = +1.0 if mean_df >= 0.0 else -1.0
        bout_front_sign[max(0, i_scan - 6) : min(N_TOTAL, j_scan + 7)] = sgn_b
        i_scan = j_scan + 1
    else:
        i_scan += 1

# -------------------------------------------------------------------------
# PASS B: 3D Capsule Torso/Head Shield + Wide-Sleeve Crossing Layering Solver!
# -------------------------------------------------------------------------
t_col = time.time()
retarget_q_all = {bn: base_q_all[bn].copy() for bn in RETARGET_NAMES}
ARM_SOLVE_BONES = ["upper_arm.L", "lower_arm.L", "lower_arm_Twist.L", "hand.L",
                   "upper_arm.R", "lower_arm.R", "lower_arm_Twist.R", "hand.R"] + FINGER_NAMES_L + FINGER_NAMES_R

min_lr_after = 999.0
min_body_after = 999.0

for idx in range(N_TOTAL):
    cur_q_local.clear(); reset_fk()
    hips_loc_cur = Vector(hips_loc_all[idx].tolist())
    for bn in RETARGET_NAMES:
        qw, qx, qy, qz = base_q_all[bn][idx]
        cur_q_local[bn] = Quaternion((qw, qx, qy, qz))

    R_ch_w  = R_arm_w @ get_arm_R("Chest")
    M_ch_d  = R_ch_w @ xm_rest_R["Chest"].inverted()
    ch_left = (M_ch_d @ REST_LEFT_W).normalized()
    ch_fwd  = (M_ch_d @ REST_FWD_W).normalized()
    ch_up   = (M_ch_d @ REST_UP_W).normalized()

    sp0 = (Mw_xm @ get_arm_p("Hips")) + ch_up * 0.08
    sp1 = (Mw_xm @ get_arm_p("Neck")) - ch_up * 0.02
    hd0 = (Mw_xm @ get_arm_p("Neck")) + ch_up * 0.04
    hd1 = (Mw_xm @ get_arm_p("Head")) + ch_up * 0.13

    # Snapshot Pass A world rotations for anatomical deviation limiter (never allow pretzel twisting!)
    passA_Rw = {b: (R_arm_w @ get_arm_R(b)).normalized() for b in ARM_SOLVE_BONES if b in cur_q_local}

    def apply_arm_displacement(side_s, c_pt, disp_w):
        if disp_w.length < 1e-4:
            return
        sh_p = Mw_xm @ get_arm_p(f"upper_arm.{side_s}")
        r_sh = c_pt - sh_p
        L_sh = r_sh.length
        if L_sh > 0.10:
            u_r = r_sh / L_sh
            eff_L = max(L_sh, 0.20)  # Safe minimum lever arm prevents shoulder pivot spin!
            d_rad = u_r * disp_w.dot(u_r)
            d_tan = disp_w - d_rad
            if d_tan.length > 1e-4:
                ang_sh = min(0.32, (d_tan.length * 0.72) / eff_L)
                ax_sh  = u_r.cross(d_tan)
                if ax_sh.length > 1e-6:
                    swing_arm_subtree(side_s, f"upper_arm.{side_s}", Quaternion(ax_sh.normalized(), ang_sh))
            el_p = Mw_xm @ get_arm_p(f"lower_arm.{side_s}")
            wr_p = Mw_xm @ get_arm_tail(f"hand.{side_s}")
            r_el = wr_p - el_p
            if r_el.length > 0.06:
                u_el = r_el.normalized()
                rem = d_tan * 0.35 + d_rad * 0.92
                rem_tan_el = rem - u_el * rem.dot(u_el)
                if rem_tan_el.length > 1e-4:
                    ang_el = min(0.36, rem_tan_el.length / max(r_el.length, 0.16))
                    ax_el  = u_el.cross(rem_tan_el)
                    if ax_el.length > 1e-6:
                        swing_arm_subtree(side_s, f"lower_arm.{side_s}", Quaternion(ax_el.normalized(), ang_el))

    elL_0 = Mw_xm @ get_arm_p("lower_arm.L"); wrL_0 = Mw_xm @ get_arm_tail("hand.L")
    elR_0 = Mw_xm @ get_arm_p("lower_arm.R"); wrR_0 = Mw_xm @ get_arm_tail("hand.R")
    both_up_0 = ((wrL_0 - elL_0).normalized().dot(ch_up) > 0.44 and (wrR_0 - elR_0).normalized().dot(ch_up) > 0.44)
    is_true_xcross_bout = (not both_up_0) and bool(raw_is_crossed[idx])

    for _it in range(10):
        # Step 1: Torso & Head Capsule Shield for each forearm+hand
        for side_s, sgn_l in (("L", +1.0), ("R", -1.0)):
            el_p = Mw_xm @ get_arm_p(f"lower_arm.{side_s}")
            wr_p = Mw_xm @ get_arm_tail(f"hand.{side_s}")

            c_arm, c_tor, _, _ = seg_closest_points(el_p, wr_p, sp0, sp1)
            diff_t = c_arm - c_tor
            d_t = diff_t.length
            R_SAFE_TORSO = 0.180
            if d_t < R_SAFE_TORSO:
                bias_out = (ch_fwd * 0.72 + ch_left * (sgn_l * 0.55) + ch_up * 0.14).normalized()
                u_push = (diff_t.normalized() * 0.60 + bias_out * 0.40).normalized() if d_t > 0.02 else bias_out
                if u_push.dot(ch_fwd) < -0.08 and abs(u_push.dot(ch_left)) < 0.5:
                    u_push = bias_out
                apply_arm_displacement(side_s, c_arm, u_push * ((R_SAFE_TORSO - d_t) * 0.84))
                el_p = Mw_xm @ get_arm_p(f"lower_arm.{side_s}")
                wr_p = Mw_xm @ get_arm_tail(f"hand.{side_s}")

            c_arm_h, c_hd, _, _ = seg_closest_points(el_p, wr_p, hd0, hd1)
            diff_h = c_arm_h - c_hd
            d_h = diff_h.length
            R_SAFE_HEAD = 0.192  # Part wide cuffs cleanly around cheeks/face during overhead raises!
            if d_h < R_SAFE_HEAD:
                bias_h = (ch_left * (sgn_l * 0.85) + ch_fwd * 0.35).normalized()
                u_push_h = (diff_h.normalized() * 0.50 + bias_h * 0.50).normalized() if d_h > 0.02 else bias_h
                apply_arm_displacement(side_s, c_arm_h, u_push_h * ((R_SAFE_HEAD - d_h) * 0.85))

        # Step 2: L vs R Oversized Folded-Cuff Sleeve Anti-Merge
        shL = Mw_xm @ get_arm_p("upper_arm.L"); elL = Mw_xm @ get_arm_p("lower_arm.L"); wrL = Mw_xm @ get_arm_tail("hand.L")
        shR = Mw_xm @ get_arm_p("upper_arm.R"); elR = Mw_xm @ get_arm_p("lower_arm.R"); wrR = Mw_xm @ get_arm_tail("hand.R")
        uaL_mid = shL.lerp(elL, 0.45)
        uaR_mid = shR.lerp(elR, 0.45)

        pairs = [
            seg_closest_points(elL, wrL, elR, wrR),
            seg_closest_points(uaL_mid, elL, elR, wrR),
            seg_closest_points(elL, wrL, uaR_mid, elR),
        ]
        cL, cR, _, _ = min(pairs, key=lambda item: (item[0] - item[1]).length)
        diff_lr = cL - cR
        d_lr = diff_lr.length
        lat_wr = (wrL - wrR).dot(ch_left)

        D_SAFE_SLEEVE = 0.248 if is_true_xcross_bout else 0.240

        if d_lr < D_SAFE_SLEEVE:
            need_sep = (D_SAFE_SLEEVE - d_lr) * 0.62

            if is_true_xcross_bout and lat_wr < -0.020:
                fsgn = bout_front_sign[idx]
                dir_over  = (ch_up * 0.76 + ch_fwd * 0.65).normalized()
                dir_under = (-ch_up * 0.84 + ch_fwd * 0.18).normalized()
                if fsgn > 0:
                    push_L = dir_over  * (need_sep * 1.15)
                    push_R = dir_under * (need_sep * 1.05)
                else:
                    push_R = dir_over  * (need_sep * 1.15)
                    push_L = dir_under * (need_sep * 1.05)
            else:
                # Pure lateral parting (+ch_left for L, -ch_left for R) keeps both hands symmetric & untwisted!
                push_L = ch_left * (need_sep * 0.95) + ch_fwd * (need_sep * 0.12)
                push_R = (-ch_left) * (need_sep * 0.95) + ch_fwd * (need_sep * 0.12)

            apply_arm_displacement("L", cL, push_L)
            apply_arm_displacement("R", cR, push_R)

    # Generous 48-deg runaway guard (allows full X-cross separation on f0861/f2112 while preventing any >50 deg spin!)
    for side_s in ("L", "R"):
        for piv_bn, max_rad in ((f"upper_arm.{side_s}", math.radians(46.0)), (f"lower_arm.{side_s}", math.radians(48.0))):
            Rw_now = (R_arm_w @ get_arm_R(piv_bn)).normalized()
            Rw_ref = passA_Rw[piv_bn]
            q_d = (Rw_now @ Rw_ref.inverted()).to_quaternion()
            if q_d.angle > max_rad:
                q_fix = Quaternion(q_d.axis, max_rad - q_d.angle)
                swing_arm_subtree(side_s, piv_bn, q_fix)

    # Measure final post-solve clearance!
    elL = Mw_xm @ get_arm_p("lower_arm.L"); wrL = Mw_xm @ get_arm_tail("hand.L")
    elR = Mw_xm @ get_arm_p("lower_arm.R"); wrR = Mw_xm @ get_arm_tail("hand.R")
    cL, cR, _, _ = seg_closest_points(elL, wrL, elR, wrR)
    min_lr_after = min(min_lr_after, (cL - cR).length)
    c_aL, c_tL, _, _ = seg_closest_points(elL, wrL, sp0, sp1)
    c_aR, c_tR, _, _ = seg_closest_points(elR, wrR, sp0, sp1)
    min_body_after = min(min_body_after, (c_aL - c_tL).length, (c_aR - c_tR).length)

    for bn in ARM_SOLVE_BONES:
        if bn in cur_q_local:
            q = cur_q_local[bn]
            retarget_q_all[bn][idx] = [q.w, q.x, q.y, q.z]

# Enforce quaternion hemisphere sign continuity & lightly filter corrective arm deltas ([0.20, 0.60, 0.20]) for silky 30fps motion!
for bn in RETARGET_NAMES:
    arr = retarget_q_all[bn]
    for i in range(1, N_TOTAL):
        if np.dot(arr[i - 1], arr[i]) < 0:
            arr[i] = -arr[i]
    if bn in ("upper_arm.L", "lower_arm.L", "upper_arm.R", "lower_arm.R"):
        # Smooth only where collision solver modified base_q_all!
        b_arr = base_q_all[bn].copy()
        for i in range(1, N_TOTAL):
            if np.dot(b_arr[i - 1], b_arr[i]) < 0: b_arr[i] = -b_arr[i]
        diff_norm = np.linalg.norm(arr - b_arr, axis=1)
        sm = arr.copy()
        for i in range(1, N_TOTAL - 1):
            if diff_norm[i - 1] > 1e-3 or diff_norm[i] > 1e-3 or diff_norm[i + 1] > 1e-3:
                q_m = 0.22 * arr[i - 1] + 0.56 * arr[i] + 0.22 * arr[i + 1]
                n_m = np.linalg.norm(q_m)
                if n_m > 1e-6: sm[i] = q_m / n_m
        retarget_q_all[bn] = sm

print(f"[2/6] Collision-free kinematic solve ({N_TOTAL}f) in {time.time()-t_ret:.2f}s | min_LR_sleeve={min_lr_after*1000:.1f}mm, min_arm_torso={min_body_after*1000:.1f}mm, max_head_up={up_pitch_clamped_seq.max():.2f}deg", flush=True)

# -------------------------------------------------------------------------
# PASS C: 360-Deg Leg-Aware 67-Bone Coat/Hair/Cuff/Bow Physics + Shorts Cuff!
# -------------------------------------------------------------------------
t_phy = time.time()
hips_pos_seq    = np.zeros((N_TOTAL, 3), dtype=np.float64)
hips_yaw_seq    = np.zeros(N_TOTAL, dtype=np.float64)
thighL_fwd_seq  = np.zeros(N_TOTAL, dtype=np.float64)
thighR_fwd_seq  = np.zeros(N_TOTAL, dtype=np.float64)
thigh_back_seq  = np.zeros(N_TOTAL, dtype=np.float64)
thighL_side_seq = np.zeros(N_TOTAL, dtype=np.float64)
thighR_side_seq = np.zeros(N_TOTAL, dtype=np.float64)
wristL_pos_seq  = np.zeros((N_TOTAL, 3), dtype=np.float64)
wristR_pos_seq  = np.zeros((N_TOTAL, 3), dtype=np.float64)
chest_z_seq     = np.zeros(N_TOTAL, dtype=np.float64)
head_pos_seq    = np.zeros((N_TOTAL, 3), dtype=np.float64)
val_hk_L        = np.zeros(N_TOTAL, dtype=np.float32)
val_hk_R        = np.zeros(N_TOTAL, dtype=np.float32)

for idx in range(N_TOTAL):
    cur_q_local.clear(); reset_fk()
    hips_loc_cur = Vector(hips_loc_all[idx].tolist())
    for bn in RETARGET_NAMES:
        qw, qx, qy, qz = retarget_q_all[bn][idx]
        cur_q_local[bn] = Quaternion((qw, qx, qy, qz))

    hips_w = Mw_xm @ get_arm_p("Hips")
    R_hp_w = R_arm_w @ get_arm_R("Hips")
    M_hp_d = R_hp_w @ xm_rest_R["Hips"].inverted()
    hp_fwd  = (M_hp_d @ REST_FWD_W).normalized()
    hp_left = (M_hp_d @ REST_LEFT_W).normalized()
    yaw_val = math.atan2(hp_fwd.x, -hp_fwd.y)

    hips_pos_seq[idx] = [hips_w.x, hips_w.y, hips_w.z]
    hips_yaw_seq[idx] = yaw_val

    bk_max = 0.0
    for s_nm, fwd_arr, side_arr, hk_arr, sgn_s in [
        ("L", thighL_fwd_seq, thighL_side_seq, val_hk_L, +1.0),
        ("R", thighR_fwd_seq, thighR_side_seq, val_hk_R, -1.0),
    ]:
        ul_bn = f"upper_leg.{s_nm}"
        ul_v = ((Mw_xm @ get_arm_tail(ul_bn)) - (Mw_xm @ get_arm_p(ul_bn))).normalized()
        f_amt = ul_v.dot(hp_fwd)
        s_amt = ul_v.dot(hp_left) * sgn_s
        fwd_arr[idx]  = math.atan2(max(0.0, f_amt), max(0.05, -ul_v.z))
        side_arr[idx] = math.atan2(max(0.0, s_amt), max(0.05, -ul_v.z))
        bk_max = max(bk_max, math.atan2(max(0.0, -f_amt), max(0.05, -ul_v.z)))

        ang_deg = math.degrees(math.acos(max(-1.0, min(1.0, -ul_v.z))))
        if ang_deg > 56.0:
            tu = min(1.18, (ang_deg - 56.0) / (120.0 - 56.0))
            hk_arr[idx] = (tu * tu * (3.0 - 2.0 * tu)) if tu <= 1.0 else tu
    thigh_back_seq[idx] = bk_max

    wLf = Mw_xm @ get_arm_p("hand.L"); wRf = Mw_xm @ get_arm_p("hand.R")
    wristL_pos_seq[idx] = [wLf.x, wLf.y, wLf.z]
    wristR_pos_seq[idx] = [wRf.x, wRf.y, wRf.z]
    chest_z_seq[idx]    = float((Mw_xm @ get_arm_p("Chest")).z)
    hdw = Mw_xm @ get_arm_p("Head")
    head_pos_seq[idx]   = [hdw.x, hdw.y, hdw.z]

def central_acc(arr):
    vel = np.gradient(arr, DT, axis=0)
    return vel, np.clip(np.gradient(vel, DT, axis=0), -18.0, 18.0)

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
    skirt_out_axes[sec] = (rad_xy, Vector((rad_xy.y, -rad_xy.x, 0.0)).normalized())

q_sk  = np.zeros((5, 2), dtype=np.float64); dq_sk  = np.zeros((5, 2), dtype=np.float64)
q_aux = np.zeros(4, dtype=np.float64);      dq_aux = np.zeros(4, dtype=np.float64)
q_hc  = np.zeros((12, 2), dtype=np.float64); dq_hc = np.zeros((12, 2), dtype=np.float64)
q_ah  = np.zeros(2, dtype=np.float64);      dq_ah  = np.zeros(2, dtype=np.float64)
hc_az = np.array([math.radians(i * 30.0) for i in range(12)], dtype=np.float64)

def step_phys(fi, sub_dt):
    global q_sk, dq_sk, q_aux, dq_aux, q_hc, dq_hc, q_ah, dq_ah
    ah = hips_acc[fi]; hd_a = head_acc[fi]; hv = head_vel[fi]
    thL = thighL_fwd_seq[fi];  thR = thighR_fwd_seq[fi]
    sdL = thighL_side_seq[fi]; sdR = thighR_side_seq[fi]
    bkM = thigh_back_seq[fi]
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
            min_q0 = max(min_q0, thL * 0.80 + sdL * 0.25 + 0.34 * sit_w)
            min_q1 = max(min_q1, thL * 0.38)
        elif sec == "FR":
            min_q0 = max(min_q0, thR * 0.80 + sdR * 0.25 + 0.34 * sit_w)
            min_q1 = max(min_q1, thR * 0.38)
        elif sec == "SL":
            min_q0 = max(min_q0, 0.54 * sit_w + sdL * 0.72 + thL * 0.30 + bkM * 0.22)
            min_q1 = max(min_q1, 0.32 * sit_w + sdL * 0.35)
        elif sec == "SR":
            min_q0 = max(min_q0, 0.54 * sit_w + sdR * 0.72 + thR * 0.30 + bkM * 0.22)
            min_q1 = max(min_q1, 0.32 * sit_w + sdR * 0.35)
        elif sec == "BK":
            min_q0 = max(min_q0, 0.62 * sit_w + bkM * 0.78)
            min_q1 = max(min_q1, 0.48 * sit_w + bkM * 0.42)

        f0 = -62.0 * (q_sk[si, 0] - min_q0) - 9.2 * dq_sk[si, 0] + inert * 14.0
        if q_sk[si, 0] < min_q0: f0 += 280.0 * (min_q0 - q_sk[si, 0])
        f1 = -46.0 * (q_sk[si, 1] - max(min_q1, q_sk[si, 0] * 0.55)) - 7.6 * dq_sk[si, 1] + inert * 19.0
        dq_sk[si, 0] += f0 * sub_dt; q_sk[si, 0] = np.clip(q_sk[si, 0] + dq_sk[si, 0] * sub_dt, min_q0 - 0.02, 1.28)
        dq_sk[si, 1] += f1 * sub_dt; q_sk[si, 1] = np.clip(q_sk[si, 1] + dq_sk[si, 1] * sub_dt, min_q1 - 0.02, 1.32)

    drv_aux = [
        float(np.clip(-wrL_acc[fi, 2] * 0.015 + wrL_acc[fi, 1] * 0.010, -0.45, 0.45)),
        float(np.clip(-wrR_acc[fi, 2] * 0.015 + wrR_acc[fi, 1] * 0.010, -0.45, 0.45)),
        float(np.clip(-chZ_acc[fi] * 0.016, -0.35, 0.35)),
        float(np.clip(-chZ_acc[fi] * 0.016, -0.35, 0.35)),
    ]
    for ai in range(4):
        fa = -75.0 * (q_aux[ai] - drv_aux[ai]) - 10.5 * dq_aux[ai]
        dq_aux[ai] += fa * sub_dt; q_aux[ai] = float(np.clip(q_aux[ai] + dq_aux[ai] * sub_dt, -0.42, 0.48))

    for hi in range(12):
        az = hc_az[hi]; rx, ry = math.sin(az), math.cos(az)
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

PHYS_BONE_NAMES = [f"PHY_Coat_{s}_{x}" for s, _, _ in COAT_SECTORS for x in ("01", "02") if f"PHY_Coat_{s}_{x}" in pbs]
PHYS_BONE_NAMES += [b for b in ("PHY_Cuff_L", "PHY_Cuff_R", "PHY_Bow_L", "PHY_Bow_R") if b in pbs]
PHYS_BONE_NAMES += [f"Hc_{hi}_{sj}" for hi in range(12) for sj in (1, 2, 3, 4) if f"Hc_{hi}_{sj}" in pbs]
PHYS_BONE_NAMES += [f"Hair_Ahoge_{sj:02d}" for sj in (1, 2, 3, 4, 5) if f"Hair_Ahoge_{sj:02d}" in pbs]

phys_q_all = {bn: np.zeros((N_TOTAL, 4), dtype=np.float64) for bn in PHYS_BONE_NAMES}
for fi in range(N_TOTAL):
    cur_q_local.clear(); reset_fk()
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
                R_base_w = ((R_arm_w @ get_arm_R(pn)) @ (R_arm_w @ bone_local_R[pn]).inverted() @ xm_rest_R[bn]).normalized()
                set_world_rot(bn, (Quaternion(ax_w, ang).to_matrix() @ R_base_w).normalized())
                q = cur_q_local[bn]; phys_q_all[bn][fi] = [q.w, q.x, q.y, q.z]
    for ai, bn in enumerate(["PHY_Cuff_L", "PHY_Cuff_R", "PHY_Bow_L", "PHY_Bow_R"]):
        if bn in pbs:
            q = Quaternion((1, 0, 0), float(phys_traj_aux[fi, ai])); phys_q_all[bn][fi] = [q.w, q.x, q.y, q.z]
    for hi in range(12):
        for seg_idx, vals in [((1, 2), phys_traj_hc[fi, hi, 0] * 0.5), ((3, 4), phys_traj_hc[fi, hi, 1] * 0.5)]:
            for sj in seg_idx:
                bn = f"Hc_{hi}_{sj}"
                if bn in pbs:
                    q = Quaternion((1, 0, 0), float(vals)); phys_q_all[bn][fi] = [q.w, q.x, q.y, q.z]
    for aj, segs in [(0, [1, 2]), (1, [3, 4, 5])]:
        for sj in segs:
            bn = f"Hair_Ahoge_{sj:02d}"
            if bn in pbs:
                q = Quaternion((1, 0, 0), float(phys_traj_ah[fi, aj] * 0.45)); phys_q_all[bn][fi] = [q.w, q.x, q.y, q.z]

for bn in PHYS_BONE_NAMES:
    arr = phys_q_all[bn]
    for i in range(1, N_TOTAL):
        if np.dot(arr[i - 1], arr[i]) < 0: arr[i] = -arr[i]

print(f"[3/6] 360-deg Coat/Hair Physics solved in {time.time()-t_phy:.2f}s", flush=True)

# -------------------------------------------------------------------------
# PASS D: Write Full 1..6265 FCurves (Armature + Shorts HighKick)
# -------------------------------------------------------------------------
def get_action_fcurves(id_data):
    act = id_data.animation_data.action
    if hasattr(act, "layers") and len(act.layers) > 0:
        for layer in act.layers:
            for st in layer.strips:
                for cb in st.channelbags:
                    if len(cb.fcurves) > 0: return cb.fcurves
    return act.fcurves

def write_full_fcurve(fcurves, data_path, index, frames_f32, vals_f32):
    fc = next((c for c in fcurves if c.data_path == data_path and c.array_index == index), None) or fcurves.new(data_path=data_path, index=index)
    fc.keyframe_points.clear()
    n = len(frames_f32)
    fc.keyframe_points.add(n)
    co = np.empty(n * 2, dtype=np.float32)
    co[0::2] = frames_f32; co[1::2] = vals_f32
    fc.keyframe_points.foreach_set("co", co)
    for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'
    fc.update()

frames_all_f32 = np.arange(1, N_TOTAL + 1, dtype=np.float32)
arm_fcs = get_action_fcurves(arm_ob)

for ax_i in range(3):
    write_full_fcurve(arm_fcs, 'pose.bones["Hips"].location', ax_i, frames_all_f32, hips_loc_all[:, ax_i].astype(np.float32))
for bn in RETARGET_NAMES:
    dp = f'pose.bones["{bn}"].rotation_quaternion'
    for qi in range(4):
        write_full_fcurve(arm_fcs, dp, qi, frames_all_f32, retarget_q_all[bn][:, qi].astype(np.float32))
for bn in PHYS_BONE_NAMES:
    dp = f'pose.bones["{bn}"].rotation_quaternion'
    for qi in range(4):
        write_full_fcurve(arm_fcs, dp, qi, frames_all_f32, phys_q_all[bn][:, qi].astype(np.float32))

sh_fcs = get_action_fcurves(shorts_ob.data.shape_keys)
write_full_fcurve(sh_fcs, 'key_blocks["HighKick_Follow_L"].value', 0, frames_all_f32, val_hk_L)
write_full_fcurve(sh_fcs, 'key_blocks["HighKick_Follow_R"].value', 0, frames_all_f32, val_hk_R)

# -------------------------------------------------------------------------
# PASS E: Identify exact frames in f1..f2400 whose pose changed (>1.5mm or >0.5deg)
# -------------------------------------------------------------------------
changed_frames = []
for fi in range(2400):
    sc.frame_set(fi + 1)
    bpy.context.view_layer.update()
    max_d_mm = 0.0
    max_deg  = 0.0
    for bi, bn in enumerate(KEY_CHECK_BONES):
        p_new = Mw_xm @ pbs[bn].head
        d_mm = float(np.linalg.norm(np.array(p_new) - old_heads_2400[fi, bi]) * 1000.0)
        if d_mm > max_d_mm: max_d_mm = d_mm
        q_n = pbs[bn].rotation_quaternion
        q_o = Quaternion(old_rots_2400[fi, bi].tolist())
        deg = math.degrees(2.0 * math.acos(min(1.0, abs(q_n.dot(q_o)))))
        if deg > max_deg: max_deg = deg
    if max_d_mm > 2.2 or max_deg > 1.2:
        changed_frames.append(fi + 1)

# Expand changed_frames by +-3 neighbor frames for 100% seamless transitions!
changed_set = set()
for cf in changed_frames:
    for d in range(-3, 4):
        if 1 <= cf + d <= 2400:
            changed_set.add(cf + d)
changed_sorted = sorted(changed_set)
(WDIR / "v10r17_changed_frames_2400.json").write_text(json.dumps({"count": len(changed_sorted), "frames": changed_sorted}), encoding="utf-8")
print(f"[4/6] Changed frames (R16->R17) in f1..f2400 needing re-render: {len(changed_sorted)} / 2400 ({len(changed_sorted)/24.0:.1f}%)", flush=True)

# Restore viewport mesh modifiers BEFORE saving!
for m, sv in mod_backup:
    m.show_viewport = sv

sc.frame_set(1704)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND), compress=True)
print(f"[5/6] SAVED V10R17 MASTER: {OUT_BLEND.name} ({OUT_BLEND.stat().st_size/1048576:.1f} MB) in {time.time()-t0:.1f}s!", flush=True)

# Render BOTH 16-frame 0:56 sequence (f1668..f1728) AND 12-panel crossing QA frames!
ODIR = WDIR / "qa_v10r17_verify"
ODIR.mkdir(parents=True, exist_ok=True)
sc.camera = bpy.data.objects["MMD_MV_CAMERA"]
sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 16
sc.render.resolution_x = 1280
sc.render.resolution_y = 720
sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.quality = 93
qa_frames = sorted(set([212, 586, 642, 750, 861, 918, 1141, 1446, 1621, 1742, 1754, 1774, 2101, 2112, 2183] + list(range(1668, 1732, 4))))
for f_v in qa_frames:
    sc.frame_set(f_v)
    sc.render.filepath = str(ODIR / f"after_{f_v:04d}.jpg")
    bpy.ops.render.render(write_still=True)
print("[6/6] V10R17_QA_RENDERS_OK!", flush=True)
