"""
V10R9 衣装修身剪裁 (Rev 4 — 原图标准姿态锁 + 真实裸臂表面中轴同轴压缩):
1) 保持 Step2 工程原始 NLA 评估姿态 (强校验 |tip_X| ≈ 450.5mm 原图同姿签名！)；
2) SailorTop / Shorts / Sock_L / Sock_R：基于真实素体柱坐标包络压缩多余空腔；
3) Coat 躯干与袖管：
   - 躯干包覆在 max(素体, 修身后水手服/短裤) 之外，压缩 50% 多余水桶厚度与宽耸肩，上肩微沉落肩、下摆微提轻盈；
   - 袖管严格围绕真实裸臂表面点云 (sv_arm_co) 拟合的 3D 弯曲中心线同轴压缩多余肥大空间，100% 零穿模！
"""
import json, math
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector, Euler
from mathutils.kdtree import KDTree

ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
ODIR = ROOT / "生成模型/组装/v010R8_全身衣装装配"
ADIR = ODIR / "audit_slim"
ADIR.mkdir(parents=True, exist_ok=True)

IN_STEP2_BLEND = ODIR / "RadDoll_V10R8_Step2_UV重建与卡通材质_PRIVATE.blend"
OUT_SLIM_STEP2 = ODIR / "RadDoll_V10R9_Step2_修身剪裁与UV材质_PRIVATE.blend"

bpy.ops.wm.open_mainfile(filepath=str(IN_STEP2_BLEND), load_ui=False, use_scripts=False)
sc = bpy.context.scene
sc.frame_set(1)
bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()

sotai_ob = bpy.data.objects["GEO-Sotai_Skin"]
eo_s = sotai_ob.evaluated_get(deps)
me_s = eo_s.to_mesh()
n_sv = len(me_s.vertices)
sv_co = np.empty(n_sv * 3, dtype=np.float32)
me_s.vertices.foreach_get("co", sv_co)
Ms = np.array(eo_s.matrix_world, dtype=np.float32)
sv_co = (sv_co.reshape(n_sv, 3) @ Ms[:3, :3].T) + Ms[:3, 3]

tip_x = float(np.abs(sv_co[:, 0]).max())
print(f"CHECK ORIG_POSE tip_x = {tip_x*1000:.2f} mm", flush=True)
assert 0.440 <= tip_x <= 0.460, f"Pose mismatch! tip_x={tip_x}"

arm_kws = ("shoulder", "arm", "elbow", "wrist", "hand", "thumb", "index", "middle", "ring", "pinky", "little", "finger", "metacarp", "palm")
arm_vg_idxs = {vg.index for vg in sotai_ob.vertex_groups if any(k in vg.name.lower() for k in arm_kws)}
is_arm_sv = np.zeros(n_sv, dtype=bool)
for vi, v in enumerate(me_s.vertices):
    if sum(g.weight for g in v.groups if g.group in arm_vg_idxs) > 0.35:
        is_arm_sv[vi] = True
eo_s.to_mesh_clear()

sv_body_co = sv_co[~is_arm_sv]
sv_arm_co  = sv_co[is_arm_sv]

def make_kd(pts):
    kd = KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(Vector((float(p[0]), float(p[1]), float(p[2]))), i)
    kd.balance()
    return kd

def get_world_verts(obj):
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    M = np.array(obj.matrix_world, dtype=np.float32)
    return (co.reshape(n, 3) @ M[:3, :3].T) + M[:3, 3]

def set_world_verts(obj, w_co):
    M_inv = np.linalg.inv(np.array(obj.matrix_world, dtype=np.float32))
    l_co = (w_co @ M_inv[:3, :3].T) + M_inv[:3, 3]
    obj.data.vertices.foreach_set("co", l_co.astype(np.float32).ravel())
    obj.data.update()

def spatial_smooth_vec(pts_ref, vec_field, k=18, max_d=0.036):
    kd = make_kd(pts_ref)
    out = np.empty_like(vec_field)
    for i, p in enumerate(pts_ref):
        hits = kd.find_n(Vector((float(p[0]), float(p[1]), float(p[2]))), k)
        ws = 0.0
        acc = np.zeros(3, dtype=np.float64)
        for _, j, d in hits:
            if d > max_d and ws > 0.0:
                continue
            w = 1.0 / (d + 0.005) ** 2
            acc += w * vec_field[j]
            ws += w
        out[i] = acc / max(ws, 1e-8)
    return out

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def build_polar_envelope(ref_pts, z_min, z_max, nz=30, ntheta=40, cx_fixed=0.0):
    zs = np.linspace(z_min, z_max, nz)
    dz = (z_max - z_min) / max(nz - 1, 1)
    cys = np.zeros(nz, dtype=np.float32)
    r_grid = np.zeros((nz, ntheta), dtype=np.float32)
    for iz, zc in enumerate(zs):
        m = (ref_pts[:, 2] >= zc - dz * 1.15) & (ref_pts[:, 2] <= zc + dz * 1.15)
        if m.sum() < 6:
            m = np.abs(ref_pts[:, 2] - zc) <= 0.045
        pts = ref_pts[m] if m.sum() > 0 else ref_pts
        cy = float(0.5 * (pts[:, 1].min() + pts[:, 1].max()))
        cys[iz] = cy
        dx = pts[:, 0] - cx_fixed
        dy = pts[:, 1] - cy
        rr = np.hypot(dx, dy)
        ang = (np.arctan2(dy, dx) + 2.0 * math.pi) % (2.0 * math.pi)
        bins = np.floor(ang / (2.0 * math.pi) * ntheta).astype(np.int32) % ntheta
        for b in range(ntheta):
            mb = (bins == b) | (bins == (b - 1) % ntheta) | (bins == (b + 1) % ntheta)
            r_grid[iz, b] = float(np.percentile(rr[mb], 98)) if mb.any() else float(np.median(rr))
    r_pad = np.pad(r_grid, ((1, 1), (0, 0)), mode='edge')
    r_grid = 0.25 * r_pad[:-2] + 0.50 * r_pad[1:-1] + 0.25 * r_pad[2:]
    r_grid = 0.25 * np.roll(r_grid, -1, axis=1) + 0.50 * r_grid + 0.25 * np.roll(r_grid, 1, axis=1)
    return zs, cys, r_grid

def query_polar_envelope(pts, zs, cys, r_grid, cx_fixed=0.0):
    ntheta = r_grid.shape[1]
    cy_p = np.interp(pts[:, 2], zs, cys).astype(np.float32)
    dx = pts[:, 0] - cx_fixed
    dy = pts[:, 1] - cy_p
    r_old = np.hypot(dx, dy)
    ang = (np.arctan2(dy, dx) + 2.0 * math.pi) % (2.0 * math.pi)

    fz = np.clip((pts[:, 2] - zs[0]) / (zs[-1] - zs[0] + 1e-8) * (len(zs) - 1), 0.0, len(zs) - 1.0001)
    iz0 = np.floor(fz).astype(np.int32)
    tz = fz - iz0

    ft = ang / (2.0 * math.pi) * ntheta
    it0 = np.floor(ft).astype(np.int32) % ntheta
    it1 = (it0 + 1) % ntheta
    tt = ft - np.floor(ft)

    r_bound = (1 - tz) * ((1 - tt) * r_grid[iz0, it0] + tt * r_grid[iz0, it1]) + \
              tz       * ((1 - tt) * r_grid[iz0+1, it0] + tt * r_grid[iz0+1, it1])
    return cy_p, dx, dy, r_old, r_bound.astype(np.float32)

# ==============================================================================
# 1) GEO-Cloth_SailorTop: 贴合斜肩与纤细胸肋 (压缩 62% 多余空腔，蝴蝶结保持饱满)
# ==============================================================================
ob_top = bpy.data.objects["GEO-Cloth_SailorTop"]
w_top = get_world_verts(ob_top)
sv_upper = sv_co[(sv_co[:, 2] >= -0.45) & (sv_co[:, 2] <= -0.14) & (np.abs(sv_co[:, 0]) <= 0.23)]
zs_top, cys_top, rg_top = build_polar_envelope(sv_upper, -0.43, -0.16, nz=30, ntheta=44, cx_fixed=0.0)

w_t = w_top.copy()
outer_sh = smoothstep(0.065, 0.145, np.abs(w_t[:, 0])) * smoothstep(-0.33, -0.22, w_t[:, 2])
w_t[:, 2] -= (0.004 + 0.015 * outer_sh)

cy_p, dx, dy, r_old, r_skin = query_polar_envelope(w_t, zs_top, cys_top, rg_top, cx_fixed=0.0)
bow_mask = (1.0 - smoothstep(0.055, 0.105, np.abs(w_t[:, 0]))) * smoothstep(0.045, 0.085, w_t[:, 1])
k_exc_top = 0.36 + 0.42 * bow_mask
min_pad_top = 0.0038

exc = np.maximum(0.0, r_old - (r_skin + min_pad_top))
r_target = np.maximum(r_skin + 0.0035, (r_skin + min_pad_top) + k_exc_top * exc)
sc_r = r_target / np.maximum(r_old, 1e-6)
w_t_target = np.stack([dx * sc_r, cy_p + dy * sc_r, w_t[:, 2]], axis=1)
w_top_final = w_top + spatial_smooth_vec(w_top, w_t_target - w_top, k=16, max_d=0.030)
# 二次包络下限钳制，杜绝平滑后局部贴肤
cy_p2, dx2, dy2, ro2, rsk2 = query_polar_envelope(w_top_final, zs_top, cys_top, rg_top, cx_fixed=0.0)
need_t = ro2 < (rsk2 + 0.0032)
if need_t.any():
    s2 = (rsk2[need_t] + 0.0032) / np.maximum(ro2[need_t], 1e-6)
    w_top_final[need_t, 0] = dx2[need_t] * s2
    w_top_final[need_t, 1] = cy_p2[need_t] + dy2[need_t] * s2
set_world_verts(ob_top, w_top_final)

# ==============================================================================
# 2) GEO-Cloth_Shorts: 贴合纤腰臀线 (压缩 64% 多余鼓包)
# ==============================================================================
ob_sh = bpy.data.objects["GEO-Cloth_Shorts"]
w_sh = get_world_verts(ob_sh)
sv_pelvis = sv_body_co[(sv_body_co[:, 2] >= -0.71) & (sv_body_co[:, 2] <= -0.46)]
zs_sh, cys_sh, rg_sh = build_polar_envelope(sv_pelvis, -0.70, -0.47, nz=24, ntheta=44, cx_fixed=0.0)

cy_p, dx, dy, r_old, r_skin = query_polar_envelope(w_sh, zs_sh, cys_sh, rg_sh, cx_fixed=0.0)
cord_mask  = (1.0 - smoothstep(0.035, 0.075, np.abs(w_sh[:, 0]))) * smoothstep(0.055, 0.090, w_sh[:, 1])
cuff_relax = smoothstep(-0.62, -0.68, w_sh[:, 2])
k_exc_sh   = 0.35 + 0.13 * cuff_relax + 0.38 * cord_mask
min_pad_sh = 0.0055

exc = np.maximum(0.0, r_old - (r_skin + min_pad_sh))
r_target = np.maximum(r_skin + 0.0050, (r_skin + min_pad_sh) + k_exc_sh * exc)
sc_r = r_target / np.maximum(r_old, 1e-6)
w_sh_target = np.stack([dx * sc_r, cy_p + dy * sc_r, w_sh[:, 2]], axis=1)
w_sh_final = w_sh + spatial_smooth_vec(w_sh, w_sh_target - w_sh, k=16, max_d=0.030)
cy_p2, dx2, dy2, ro2, rsk2 = query_polar_envelope(w_sh_final, zs_sh, cys_sh, rg_sh, cx_fixed=0.0)
need_s = ro2 < (rsk2 + 0.0045)
if need_s.any():
    s2 = (rsk2[need_s] + 0.0045) / np.maximum(ro2[need_s], 1e-6)
    w_sh_final[need_s, 0] = dx2[need_s] * s2
    w_sh_final[need_s, 1] = cy_p2[need_s] + dy2[need_s] * s2
set_world_verts(ob_sh, w_sh_final)

# ==============================================================================
# 3) GEO-Cloth_Coat (Rev 6 — 无尖角顺滑落肩领 + 重力自然垂坠瘦袖 + 零穿模锁):
#    - 消除 |X|≈0.19m 领肩交界处的小尖角：将袖管分离线在 Z>-0.38m 处平滑外推至 0.225m，并对袖根同步施加落肩重力下沉；
#    - 真实重力垂坠 (Gravity Drape)：宽松袖管受重力自然下沉 (-16mm)，使袖管上壁轻贴裸臂上方 (净空 5.5mm 零穿模)，彻底消除上臂外侧红区鼓包！
# ==============================================================================
def build_skin_arm_polyline(sv_arm_side):
    u_coord = 0.55 * np.abs(sv_arm_side[:, 0]) - 0.835 * sv_arm_side[:, 2]
    u_min, u_max = float(np.percentile(u_coord, 2)), float(np.percentile(u_coord, 98))
    us = np.linspace(u_min, u_max, 9)
    du = (u_max - u_min) / 8.0
    nodes = []
    for uc in us:
        m = np.abs(u_coord - uc) <= du * 0.85
        pts = sv_arm_side[m] if m.sum() >= 5 else sv_arm_side[np.argsort(np.abs(u_coord - uc))[:12]]
        nodes.append(np.array([
            0.5 * (float(pts[:, 0].min()) + float(pts[:, 0].max())),
            0.5 * (float(pts[:, 1].min()) + float(pts[:, 1].max())),
            0.5 * (float(pts[:, 2].min()) + float(pts[:, 2].max())),
        ], dtype=np.float32))
    v_end = nodes[-1] - nodes[-2]
    v_end /= max(float(np.linalg.norm(v_end)), 1e-6)
    nodes.append(nodes[-1] + 0.055 * v_end)
    return nodes

poly_L = build_skin_arm_polyline(sv_arm_co[sv_arm_co[:, 0] >  0.14])
poly_R = build_skin_arm_polyline(sv_arm_co[sv_arm_co[:, 0] < -0.14])

def closest_on_polyline(pts, poly):
    N = len(pts)
    best_d2 = np.full(N, 1e9, dtype=np.float32)
    best_q  = np.zeros((N, 3), dtype=np.float32)
    best_s  = np.zeros(N, dtype=np.float32)
    seg_lens = [float(np.linalg.norm(poly[k+1] - poly[k])) for k in range(len(poly)-1)]
    tot_len  = sum(seg_lens)
    cum = 0.0
    for k in range(len(poly) - 1):
        a, b = poly[k], poly[k+1]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        t = np.clip(((pts - a) @ ab) / max(L2, 1e-8), 0.0, 1.0)
        q = a[None, :] + t[:, None] * ab[None, :]
        d2 = np.sum((pts - q) ** 2, axis=1)
        m = d2 < best_d2
        best_d2[m] = d2[m]
        best_q[m]  = q[m]
        best_s[m]  = (cum + t[m] * seg_lens[k]) / tot_len
        cum += seg_lens[k]
    return best_q, best_s

bpy.context.view_layer.update()
top_for_env = w_top_final[w_top_final[:, 1] < 0.092]
torso_union = np.vstack([
    sv_body_co[(sv_body_co[:, 2] >= -0.92) & (sv_body_co[:, 2] <= -0.20)],
    top_for_env,
    w_sh_final,
])
zs_ct, cys_ct, rg_ct = build_polar_envelope(torso_union, -0.90, -0.22, nz=34, ntheta=48, cx_fixed=0.0)

ob_coat = bpy.data.objects["GEO-Cloth_Coat"]
w_c = get_world_verts(ob_coat)
ax = np.abs(w_c[:, 0])
pz = w_c[:, 2]

# 关键改进：在肩领上方 (Z > -0.38m) 将分界平滑外推至 0.225m，使整个落肩大翻领作为一个整体顺滑过渡，彻底消除领角小尖峰！
upper_collar_push = 0.038 * smoothstep(-0.42, -0.28, pz)
x_split = np.where(pz >= -0.43, 0.188 + upper_collar_push, 0.188 + 0.185 * (-0.43 - pz))
sleeve_w = smoothstep(x_split - 0.042, x_split + 0.042, ax) * smoothstep(-0.70, -0.66, pz)

# A. 躯干与落肩翻领目标：上领自然落肩下沉 -20mm，下摆微提 +8mm，多余水桶空腔压缩至 44%~60%（保留微A字轻盈衣摆）
zt_coat = smoothstep(-0.85, -0.27, pz)
z_c_torso = pz + (0.008 - 0.028 * zt_coat)
w_c_q = np.stack([w_c[:, 0], w_c[:, 1], z_c_torso], axis=1)
cy_ct, dx_ct, dy_ct, r_old_ct, r_in_ct = query_polar_envelope(w_c_q, zs_ct, cys_ct, rg_ct, cx_fixed=0.0)

k_exc_torso = 0.60 - 0.16 * zt_coat
min_pad_torso = 0.0080
exc_ct = np.maximum(0.0, r_old_ct - (r_in_ct + min_pad_torso))
r_new_ct = np.maximum(r_in_ct + 0.0070, (r_in_ct + min_pad_torso) + k_exc_torso * exc_ct)
sc_ct = r_new_ct / np.maximum(r_old_ct, 1e-6)
torso_target = np.stack([dx_ct * sc_ct, cy_ct + dy_ct * sc_ct, z_c_torso], axis=1)

# B. 左右宽袖管目标：同轴收窄 + 真实重力自然垂落 (上壁轻贴手臂上方，下壁垂顺贴合《原图》袖底线！)
def slim_sleeve_side(pts, sv_arm_side, poly):
    q_sk, s_sk = closest_on_polyline(sv_arm_side, poly)
    r_sk = np.linalg.norm(sv_arm_side - q_sk, axis=1)
    ns = 24
    s_grid = np.linspace(0.0, 1.0, ns)
    r_prof = np.full(ns, 0.033, dtype=np.float32)
    for i_s, sc_val in enumerate(s_grid):
        ms = np.abs(s_sk - sc_val) <= 0.080
        if ms.sum() >= 4:
            r_prof[i_s] = float(np.percentile(r_sk[ms], 98))

    q_cl, s_cl = closest_on_polyline(pts, poly)
    rv = pts - q_cl
    r_old = np.linalg.norm(rv, axis=1)
    r_hat = rv / np.maximum(r_old[:, None], 1e-6)
    r_arm = np.interp(s_cl, s_grid, r_prof).astype(np.float32)

    cuff_f = smoothstep(0.68, 0.92, s_cl)
    up_f   = np.clip(r_hat[:, 2], 0.0, 1.0)     # 上外侧
    dn_f   = np.clip(-r_hat[:, 2], 0.0, 1.0)    # 下内侧垂坠
    # 上壁紧贴收窄 (k=0.30)，前后收窄 (k=0.48)，下壁保留自然垂感 (k=0.58)，翻折袖口微留层次 (+0.10)
    k_slv  = (0.48 - 0.18 * up_f + 0.10 * dn_f) + 0.10 * cuff_f
    min_pad = 0.0075 + 0.005 * dn_f

    exc = np.maximum(0.0, r_old - (r_arm + min_pad))
    r_new = np.maximum(r_arm + 0.0068, (r_arm + min_pad) + k_slv * exc)
    q_cl_trim = poly[0][None, :] + (q_cl - poly[0][None, :]) * 0.990
    out_pts = q_cl_trim + r_hat * r_new[:, None]
    # 重力微垂：袖根与领口同步下沉 -16mm，袖身下垂 -12mm（紧接着由下方 360° 裸臂安全锁保证上壁绝不穿入裸臂！）
    out_pts[:, 2] -= (0.016 - 0.004 * s_cl)
    return out_pts, s_grid, r_prof

sleeve_target = w_c.copy()
mL = w_c[:, 0] >= 0.0
mR = ~mL
sleeve_target[mL], sg_L, rp_L = slim_sleeve_side(w_c[mL], sv_arm_co[sv_arm_co[:, 0] >  0.14], poly_L)
sleeve_target[mR], sg_R, rp_R = slim_sleeve_side(w_c[mR], sv_arm_co[sv_arm_co[:, 0] < -0.14], poly_R)

coat_combined = (1.0 - sleeve_w[:, None]) * torso_target + sleeve_w[:, None] * sleeve_target
w_coat_final = w_c + spatial_smooth_vec(w_c, coat_combined - w_c, k=26, max_d=0.048)

# 裸臂 360° 绝对安全锁（平滑推离，保证上壁贴在裸臂上方 >= 6.5mm 而不穿肉！）
push_arm = np.zeros_like(w_coat_final)
for side_m, poly, sg, rp in [(w_coat_final[:, 0] >= 0.0, poly_L, sg_L, rp_L), (w_coat_final[:, 0] < 0.0, poly_R, sg_R, rp_R)]:
    idx_s = np.where(side_m & (sleeve_w > 0.15))[0]
    pts_s = w_coat_final[idx_s]
    q_s, s_s = closest_on_polyline(pts_s, poly)
    rv_s = pts_s - q_s
    ro_s = np.linalg.norm(rv_s, axis=1)
    req_r = np.interp(s_s, sg, rp) + 0.0068
    need = ro_s < req_r
    if need.any():
        target_s = q_s[need] + (rv_s[need] / np.maximum(ro_s[need, None], 1e-6)) * req_r[need, None]
        push_arm[idx_s[need]] = target_s - pts_s[need]

if np.max(np.linalg.norm(push_arm, axis=1)) > 1e-5:
    sm_pa = spatial_smooth_vec(w_coat_final, push_arm, k=16, max_d=0.032)
    m1 = np.linalg.norm(push_arm, axis=1, keepdims=True)
    m2 = np.linalg.norm(sm_pa,    axis=1, keepdims=True)
    w_coat_final = w_coat_final + np.where(m1 > m2, push_arm, sm_pa)

# 躯干后背与侧胯安全锁
idx_t = np.where((sleeve_w < 0.35) & ~((np.abs(w_coat_final[:, 0]) < 0.105) & (w_coat_final[:, 1] > 0.03)))[0]
cy_2, dx_2, dy_2, ro_2, rin_2 = query_polar_envelope(w_coat_final[idx_t], zs_ct, cys_ct, rg_ct, cx_fixed=0.0)
need_ct = ro_2 < (rin_2 + 0.0055)
if need_ct.any():
    s2 = (rin_2[need_ct] + 0.0055) / np.maximum(ro_2[need_ct], 1e-6)
    sub = idx_t[need_ct]
    w_coat_final[sub, 0] = dx_2[need_ct] * s2
    w_coat_final[sub, 1] = cy_2[need_ct] + dy_2[need_ct] * s2

set_world_verts(ob_coat, w_coat_final)

# ==============================================================================
# 4) GEO-Cloth_Sock_L/R & GEO-Shoe_L/R
# ==============================================================================
for side_sgn, s_name, sh_name in [(-1.0, "GEO-Cloth_Sock_L", "GEO-Shoe_L"), (+1.0, "GEO-Cloth_Sock_R", "GEO-Shoe_R")]:
    ob_sk = bpy.data.objects[s_name]
    w_sk = get_world_verts(ob_sk)
    leg_pts = sv_body_co[(sv_body_co[:, 2] >= -1.31) & (sv_body_co[:, 2] <= -1.01) & (sv_body_co[:, 0] * side_sgn > 0.02)]
    cx_leg = float(np.median(leg_pts[:, 0]))
    zs_lk, cys_lk, rg_lk = build_polar_envelope(leg_pts, -1.30, -1.02, nz=20, ntheta=36, cx_fixed=cx_leg)
    cy_p, dx, dy, r_old, r_skin = query_polar_envelope(w_sk, zs_lk, cys_lk, rg_lk, cx_fixed=cx_leg)
    min_pad_sk = 0.0045
    exc_sk = np.maximum(0.0, r_old - (r_skin + min_pad_sk))
    r_new_sk = np.maximum(r_skin + 0.0042, (r_skin + min_pad_sk) + 0.42 * exc_sk)
    sc_sk = r_new_sk / np.maximum(r_old, 1e-6)
    w_sk_t = np.stack([cx_leg + dx * sc_sk, cy_p + dy * sc_sk, w_sk[:, 2]], axis=1)
    w_sk_fin = w_sk + spatial_smooth_vec(w_sk, w_sk_t - w_sk, k=14, max_d=0.025)
    cy_2, dx_2, dy_2, ro_2, rsk_2 = query_polar_envelope(w_sk_fin, zs_lk, cys_lk, rg_lk, cx_fixed=cx_leg)
    nd = ro_2 < (rsk_2 + 0.0038)
    if nd.any():
        s2 = (rsk_2[nd] + 0.0038) / np.maximum(ro_2[nd], 1e-6)
        w_sk_fin[nd, 0] = cx_leg + dx_2[nd] * s2
        w_sk_fin[nd, 1] = cy_2[nd] + dy_2[nd] * s2
    set_world_verts(ob_sk, w_sk_fin)

    ob_shoe = bpy.data.objects[sh_name]
    w_shoe = get_world_verts(ob_shoe)
    z_floor = float(w_shoe[:, 2].min())
    cx_sh = float(np.median(w_shoe[:, 0]))
    ank_m = (sv_body_co[:, 2] >= -1.31) & (sv_body_co[:, 2] <= -1.25) & (sv_body_co[:, 0] * side_sgn > 0.02)
    cy_ank = float(np.median(sv_body_co[ank_m, 1]))
    w_shoe_new = w_shoe.copy()
    w_shoe_new[:, 0] = cx_sh  + (w_shoe[:, 0] - cx_sh)  * 0.895
    w_shoe_new[:, 1] = cy_ank + (w_shoe[:, 1] - cy_ank) * 0.925
    w_shoe_new[:, 2] = z_floor + (w_shoe[:, 2] - z_floor) * 0.935
    set_world_verts(ob_shoe, w_shoe_new)

bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_SLIM_STEP2), compress=True)

# 渲染四组质检图
K_ORIG = 0.000487708
V_CENTER = 0.5 * (380.0 + 3756.0)
cam = bpy.data.objects["REVIEW_CAMERA"]
sc.camera = cam
cam.data.type = 'ORTHO'
sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 24
sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGBA'
sc.render.film_transparent = True

cam.location = Vector((0.0, 3.2, -(V_CENTER - 788.7) * K_ORIG))
cam.rotation_euler = Euler((math.radians(90.0), 0.0, math.radians(180.0)), 'XYZ')
cam.data.ortho_scale = 3376.0 * K_ORIG
sc.render.resolution_x = 1200
sc.render.resolution_y = 1978
sc.render.filepath = str(ADIR / "slim_front_rgba.png")
bpy.ops.render.render(write_still=True)

cam.location = Vector((-3.2, 0.0, -0.63))
cam.rotation_euler = Euler((math.radians(90.0), 0.0, math.radians(-90.0)), 'XYZ')
cam.data.ortho_scale = 1.72
sc.render.resolution_x = 900
sc.render.resolution_y = 1500
sc.render.filepath = str(ADIR / "slim_side_rgba.png")
bpy.ops.render.render(write_still=True)

sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.film_transparent = False
cam.location = Vector((-1.84, 2.62, -0.63))
cam.rotation_euler = Euler((math.radians(90.0), 0.0, math.radians(-145.0)), 'XYZ')
sc.render.filepath = str(ADIR / "slim_obliq35.jpg")
bpy.ops.render.render(write_still=True)

cam.location = Vector((0.0, -3.2, -0.63))
cam.rotation_euler = Euler((math.radians(90.0), 0.0, math.radians(0.0)), 'XYZ')
sc.render.filepath = str(ADIR / "slim_back180.jpg")
bpy.ops.render.render(write_still=True)

print("SLIM_R4_DONE", flush=True)
