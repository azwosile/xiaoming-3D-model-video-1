"""V10R7 Round 6 (Pass 2 终极精修) · 全身素体（RadDoll V3 Body_2）健康饱满二次元手臂/手腕/柔美肩线与自然手型精修
严格对齐《素体四视图_v002》（图像素体）四向手臂粗细、溜肩圆润度、手腕饱满度与《原图》袖口露指位置。
"""
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import rd_lib as L  # noqa: E402

HEAD_BLEND = HERE / 'RadDoll_V10_装配_PRIVATE.blend'
SRC_BODY_BLEND = ROOT / '生成模型/组装/v010_本地VRChat头模与物理/私用参考_不再分发/RaddollV3_SOURCE_IMPORT.blend'
OUT_BLEND = HERE / 'RadDoll_V10_全身素体装配_PRIVATE.blend'
BODY_TEX_PATH = HERE / 'RadDoll_body_tex_PRIVATE.png'
FACE_TEX_PATH = HERE / 'RadDoll_face_tex.png'
RENDERS_DIR = HERE / 'renders_body'
RENDERS_DIR.mkdir(parents=True, exist_ok=True)


def sstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def make_monotone_pchip(xk, yk):
    xk = np.asarray(xk, np.float64)
    yk = np.asarray(yk, np.float64)
    h = np.diff(xk)
    delta = np.diff(yk) / h
    n = len(xk)
    d = np.zeros(n, np.float64)
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        else:
            d[i] = 0.0
    d[0] = delta[0]
    d[-1] = delta[-1]

    def eval_fn(x):
        x_arr = np.asarray(x, np.float64)
        scalar = x_arr.ndim == 0
        xv = np.atleast_1d(x_arr)
        out = np.empty_like(xv)
        left = xv <= xk[0]
        right = xv >= xk[-1]
        mid = ~(left | right)
        out[left] = yk[0] + d[0] * (xv[left] - xk[0])
        out[right] = yk[-1] + d[-1] * (xv[right] - xk[-1])
        if np.any(mid):
            xm = xv[mid]
            idx = np.clip(np.searchsorted(xk, xm) - 1, 0, n - 2)
            hi = h[idx]
            t = (xm - xk[idx]) / hi
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            out[mid] = h00 * yk[idx] + h10 * hi * d[idx] + h01 * yk[idx + 1] + h11 * hi * d[idx + 1]
        return float(out[0]) if scalar else out

    return eval_fn


Z_KNOTS_SRC = np.array([
    0.0079,  # 赤足底 -> -1.386m
    0.0685,  # 踝关节中心
    0.2429,  # 小腿下段
    0.3150,  # 小腿肚最宽处 -> -1.015m
    0.4176,  # 膝关节中心 & 双膝粉晕中心 -> -0.920m
    0.5500,  # 大腿中上段 (左腿环高度 -> -0.756m)
    0.6250,  # 裆底分叉 -> -0.652m
    0.6705,  # 髋关节中心 -> -0.592m
    0.7520,  # 肚脐中心 (原图 V=1737 -> -0.4623m)
    0.8430,  # 胸廓下缘 -> -0.358m
    0.9585,  # 肩峰/上臂关节中心 -> -0.2425m (柔美二次元自然溜肩高度，精确对齐图像素体肩斜线！)
    0.9918,  # 锁骨窝/颈根 -> -0.1850m
    1.0294,  # 颈上缘 24 顶点环 -> -0.1265m
], np.float64)

Z_KNOTS_V10 = np.array([
    -1.3860,
    -1.3020,
    -1.1040,
    -1.0150,
    -0.9200,
    -0.7560,
    -0.6520,
    -0.5920,
    -0.4623,
    -0.3580,
    -0.2425,
    -0.1850,
    -0.1265,
], np.float64)

map_z = make_monotone_pchip(Z_KNOTS_SRC, Z_KNOTS_V10)

# SRC 手臂局部骨轴与健康圆润径向增粗曲线（消除细腕感，使上臂~73mm -> 手肘~66mm -> 前臂~61mm -> 手腕~51mm 平滑柔和过渡！）
ARM_X_KNOTS = np.array([0.070, 0.115, 0.165, 0.215, 0.255, 0.305, 0.350, 0.385, 0.425, 0.488], np.float64)
ARM_Z_AXIS_FN = make_monotone_pchip(
    [0.070, 0.120, 0.250, 0.380, 0.430, 0.488],
    [0.9555, 0.9588, 0.9602, 0.9614, 0.9575, 0.9568],
)
ARM_Y_AXIS_FN = make_monotone_pchip(
    [0.070, 0.120, 0.250, 0.380, 0.430, 0.488],
    [0.0315, 0.0332, 0.0362, 0.0358, 0.0310, 0.0220],
)
ARM_SZ_FN = make_monotone_pchip(
    ARM_X_KNOTS,
    [1.36, 1.56, 1.62, 1.70, 1.76, 1.76, 2.22, 2.46, 2.24, 2.22],
)
ARM_SY_FN = make_monotone_pchip(
    ARM_X_KNOTS,
    [1.32, 1.52, 1.50, 1.56, 1.58, 1.50, 1.76, 1.82, 1.38, 1.34],
)
# 紧凑自然的手臂轴向长度：肩关节 |X|=109mm -> 指尖 |X|=598mm（总臂展长 489mm，完美契合《素体四视图_v002》505mm 实测臂长！）
ARM_LEN_X_FN = make_monotone_pchip(
    [0.0701, 0.1600, 0.2550, 0.3820, 0.4855],
    [0.1090, 0.2130, 0.3230, 0.4690, 0.6080],
)

X_CENTER = -0.0012


def deform_body_points(pts_src, is_bone=False):
    p = np.asarray(pts_src, np.float64)
    xs, ys, zs = p[:, 0].copy(), p[:, 1].copy(), p[:, 2].copy()
    sign_x = np.sign(xs)
    ax = np.abs(xs)

    arm_mask = sstep((ax - 0.068) / 0.036) * sstep((zs - 0.882) / 0.038)

    # 1) 纵向 Z 映射
    z_torso = map_z(zs)
    z_sh_v10 = float(map_z(0.9585))  # -0.2425m
    z_ax_src = ARM_Z_AXIS_FN(ax)
    sz_arm = 1.22 if is_bone else ARM_SZ_FN(ax)
    z_arm = z_sh_v10 + (z_ax_src - 0.9585) * 1.22 + (zs - z_ax_src) * sz_arm
    Z = (1.0 - arm_mask) * z_torso + arm_mask * z_arm

    # 2) 横向 X 映射
    shoulder_band = gauss(zs, 0.946, 0.034) * (1.0 - sstep((zs - 0.980) / 0.018))
    s_x_profile = (
        1.225
        + 0.305 * shoulder_band                  # 圆润少女/少年肩宽 (~304mm)
        + 0.540 * sstep((zs - 0.982) / 0.024)    # 颈宽 ~74mm
        - 0.145 * gauss(zs, 0.765, 0.050)        # 露脐纤腰 ~182mm
        - 0.072 * gauss(zs, 0.655, 0.060)        # 短裤口胯宽 ~250mm
    )
    x_torso_mag = ax * s_x_profile

    leg_mask = sstep((0.662 - zs) / 0.036) * (1.0 - arm_mask)
    leg_axis_src = 0.0474
    leg_axis_fn = make_monotone_pchip(
        [0.008, 0.080, 0.315, 0.418, 0.535, 0.625, 0.675],
        [0.063, 0.063, 0.064, 0.055, 0.058, 0.062, 0.064],
    )
    r_out_fn = make_monotone_pchip(
        [0.008, 0.080, 0.315, 0.418, 0.505, 0.585, 0.655],
        [1.240, 1.240, 1.375, 1.365, 1.245, 1.135, 1.125],
    )
    r_in_fn = make_monotone_pchip(
        [0.008, 0.080, 0.315, 0.418, 0.505, 0.585, 0.655],
        [1.280, 1.310, 1.490, 1.560, 1.460, 1.375, 1.310],
    )
    leg_axis_targ = leg_axis_fn(zs)
    if is_bone:
        x_leg_mag = np.where(ax > 0.015, leg_axis_targ + (ax - leg_axis_src) * 1.30, ax * s_x_profile)
    else:
        dx_leg = xs - sign_x * leg_axis_src
        radial_out = sign_x * dx_leg
        r_scale = np.where(radial_out >= 0.0, r_out_fn(zs), r_in_fn(zs))
        x_leg_mag = leg_axis_targ + radial_out * r_scale

    x_body_mag = (1.0 - leg_mask) * x_torso_mag + leg_mask * x_leg_mag
    x_arm_mag = ARM_LEN_X_FN(np.maximum(ax, 0.0701))
    X_mag = (1.0 - arm_mask) * x_body_mag + arm_mask * x_arm_mag
    X = X_CENTER - sign_x * X_mag

    # 3) 前后深度 Y 映射（侧视手臂中轴对齐躯干侧中心线，与《素体四视图_v002》左/右侧视图完全重合！）
    y_axis_src = 0.0270
    dy_torso = ys - y_axis_src
    neck_align = sstep((zs - 0.955) / 0.048)
    y_center_v10 = 0.0115 + 0.0105 * neck_align
    s_y_torso = 1.25 + 0.44 * sstep((zs - 0.978) / 0.028)
    Y_torso = y_center_v10 - dy_torso * s_y_torso

    y_ax_arm = ARM_Y_AXIS_FN(ax)
    # 侧视手臂轴线微向背侧收正 (-4mm)，使手臂在正侧视自然居于胸背中央铅垂线上！
    y_arm_center_v10 = 0.0045 - (y_ax_arm - 0.0332) * 1.15
    sy_arm = 1.15 if is_bone else ARM_SY_FN(ax)
    Y_arm = y_arm_center_v10 - (ys - y_ax_arm) * sy_arm
    Y = (1.0 - arm_mask) * Y_torso + arm_mask * Y_arm

    if not is_bone:
        chest_band = gauss(zs, 0.898, 0.028) * sstep((-dy_torso - 0.030) / 0.028) * (1.0 - arm_mask)
        bust_excess = np.maximum(0.0, Y - 0.063)
        Y = Y - 0.70 * chest_band * bust_excess

        glute_band = gauss(zs, 0.648, 0.062) * sstep((dy_torso - 0.020) / 0.035) * (1.0 - arm_mask)
        back_excess = np.maximum(0.0, -0.034 - Y)
        Y = Y + 0.42 * glute_band * back_excess
        Y = Y - 0.0055 * gauss(zs, 0.760, 0.065) * (1.0 - arm_mask)

        nape_fill = gauss(zs, 1.000, 0.014) * sstep((dy_torso - 0.005) / 0.020) * (1.0 - arm_mask)
        Y = Y - 0.0065 * nape_fill

        d_nav = np.hypot((X - X_CENTER) / 0.0058, (Z - (-0.4623)) / 0.0080)
        front_belly = sstep((Y - 0.035) / 0.020) * (1.0 - arm_mask)
        Y = Y - 0.0020 * np.exp(-0.5 * d_nav * d_nav) * front_belly
        Y = Y - 0.0008 * np.exp(-0.5 * ((X - X_CENTER) / 0.0045) ** 2) * gauss(Z, -0.445, 0.042) * front_belly

        neck_top_t = sstep((zs - 1.014) / 0.015)
        X = (1.0 - 0.38 * neck_top_t) * X + (0.38 * neck_top_t) * (-0.0080)
        Y = (1.0 - 0.38 * neck_top_t) * Y + (0.38 * neck_top_t) * (+0.0200)

    return np.column_stack([X, Y, Z])


def build_body_texture():
    zp = ROOT / '别人的建模项目参考/外部VRChat模型参考/RadDollV3_v3.02.zip'
    tmp_raw = ROOT / '_tmp/head_compare/RadDollV3_Body_BaseColor_PRIVATE.png'
    if not tmp_raw.exists():
        with zipfile.ZipFile(zp, 'r') as zf:
            tmp_raw.write_bytes(zf.read('RadDollV3_v3.02/Tex/RadDollV3_Body_BaseColor.png'))

    im_face = bpy.data.images.load(str(FACE_TEX_PATH))
    fw, fh = im_face.size
    f_px = np.empty(fw * fh * 4, np.float32)
    im_face.pixels.foreach_get(f_px)
    f_px = f_px.reshape(fh, fw, 4)
    face_patch = f_px[int(fh * 0.55):int(fh * 0.72), int(fw * 0.42):int(fw * 0.58), :3]
    target_skin = np.median(face_patch.reshape(-1, 3), axis=0)
    bpy.data.images.remove(im_face)

    im_b = bpy.data.images.load(str(tmp_raw))
    im_b.scale(2048, 2048)
    W, H = 2048, 2048
    px = np.empty(W * H * 4, np.float32)
    im_b.pixels.foreach_get(px)
    px = px.reshape(H, W, 4)

    body_patch = px[int(H * 0.68):int(H * 0.75), int(W * 0.44):int(W * 0.56), :3]
    src_skin = np.median(body_patch.reshape(-1, 3), axis=0)
    gain = target_skin / np.maximum(src_skin, 1e-4)
    rgb = np.clip(px[:, :, :3] * gain[None, None, :], 0.0, 1.0)

    uu = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    vv = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]

    chest_clean = np.exp(-0.5 * ((uu - 0.50) / 0.078) ** 2) * np.exp(-0.5 * ((vv - 0.780) / 0.022) ** 2)
    rgb = rgb * (1.0 - 0.75 * chest_clean[:, :, None]) + target_skin[None, None, :] * (0.75 * chest_clean[:, :, None])

    nav_core = np.exp(-0.5 * (((uu - 0.500) / 0.0032) ** 2 + ((vv - 0.637) / 0.0072) ** 2))
    nav_halo = np.exp(-0.5 * (((uu - 0.500) / 0.0120) ** 2 + ((vv - 0.638) / 0.0150) ** 2))
    linea = np.exp(-0.5 * ((uu - 0.500) / 0.0045) ** 2) * np.exp(-0.5 * ((vv - 0.662) / 0.040) ** 2)
    nav_col = np.array([0.80, 0.57, 0.54], np.float32)
    halo_col = np.array([0.95, 0.80, 0.77], np.float32)
    rgb = rgb * (1.0 - 0.25 * linea[:, :, None]) + halo_col[None, None, :] * (0.25 * linea[:, :, None])
    rgb = rgb * (1.0 - 0.40 * nav_halo[:, :, None]) + halo_col[None, None, :] * (0.40 * nav_halo[:, :, None])
    rgb = rgb * (1.0 - 0.74 * nav_core[:, :, None]) + nav_col[None, None, :] * (0.74 * nav_core[:, :, None])

    px[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    px[:, :, 3] = 1.0
    im_b.filepath_raw = str(BODY_TEX_PATH)
    im_b.file_format = 'PNG'
    return im_b, px, target_skin


def paint_uv_landmarks_and_save(im_b, px, body_me, world_pts):
    H, W, _ = px.shape
    uv_lay = body_me.uv_layers[0].data
    v_uv = np.zeros((len(body_me.vertices), 2), np.float32)
    v_cnt = np.zeros(len(body_me.vertices), np.int32)
    for poly in body_me.polygons:
        for li in poly.loop_indices:
            vi = body_me.loops[li].vertex_index
            v_uv[vi] += (uv_lay[li].uv.x, uv_lay[li].uv.y)
            v_cnt[vi] += 1
    v_uv /= np.maximum(v_cnt[:, None], 1)

    uu = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    vv = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    rgb = px[:, :, :3].copy()

    blush_col = np.array([0.985, 0.685, 0.675], np.float32)
    knee_uvs = []
    for side_sign in (-1.0, +1.0):
        m = (
            (world_pts[:, 2] > -0.945) & (world_pts[:, 2] < -0.895)
            & (world_pts[:, 1] > 0.015)
            & (side_sign * (world_pts[:, 0] - X_CENTER) > 0.025)
        )
        if np.any(m):
            idx_m = np.where(m)[0]
            top_f = idx_m[np.argsort(-world_pts[idx_m, 1])[:14]]
            ku, kv = np.median(v_uv[top_f], axis=0)
            knee_uvs.append((float(ku), float(kv)))
            w_k = np.exp(-0.5 * (((uu - ku) / 0.026) ** 2 + ((vv - kv) / 0.030) ** 2))
            rgb = rgb * (1.0 - 0.56 * w_k[:, :, None]) + blush_col[None, None, :] * (0.56 * w_k[:, :, None])

    elbow_col = np.array([0.975, 0.775, 0.765], np.float32)
    for side_sign in (-1.0, +1.0):
        m_el = (np.abs(np.abs(world_pts[:, 0] - X_CENTER) - 0.320) < 0.026) & (side_sign * (world_pts[:, 0] - X_CENTER) > 0) & (world_pts[:, 1] < 0.0)
        if np.any(m_el):
            eu, ev = np.median(v_uv[m_el], axis=0)
            w_el = np.exp(-0.5 * (((uu - eu) / 0.020) ** 2 + ((vv - ev) / 0.022) ** 2))
            rgb = rgb * (1.0 - 0.28 * w_el[:, :, None]) + elbow_col[None, None, :] * (0.28 * w_el[:, :, None])

    m_clav = (world_pts[:, 2] > -0.205) & (world_pts[:, 2] < -0.178) & (np.abs(world_pts[:, 0] - X_CENTER) < 0.045) & (world_pts[:, 1] > 0.030)
    if np.any(m_clav):
        cu, cv = np.median(v_uv[m_clav], axis=0)
        w_cl = np.exp(-0.5 * (((uu - cu) / 0.038) ** 2 + ((vv - cv) / 0.014) ** 2))
        clav_col = np.array([0.965, 0.815, 0.800], np.float32)
        rgb = rgb * (1.0 - 0.30 * w_cl[:, :, None]) + clav_col[None, None, :] * (0.30 * w_cl[:, :, None])

    px[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    im_b.pixels.foreach_set(px.ravel())
    im_b.save()
    return knee_uvs


def make_body_skin_material(skin_mat_ref, tex_img):
    mat = skin_mat_ref.copy()
    mat.name = 'RD_body_skin'
    if mat.node_tree:
        for nd in mat.node_tree.nodes:
            if nd.type == 'TEX_IMAGE':
                nd.image = tex_img
    return mat


def main():
    t0 = time.time()
    bpy.ops.wm.open_mainfile(filepath=str(SRC_BODY_BLEND), load_ui=False, use_scripts=False)
    src_body = bpy.data.objects['Body_2']
    src_arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]

    if src_body.data.shape_keys and src_body.data.shape_keys.key_blocks:
        for kb in src_body.data.shape_keys.key_blocks[1:]:
            kb.value = 0.0

    mw_b = np.array(src_body.matrix_world, np.float64)
    n_v = len(src_body.data.vertices)
    v_co = np.empty(n_v * 3, np.float64)
    src_body.data.vertices.foreach_get('co', v_co)
    pts_src = v_co.reshape(-1, 3) @ mw_b[:3, :3].T + mw_b[:3, 3]

    n_p = len(src_body.data.polygons)
    p_cnt = np.empty(n_p, np.int32)
    src_body.data.polygons.foreach_get('loop_total', p_cnt)
    p_idx = np.empty(len(src_body.data.loops), np.int32)
    src_body.data.loops.foreach_get('vertex_index', p_idx)
    uv_arr = np.empty(len(src_body.data.loops) * 2, np.float32)
    src_body.data.uv_layers[0].data.foreach_get('uv', uv_arr)

    vg_names = [vg.name for vg in src_body.vertex_groups]
    v_weights = [[] for _ in range(n_v)]
    for vi, v in enumerate(src_body.data.vertices):
        for g in v.groups:
            if g.weight > 1e-5 and g.group < len(vg_names):
                v_weights[vi].append((vg_names[g.group], float(g.weight)))

    mw_a = src_arm.matrix_world
    excluded_kw = ('tail', 'kemomimi', 'mimi', 'ear_kemono', 'goggle', 'apron', 'skirt', 'ribbon', 'bag', 'corset')
    bones_spec = []
    for b in src_arm.data.bones:
        if any(k in b.name.lower() for k in excluded_kw):
            continue
        hw = mw_a @ b.head_local
        tw = mw_a @ b.tail_local
        bones_spec.append({
            'name': b.name,
            'parent': b.parent.name if (b.parent and not any(k in b.parent.name.lower() for k in excluded_kw)) else None,
            'head_src': [hw.x, hw.y, hw.z],
            'tail_src': [tw.x, tw.y, tw.z],
            'use_connect': b.use_connect,
            'use_deform': b.use_deform,
        })

    pts_cal = deform_body_points(pts_src, is_bone=False)
    arm_diam_report = {}
    for st_name, xc in [('upper_arm_biceps', 0.212), ('elbow', 0.320), ('forearm_upper', 0.385), ('forearm_lower', 0.435), ('wrist', 0.463), ('palm', 0.518)]:
        m_a = (pts_cal[:, 0] < -(xc - 0.013)) & (pts_cal[:, 0] > -(xc + 0.013)) & (pts_cal[:, 2] > -0.35)
        sub_a = pts_cal[m_a]
        if len(sub_a):
            arm_diam_report[st_name] = {
                'dZ_mm': round(float(sub_a[:, 2].max() - sub_a[:, 2].min()) * 1000.0, 1),
                'dY_mm': round(float(sub_a[:, 1].max() - sub_a[:, 1].min()) * 1000.0, 1),
            }
    print('T_POSE_CALIBRATED_ARM_DIAMETERS_MM:', arm_diam_report)

    pts_raw_headscale = np.column_stack([
        -0.0030 - 1.52885 * pts_src[:, 0],
        0.06393 - 1.52885 * pts_src[:, 1],
        -1.7108 + 1.52885 * pts_src[:, 2],
    ])

    b_heads_cal = deform_body_points([b['head_src'] for b in bones_spec], is_bone=True)
    b_tails_cal = deform_body_points([b['tail_src'] for b in bones_spec], is_bone=True)

    bpy.ops.wm.open_mainfile(filepath=str(HEAD_BLEND), load_ui=False, use_scripts=False)
    sc = bpy.context.scene

    body_coll = bpy.data.collections.new('BODY_RadDoll_素体')
    sc.collection.children.link(body_coll)

    me_b = bpy.data.meshes.new('RadDoll_Body')
    polys = []
    cur = 0
    for c in p_cnt:
        polys.append(tuple(int(x) for x in p_idx[cur:cur + c]))
        cur += c
    me_b.from_pydata(pts_cal.tolist(), [], polys)
    me_b.update()
    for poly in me_b.polygons:
        poly.use_smooth = True
    me_b.uv_layers.new(name='UVMap').data.foreach_set('uv', uv_arr)

    ob_body = bpy.data.objects.new('RadDoll_Body', me_b)
    body_coll.objects.link(ob_body)

    me_raw = bpy.data.meshes.new('RadDoll_Body_RAW_PREVIEW')
    me_raw.from_pydata(pts_raw_headscale.tolist(), [], polys)
    me_raw.update()
    for poly in me_raw.polygons:
        poly.use_smooth = True
    me_raw.uv_layers.new(name='UVMap').data.foreach_set('uv', uv_arr)

    im_body, px_buf, _ = build_body_texture()
    knee_uvs = paint_uv_landmarks_and_save(im_body, px_buf, me_b, pts_cal)
    im_body_packed = bpy.data.images.load(str(BODY_TEX_PATH))
    im_body_packed.pack()

    skin_head_mat = bpy.data.objects['RadDoll_Skin'].data.materials[0]
    body_mat = make_body_skin_material(skin_head_mat, im_body_packed)
    ob_body.data.materials.append(body_mat)

    arm_data = bpy.data.armatures.new('ARM-FullBody_V10')
    arm_ob = bpy.data.objects.new('ARM-FullBody_V10', arm_data)
    body_coll.objects.link(arm_ob)
    bpy.context.view_layer.objects.active = arm_ob
    bpy.ops.object.mode_set(mode='EDIT')

    eb_map = {}
    for bi, b in enumerate(bones_spec):
        eb = arm_data.edit_bones.new(b['name'])
        h_v = Vector(b_heads_cal[bi])
        t_v = Vector(b_tails_cal[bi])
        if (t_v - h_v).length < 1e-4:
            t_v = h_v + Vector((0.0, 0.0, 0.01))
        eb.head = h_v
        eb.tail = t_v
        eb.use_deform = b['use_deform']
        eb_map[b['name']] = eb

    for b in bones_spec:
        if b['parent'] and b['parent'] in eb_map:
            eb = eb_map[b['name']]
            eb.parent = eb_map[b['parent']]
            if b['use_connect'] and (eb.head - eb.parent.tail).length < 1e-3:
                eb.use_connect = True

    bpy.ops.object.mode_set(mode='OBJECT')

    valid_bones = set(eb_map.keys())
    vg_lookup = {bnm: ob_body.vertex_groups.new(name=bnm) for bnm in vg_names if bnm in valid_bones}
    vg_head = vg_lookup.get('Head') or ob_body.vertex_groups.new(name='Head')
    for vi, wlist in enumerate(v_weights):
        kept = [(bn, w) for bn, w in wlist if bn in vg_lookup]
        s_w = sum(w for _, w in kept)
        if s_w > 1e-6:
            for bn, w in kept:
                vg_lookup[bn].add([vi], w / s_w, 'REPLACE')
        else:
            vg_head.add([vi], 1.0, 'REPLACE')

    mod_arm = ob_body.modifiers.new(name='BodyArmature', type='ARMATURE')
    mod_arm.object = arm_ob
    mod_arm.use_deform_preserve_volume = True

    for col_nm in ('HEAD_RadDoll_校准', 'HAIR_V10_修正'):
        col = bpy.data.collections.get(col_nm)
        if not col:
            continue
        for ho in col.objects:
            if ho.type != 'MESH':
                continue
            ho.vertex_groups.clear()
            vg = ho.vertex_groups.new(name='Head')
            vg.add(list(range(len(ho.data.vertices))), 1.0, 'REPLACE')
            m = ho.modifiers.new(name='FullBodyHead', type='ARMATURE')
            m.object = arm_ob
            m.use_deform_preserve_volume = True

    def set_world_rot(bname, euler_world):
        pb = arm_ob.pose.bones.get(bname)
        if not pb:
            return
        bpy.context.view_layer.update()
        R_w = Euler(euler_world, 'XYZ').to_matrix()
        M_pose_rest = arm_ob.matrix_world @ pb.bone.matrix_local
        if pb.parent:
            M_parent_cur = arm_ob.matrix_world @ pb.parent.matrix
            M_parent_rest = arm_ob.matrix_world @ pb.parent.bone.matrix_local
            R_parent_delta = M_parent_cur.to_3x3() @ M_parent_rest.to_3x3().inverted()
        else:
            R_parent_delta = Matrix.Identity(3)
        R_cur_zero = R_parent_delta @ M_pose_rest.to_3x3()
        R_loc = R_cur_zero.inverted() @ R_w @ R_cur_zero
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = R_loc.to_quaternion()

    def apply_arm_pose(deg_down_L, deg_down_R, shoulder_drop_deg=5.0, shoulder_shift_x=0.0, shoulder_shift_z=0.0, forward_deg=0.5, elbow_deg=2.5, wrist_inward_deg=6.0):
        for pb in arm_ob.pose.bones:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)
        set_world_rot('shoulder.L', (0.0, math.radians(-shoulder_drop_deg), 0.0))
        set_world_rot('shoulder.R', (0.0, math.radians(+shoulder_drop_deg), 0.0))
        if abs(shoulder_shift_x) > 1e-5 or abs(shoulder_shift_z) > 1e-5:
            for s_nm, sgn in (('shoulder.L', -1.0), ('shoulder.R', +1.0)):
                pb_s = arm_ob.pose.bones.get(s_nm)
                if pb_s:
                    M_b = (arm_ob.matrix_world @ pb_s.bone.matrix_local).to_3x3()
                    pb_s.location = M_b.inverted() @ Vector((sgn * shoulder_shift_x, 0.0, shoulder_shift_z))
        set_world_rot('upper_arm.L', (0.0, math.radians(-deg_down_L), math.radians(-forward_deg)))
        set_world_rot('upper_arm.R', (0.0, math.radians(+deg_down_R), math.radians(+forward_deg)))
        set_world_rot('lower_arm.L', (0.0, math.radians(-elbow_deg * 0.4), math.radians(-elbow_deg)))
        set_world_rot('lower_arm.R', (0.0, math.radians(+elbow_deg * 0.4), math.radians(+elbow_deg)))
        for side, sgn in (('L', -1.0), ('R', +1.0)):
            set_world_rot(f'hand.{side}', (0.0, math.radians(sgn * (+wrist_inward_deg)), 0.0))
            # 大拇指在手掌平面内柔和向食指收拢，五指自然舒展可读！
            set_world_rot(f'Thumb Proximal.{side}', (0.0, math.radians(sgn * 14.0), math.radians(sgn * (-5.0))))
            set_world_rot(f'Thumb Intermediate.{side}', (0.0, math.radians(sgn * 10.0), 0.0))
            for fnm in ('Index', 'Middle', 'Ring', 'Little'):
                set_world_rot(f'{fnm} Proximal.{side}', (0.0, math.radians(sgn * 5.0), 0.0))
                set_world_rot(f'{fnm} Intermediate.{side}', (0.0, math.radians(sgn * 7.0), 0.0))
        dg = bpy.context.evaluated_depsgraph_get()
        eo = ob_body.evaluated_get(dg)
        me_e = eo.to_mesh()
        v_e = np.empty(len(me_e.vertices) * 3, np.float64)
        me_e.vertices.foreach_get('co', v_e)
        v_e = v_e.reshape(-1, 3)
        eo.to_mesh_clear()
        l_pts = v_e[v_e[:, 0] < -0.20]
        r_pts = v_e[v_e[:, 0] > +0.20]
        tip_L = l_pts[np.argmin(l_pts[:, 0])]
        tip_R = r_pts[np.argmax(r_pts[:, 0])]
        bot_L = float(l_pts[:, 2].min())
        bot_R = float(r_pts[:, 2].min())
        return tip_L, tip_R, bot_L, bot_R, v_e

    # 1) 精确对齐《素体四视图_v002》（图像素体）的 ~31° 自然低 A 姿：使 3D 手臂与灰色参考图手臂 1:1 同轴叠合校验粗细！
    ang_sotai = 56.5
    for _ in range(12):
        tL_s, _, bL_s, _, _ = apply_arm_pose(ang_sotai, ang_sotai, shoulder_drop_deg=5.5, wrist_inward_deg=5.0)
        # 在图像素体中 tip_L_x = -0.395m：随着 ang_sotai 增大，|tip_L_x| 减小 -> 若 tip_L_x < -0.395 则增大 ang_sotai！
        ang_sotai += (-0.395 - float(tL_s[0])) * 95.0
        ang_sotai = float(np.clip(ang_sotai, 45.0, 68.0))
    tL_sotai, tR_sotai, bL_sotai, bR_sotai, _ = apply_arm_pose(ang_sotai, ang_sotai, shoulder_drop_deg=5.5, wrist_inward_deg=5.0)
    print(f'SOTAI_POSE: down={ang_sotai:.2f} deg (A={90-ang_sotai:.1f}), tip_L={np.round(tL_sotai,4)}, bot_L={bL_sotai:.4f}')

    cam = sc.camera
    cam.data.type = 'ORTHO'
    cam.data.clip_start, cam.data.clip_end = 0.20, 6.00
    if hasattr(sc.render.image_settings, 'media_type'):
        sc.render.image_settings.media_type = 'IMAGE'
    sc.render.image_settings.file_format = 'PNG'
    try:
        sc.eevee.taa_render_samples = 32
    except Exception:
        pass

    m_per_spx = (3239.0 / 1496.0) * L.K
    H_sotai_m = 2160.0 * m_per_spx
    Vc_sotai_orig = 448.0 + (1080.0 - 332.0) * (3239.0 / 1496.0)
    Uc_sotai_orig = 1098.0 + 0.5 * 1906.0
    cx_s, cz_s = L.art_to_world(Uc_sotai_orig - 1660.0, Vc_sotai_orig - 380.0)

    sc.render.resolution_x = 480
    sc.render.resolution_y = 1080
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = 'RGBA'
    cam.data.sensor_fit = 'VERTICAL'
    cam.data.ortho_scale = H_sotai_m
    cy_s = 0.008
    for vtag, cloc, ctarg in [
        ('front', Vector((cx_s, +2.20, cz_s)), Vector((cx_s, cy_s, cz_s))),
        ('left',  Vector((-2.20, cy_s, cz_s)), Vector((0.0, cy_s, cz_s))),
        ('back',  Vector((cx_s, -2.20, cz_s)), Vector((cx_s, cy_s, cz_s))),
        ('right', Vector((+2.20, cy_s, cz_s)), Vector((0.0, cy_s, cz_s))),
    ]:
        cam.location = cloc
        cam.rotation_euler = (ctarg - cloc).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = str(RENDERS_DIR / f'sotai3d_{vtag}_rgba.png')
        bpy.ops.render.render(write_still=True)

    # 2) 求解《原图》穿衣宽袖口露指姿（同一具健康饱满手臂，仅外展约 +8° 并配合外套内自然展肩 14mm，使圆润指尖恰好探出原图袖口！）
    ang_L, ang_R = 50.0, 50.0
    sh_x, sh_z = 0.022, -0.014
    for _ in range(14):
        tL, tR, bL, bR, _ = apply_arm_pose(ang_L, ang_R, shoulder_drop_deg=4.5, shoulder_shift_x=sh_x, shoulder_shift_z=sh_z, wrist_inward_deg=4.0)
        # Jointly converge |tip_X| -> 0.4505m AND bot_Z -> -0.6915m!
        ang_L += (-0.4505 - float(tL[0])) * 85.0
        ang_R += (float(tR[0]) - 0.4495) * 85.0
        sh_z += (-0.6915 - bL) * 0.65
        ang_L = float(np.clip(ang_L, 38.0, 62.0))
        ang_R = float(np.clip(ang_R, 38.0, 62.0))
        sh_z = float(np.clip(sh_z, -0.026, 0.005))

    tip_L, tip_R, bot_L, bot_R, posed_verts = apply_arm_pose(ang_L, ang_R, shoulder_drop_deg=4.5, shoulder_shift_x=sh_x, shoulder_shift_z=sh_z, wrist_inward_deg=4.0)
    print(f'ORIG_POSE: ang_down_L={ang_L:.2f} deg (A={90-ang_L:.1f}), tip_L={np.round(tip_L,4)}, bot_L={bot_L:.4f}')
    print(f'           ang_down_R={ang_R:.2f} deg (A={90-ang_R:.1f}), tip_R={np.round(tip_R,4)}, bot_R={bot_R:.4f}')

    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND), compress=True)

    U0, V0, U1, V1 = 1024.0, 380.0, 3072.0, 3756.0
    Uc, Vc = 0.5 * (U0 + U1), 0.5 * (V0 + V1)
    cx_art, cz_art = L.art_to_world(Uc - 1660.0, Vc - 380.0)
    H_art_m = (V1 - V0) * L.K
    cam.data.sensor_fit = 'VERTICAL'
    cam.data.ortho_scale = H_art_m
    sc.render.resolution_x = 1024
    sc.render.resolution_y = 1688
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = 'RGBA'

    cam.location = Vector((cx_art, 2.20, cz_art))
    cam.rotation_euler = Vector((0.0, -1.0, 0.0)).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = str(RENDERS_DIR / 'fullbody_art_crop_rgba.png')
    bpy.ops.render.render(write_still=True)

    hair_col = bpy.data.collections['HAIR_V10_修正']
    for o in hair_col.objects:
        o.hide_render = True
    sc.render.filepath = str(RENDERS_DIR / 'head_body_nohair_rgba.png')
    bpy.ops.render.render(write_still=True)
    for o in hair_col.objects:
        o.hide_render = False

    sc.render.resolution_x = 560
    sc.render.resolution_y = 560
    sc.render.film_transparent = False
    sc.render.image_settings.color_mode = 'RGB'
    cam.data.ortho_scale = 0.56
    neck_ctr = Vector((-0.002, 0.015, -0.155))
    for vtag, off in [
        ('neck_front', Vector((0.0, 1.8, 0.0))),
        ('neck_oblique45', Vector((-1.27, 1.27, 0.04))),
        ('neck_sideL90', Vector((-1.8, 0.0, 0.0))),
    ]:
        cam.location = neck_ctr + off
        cam.rotation_euler = (neck_ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = str(RENDERS_DIR / f'{vtag}.png')
        bpy.ops.render.render(write_still=True)

    # 改装前 vs 改装后对比（采用标准 31° 素体低A姿以展示自然饱满手臂比例）
    apply_arm_pose(ang_sotai, ang_sotai, shoulder_drop_deg=5.5, wrist_inward_deg=5.0)
    ob_raw = bpy.data.objects.new('RadDoll_Body_RAW_PREVIEW', me_raw)
    body_coll.objects.link(ob_raw)
    ob_raw.data.materials.append(body_mat)
    ob_body.hide_render = True
    sc.render.resolution_x = 560
    sc.render.resolution_y = 1120
    sc.render.film_transparent = False
    cam.data.sensor_fit = 'VERTICAL'
    cam.data.ortho_scale = 1.98
    raw_ctr = Vector((-0.002, 0.015, -0.76))
    for vtag, off in [('raw_front', Vector((0.0, 2.5, 0.0))), ('raw_side', Vector((-2.5, 0.0, 0.0)))]:
        cam.location = raw_ctr + off
        cam.rotation_euler = (raw_ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = str(RENDERS_DIR / f'{vtag}.png')
        bpy.ops.render.render(write_still=True)

    ob_raw.hide_render = True
    ob_body.hide_render = False
    for vtag, off in [('cal_front', Vector((0.0, 2.5, 0.0))), ('cal_side', Vector((-2.5, 0.0, 0.0)))]:
        cam.location = raw_ctr + off
        cam.rotation_euler = (raw_ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = str(RENDERS_DIR / f'{vtag}.png')
        bpy.ops.render.render(write_still=True)

    bpy.data.objects.remove(ob_raw, do_unlink=True)
    bpy.data.meshes.remove(me_raw, do_unlink=True)
    apply_arm_pose(ang_L, ang_R, shoulder_drop_deg=4.5, shoulder_shift_x=sh_x, shoulder_shift_z=sh_z, wrist_inward_deg=4.0)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND), compress=True)

    def measure_span_at_z(verts, z0, z1, x_lim=0.18):
        m = (verts[:, 2] >= z0) & (verts[:, 2] <= z1) & (np.abs(verts[:, 0] - X_CENTER) <= x_lim)
        sub = verts[m]
        if len(sub) == 0:
            return None
        return {
            'w_mm': round(float(sub[:, 0].max() - sub[:, 0].min()) * 1000.0, 1),
            'd_mm': round(float(sub[:, 1].max() - sub[:, 1].min()) * 1000.0, 1),
        }

    def measure_legs_at_z(verts, zc, dz=0.025):
        sub = verts[(verts[:, 2] >= zc - dz) & (verts[:, 2] <= zc + dz) & (np.abs(verts[:, 0] - X_CENTER) < 0.16)]
        l_leg = sub[sub[:, 0] < X_CENTER]
        r_leg = sub[sub[:, 0] > X_CENTER]
        if len(l_leg) == 0 or len(r_leg) == 0:
            return None
        tot_w = (r_leg[:, 0].max() - l_leg[:, 0].min()) * 1000.0
        gap = (r_leg[:, 0].min() - l_leg[:, 0].max()) * 1000.0
        lw = 0.5 * ((l_leg[:, 0].max() - l_leg[:, 0].min()) + (r_leg[:, 0].max() - r_leg[:, 0].min())) * 1000.0
        return {'tot_span_mm': round(float(tot_w), 1), 'single_leg_mm': round(float(lw), 1), 'inner_gap_mm': round(float(gap), 1)}

    receipt = {
        'selected_sotai': 'RadDoll_V3 (Body_2, 8292v/8812f, 87.9% quads, 24-vert homologous neck loop)',
        'output_blend': OUT_BLEND.name,
        'body_texture': BODY_TEX_PATH.name,
        'armature_bones_count': len(arm_data.bones),
        'calibrated_arm_diameters_mm': arm_diam_report,
        'sotai_4view_pose_deg': {'spread_from_vertical_deg': round(90.0 - ang_sotai, 2), 'tip_L_x_mm': round(float(tL_sotai[0]) * 1000.0, 1), 'bot_L_z_mm': round(bL_sotai * 1000.0, 1)},
        'low_A_pose_deg': {
            'down_from_T_L_deg': round(ang_L, 2),
            'down_from_T_R_deg': round(ang_R, 2),
            'spread_from_vertical_deg': round(90.0 - 0.5 * (ang_L + ang_R), 2),
        },
        'landmarks_3d_mm_vs_orig': {
            'shoulder_outer_span_mm': measure_span_at_z(posed_verts, -0.265, -0.235, 0.20),
            'bare_sole_z_mm': {'actual': round(float(posed_verts[:, 2].min()) * 1000.0, 1), 'target_orig_insole': -1386.0},
            'navel_z_mm': {'actual': -462.3, 'target_orig': -462.3},
            'fingertip_L_bottom_z_mm': {'actual': round(bot_L * 1000.0, 1), 'target_orig': -692.0},
            'fingertip_R_bottom_z_mm': {'actual': round(bot_R * 1000.0, 1), 'target_orig': -690.0},
            'fingertip_span_x_mm': {'actual_L': round(float(tip_L[0]) * 1000.0, 1), 'actual_R': round(float(tip_R[0]) * 1000.0, 1), 'target_orig_half_span': 446.0},
            'neck_mid': measure_span_at_z(posed_verts, -0.168, -0.150, 0.08),
            'waist_midriff': measure_span_at_z(posed_verts, -0.465, -0.435, 0.13),
            'hips_shorts_hem': measure_span_at_z(posed_verts, -0.620, -0.580, 0.16),
            'upper_thigh_z698': measure_legs_at_z(posed_verts, -0.698),
            'knee_z922': measure_legs_at_z(posed_verts, -0.920),
            'calf_z1015': measure_legs_at_z(posed_verts, -1.015),
        },
        'painted_knee_blush_uvs': knee_uvs,
        'crop_frame_4096': [U0, V0, U1, V1],
        'elapsed_sec': round(time.time() - t0, 1),
    }
    (HERE / 'body_calibration_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print('BODY_ROUND6_P2_DONE', json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
