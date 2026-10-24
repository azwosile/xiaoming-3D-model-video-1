"""V10R7 · 眼睛与面部贴图生成：基于 head_fit.py 导出的变形后精确 UV↔世界坐标三角网（重心逆映射），
零畸变绘制 2048×2048 的 RadDoll_eye_tex.png（亮白巩膜 + 上睑淡紫投影 + 圆润钴蓝→冰蓝渐变虹膜 + 柔和靛蓝瞳仁 + 双点高光）
与 RadDoll_face_tex.png（瓷白肤色 + 眼窝底无缝亮白巩膜 + 柔粉腮红 + 鼻尖/唇线淡彩 + 颌下暖调阴影）。
Windows Python + NumPy + Pillow。
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
SNAP = HERE.parents[2] / '_tmp/head_compare/snap/head_uv_deformed.npz'
W_EYE = 2048
W_FACE = 2048

# 原图实测双瞳几何（820×820 裁剪系）：左右虹膜圆润饱满（rx=30.2, ry=28.5 px），瞳仁柔和竖椭圆（非猫眼细缝）
EYES = {
    'L': {'uc': 301.8, 'vc': 546.8, 'rx': 29.6, 'ry': 28.4, 'vlash': 529.5},
    'R': {'uc': 486.8, 'vc': 536.2, 'rx': 28.8, 'ry': 27.6, 'vlash': 519.5},
}


def sstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp3(a, b, t):
    t = np.asarray(t)[..., None]
    return np.asarray(a, np.float32) * (1.0 - t) + np.asarray(b, np.float32) * t


def rasterize_uv_to_world(tris_uv, tris_w, res):
    """把 UV 三角网光栅化到 res×res 网格，返回 (valid_mask, world_xyz_map[res,res,3])。"""
    xyz = np.zeros((res, res, 3), np.float32)
    valid = np.zeros((res, res), bool)
    # UV 像素坐标：ix = u * res - 0.5, iy = (1 - v) * res - 0.5
    U = tris_uv[:, :, 0] * res - 0.5
    V = (1.0 - tris_uv[:, :, 1]) * res - 0.5
    xmin = np.maximum(0, np.floor(U.min(axis=1)).astype(int))
    xmax = np.minimum(res - 1, np.ceil(U.max(axis=1)).astype(int))
    ymin = np.maximum(0, np.floor(V.min(axis=1)).astype(int))
    ymax = np.minimum(res - 1, np.ceil(V.max(axis=1)).astype(int))

    x0, y0 = U[:, 0], V[:, 0]
    x1, y1 = U[:, 1], V[:, 1]
    x2, y2 = U[:, 2], V[:, 2]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

    for ti in range(len(tris_uv)):
        d = float(den[ti])
        if abs(d) < 1e-8 or xmin[ti] > xmax[ti] or ymin[ti] > ymax[ti]:
            continue
        gx, gy = np.meshgrid(np.arange(xmin[ti], xmax[ti] + 1), np.arange(ymin[ti], ymax[ti] + 1))
        w0 = ((y1[ti] - y2[ti]) * (gx - x2[ti]) + (x2[ti] - x1[ti]) * (gy - y2[ti])) / d
        w1 = ((y2[ti] - y0[ti]) * (gx - x2[ti]) + (x0[ti] - x2[ti]) * (gy - y2[ti])) / d
        w2 = 1.0 - w0 - w1
        m = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not m.any():
            continue
        ys, xs = gy[m], gx[m]
        b0, b1, b2 = w0[m, None], w1[m, None], w2[m, None]
        tw = tris_w[ti]
        xyz[ys, xs] = b0 * tw[0] + b1 * tw[1] + b2 * tw[2]
        valid[ys, xs] = True
    return valid, xyz


def dilate_rgb(rgb, valid, steps=16):
    out = rgb.copy()
    v = valid.copy()
    for _ in range(steps):
        if v.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(v.shape, np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            sv = np.roll(np.roll(v, dy, axis=0), dx, axis=1)
            sc = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            m = (~v) & sv
            acc[m] += sc[m]
            cnt[m] += 1.0
        fill = (~v) & (cnt > 0)
        out[fill] = acc[fill] / cnt[fill, None]
        v |= fill
    return out


def shade_eyes(u_art, v_art, mid_u):
    """输入任意形状的 (u_art, v_art) 数组，返回 sRGB [0..255] float32。"""
    sclera = np.array([249.0, 243.0, 245.0], np.float32)
    sclera_sh = np.array([208.0, 195.0, 213.0], np.float32)
    out = np.broadcast_to(sclera, u_art.shape + (3,)).copy()

    for side, cfg in EYES.items():
        mask_side = (u_art < mid_u) if side == 'L' else (u_art >= mid_u)
        if not np.any(mask_side):
            continue
        u = u_art[mask_side]
        v = v_art[mask_side]
        uc, vc, rx, ry, vlash = cfg['uc'], cfg['vc'], cfg['rx'], cfg['ry'], cfg['vlash']

        # 1) 巩膜：洁白偏暖 + 上睫毛下缘柔和淡紫灰阴影
        t_lid = sstep((vlash + 14.5 - v) / 16.5)
        col = lerp3(sclera, sclera_sh, 0.85 * t_lid)

        # 2) 虹膜主体（近似正圆，上钴蓝→中矢车菊蓝→下冰晶蓝）
        du = (u - uc) / rx
        dv = (v - vc) / ry
        rn = np.hypot(du, dv)
        in_iris = rn < 1.06
        if np.any(in_iris):
            du_i, dv_i, rn_i, v_i, u_i = du[in_iris], dv[in_iris], rn[in_iris], v[in_iris], u[in_iris]
            # 竖向三段柔和渐变
            g1 = sstep((dv_i + 0.85) / 0.85)
            g2 = sstep((dv_i - 0.00) / 0.88)
            ic = lerp3([42, 64, 160], [82, 124, 210], g1)
            ic = lerp3(ic, [214, 239, 255], g2)

            # 放射状细微虹膜纹理 + 下半圈月牙提亮
            ang = np.arctan2(dv_i, du_i)
            spokes = 0.5 + 0.5 * np.cos(22.0 * ang + 0.35 * np.sin(6.0 * ang))
            w_spoke = sstep((rn_i - 0.30) / 0.20) * sstep((0.86 - rn_i) / 0.14) * sstep((dv_i + 0.35) / 0.8)
            ic = lerp3(ic, [226, 245, 255], 0.16 * spokes * w_spoke)
            w_glow = np.exp(-((rn_i - 0.67) / 0.15) ** 2) * sstep((dv_i - 0.08) / 0.55)
            ic = lerp3(ic, [232, 248, 255], 0.42 * w_glow)

            # 外圈暗环（上粗深靛蓝、下细柔灰蓝，还原原图笔触）
            limbal_start = np.where(dv_i < 0, 0.79, 0.85)
            w_lim = sstep((rn_i - limbal_start) / (1.0 - limbal_start))
            lim_col = lerp3([36, 50, 120], [74, 96, 162], sstep((dv_i + 0.25) / 0.85))
            ic = lerp3(ic, lim_col, 0.88 * w_lim)

            # 上眼睑落在虹膜顶部的柔和暗影
            w_top_sh = sstep((vlash + 11.0 - v_i) / 12.0)
            ic = lerp3(ic, [38, 54, 128], 0.36 * w_top_sh)

            # 3) 柔和椭圆形靛蓝瞳仁（中心略偏上，带半透明紫蓝内芯，告别猫眼黑竖缝）
            pdu = (u_i - uc) / 7.5
            pdv = (v_i - (vc - 1.2)) / 9.1
            pr = np.hypot(pdu, pdv)
            w_pup = sstep((1.08 - pr) / 0.24)
            pup_col = lerp3([36, 50, 118], [88, 108, 182], 0.68 * sstep((0.66 - np.hypot(pdu, pdv - 0.12)) / 0.42))
            ic = lerp3(ic, pup_col, 0.92 * w_pup)

            # 4) 二次元高光点（右上主高光 + 左上副高光 + 右下水光点）
            # 主高光（椭圆微倾）
            hx1 = (u_i - (uc + 5.0))
            hy1 = (v_i - (vc - 4.7))
            ca, sa = math.cos(-0.28), math.sin(-0.28)
            hr1 = np.hypot((hx1 * ca - hy1 * sa) / 4.2, (hx1 * sa + hy1 * ca) / 3.4)
            ic = lerp3(ic, [255, 255, 255], sstep((1.08 - hr1) / 0.22))
            # 左上小高光
            hr2 = np.hypot((u_i - (uc - 8.2)) / 2.05, (v_i - (vc - 3.2)) / 2.05)
            ic = lerp3(ic, [255, 255, 255], 0.95 * sstep((1.12 - hr2) / 0.28))
            # 右下柔水光
            hr3 = np.hypot((u_i - (uc + 8.8)) / 2.65, (v_i - (vc + 7.2)) / 1.8)
            ic = lerp3(ic, [245, 252, 255], 0.78 * sstep((1.12 - hr3) / 0.35))

            # 虹膜外缘约 1.1 原图像素的平滑抗锯齿过渡到眼白
            aa = sstep((1.0 - rn_i) * rx / 1.15 + 0.5)
            col_i = col[in_iris]
            col[in_iris] = lerp3(col_i, ic, aa)

        out[mask_side] = col
    return out


def main():
    z = np.load(SNAP)
    EX, EZ, K, MID_U, MID_V, CHIN_Z = z['meta']

    def to_art(xyz):
        return MID_U - (xyz[..., 0] - EX) / K, MID_V - (xyz[..., 2] - EZ) / K

    # -------- 1. 眼睛高精度贴图 RadDoll_eye_tex.png --------
    v_eye, xyz_eye = rasterize_uv_to_world(z['eye_uv'], z['eye_w'], W_EYE)
    eye_rgb = np.full((W_EYE, W_EYE, 3), [249.0, 243.0, 245.0], np.float32)
    ue, ve = to_art(xyz_eye[v_eye])
    eye_rgb[v_eye] = shade_eyes(ue, ve, MID_U)
    eye_rgb = dilate_rgb(eye_rgb, v_eye, steps=18)
    Image.fromarray(np.clip(np.round(eye_rgb), 0, 255).astype(np.uint8), 'RGB').save(HERE / 'RadDoll_eye_tex.png')

    # -------- 2. 面部贴图 RadDoll_face_tex.png（二次元精致妆容：双层蜜桃腮红+左颊水滴高光+上下眼影卧蚕+鼻梁高光鼻头红晕+水润唇妆+下巴暖晕+无缝眼窝白底）--------
    v_sk, xyz_sk = rasterize_uv_to_world(z['skin_uv'], z['skin_w'], W_FACE)
    skin_base = np.array([251.0, 234.0, 232.5], np.float32)
    face_rgb = np.full((W_FACE, W_FACE, 3), skin_base, np.float32)

    pts = xyz_sk[v_sk]
    us, vs = to_art(pts)
    col_s = np.broadcast_to(skin_base, pts.shape[:-1] + (3,)).copy()

    # 正面面部皮肤遮罩（排除后脑、耳后与眼窝深底，确保妆容只落在外表面）
    front_m = sstep((pts[:, 1] - 0.050) / 0.024)
    outer_skin_m = sstep((pts[:, 1] - 0.086) / 0.006)
    ca_r, sa_r = math.cos(math.radians(-3.28)), math.sin(math.radians(-3.28))

    # 2a) 额前发际柔和暖调（让露肤窗与银白刘海自然衔接，不惨白）
    w_fore = sstep((502.0 - vs) / 42.0) * sstep((vs - 430.0) / 30.0) * front_m
    col_s = lerp3(col_s, [249.0, 224.0, 222.0], 0.22 * w_fore)

    # 2b) 双层蜜桃珊瑚腮红（广域柔雾晕染 + 眼下微醺红晕核 + 极淡二次元斜线笔触 + 原画左颊水滴珍珠高光）
    blush_specs = (
        # (broad_u, broad_v, brx, bry, b_w, core_u, core_v, crx, cry, c_w)
        (288.0, 588.0, 47.0, 25.0, 0.48, 277.0, 578.5, 25.0, 11.5, 0.40),
        (496.0, 576.0, 45.0, 24.0, 0.45, 505.0, 567.0, 24.0, 11.0, 0.38),
    )
    for bu, bv, brx, bry, bw, cu, cv, crx, cry, cw in blush_specs:
        du_b = (us - bu) * ca_r - (vs - bv) * sa_r
        dv_b = (us - bu) * sa_r + (vs - bv) * ca_r
        r_b = np.hypot(du_b / brx, dv_b / bry)
        w_b = bw * np.exp(-1.15 * (r_b ** 2)) * sstep((1.45 - r_b) / 0.45) * front_m
        col_s = lerp3(col_s, [247.0, 185.0, 179.0], w_b)

        du_c = (us - cu) * ca_r - (vs - cv) * sa_r
        dv_c = (us - cu) * sa_r + (vs - cv) * ca_r
        r_c = np.hypot(du_c / crx, dv_c / cry)
        w_c = cw * np.exp(-1.25 * (r_c ** 2)) * front_m
        col_s = lerp3(col_s, [240.0, 170.0, 166.0], w_c)

        # 细微二次元腮红斜向笔触（仅在红晕核内轻微浮现）
        slash_t = (du_c * 0.84 + dv_c * 0.54)
        stripes = 0.5 + 0.5 * np.cos(slash_t * (2.0 * math.pi / 7.5))
        w_sl = 0.09 * stripes * np.exp(-(r_c / 0.75) ** 2) * front_m
        col_s = lerp3(col_s, [228.0, 148.0, 152.0], w_sl)

    # 原画标志性细节：左眼外下方腮红上的小水滴高光点（u≈263.5, v≈589.2）
    r_drop_rim = np.hypot((us - 264.8) / 3.0, (vs - 588.8) / 3.5)
    col_s = lerp3(col_s, [210.0, 168.0, 174.0], 0.42 * sstep((1.15 - r_drop_rim) / 0.35) * front_m)
    r_drop_hl = np.hypot((us - 263.3) / 2.05, (vs - 589.3) / 2.55)
    col_s = lerp3(col_s, [255.0, 252.0, 254.0], 0.84 * sstep((1.12 - r_drop_hl) / 0.32) * front_m)

    # 2c) 眼周精致妆容：上睑双眼皮褶暖杏粉眼影 + 外眼角红棕三角晕染 + 内眼角淡粉 + 下睑蜜桃卧蚕
    eye_makeup = (
        # (u_mid, u_half, v_top_arch, v_low_arch, out_u, out_v, in_u, in_v)
        (295.0, 43.0, 516.5, 566.5, 250.0, 548.0, 338.0, 553.0),
        (488.0, 42.0, 506.0, 555.5, 529.0, 537.0, 448.0, 543.0),
    )
    for um, uh, vt, vl, ou, ov, iu, iv in eye_makeup:
        xn = (us - um) / uh
        in_span = sstep((1.18 - np.abs(xn)) / 0.28) * outer_skin_m
        # 上眼睑暖杏粉眼影带（连接上睫毛根与双眼皮线）
        v_up_curve = vt + 6.2 * (xn ** 2)
        w_up_es = 0.35 * np.exp(-((vs - v_up_curve) / 7.8) ** 2) * in_span
        col_s = lerp3(col_s, [238.0, 188.0, 186.0], w_up_es)

        # 外眼角红棕微醺三角晕染
        r_out = np.hypot((us - ou) / 15.5, (vs - ov) / 10.5)
        col_s = lerp3(col_s, [208.0, 146.0, 148.0], 0.38 * np.exp(-1.35 * (r_out ** 2)) * outer_skin_m)

        # 内眼角淡粉血色
        r_in = np.hypot((us - iu) / 8.5, (vs - iv) / 6.5)
        col_s = lerp3(col_s, [228.0, 176.0, 180.0], 0.28 * np.exp(-1.4 * (r_in ** 2)) * outer_skin_m)

        # 下睑根部蜜桃粉眼影 + 卧蚕柔亮带 + 极浅卧蚕暖影弧
        v_lo_curve = vl - 2.4 * (xn ** 2)
        span_lo = sstep((1.02 - np.abs(xn)) / 0.25) * outer_skin_m
        w_lo_es = 0.36 * np.exp(-((vs - v_lo_curve) / 4.2) ** 2) * span_lo
        col_s = lerp3(col_s, [234.0, 178.0, 180.0], w_lo_es)
        w_aegyo_hl = 0.30 * np.exp(-((vs - (v_lo_curve + 2.2)) / 1.8) ** 2) * sstep((0.82 - np.abs(xn)) / 0.25) * outer_skin_m
        col_s = lerp3(col_s, [255.0, 246.0, 244.0], w_aegyo_hl)
        w_aegyo_sh = 0.14 * np.exp(-((vs - (v_lo_curve + 5.5)) / 1.6) ** 2) * sstep((0.72 - np.abs(xn)) / 0.25) * outer_skin_m
        col_s = lerp3(col_s, [222.0, 182.0, 184.0], w_aegyo_sh)

    # 2d) 鼻部妆容：鼻头蜜桃暖红晕 + 鼻梁左侧柔白竖高光 + 居中轻柔鼻尖小暖点（修正旧版 u=404.5 偏右误差）
    nose_m = sstep((pts[:, 1] - 0.112) / 0.008)
    r_nblush = np.hypot((us - 395.8) / 13.0, (vs - 593.0) / 10.0)
    col_s = lerp3(col_s, [246.0, 194.0, 192.0], 0.36 * np.exp(-1.2 * (r_nblush ** 2)) * nose_m)
    r_nhl = np.hypot((us - 392.2) / 2.4, (vs - 583.5) / 7.8)
    col_s = lerp3(col_s, [255.0, 251.0, 250.0], 0.56 * np.exp(-1.3 * (r_nhl ** 2)) * nose_m)
    r_ndot = np.hypot((us - 396.0) / 2.0, (vs - 595.0) / 3.0)
    col_s = lerp3(col_s, [206.0, 160.0, 162.0], 0.38 * np.exp(-1.4 * (r_ndot ** 2)) * sstep((pts[:, 1] - 0.120) / 0.005))

    # 2e) 水润蜜桃唇妆：精确对齐 3D 真实唇缝（u=396.0, v=653.6），张口 vrc.v_aa 时唇彩随上下唇自然分开无残线
    lip_m = sstep((pts[:, 1] - 0.1095) / 0.0030)
    seam_m = lip_m * sstep((0.1162 - pts[:, 1]) / 0.0028)
    r_ltint = np.hypot((us - 396.0) / 13.2, (vs - 653.6) / 4.2)
    col_s = lerp3(col_s, [238.0, 170.0, 174.0], 0.48 * np.exp(-1.2 * (r_ltint ** 2)) * lip_m)
    v_lip_arc = 653.5 + 1.1 * (((us - 396.0) / 12.0) ** 2)
    w_lline = 0.52 * np.exp(-((vs - v_lip_arc) / 1.05) ** 2) * sstep((13.0 - np.abs(us - 396.0)) / 3.0) * seam_m
    col_s = lerp3(col_s, [182.0, 118.0, 124.0], w_lline)
    r_lhl = np.hypot((us - 395.4) / 3.6, (vs - 655.6) / 1.2)
    col_s = lerp3(col_s, [255.0, 249.0, 250.0], 0.62 * np.exp(-1.4 * (r_lhl ** 2)) * lip_m)

    # 2f) 下巴尖柔和蜜桃血色晕染 + 颌下脖子暖阴影
    r_chin = np.hypot((us - 395.0) / 24.0, (vs - 702.0) / 11.0)
    col_s = lerp3(col_s, [244.0, 196.0, 198.0], 0.30 * np.exp(-1.2 * (r_chin ** 2)) * sstep((pts[:, 1] - 0.080) / 0.012))
    w_neck = sstep(((CHIN_Z + 0.002) - pts[:, 2]) / 0.008) * sstep((0.062 - pts[:, 1]) / 0.020)
    col_s = lerp3(col_s, [214.0, 174.0, 179.0], 0.72 * w_neck)

    # 2g) 关键修复：仅正面眼窝凹陷底壁（58 mm < y < 86.5 mm 且位于眼框窗口内）绘入同源亮白巩膜色，严禁穿透到后脑勺（y < 0）或覆盖外表面眼影！
    in_l_win = (us >= 251.0) & (us <= 340.0) & (vs >= 526.0) & (vs <= 566.0)
    in_r_win = (us >= 447.0) & (us <= 531.0) & (vs >= 516.0) & (vs <= 556.0)
    in_win = (in_l_win | in_r_win) & (pts[:, 1] > 0.055)
    if np.any(in_win):
        y_w = pts[in_win, 1]
        w_sock = sstep((0.0858 - y_w) / 0.0028) * sstep((y_w - 0.0580) / 0.0050)
        vl_ref = np.where(us[in_win] < MID_U, EYES['L']['vlash'], EYES['R']['vlash'])
        t_sh = sstep((vl_ref + 14.0 - vs[in_win]) / 16.0)
        eye_sock_col = lerp3([249.0, 243.0, 245.0], [212.0, 200.0, 216.0], 0.80 * t_sh)
        col_s[in_win] = lerp3(col_s[in_win], eye_sock_col, w_sock)

    face_rgb[v_sk] = col_s
    face_rgb = dilate_rgb(face_rgb, v_sk, steps=16)
    Image.fromarray(np.clip(np.round(face_rgb), 0, 255).astype(np.uint8), 'RGB').save(HERE / 'RadDoll_face_tex.png')
    print(f'TEX saved: RadDoll_eye_tex.png ({W_EYE}x{W_EYE}, valid={int(v_eye.sum())}), RadDoll_face_tex.png ({W_FACE}x{W_FACE}, valid={int(v_sk.sum())})')


if __name__ == '__main__':
    main()
