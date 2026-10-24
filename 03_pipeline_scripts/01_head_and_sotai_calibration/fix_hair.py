"""V10R7 · 修正发片（对照原图与校准后的 RadDoll 头）：
 H1 清理埋头：从头表面定义“推出场” F(q)（越深推得越多、沿头部平滑法线向外、随离头距离衰减，保持发帽多层的先后顺序）；
 H2 重建眼上方分缕刘海发束：按原画真实刘海走向与倒 V / 拱形露眉分缝，生成带 3D 中脊微拱截面的柔顺层次发束（取代早期遮死眉毛的水平齐切平板）；
 H3 呆毛各向异性校准；
 H4 贴合前刘海表面的二次元“刘海透眉”薄片（仅在发丝遮挡段贴合发面 +0.45mm，配合视线角衰减材质）。
输入：V10 惯性分支第 1 帧发片（含发帽）+ head_fit.json 的头模参数。输出 hair_fix.json 与 _tmp/head_compare/snap/V10hair_R7.npz。只读源文件。
Run: Blender -b --factory-startup --disable-autoexec --python-exit-code 1 --python this_file
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_lib as L  # noqa: E402
import head_fit as HFIT  # noqa: E402
import head_verify as HV  # noqa: E402

HERE = L.HERE
C_TARGET = 0.002                 # 目标间隙 2 mm
T_DECAY = 0.052                  # 推出场随离头距离衰减到 0 的距离
Z_HAIRLINE_TOP = L.EZ - (425 - L.MID_V) * L.K   # 额顶分缝上限（原图 y≈425，此线以上不应露头皮空洞；y=430..515 为自然露眉分缝）
OUT_NPZ = L.ROOT / '_tmp/head_compare/snap/V10hair_R7.npz'


def vertex_normals(V, counts, idx):
    t = L.tris_of(counts, idx)
    fn = np.cross(V[t[:, 1]] - V[t[:, 0]], V[t[:, 2]] - V[t[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, t[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12), t


def adjacency_lists(n, t):
    nb = [set() for _ in range(n)]
    for a, b, c in t:
        nb[a].update((b, c)); nb[b].update((a, c)); nb[c].update((a, b))
    return [np.array(sorted(s)) for s in nb]


def bary(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-18:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.clip(np.array([1 - v - w, v, w]), 0, 1)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def catmull_rom_chain(knots, n_samples):
    """输入 shape=(K, D) 控制点，返回平滑 C¹ 样条采样 shape=(n_samples, D)。"""
    P = np.asarray(knots, dtype=np.float64)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    segs = len(P) - 1
    ts = np.linspace(0.0, float(segs), n_samples)
    out = np.empty((n_samples, P.shape[1]), dtype=np.float64)
    for idx_s, g in enumerate(ts):
        i = min(int(math.floor(g)), segs - 1)
        t = g - i
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t2, t3 = t * t, t * t * t
        out[idx_s] = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
    return out


def skin_y_at(tree, x, z, y_fallback=0.095):
    d = Vector((0.0, -1.0, 0.0))
    h = tree.ray_cast(Vector((float(x), 0.5, float(z))), d, 2.0)
    if h[0] is not None:
        return float(h[0].y)
    loc, _nrm, _fi, _dist = tree.find_nearest(Vector((float(x), y_fallback, float(z))))
    return float(loc.y)


def ensure_clearance(tree, p3, min_clear=0.0028):
    loc, nrm, _fi, dist = tree.find_nearest(Vector(p3))
    sgn = 1.0 if (Vector(p3) - loc).dot(nrm) >= 0 else -1.0
    if sgn * dist < min_clear:
        p3 = np.array(loc, dtype=np.float64) + np.array(nrm, dtype=np.float64) * (min_clear + 0.0002)
    return p3


def build_quad_piece(name, V_grid):
    """V_grid: shape=(NS, NW, 3)，生成法线朝 +Y 的光滑四边形网格。"""
    NS, NW, _ = V_grid.shape
    V = V_grid.reshape(-1, 3)
    quads = []
    for ks in range(NS - 1):
        for kw in range(NW - 1):
            a = ks * NW + kw
            b = ks * NW + (kw + 1)
            c = (ks + 1) * NW + (kw + 1)
            e = (ks + 1) * NW + kw
            quads.append((a, b, c, e))
    quads = np.array(quads, dtype=np.int32)
    q0 = quads[len(quads) // 2]
    nrm = np.cross(V[q0[1]] - V[q0[0]], V[q0[3]] - V[q0[0]])
    if nrm[1] < 0:
        quads = quads[:, ::-1]
    counts = np.full(len(quads), 4, dtype=np.int32)
    idx = quads.reshape(-1).astype(np.int32)
    return {'name': name, 'v': V, 'counts': counts, 'idx': idx}


def build_tapered_lock(name, knots_uvhw, base_off_mm, tip_extra_mm, ridge_mm, tree, ns=28, nw=9, ref_bvh=None):
    """按原图 (u, v, half_width_px) 样条生成带 3D 棱脊截面的锥形二次元刘海发束。"""
    sp = catmull_rom_chain(knots_uvhw, ns)
    uc, vc, hw = sp[:, 0], sp[:, 1], np.maximum(sp[:, 2], 0.5)
    du = np.gradient(uc)
    dv = np.gradient(vc)
    ln = np.maximum(np.hypot(du, dv), 1e-6)
    # 图像平面法向（指向发束右侧 +u 侧）
    nu, nv = dv / ln, -du / ln
    if nu.mean() < 0:
        nu, nv = -nu, -nv

    s_arr = np.linspace(0.0, 1.0, ns)
    tw_arr = np.linspace(-1.0, 1.0, nw)
    V_grid = np.empty((ns, nw, 3), dtype=np.float64)
    d_ray = Vector((0.0, -1.0, 0.0))

    # 先沿中心脊采样平滑基础深度 y_spine(s)，避免横向跨越眼窝边缘时产生突变折痕
    y_spine = np.empty(ns, dtype=np.float64)
    for ks in range(ns):
        xc, zc = L.art_to_world(uc[ks], vc[ks])
        ys = skin_y_at(tree, xc, zc)
        if ref_bvh is not None:
            hr = ref_bvh.ray_cast(Vector((float(xc), 0.5, float(zc))), d_ray, 2.0)
            if hr[0] is not None and hr[0].y > ys and hr[0].y < ys + 0.024:
                ys = max(ys, float(hr[0].y) - 0.0035)
        y_spine[ks] = ys
    for _ in range(5):
        y_spine[1:-1] = 0.25 * y_spine[:-2] + 0.50 * y_spine[1:-1] + 0.25 * y_spine[2:]

    for ks, s in enumerate(s_arr):
        off_s = (base_off_mm + tip_extra_mm * smoothstep(s)) * 1e-3
        ridge_s = (ridge_mm * 1e-3) * (1.0 - 0.55 * (s ** 1.8))
        for kw, tw in enumerate(tw_arr):
            u_pt = uc[ks] + tw * hw[ks] * nu[ks]
            v_pt = vc[ks] + tw * hw[ks] * nv[ks]
            x_pt, z_pt = L.art_to_world(u_pt, v_pt)
            yk = max(y_spine[ks] - 0.0020 * (tw ** 2), skin_y_at(tree, x_pt, z_pt))
            # 3D 截面拱起：中心脊鼓起 +ridge_s，两侧收边产生 ±X 法线偏转，激活二次元发束边缘分缝线
            arch = ridge_s * (1.0 - tw * tw)
            p3 = np.array([x_pt, yk + off_s + arch, z_pt], dtype=np.float64)
            V_grid[ks, kw] = ensure_clearance(tree, p3, min_clear=0.0028)

    return build_quad_piece(name, V_grid)


def build_right_upper_arch(name, tree, nu_cols=30, nv_rows=14):
    """右额发际斜分弧形发束：填补 u=432..548 额顶发帽分界，下沿精确停在右眉上方（v=414..456），绝不压眉。"""
    u_arr = np.linspace(432.0, 548.0, nu_cols)
    # 下沿平滑控制点（严格高于右眉 v=457..472，还原原画右额柔弧露眉窗）
    bot_knots = np.array([
        [432.0, 412.0],
        [454.0, 420.0],
        [478.0, 430.0],
        [502.0, 441.0],
        [525.0, 454.0],
        [548.0, 475.0],
    ], dtype=np.float64)
    sp_bot = catmull_rom_chain(bot_knots, nu_cols)
    v_bot = np.interp(u_arr, sp_bot[:, 0], sp_bot[:, 1])
    v_top = np.full_like(u_arr, 328.0)

    sv_arr = np.linspace(0.0, 1.0, nv_rows)
    V_grid = np.empty((nu_cols, nv_rows, 3), dtype=np.float64)
    for ku, u_val in enumerate(u_arr):
        tu = (u_val - u_arr[0]) / (u_arr[-1] - u_arr[0])
        # 两道柔和斜向发缕棱脊，使大片弧面也具备细腻发丝立体起伏
        groove = 0.0014 * (0.5 + 0.5 * math.cos(2.0 * math.pi * (2.2 * tu))) * math.sin(math.pi * tu)
        edge_tuck = -0.0018 * ((2.0 * tu - 1.0) ** 4)
        for kv, sv in enumerate(sv_arr):
            v_val = v_top[ku] * (1.0 - sv) + v_bot[ku] * sv
            x_pt, z_pt = L.art_to_world(u_val, v_val)
            yh = skin_y_at(tree, x_pt, z_pt)
            off = 0.0052 + 0.0024 * math.sin(math.pi * sv) + groove * (0.4 + 0.6 * sv) + edge_tuck
            p3 = np.array([x_pt, yh + off, z_pt], dtype=np.float64)
            V_grid[ku, kv] = ensure_clearance(tree, p3, min_clear=0.0028)

    return build_quad_piece(name, V_grid)


def build_conformal_brow_seethrough(fac_world, cls_v, hair_pieces, skin_tree):
    """提取校准后 3D 眉毛的真实上下缘，仅在前刘海发束遮挡区沿发片外表面 +0.45mm 生成贴合透眉片。"""
    verts_h, tris_h, base = [], [], 0
    for hp in hair_pieces:
        verts_h.append(hp['v'])
        tris_h.append(L.tris_of(hp['counts'], hp['idx']) + base)
        base += len(hp['v'])
    hbvh = BVHTree.FromPolygons(np.concatenate(verts_h).tolist(), np.concatenate(tris_h).tolist(), all_triangles=True)

    brow_pts = fac_world[cls_v == 'brow']
    d_ray = Vector((0.0, -1.0, 0.0))
    V_all, quads_all, alpha_all = [], [], []
    NV = 5
    sv_grid = np.linspace(0.0, 1.0, NV)

    for side_cond in (brow_pts[:, 0] > L.EX, brow_pts[:, 0] <= L.EX):
        bp = brow_pts[side_cond]
        if len(bp) < 6:
            continue
        x_min, x_max = float(bp[:, 0].min()), float(bp[:, 0].max())
        NC = 44
        xs = np.linspace(x_min, x_max, NC)
        # 分桶求该侧眉毛每列的上缘、下缘与深度 y
        half_bin = max((x_max - x_min) / 14.0, 0.0025)
        z_lo = np.empty(NC)
        z_hi = np.empty(NC)
        y_br = np.empty(NC)
        for kc, xv in enumerate(xs):
            near = np.abs(bp[:, 0] - xv) <= half_bin
            if not np.any(near):
                idx_n = np.argsort(np.abs(bp[:, 0] - xv))[:4]
                sub = bp[idx_n]
            else:
                sub = bp[near]
            z_lo[kc] = float(np.percentile(sub[:, 2], 10))
            z_hi[kc] = float(np.percentile(sub[:, 2], 90))
            y_br[kc] = float(np.median(sub[:, 1]))
        for arr in (z_lo, z_hi, y_br):
            for _ in range(4):
                arr[1:-1] = 0.25 * arr[:-2] + 0.50 * arr[1:-1] + 0.25 * arr[2:]
        # 确保最小眉条高度 1.5mm（约 3 px），两端自然收尖
        t_arch = np.linspace(0.0, 1.0, NC)
        end_env = np.sin(math.pi * t_arch) ** 0.65
        z_mid = 0.5 * (z_lo + z_hi)
        half_h = np.maximum(0.5 * (z_hi - z_lo), 0.00095 * end_env)
        z_lo = z_mid - half_h
        z_hi = z_mid + half_h

        # 逐列检测前刘海遮挡表面 y_hit
        occ = np.zeros(NC, dtype=bool)
        y_hit = np.zeros(NC, dtype=np.float64)
        for kc, xv in enumerate(xs):
            h = hbvh.ray_cast(Vector((float(xv), 0.5, float(z_mid[kc]))), d_ray, 2.0)
            if h[0] is not None and float(h[0].y) > y_br[kc] + 0.0006:
                occ[kc] = True
                y_hit[kc] = float(h[0].y)

        # 按连续遮挡段各自生成贴合薄带（绝不跨越发片与皮肤之间的空气间隙）
        kc = 0
        while kc < NC:
            if not occ[kc]:
                kc += 1
                continue
            k_start = kc
            while kc < NC and occ[kc]:
                kc += 1
            k_end = kc  # [k_start, k_end)
            seg_len = k_end - k_start
            if seg_len < 2:
                continue
            # 平滑该段 y_hit（避免发片交叠处 1mm 台阶），但保证不低于原 y_hit
            yh_seg = y_hit[k_start:k_end].copy()
            if seg_len >= 4:
                yh_sm = yh_seg.copy()
                for _ in range(3):
                    yh_sm[1:-1] = 0.25 * yh_sm[:-2] + 0.50 * yh_sm[1:-1] + 0.25 * yh_sm[2:]
                yh_seg = np.maximum(yh_seg, yh_sm)

            v_base = len(V_all)
            for j, col_i in enumerate(range(k_start, k_end)):
                # 段首尾 2 列柔和羽化，与裸肤真眉无缝衔接
                seg_t = (j + 0.5) / seg_len
                tip_w = float(end_env[col_i])
                for kv, sv in enumerate(sv_grid):
                    zv = z_hi[col_i] * (1.0 - sv) + z_lo[col_i] * sv
                    # 在该 (x, zv) 再做一次精确前发面探测，缺失则回退 yh_seg[j]
                    hh = hbvh.ray_cast(Vector((float(xs[col_i]), 0.5, float(zv))), d_ray, 2.0)
                    yv = float(hh[0].y) if (hh[0] is not None and float(hh[0].y) > y_br[col_i]) else yh_seg[j]
                    yv = min(max(yv, yh_seg[j] - 0.0012), y_br[col_i] + 0.0115) + 0.00035
                    p3 = ensure_clearance(skin_tree, np.array([xs[col_i], yv, zv], dtype=np.float64), min_clear=0.0025)
                    V_all.append(p3)
                    v_prof = math.sin(math.pi * sv) ** 0.75
                    alpha_all.append(float(np.clip(tip_w * v_prof, 0.0, 1.0)))

            for j in range(seg_len - 1):
                for kv in range(NV - 1):
                    a = v_base + j * NV + kv
                    b = v_base + (j + 1) * NV + kv
                    c = v_base + (j + 1) * NV + (kv + 1)
                    e = v_base + j * NV + (kv + 1)
                    # 法线朝 +Y
                    nrm = np.cross(V_all[b] - V_all[a], V_all[e] - V_all[a])
                    if nrm[1] < 0:
                        quads_all.append((a, e, c, b))
                    else:
                        quads_all.append((a, b, c, e))

    if not quads_all:
        return None, None
    V_arr = np.array(V_all, dtype=np.float64)
    q_arr = np.array(quads_all, dtype=np.int32)
    counts = np.full(len(q_arr), 4, dtype=np.int32)
    idx = q_arr.reshape(-1).astype(np.int32)
    return {'name': 'GEO-Hair-Brow_SeeThrough', 'v': V_arr, 'counts': counts, 'idx': idx}, np.array(alpha_all, dtype=np.float32)


def main():
    spec_fit = json.loads((HERE / 'head_fit.json').read_text(encoding='utf-8'))['fitted']
    rd = L.RadDoll()
    place = L.make_xf(dy=spec_fit['dy_mm'] / 1000, pitch_deg=spec_fit['pitch_deg'], roll_deg=spec_fit['roll_deg'], scale=spec_fit['scale'])
    a_param = [x / 1000.0 for x in spec_fit['eye_bulge_AL_AR_mm']] if 'eye_bulge_AL_AR_mm' in spec_fit else (spec_fit['eye_bulge_A_mm'] / 1000.0)
    p_centers = HFIT.placed_centers(place)
    xf = L.compose(place, L.make_deform(A=a_param, lid=spec_fit['eye_lower_lid'], s_low=spec_fit['s_low'], s_up=spec_fit.get('s_up', 1.0), centers=p_centers, jaw=spec_fit.get('jaw_strength', 0.0)))
    S = rd.world('skin', {}, xf)
    F_fac = L.refine_fac_parts(rd.world('fac', {}, xf), rd.cls_of_vertex, p_centers)
    hc, hi = rd.polys('skin')
    tree = L.bvh(S, hc, hi)
    tri = L.tris_of(hc, hi)
    vn, _ = vertex_normals(S, hc, hi)
    nbh = adjacency_lists(len(S), tri)
    hair = L.load_hair_calibrated('V10hair')
    pts = np.concatenate([p['v'] for p in hair])
    lo, hi_ = S.min(0) - 0.01, S.max(0) + 0.01
    rec = {'hair_upper_compress': L.S_HAIR_UP, 'params': {k: spec_fit[k] for k in ('scale', 'dy_mm', 'pitch_deg', 'roll_deg', 'eye_bulge_A_mm', 'eye_lower_lid', 's_low', 's_up')}, 'c_target_mm': C_TARGET * 1000, 'decay_mm': T_DECAY * 1000}

    def measure(P):
        par = HV.parity_inside(tree, P, lo, hi_)
        d = np.empty(len(P))
        for i, q in enumerate(P):
            d[i] = tree.find_nearest(Vector(q))[3]
        sd = np.where(par, -d, d)
        return par, sd

    par0, sd0 = measure(pts)
    rec['before'] = {'inside_share': round(float(par0.mean()), 4), 'inside': int(par0.sum()), 'max_depth_mm': round(float(-sd0.min()) * 1000, 1),
                     'closer_than_2mm_outside_share': round(float(((~par0) & (sd0 < C_TARGET)).mean()), 4)}
    print('FIX before', json.dumps(rec['before']), flush=True)

    # ---- H1 推出场 ----
    need = np.maximum(0.0, C_TARGET - sd0)
    F = np.zeros(len(S))
    for i in np.nonzero(need > 0)[0]:
        loc, nrm, fi, dist = tree.find_nearest(Vector(pts[i]))
        for v in tri[fi]:
            F[v] = max(F[v], need[i])
    for _ in range(26):
        Fm = np.array([F[nb].mean() if len(nb) else F[i] for i, nb in enumerate(nbh)])
        F = np.maximum(F, 0.5 * F + 0.5 * Fm)
    D = np.zeros_like(pts)
    for i, p in enumerate(pts):
        loc, nrm, fi, dist = tree.find_nearest(Vector(p))
        a, b, c = (S[k] for k in tri[fi])
        w3 = bary(np.array(loc), a, b, c)
        Fq = float(w3 @ F[tri[fi]])
        if Fq <= 1e-6:
            continue
        n = w3 @ vn[tri[fi]]
        n /= max(np.linalg.norm(n), 1e-12)
        sd = sd0[i]
        w = 1.0 if sd <= 0 else 1.0 - float(smoothstep((sd - 0.0) / T_DECAY))
        D[i] = n * Fq * w
    P1 = pts + D
    par1, sd1 = measure(P1)
    need2 = np.maximum(0.0, C_TARGET - sd1)
    if (need2 > 1e-4).any():
        for i in np.nonzero(need2 > 1e-4)[0]:
            loc, nrm, fi, dist = tree.find_nearest(Vector(P1[i]))
            a, b, c = (S[k] for k in tri[fi])
            w3 = bary(np.array(loc), a, b, c)
            n = w3 @ vn[tri[fi]]
            n /= max(np.linalg.norm(n), 1e-12)
            P1[i] = P1[i] + n * need2[i]
        par1, sd1 = measure(P1)
    rec['after'] = {'inside_share': round(float(par1.mean()), 4), 'inside': int(par1.sum()), 'closer_than_2mm_outside_share': round(float(((~par1) & (sd1 < C_TARGET - 1e-4)).mean()), 4),
                    'moved_vertices': int((np.linalg.norm(D, axis=1) > 1e-4).sum()), 'max_move_mm': round(float(np.linalg.norm(P1 - pts, axis=1).max()) * 1000, 1),
                    'mean_move_of_moved_mm': round(float(np.linalg.norm(P1 - pts, axis=1)[np.linalg.norm(P1 - pts, axis=1) > 1e-4].mean()) * 1000, 2)}
    print('FIX after', json.dumps(rec['after']), flush=True)

    k0, per = 0, {}
    new_hair = []
    for p in hair:
        n = len(p['v'])
        Q = P1[k0:k0 + n].copy()
        nm = p['name'].replace('GEO-Hair-', '')
        # 微调：将 Fringe_R_Main 在右外眼角处向内戳入眼白的小尖角（u∈[524,556], v∈[518,554]）柔和外拨 4.8mm，还原原画干净右外眼角
        if nm == 'Fringe_R_Main':
            ua, va = L.world_to_art(Q[:, 0], Q[:, 2])
            wu = smoothstep((ua - 518.0) / 14.0) * (1.0 - smoothstep((ua - 552.0) / 14.0))
            wv = smoothstep((va - 512.0) / 12.0) * (1.0 - smoothstep((va - 548.0) / 12.0))
            wy = smoothstep((Q[:, 1] - 0.065) / 0.020)
            w_tuck = wu * wv * wy
            if np.any(w_tuck > 1e-3):
                Q[:, 0] -= 0.0052 * w_tuck
                for idx_q in np.nonzero(w_tuck > 1e-3)[0]:
                    Q[idx_q] = ensure_clearance(tree, Q[idx_q], min_clear=0.0024)
        # 微调：将 Fringe_R_Thin 底部过长且过细的孤立下刺（v > 522）柔和收拢至内侧刘海弧线内，消除原分屏截图中的细碎毛刺感
        if nm == 'Fringe_R_Thin':
            ua, va = L.world_to_art(Q[:, 0], Q[:, 2])
            wv = smoothstep((va - 500.0) / 28.0)
            if np.any(wv > 1e-3):
                Q[:, 2] += 0.011 * wv
                for idx_q in np.nonzero(wv > 1e-3)[0]:
                    Q[idx_q] = ensure_clearance(tree, Q[idx_q], min_clear=0.0024)

        # 对顶层叠交发束施加微米级外层序偏置 (+0.8~1.2mm)，消除与底层发帽/邻束 0.0mm 共面引起的旋转 Z-fighting
        TOP_LOCKS = {'Fringe_C_Overlap': 0.0012, 'Crown_R_Main': 0.0010, 'Back_C_Surface': 0.0010, 'Temple_R_Outer': 0.0009, 'Side_R_Flick': 0.0009}
        if nm in TOP_LOCKS:
            rad_xy = Q[:, :2] - np.array([L.EX, 0.010])
            rn = np.maximum(np.linalg.norm(rad_xy, axis=1, keepdims=True), 1e-6)
            Q[:, :2] += (rad_xy / rn) * TOP_LOCKS[nm]

        mv = np.linalg.norm(Q - p['v'], axis=1)
        t = L.tris_of(p['counts'], p['idx'])
        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        l0 = np.linalg.norm(p['v'][e[:, 0]] - p['v'][e[:, 1]], axis=1)
        l1 = np.linalg.norm(Q[e[:, 0]] - Q[e[:, 1]], axis=1)
        r = l1 / np.maximum(l0, 1e-9)
        per[nm] = {'moved': int((mv > 1e-4).sum()), 'max_move_mm': round(float(mv.max()) * 1000, 1), 'inside_after': int(measure(Q)[0].sum()),
                   'edge_ratio_min': round(float(r[l0 > 1e-6].min()), 2) if (l0 > 1e-6).any() else 1.0, 'edge_ratio_max': round(float(r[l0 > 1e-6].max()), 2) if (l0 > 1e-6).any() else 1.0}
        new_hair.append({'name': p['name'], 'v': Q, 'counts': p['counts'], 'idx': p['idx']})
        k0 += n
    rec['per_object'] = per

    # ---- H3 呆毛对照原图 ----
    ah_i = next(i for i, h in enumerate(new_hair) if h['name'].endswith('Ahoge'))
    ah = new_hair[ah_i]
    av = ah['v']
    dd = np.array([tree.find_nearest(Vector(q))[3] for q in av])
    root = av[np.argsort(dd)[:20]].mean(0)
    apex3 = av[np.argmax(av[:, 2])]
    Uc, _Vc = L.world_to_art(av[:, 0], av[:, 2])
    cand = np.nonzero(Uc < 235)[0]
    tip_idx = int(cand[np.argmin(av[cand, 2])])
    tip3 = av[tip_idx]
    A_apex = np.array([L.px_to_world_x(L.ART['ahoge']['apex_px'][0]), L.px_to_world_z(L.ART['ahoge']['apex_px'][1])])
    A_tip = np.array([L.px_to_world_x(L.ART['ahoge']['tip_px'][0]), L.px_to_world_z(L.ART['ahoge']['tip_px'][1])])
    sx = (A_apex[0] - root[0]) / (apex3[0] - root[0])
    sz = (A_apex[1] - root[2]) / (apex3[2] - root[2])
    sy = math.sqrt(abs(sx * sz))

    def step1(P):
        return np.stack([root[0] + sx * (P[:, 0] - root[0]), root[1] + sy * (P[:, 1] - root[1]), root[2] + sz * (P[:, 2] - root[2])], 1)
    P1a = step1(av)
    tip1 = P1a[tip_idx]
    dX, dZ = A_tip[0] - tip1[0], A_tip[1] - tip1[2]
    t_ = np.clip((P1a[:, 0] - A_apex[0]) / max(tip1[0] - A_apex[0], 1e-6), 0, 1)
    w_ = smoothstep(t_)
    P2a = P1a.copy()
    P2a[:, 0] += dX * w_
    P2a[:, 2] += dZ * w_
    before_apex_px = L.world_to_art(apex3[0], apex3[2])
    before_tip_px = L.world_to_art(tip3[0], tip3[2])
    ah['v'] = P2a
    tip_after = P2a[tip_idx]
    apex_after = P2a[np.argmax(av[:, 2])]
    rec['ahoge'] = {'scale_x': round(float(sx), 3), 'scale_z': round(float(sz), 3), 'scale_y': round(float(sy), 3), 'bend_dx_mm': round(float(dX) * 1000, 1), 'bend_dz_mm': round(float(dZ) * 1000, 1),
                    'root_xyz': [round(float(x), 4) for x in root],
                    'apex_px': {'art': L.ART['ahoge']['apex_px'], 'model_before': [round(x, 1) for x in before_apex_px], 'model_after': [round(x, 1) for x in L.world_to_art(apex_after[0], apex_after[2])]},
                    'tip_px': {'art': L.ART['ahoge']['tip_px'], 'model_before': [round(x, 1) for x in before_tip_px], 'model_after': [round(x, 1) for x in L.world_to_art(tip_after[0], tip_after[2])]},
                    'note': '各向异性缩放+下弯对齐原图最高点与卷尖'}
    print('FIX ahoge', json.dumps(rec['ahoge']), flush=True)

    # ---- 构建现有前发 BVH（供新增层次刘海自然衔接深度与测量额顶空洞）----
    verts0, tris0, base0 = [], [], 0
    for h in new_hair:
        verts0.append(h['v'])
        tris0.append(L.tris_of(h['counts'], h['idx']) + base0)
        base0 += len(h['v'])
    htree = BVHTree.FromPolygons(np.concatenate(verts0).tolist(), np.concatenate(tris0).tolist(), all_triangles=True)

    g = 0.001
    xs = np.arange(-0.09, 0.09 + 1e-9, g)
    zs = np.arange(0.07, -0.13 - 1e-9, -g)
    void = np.zeros((len(zs), len(xs)), bool)
    d = Vector((0, -1, 0))
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            h = htree.ray_cast(Vector((x, 0.5, z)), d, 2.0)
            if h[0] is not None and h[1].dot(d) > 0:
                void[iz, ix] = True
    above = void & (zs[:, None] > Z_HAIRLINE_TOP)
    rec['void_after_push_cm2'] = {'total_in_window': round(float(void.sum()) * g * g * 1e4, 1), 'above_brow': round(float(above.sum()) * g * g * 1e4, 1)}
    print('FIX void', json.dumps(rec['void_after_push_cm2']), flush=True)

    # ---- H2 按原画真实走向构建眼上方分缕立体刘海发束（保留自然倒 V 与右额拱形露眉窗）----
    new_pieces = [
        # 1) 左眼外上侧主斜垂发束（↙ 垂至左外眼角上方 u=234, v=524）
        build_tapered_lock(
            'GEO-Hair-Fringe_L_OuterLock_NEW',
            [(256.0, 334.0, 38.0), (248.0, 388.0, 34.0), (240.0, 440.0, 25.0), (234.0, 484.0, 13.5), (230.0, 524.0, 0.9)],
            base_off_mm=4.6, tip_extra_mm=1.5, ridge_mm=2.0, tree=tree, ns=30, nw=9
        ),
        # 2) 左眼上中部分叉副发缕（↙ 垂至 u=250, v=510，叠加在主束前形成原画双尖层次）
        build_tapered_lock(
            'GEO-Hair-Fringe_L_MidWisp_NEW',
            [(267.0, 346.0, 18.0), (260.0, 402.0, 14.5), (253.0, 452.0, 9.5), (248.0, 485.0, 5.0), (245.0, 509.0, 0.8)],
            base_off_mm=6.0, tip_extra_mm=1.5, ridge_mm=1.8, tree=tree, ns=26, nw=7
        ),
        # 3) 左眼内上侧斜分细束（↘ 垂至左瞳上方 u=316, v=520，与外束围出倒 V 左眉透肤窗）
        build_tapered_lock(
            'GEO-Hair-Fringe_L_InnerLock_NEW',
            [(286.0, 340.0, 25.0), (293.0, 396.0, 19.5), (301.0, 448.0, 12.0), (309.0, 488.0, 6.5), (317.0, 521.0, 0.8)],
            base_off_mm=5.4, tip_extra_mm=1.6, ridge_mm=1.9, tree=tree, ns=28, nw=7
        ),
        # 4) 右额发际斜分弧形发束（↘ 停在右眉上方 v=412..455，敞开原画右眉额窗并消除发帽顶空洞）
        build_right_upper_arch('GEO-Hair-Fringe_R_UpperArch_NEW', tree=tree, nu_cols=30, nv_rows=14),
        # 5) 右眼内上侧柔弧斜垂发束（↙ 紧贴额面自然垂落，不超出鼻尖侧影）
        build_tapered_lock(
            'GEO-Hair-Fringe_R_InnerSweep_NEW',
            [(455.0, 344.0, 22.0), (454.0, 402.0, 19.0), (448.0, 452.0, 13.5), (438.0, 492.0, 7.5), (427.0, 522.0, 0.8)],
            base_off_mm=5.8, tip_extra_mm=1.7, ridge_mm=1.8, tree=tree, ns=30, nw=9
        ),
        # 6) 右眼外上侧发夹下缘贴鬓轻缕（↘ 衔接鲸鱼发夹下缘与右侧鬓发）
        build_tapered_lock(
            'GEO-Hair-Fringe_R_OuterWisp_NEW',
            [(516.0, 418.0, 15.0), (531.0, 456.0, 13.0), (544.0, 490.0, 8.5), (552.0, 521.0, 0.9)],
            base_off_mm=5.0, tip_extra_mm=1.5, ridge_mm=1.6, tree=tree, ns=22, nw=7
        ),
    ]

    # ---- H4 贴合前刘海表面的二次元“刘海透眉”薄片 ----
    brow_piece, brow_alpha = build_conformal_brow_seethrough(F_fac, rd.cls_of_vertex, new_hair + new_pieces, tree)
    if brow_piece is not None:
        new_pieces.append(brow_piece)

    for pc in new_pieces:
        nm_short = pc['name'].replace('GEO-Hair-', '')
        t_pc = L.tris_of(pc['counts'], pc['idx'])
        area = 0.5 * np.linalg.norm(np.cross(pc['v'][t_pc][:, 1] - pc['v'][t_pc][:, 0], pc['v'][t_pc][:, 2] - pc['v'][t_pc][:, 0]), axis=1).sum()
        par, sd = measure(pc['v'])
        rec.setdefault('new_pieces', {})[nm_short] = {
            'verts': int(len(pc['v'])), 'faces': int(len(pc['counts'])), 'area_cm2': round(float(area) * 1e4, 1),
            'z_range_mm': [round(float(pc['v'][:, 2].min()) * 1000, 1), round(float(pc['v'][:, 2].max()) * 1000, 1)],
            'x_range_mm': [round(float(pc['v'][:, 0].min()) * 1000, 1), round(float(pc['v'][:, 0].max()) * 1000, 1)],
            'inside': int(par.sum()),
            'clearance_mm_min_med_max': [round(float(np.min(sd)) * 1000, 1), round(float(np.median(sd)) * 1000, 1), round(float(np.max(sd)) * 1000, 1)],
        }
        print('FIX piece', pc['name'], json.dumps(rec['new_pieces'][nm_short]), flush=True)

    # 修后额顶空洞统计（不含透眉片）
    solid_all = [h for h in (new_hair + new_pieces) if h['name'] != 'GEO-Hair-Brow_SeeThrough']
    verts2, tris2, base2 = [], [], 0
    for h in solid_all:
        verts2.append(h['v'])
        tris2.append(L.tris_of(h['counts'], h['idx']) + base2)
        base2 += len(h['v'])
    tree2 = BVHTree.FromPolygons(np.concatenate(verts2).tolist(), np.concatenate(tris2).tolist(), all_triangles=True)
    void2 = np.zeros_like(void)
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            h = tree2.ray_cast(Vector((x, 0.5, z)), d, 2.0)
            if h[0] is not None and h[1].dot(d) > 0:
                void2[iz, ix] = True
    rec['void_final_cm2'] = {'total_in_window': round(float(void2.sum()) * g * g * 1e4, 1), 'above_brow': round(float((void2 & (zs[:, None] > Z_HAIRLINE_TOP)).sum()) * g * g * 1e4, 1)}
    print('FIX void_final', json.dumps(rec['void_final_cm2']), flush=True)

    allp = new_hair + new_pieces
    names = [h['name'] for h in allp]
    extra = {'brow_thru_alpha': brow_alpha} if brow_alpha is not None else {}
    np.savez_compressed(OUT_NPZ, names=np.array(names), **{f'{i}_v': h['v'] for i, h in enumerate(allp)}, **{f'{i}_counts': h['counts'] for i, h in enumerate(allp)}, **{f'{i}_idx': h['idx'] for i, h in enumerate(allp)}, **extra)
    (HERE / 'hair_fix.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    print('HAIRFIX_DONE', flush=True)


if __name__ == '__main__':
    main()
