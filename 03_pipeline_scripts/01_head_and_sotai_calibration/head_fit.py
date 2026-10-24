"""V10R7 · 头模校准 v2（RadDoll）：
 · 对照原图：眼睛局部变形（内眼角与虹膜不动，只拉长外侧眼白、抬下眼睑）+ 下半脸竖向压缩；目标 = 原图量得的眼开口、嘴、下巴、虹膜连线倾斜；
 · 对照发束：在 (缩放, 深度 dy, 点头 pitch, 侧倾 roll) 上网格搜索，目标函数 = 发片埋头程度 + 五官偏离原图 + 与 V7（发束原配头）的“发片—头皮距离分布”之差。
输出 head_fit.json、renders/head_art_frame_<tag>.png。只读源文件。
Run: Blender -b --factory-startup --disable-autoexec --python-exit-code 1 --python this_file
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_lib as L  # noqa: E402
import head_frames as HF  # noqa: E402

HERE = L.HERE
OUT = HERE / 'renders'
K, ART = L.K, L.ART
MOUTH_Z = L.EZ - (ART['mouth']['center'][1] - L.MID_V) * K
CHIN_Z = L.EZ - (720 - L.MID_V) * K                                  # 下巴尖 y≈720（±10 px）
IRIS = {'L': (L.px_to_world_x(ART['iris']['L']['center'][0]), L.px_to_world_z(ART['iris']['L']['center'][1])),
        'R': (L.px_to_world_x(ART['iris']['R']['center'][0]), L.px_to_world_z(ART['iris']['R']['center'][1]))}
ROLL_ART = math.degrees(math.atan2(ART['iris']['L']['center'][1] - ART['iris']['R']['center'][1], ART['iris']['R']['center'][0] - ART['iris']['L']['center'][0]))
OPEN_TARGET = {'img_left_eye': (81, 36), 'img_right_eye': (79, 35)}          # 原图实测真实眼白/虹膜开口 宽×高(px)，不含外挑长睫毛尖
BASE_OPEN_W_PX = 83                                                     # R3 对齐基线的可见眼开口宽(px)


def visible_eye_boxes(png):
    im = L.read_rgba(png)[::-1]
    red = (im[..., 3] > 0.5) & (im[..., 0] > 0.8) & (im[..., 1] < 0.2) & (im[..., 2] < 0.2)
    out = {}
    for name, sl in (('img_left_eye', slice(0, int(L.MID_U))), ('img_right_eye', slice(int(L.MID_U), 820))):
        ys, xs = np.nonzero(red[:, sl])
        xs = xs + (sl.start or 0)
        out[name] = {'bbox_px': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], 'size_px': [int(xs.max() - xs.min()), int(ys.max() - ys.min())],
                     'size_mm': [round((xs.max() - xs.min()) * K * 1000, 1), round((ys.max() - ys.min()) * K * 1000, 1)]}
    return out


def placed_centers(place):
    return [tuple(place(np.array([c]))[0]) for c in L.EYE_CENTERS]


def final_xf(rd, place, A, lid, s_low=None, s_up=1.0):
    if s_low is None:
        lm = HF.landmarks(rd, {}, xf=place)
        s_m = (L.EZ - MOUTH_Z) / (L.EZ - lm['mouth']['center'][2])
        s_c = (L.EZ - CHIN_Z) / (L.EZ - lm['chin_tip']['z'])
        s_low = 0.5 * (s_m + s_c)
    deform = L.make_deform(A=A, lid=lid, s_low=s_low, s_up=s_up, centers=placed_centers(place), jaw=1.0)
    return L.compose(place, deform), s_low


def hair_stats(rd, xf, hair_pts, reg_mask, crown_mask):
    S = rd.world('skin', {}, xf)
    c, i = rd.polys('skin')
    tree = L.bvh(S, c, i)
    sd = L.signed_depth(tree, hair_pts)
    ins = sd < 0
    dep = -sd[ins] * 1000
    near = sd[reg_mask & (sd < 0.06)] * 1000
    q = [float(np.percentile(near, p)) for p in (25, 50, 75)] if len(near) else [0, 0, 0]
    nc = sd[crown_mask & (sd < 0.06)] * 1000
    qc = [float(np.percentile(nc, p)) for p in (25, 50, 75)] if len(nc) else [0, 0, 0]
    return {'inside_share': float(ins.mean()), 'mean_depth_mm': float(dep.mean()) if len(dep) else 0.0,
            'p95_depth_mm': float(np.percentile(dep, 95)) if len(dep) else 0.0, 'max_depth_mm': float(dep.max()) if len(dep) else 0.0,
            'face_region_q25_q50_q75_mm': [round(x, 1) for x in q], 'crown_region_q25_q50_q75_mm': [round(x, 1) for x in qc]}, sd


def main():
    rd = L.RadDoll()
    hair_raw = L.load_hair('V10hair')
    raw_pts = np.concatenate([p['v'] for p in hair_raw])                 # V7 参考分布用“未压缩”的发片（发束原配头的自然关系）
    hair = L.load_hair_calibrated('V10hair')                             # 候选头模用“对照原图压缩过头顶”的发片打分
    hair_pts = np.concatenate([p['v'] for p in hair])
    reg = L.face_region_mask(hair_pts)
    crown = hair_pts[:, 2] > 0.05
    sub = np.arange(0, len(hair_pts), 2)
    hp, rg, cr = hair_pts[sub], reg[sub], crown[sub]
    # ---- V7 参照分布 ----
    v7v, v7c, v7i = L.load_head_snapshot('V7Head')
    sd7 = L.signed_depth(L.bvh(v7v, v7c, v7i), raw_pts)
    ins7 = sd7 < 0
    reg7, crown7 = L.face_region_mask(raw_pts), raw_pts[:, 2] > 0.05
    near7 = sd7[reg7 & (sd7 < 0.06)] * 1000
    nc7 = sd7[crown7 & (sd7 < 0.06)] * 1000
    v7ref = {'inside_share': float(ins7.mean()), 'face_region_q25_q50_q75_mm': [round(float(np.percentile(near7, p)), 1) for p in (25, 50, 75)],
             'crown_region_q25_q50_q75_mm': [round(float(np.percentile(nc7, p)), 1) for p in (25, 50, 75)], 'skull_top_z': round(float(v7v[:, 2].max()), 4)}
    q7 = v7ref['face_region_q25_q50_q75_mm']
    qc7 = v7ref['crown_region_q25_q50_q75_mm']
    print('FIT v7ref', json.dumps(v7ref), flush=True)
    res = {'targets': {'mouth_z': round(MOUTH_Z, 4), 'chin_z': round(CHIN_Z, 4), 'roll_art_deg': round(ROLL_ART, 2), 'eye_open_px': OPEN_TARGET}, 'v7_reference': v7ref}
    # ---- 基线（R3 对齐，不变形）----
    ident = L.make_xf()
    st, _ = hair_stats(rd, ident, hair_pts, reg, crown)
    res['baseline'] = {'landmarks': HF.landmarks(rd, {}), 'hair': st}
    HF.render_frame(rd, {}, 'base')
    res['baseline']['eye_visible'] = visible_eye_boxes(OUT / 'head_art_frame_base.png')
    print('FIT baseline', json.dumps(res['baseline']['hair']), json.dumps(res['baseline']['eye_visible']), flush=True)

    def J(h, lm, place_centers, scale, pitch):
        mouth_err = abs(lm['mouth']['center'][2] - MOUTH_Z) * 1000
        chin_err = abs(lm['chin_tip']['z'] - CHIN_Z) * 1000
        # 瞳孔（=放置后的眼中心）对原图虹膜的位置偏差
        eL, eR = place_centers
        eye_err = math.hypot(eL[0] - IRIS['L'][0], eL[2] - IRIS['L'][1]) * 1000 + math.hypot(eR[0] - IRIS['R'][0], eR[2] - IRIS['R'][1]) * 1000
        dist = abs(h['face_region_q25_q50_q75_mm'][1] - q7[1]) + 0.5 * abs(h['face_region_q25_q50_q75_mm'][0] - q7[0])
        dist += abs(h['crown_region_q25_q50_q75_mm'][1] - qc7[1]) + 0.5 * abs(h['crown_region_q25_q50_q75_mm'][0] - qc7[0])
        return h['inside_share'] * 100 + 0.3 * h['mean_depth_mm'] + 0.2 * h['max_depth_mm'] + 0.4 * (mouth_err + chin_err) + 1.5 * eye_err + 1.0 * dist + 60 * abs(scale - 1) + 0.2 * abs(pitch), \
            {'mouth_err_mm': round(mouth_err, 1), 'chin_err_mm': round(chin_err, 1), 'eye_pos_err_mm': round(eye_err, 1), 'dist_err_mm': round(dist, 1)}

    t0 = time.time()
    rows = []
    for scale in (1.0, 0.98):
        A0 = (K * (81 - BASE_OPEN_W_PX * scale), K * (79 - BASE_OPEN_W_PX * scale))
        for dy in (-0.016, -0.012, -0.008, -0.004, 0.0, 0.004):
            for pitch in (-2.0, 0.0, 2.0, 4.0, 6.0):          # 点头角封顶 6°：大角度仰头不自然，剩余穿模交给修发片
                for roll in (0.0, ROLL_ART):
                  for s_up in (1.0, 0.94, 0.90, 0.86):
                    place = L.make_xf(dy=dy, pitch_deg=pitch, roll_deg=roll, scale=scale)
                    xf, s_low = final_xf(rd, place, A0, 0.74, s_up=s_up)
                    h, _ = hair_stats(rd, xf, hp, rg, cr)
                    lm = HF.landmarks(rd, {}, xf=xf)
                    j, parts = J(h, lm, placed_centers(place), scale, pitch)
                    rows.append({'scale': scale, 'dy': dy, 'pitch': pitch, 'roll': round(roll, 2), 's_low': round(s_low, 3), 's_up': s_up, 'jaw_strength': 1.0, 'jaw_knots_y': [list(k) for k in L.JAW_KNOTS_Y], 'hair': h, 'J': round(j, 2), **parts})
    rows.sort(key=lambda r: r['J'])
    print('FIT search %d candidates in %.0fs' % (len(rows), time.time() - t0), flush=True)
    for r in rows[:5]:
        print('FIT top %.2f dy=%+.0f pitch=%+.0f roll=%.1f s_low=%.3f inside=%.1f%% mean=%.1f max=%.1f q=%s J=%.1f %s' % (r['scale'], r['dy'] * 1000, r['pitch'], r['roll'], r['s_low'], 100 * r['hair']['inside_share'], r['hair']['mean_depth_mm'], r['hair']['max_depth_mm'], r['hair']['face_region_q25_q50_q75_mm'], r['J'], {k: r[k] for k in ('mouth_err_mm', 'chin_err_mm', 'eye_pos_err_mm', 'dist_err_mm')}), flush=True)
    best = rows[0]
    place = L.make_xf(dy=best['dy'], pitch_deg=best['pitch'], roll_deg=best['roll'], scale=best['scale'])
    AL = K * (OPEN_TARGET['img_left_eye'][0] - BASE_OPEN_W_PX * best['scale'])
    AR = K * (OPEN_TARGET['img_right_eye'][0] - BASE_OPEN_W_PX * best['scale'])
    A, lid, s_up = (AL, AR), 0.74, best['s_up']
    # ---- 眼开口左右独立迭代校准（渲染→量→调）----
    for it in range(4):
        xf, s_low = final_xf(rd, place, A, lid, s_up=s_up)
        HF.render_frame(rd, {}, 'fitted', xf=xf, centers=placed_centers(place))
        vis = visible_eye_boxes(OUT / 'head_art_frame_fitted.png')
        wL, hL = vis['img_left_eye']['size_px']
        wR, hR = vis['img_right_eye']['size_px']
        twL, thL = OPEN_TARGET['img_left_eye']
        twR, thR = OPEN_TARGET['img_right_eye']
        h_ = 0.5 * (hL + hR)
        th = 0.5 * (thL + thR)
        print('FIT eye-iter', it, 'A=(%.2f, %.2f) mm lid=%.3f -> L=%dx%d R=%dx%d px (目标 L=%dx%d R=%dx%d)' % (A[0]*1000, A[1]*1000, lid, wL, hL, wR, hR, twL, thL, twR, thR), flush=True)
        if abs(wL - twL) <= 1 and abs(wR - twR) <= 1 and abs(h_ - th) <= 1.5:
            break
        A = (A[0] + K * (twL - wL), A[1] + K * (twR - wR))
        lid = float(np.clip(lid * (th / h_) ** 1.25, 0.52, 0.95))
    xf, s_low = final_xf(rd, place, A, lid, s_up=s_up)
    HF.render_frame(rd, {}, 'fitted', xf=xf, centers=placed_centers(place))
    st, sd = hair_stats(rd, xf, hair_pts, reg, crown)
    # 各对象埋入
    S = rd.world('skin', {}, xf)
    diag, k0 = {}, 0
    for p in hair:
        n = len(p['v'])
        s_ = sd[k0:k0 + n]
        k0 += n
        ins = s_ < 0
        j = int(np.argmin(s_))
        diag[p['name'].replace('GEO-Hair-', '')] = {'verts': n, 'inside': int(ins.sum()), 'inside_share': round(float(ins.mean()), 3),
                                                    'max_depth_mm': round(float(-s_.min()) * 1000, 1) if ins.any() else 0.0,
                                                    'deepest_at_xyz': [round(float(x), 3) for x in p['v'][j]] if ins.any() else None}
    res['fitted'] = {'scale': best['scale'], 'dy_mm': round(best['dy'] * 1000, 1), 'pitch_deg': best['pitch'], 'roll_deg': best['roll'],
                     'eye_bulge_A_mm': round(0.5 * (A[0] + A[1]) * 1000, 2), 'eye_bulge_AL_AR_mm': [round(A[0] * 1000, 2), round(A[1] * 1000, 2)], 'eye_lower_lid': round(lid, 3), 's_low': round(s_low, 3), 's_up': s_up, 'jaw_strength': 1.0, 'jaw_knots_y': [list(k) for k in L.JAW_KNOTS_Y], 'landmarks': HF.landmarks(rd, {}, xf=xf),
                     'eye_visible': visible_eye_boxes(OUT / 'head_art_frame_fitted.png'), 'hair_full': st, 'per_object': diag}
    res['search_top10'] = rows[:10]
    (HERE / 'head_fit.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    # 导出变形后眼片与脸皮的精确 UV ↔ 世界坐标三角网，供 make_eye_texture.py 做零畸变重心逆映射
    F_def = L.refine_fac_parts(rd.world('fac', {}, xf), rd.cls_of_vertex, placed_centers(place))
    def _uv_tris(verts_w, tag, only_cls=None):
        c, idx = rd.polys(tag)
        uv = rd.p[tag]['uv']
        starts = np.cumsum(c) - c
        tris_w, tris_uv = [], []
        for s0, cnt in zip(starts, c):
            if only_cls is not None and rd.cls_of_vertex[idx[s0]] != only_cls:
                continue
            for k_ in range(1, int(cnt) - 1):
                ls = [s0, s0 + k_, s0 + k_ + 1]
                tris_w.append(verts_w[idx[ls]])
                tris_uv.append(uv[ls])
        return np.array(tris_w, np.float32), np.array(tris_uv, np.float32)
    ew, euv = _uv_tris(F_def, 'fac', 'eye')
    sw, suv = _uv_tris(S, 'skin', None)
    np.savez_compressed(L.SNAP / 'head_uv_deformed.npz', eye_w=ew, eye_uv=euv, skin_w=sw, skin_uv=suv,
                        meta=np.array([L.EX, L.EZ, L.K, L.MID_U, L.MID_V, CHIN_Z], np.float64))
    print('HEADFIT_DONE', flush=True)


if __name__ == '__main__':
    main()
