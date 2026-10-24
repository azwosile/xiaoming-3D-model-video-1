"""V10R7 共用库：读 RadDoll（R1 静态装配里已按 V10 坐标对齐的皮肤+脸部件）、按形态键求值、
按“原图坐标”渲染、发片载入、穿头检验。只读源文件。仅供同目录脚本 import（需在 Blender 内运行）。
"""
import json
import math
import re
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
B = ROOT / '生成模型/组装'
R1_STATIC = B / 'v010R1_头模对照与头发修整/RadDoll_V10_static_PRIVATE.blend'
SNAP = ROOT / '_tmp/head_compare/snap'
SPEC = json.loads((B / 'v010R1_头模对照与头发修整/spec.json').read_text(encoding='utf-8'))
ART = json.loads((HERE / 'landmarks_art.json').read_text(encoding='utf-8'))
CAP = 'GEO-Hair-Cap_Shared'

# ---------------- 原图 ↔ 发束坐标系（V10 帧）的配准：只对齐两眼（与 R6 相同）----------------
EYE_L_PX, EYE_R_PX = tuple(ART['iris']['L']['center']), tuple(ART['iris']['R']['center'])
EX, EZ = SPEC['target_eye_proxy'][0], SPEC['target_eye_proxy'][2]
EY = SPEC['target_eye_proxy'][1]
K = float(SPEC['target_eye_spacing']) / (EYE_R_PX[0] - EYE_L_PX[0])         # 米 / 原图像素
MID_U, MID_V = (EYE_L_PX[0] + EYE_R_PX[0]) / 2, (EYE_L_PX[1] + EYE_R_PX[1]) / 2


def art_to_world(u, v):
    return EX - (u - MID_U) * K, EZ - (v - MID_V) * K          # 图右 = -x


def world_to_art(x, z):
    return MID_U - (x - EX) / K, MID_V - (z - EZ) / K


# ---------------- 通用几何 ----------------
def tris_of(counts, idx):
    starts = np.cumsum(counts) - counts
    out = []
    for c in np.unique(counts):
        sel = starts[counts == c]
        for k in range(1, int(c) - 1):
            out.append(np.stack([idx[sel], idx[sel + k], idx[sel + k + 1]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), np.int32)


def bvh(verts, counts, idx):
    return BVHTree.FromPolygons(np.asarray(verts).tolist(), tris_of(counts, idx).tolist(), all_triangles=True)


def load_hair(label='V10hair'):
    meta = json.loads((SNAP / f'{label}.json').read_text(encoding='utf-8'))
    z = np.load(SNAP / f'{label}.npz')
    return [{'name': pm['name'], 'v': z[f'{i}_v'].astype(np.float64), 'counts': z[f'{i}_counts'], 'idx': z[f'{i}_idx']} for i, pm in enumerate(meta['parts'])]


# ---------------- RadDoll ----------------
class RadDoll:
    """skin / fac 两个物件；键值 = {键名: 权重}；返回世界坐标（V10 帧）。"""

    def __init__(self):
        bpy.ops.wm.open_mainfile(filepath=str(R1_STATIC), load_ui=False, use_scripts=False)
        sk, fa = bpy.data.objects['RadDoll_Head_0'], bpy.data.objects['RadDoll_Head_1']
        self.M = np.array(sk.matrix_world)
        assert np.allclose(self.M, np.array(fa.matrix_world))
        self.p = {}
        for tag, o in (('skin', sk), ('fac', fa)):
            me = o.data
            n = len(me.vertices)
            keys = me.shape_keys.key_blocks

            def co(kb):
                a = np.empty(n * 3)
                kb.data.foreach_get('co', a)
                return a.reshape(-1, 3)
            b0 = co(keys[0])
            counts = np.empty(len(me.polygons), np.int32)
            me.polygons.foreach_get('loop_total', counts)
            idx = np.empty(len(me.loops), np.int32)
            me.loops.foreach_get('vertex_index', idx)
            deltas = {}
            for kb in keys[1:]:
                if kb.name.startswith('==='):
                    continue
                d = co(kb) - b0
                if np.abs(d).max() > 1e-9:
                    deltas[kb.name] = d.astype(np.float32)
            order = [kb.name for kb in keys[1:] if not kb.name.startswith('===')]
            uv = None
            if me.uv_layers:
                uvf = np.empty(len(me.loops) * 2, np.float32)
                me.uv_layers[0].data.foreach_get('uv', uvf)
                uv = uvf.reshape(-1, 2)
            self.p[tag] = {'basis': b0, 'counts': counts, 'idx': idx, 'deltas': deltas, 'n': n, 'order': order, 'uv': uv}
        self._components()

    def _components(self):
        fa = self.p['fac']
        n = fa['n']
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        starts = np.cumsum(fa['counts']) - fa['counts']
        for s0, c in zip(starts, fa['counts']):
            vs = fa['idx'][s0:s0 + c]
            for a, b in zip(vs, vs[1:]):
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[ra] = rb
        roots = np.array([find(i) for i in range(n)])
        self.comp_of = roots
        W = self.world('fac', {})
        cls = {}
        for r in np.unique(roots):
            ix = np.nonzero(roots == r)[0]
            c = W[ix].mean(0)
            cnt = len(ix)
            if cnt == 301:
                k = 'eye'
            elif cnt == 219:
                k = 'ear'
            elif cnt == 143:
                k = 'lash'
            elif c[2] < -0.115:
                k = 'mouth'
            elif cnt == 34 and c[2] > -0.04:
                k = 'brow'
            elif cnt == 16 and c[1] > 0.100 and -0.055 < c[2] < -0.038:
                k = 'double_lid'
            else:
                # 其余小矩形片为 RadDoll 原厂需贴图 alpha 遮罩的睫毛尖/高光/漫符贴纸/猫眼竖瞳片；
                # 高光与圆润瞳仁直接绘入高精度眼睛贴图，此处归入 hidden 使用全透明材质，完整保留网格与形态键。
                k = 'hidden'
            cls[int(r)] = k
        self.cls_of_vertex = np.array([cls[int(r)] for r in roots])

    def local(self, tag, values):
        p = self.p[tag]
        co = p['basis'].astype(np.float64).copy()
        for k, w in values.items():
            if w and k in p['deltas']:
                co += w * p['deltas'][k]
        return co

    def world(self, tag, values, xf=None):
        co = self.local(tag, values) @ self.M[:3, :3].T + self.M[:3, 3]
        return xf(co) if xf else co

    def polys(self, tag):
        return self.p[tag]['counts'], self.p[tag]['idx']


def make_xf(dx=0.0, dy=0.0, dz=0.0, pitch_deg=0.0, roll_deg=0.0, scale=1.0, pivot=None):
    """绕眼中心点：先缩放，再点头(pitch，绕 X)与侧倾(roll，绕 Y)，再平移。
    pitch>0：额头后退、下巴前送；roll>0：画面右眼(世界 -x)上抬、画面左眼下沉。"""
    pv = np.array(pivot if pivot is not None else (EX, EY, EZ))
    th, ph = math.radians(pitch_deg), math.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    Ry = np.array([[math.cos(ph), 0, math.sin(ph)], [0, 1, 0], [-math.sin(ph), 0, math.cos(ph)]])
    R = Ry @ Rx

    def f(P):
        return (scale * (P - pv)) @ R.T + pv + np.array([dx, dy, dz])
    return f


# ---------------- 渲染 ----------------
def new_scene(res=800, flat=False, bg=(0.82, 0.84, 0.87), exposure=0.5, transparent=False, color_type='OBJECT'):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.image_settings.file_format = 'PNG'
    sc.view_settings.view_transform = 'Standard'
    sh = sc.display.shading
    sh.light, sh.color_type, sh.show_cavity, sh.show_specular_highlight = ('FLAT' if flat else 'STUDIO'), color_type, False, False
    sh.show_shadows = not flat
    sh.background_type = 'WORLD'
    sc.world = bpy.data.worlds.new('W')
    sc.world.color = bg
    if flat:
        sc.render.film_transparent = True
        sc.render.image_settings.color_mode = 'RGBA'
        sc.display.render_aa = 'OFF'
    else:
        sc.view_settings.exposure = exposure
        if transparent:
            sc.render.film_transparent = True
            sc.render.image_settings.color_mode = 'RGBA'
    cam = bpy.data.objects.new('CAM', bpy.data.cameras.new('CAM'))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = 'ORTHO'
    cam.data.clip_start, cam.data.clip_end = 0.001, 20
    return sc, cam


def add_mesh(name, verts, counts, idx, color, keep_polys=None):
    me = bpy.data.meshes.new(name)
    polys, i = [], 0
    for c in counts:
        polys.append(tuple(int(x) for x in idx[i:i + c]))
        i += c
    if keep_polys is not None:
        polys = [polys[k] for k in keep_polys]
    me.from_pydata(np.asarray(verts).tolist(), [], polys)
    me.update()
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.color = color
    return ob


def shoot(sc, cam, center, scale, vec, path, up_top=False):
    cam.data.ortho_scale = scale
    cam.location = Vector(center) + Vector(vec)
    if up_top:
        cam.rotation_euler = (0, 0, 0)
    else:
        cam.rotation_euler = (Vector(center) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def art_frame_camera():
    """与原图头部特写 820×820 像素一一对应的正射相机：返回 (center_xyz, ortho_scale, vec)。"""
    xc, zc = art_to_world(410, 410)
    return (xc, 0.0, zc), 820 * K, (0, 2, 0)


# ---------------- 对世界坐标的点对点变形（不动形态键；烘焙装配时对每个键的绝对坐标同样套用）----------------
def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


EYE_CENTERS = [(0.0438, 0.0955, -0.0650), (-0.0498, 0.0955, -0.0650)]          # 画面左眼、右眼（V10 帧，R3 对齐后的眼部件中心）


def px_to_world_x(u):
    return EX - (u - MID_U) * K


def px_to_world_z(v):
    return EZ - (v - MID_V) * K


def eye_bulge(P, centers, A=0.0052, lid=0.68, u0=0.010, u1=0.021):
    """局部变形（每只眼）：支持左右独立外扩量 A=(A_L, A_R) 或标量 A：
       · u0..u1 之间平滑地把外侧眼白向外推，推移量在 u1 附近达到 A_i，之后在约 26 mm 内衰减到 0（虹膜与内眼角不动）；
       · 下眼睑：瞳孔以下的竖向距离乘以 lid（<1 即抬高下缘），收紧纵向眼裂高度至原画 40-41 px；
       · 上眼睑中段轻柔微拱（+1.0 mm），自然贴合原画上睫弧度而不放大眼眶。"""
    Q = P.copy()
    a_pair = (float(A[0]), float(A[1])) if isinstance(A, (tuple, list, np.ndarray)) else (float(A), float(A))
    for idx_c, c in enumerate(centers):
        Ai = a_pair[idx_c]
        s_out = 1.0 if c[0] > EX else -1.0
        u = s_out * (P[:, 0] - c[0])
        wz = 1.0 - smoothstep((np.abs(P[:, 2] - c[2]) - 0.020) / 0.020)
        ramp = smoothstep((u - u0) / (u1 - u0))
        decay = 1.0 - smoothstep((u - (u1 + 0.002)) / 0.026)
        dx = Ai * ramp * np.where(u > u1, decay, 1.0) * wz
        Q[:, 0] += s_out * dx
        # 下眼睑抬高（紧凑柔和杏眼下缘）
        d = c[2] - P[:, 2]
        m = d > 0.0025
        wx = 1.0 - smoothstep((np.abs(u) - 0.026) / 0.018)
        wd = 1.0 - smoothstep((d - 0.028) / 0.028)
        k = (1.0 - lid) * wx * wd * smoothstep((d - 0.0025) / 0.0055)
        Q[m, 2] += (d * k)[m]
        # 内眼角轻微向瞳孔侧收拢（约 1.2 mm），配合外眼角内收使总眼裂更小巧精致
        u_in = -u
        ramp_in = smoothstep((u_in - 0.009) / 0.010) * (1.0 - smoothstep((u_in - 0.020) / 0.016))
        Q[:, 0] += s_out * (0.0012 * ramp_in * wz)
        # 上眼睑轻度下收（两侧眼角下收 -0.9 mm，中段仅保留柔和圆弧），收紧纵向眼眶同时不挡主高光
        du_c = P[:, 0] - c[0]
        dz_up = P[:, 2] - c[2]
        m_up = (dz_up > 0.0015) & (dz_up < 0.0185) & (P[:, 1] > 0.074)
        w_span_x = 1.0 - smoothstep((np.abs(du_c) - 0.022) / 0.014)
        w_mid_x = np.exp(-((du_c / 0.0135) ** 2))
        w_arch_z = smoothstep((dz_up - 0.0015) / 0.0045) * (1.0 - smoothstep((dz_up - 0.0120) / 0.0065))
        dz_shift = (-0.0009 * w_span_x + 0.0017 * w_mid_x) * w_arch_z
        Q[m_up, 2] += dz_shift[m_up]
    return Q


def refine_fac_parts(P, cls_v, centers):
    """针对 RadDoll_FacialParts 子网格的 C¹ 单调保向修饰（保证 flipped_by_deform == 0）：
       1. lash（上睫毛）：C¹ 平滑内收外挑眼尾并适度收纤上沿厚度，告别粗重黑框；
       2. brow（眉毛）：基于二次抛物线拟合眉弓中心线，温和加粗眉条并贴肤微前移 +0.5mm。"""
    Q = P.copy()
    m_lash = (cls_v == 'lash')
    if np.any(m_lash):
        pts = Q[m_lash].copy()
        for c in centers:
            s_out = 1.0 if c[0] > EX else -1.0
            side = (pts[:, 0] > EX) if s_out > 0 else (pts[:, 0] <= EX)
            if not np.any(side):
                continue
            u = s_out * (pts[side, 0] - c[0])
            # C¹ 平滑大幅收短外挑睫毛翼（-4.3 mm）+ 内眼角微收（-1.0 mm），使眼型整体小巧精致且零翻转面
            pts[side, 0] -= s_out * (0.0056 * smoothstep((u - 0.012) / 0.020))
            pts[side, 0] += s_out * (0.0010 * smoothstep((-u - 0.010) / 0.012))
            pts[side, 2] -= 0.0007 * smoothstep((pts[side, 2] - c[2] - 0.002) / 0.010)
        Q[m_lash] = pts

    m_dlid = (cls_v == 'double_lid')
    if np.any(m_dlid):
        pts = Q[m_dlid].copy()
        for c in centers:
            s_out = 1.0 if c[0] > EX else -1.0
            side = (pts[:, 0] > EX) if s_out > 0 else (pts[:, 0] <= EX)
            if not np.any(side):
                continue
            pts[side, 2] -= 0.0013
            pts[side, 0] = c[0] + 0.90 * (pts[side, 0] - c[0])
        Q[m_dlid] = pts

    m_brow = (cls_v == 'brow')
    if np.any(m_brow):
        pts = Q[m_brow].copy()
        for s_side in (1.0, -1.0):
            side = (pts[:, 0] > EX) if s_side > 0 else (pts[:, 0] <= EX)
            if not np.any(side):
                continue
            ps = pts[side].copy()
            xs = ps[:, 0]
            coef = np.polyfit(xs, ps[:, 2], 2)
            z_mid = np.polyval(coef, xs)
            x_span = max(xs.max() - xs.min(), 1e-6)
            t_mid = np.sin(math.pi * np.clip((xs - xs.min()) / x_span, 0.0, 1.0))
            ps[:, 2] = z_mid + (ps[:, 2] - z_mid) * (1.08 + 0.22 * t_mid) + 0.0005
            ps[:, 1] += 0.0005
            pts[side] = ps
        Q[m_brow] = pts
    return Q


def lower_compress(P, s0, z_pivot=EZ, blend=0.020):
    """眼线以下的竖向压缩：眼线处不变，向下 blend 米内平滑过渡到 s0 倍，再往下为常数 s0。"""
    Q = P.copy()
    d = z_pivot - P[:, 2]                                   # 眼线以下为正
    m = d > 0
    t = np.clip(d[m] / blend, 0, 1)
    s = 1.0 - (1.0 - s0) * smoothstep(t)
    Q[m, 2] = z_pivot - d[m] * s
    return Q


def upper_compress(P, s_up, z_pivot=EZ, blend=0.030):
    """眼线以上的竖向压缩（颅顶）：眼线处不变，向上 blend 米内平滑过渡到 s_up 倍，再往上为常数 s_up。"""
    Q = P.copy()
    d = P[:, 2] - z_pivot
    m = d > 0
    t = np.clip(d[m] / blend, 0, 1)
    s = 1.0 - (1.0 - s_up) * smoothstep(t)
    Q[m, 2] = z_pivot + d[m] * s
    return Q


# 下半脸宽度剖面（x 向缩放系数，绕脸中线 x=EX）：原图 y（像素）→ 系数。
# 依据：逐行追踪原图真实下颌/脸颊墨线内侧宽度（含暖色下巴阴影）与未收窄 3D 头皮轮廓比值，采用单调平滑凸曲线避免下巴内凹夹尖。
JAW_KNOTS_Y = (
    (505, 1.000),
    (545, 0.965),
    (580, 0.905),
    (610, 0.855),
    (635, 0.822),
    (655, 0.802),
    (675, 0.768),
    (690, 0.695),
    (702, 0.632),
    (716, 0.630),
    (760, 0.630),
)


def jaw_narrow(P, strength=1.0, mid=EX):
    ys = np.array([k[0] for k in JAW_KNOTS_Y], float)
    fs = np.array([k[1] for k in JAW_KNOTS_Y], float)
    zk = np.array([px_to_world_z(y) for y in ys])               # 递减
    order = np.argsort(zk)
    zk, fk = zk[order], fs[order]
    z = P[:, 2]
    f = np.ones(len(P))
    for i in range(len(zk) - 1):
        m = (z >= zk[i]) & (z < zk[i + 1])
        t = (z[m] - zk[i]) / (zk[i + 1] - zk[i])
        f[m] = fk[i] + (fk[i + 1] - fk[i]) * smoothstep(t)
    f[z < zk[0]] = fk[0]
    f[z >= zk[-1]] = 1.0
    f = 1.0 - strength * (1.0 - f)
    # 后脑勺平滑减弱横向收窄，消除侧/45° 视图后脑水平过渡折痕，保持颅骨圆润
    w_ant = 0.35 + 0.65 * smoothstep((P[:, 1] + 0.025) / 0.075)
    f_eff = 1.0 - (1.0 - f) * w_ant
    Q = P.copy()
    dx = P[:, 0] - mid
    # 补偿 -3° 侧倾带来的下巴尖向左微偏（约 3.2 mm），使正视图左右下颌线与下巴尖精确居中对齐原图
    z_chin_start = px_to_world_z(635)
    w_front = smoothstep((P[:, 1] - 0.005) / 0.035)
    chin_shift = -0.0032 * strength * smoothstep((z_chin_start - z) / 0.040) * w_front
    Q[:, 0] = mid + f_eff * dx + chin_shift
    # 侧面/45° 脸颊深度联动微收，消除正面收窄后外颊鼓包，但完全不动鼻梁与嘴唇中线区（|dx|<=18mm）
    lat = np.maximum(0.0, np.abs(dx) - 0.018)
    w_cheek_y = smoothstep((P[:, 1] - 0.020) / 0.035)
    Q[:, 1] = P[:, 1] - 0.16 * (1.0 - f_eff) * lat * w_cheek_y

    # 侧面下颌角上扬与耳后/枕底收颈（严格限制在 y < 38mm 正脸轮廓线后方，正视图 0.0mm 零影响）：
    # 消除侧视图下巴后方水平直切粉块，塑造自然上扬的二次元下颌角与紧致后颈底
    w_post_jaw = smoothstep((0.038 - Q[:, 1]) / 0.072)
    w_low_slab = smoothstep((-0.072 - Q[:, 2]) / 0.068)
    Q[:, 2] += 0.031 * strength * w_post_jaw * w_low_slab
    w_neck_in = smoothstep((0.026 - Q[:, 1]) / 0.065) * smoothstep((-0.086 - Q[:, 2]) / 0.056)
    Q[:, 0] = mid + (Q[:, 0] - mid) * (1.0 - 0.18 * strength * w_neck_in)
    return Q


def make_deform(A=0.0, lid=1.0, s_low=1.0, s_up=1.0, centers=None, jaw=0.0):
    cs = centers if centers is not None else EYE_CENTERS

    def f(P):
        has_A = (any(A) if isinstance(A, (tuple, list, np.ndarray)) else bool(A))
        if has_A or lid != 1.0:
            P = eye_bulge(P, cs, A=A, lid=lid)
        if s_low != 1.0:
            P = lower_compress(P, s_low)
        if s_up != 1.0:
            P = upper_compress(P, s_up)
        if jaw:
            P = jaw_narrow(P, jaw)
        return P
    return f


def compose(*fs):
    def f(P):
        for g in fs:
            P = g(P)
        return P
    return f


def read_rgba(path):
    img = bpy.data.images.load(str(path))
    w, h = img.size
    buf = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)


def signed_depth(tree, pts):
    """近似符号距离（正=在头外）：最近点法线判符号。仅用于搜索/优化；最终用奇偶射线复核。"""
    out = np.empty(len(pts))
    for i, p in enumerate(pts):
        v = Vector(p)
        loc, nrm, _, dist = tree.find_nearest(v)
        out[i] = dist if (v - loc).dot(nrm) >= 0 else -dist
    return out


def load_head_snapshot(label):
    """快照里的头皮（prefit 的 V7 已在 V10 帧）。返回 (verts, counts, idx)。"""
    meta = json.loads((SNAP / f'{label}.json').read_text(encoding='utf-8'))
    z = np.load(SNAP / f'{label}.npz')
    for k, pm in enumerate(meta['parts']):
        if pm.get('role') == 'skin':
            i = pm.get('i', k)
            return z[f'{i}_v'].astype(np.float64), z[f'{i}_counts'], z[f'{i}_idx']
    raise KeyError('no skin')


def face_region_mask(v):
    return (v[:, 1] > 0.02) & (np.abs(v[:, 0]) < 0.12) & (v[:, 2] > -0.11) & (v[:, 2] < 0.07)


# 发片头顶高度对照原图：右侧头顶 5 个位置上“原图头顶离眼线的距离 / 3D 发型头顶离眼线的距离”= 0.928–0.936（中位 0.932），
# 是一个干净的竖向比例差，故对眼线以上的发片做竖向压缩（与头模的颅顶压缩同类，枢轴为眼线）。
S_HAIR_UP = 0.932


def load_hair_calibrated(label='V10hair'):
    out = []
    for p in load_hair(label):
        q = dict(p)
        q['v'] = upper_compress(p['v'], S_HAIR_UP, blend=0.030)
        out.append(q)
    return out
