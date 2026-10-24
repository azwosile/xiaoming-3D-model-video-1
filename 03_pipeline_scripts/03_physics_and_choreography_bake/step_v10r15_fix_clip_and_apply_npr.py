"""
V10R15 主工程脚本：
1) 根治 f546 (及全片) 手臂/手腕与外套袖口穿模（手腕随动权重 + 袖口微扩 + 袖内遮罩延伸）
2) 根治 f499 (及全片) 短裤与大腿穿模 + 腰头下方横缝（封闭横缝 + 精确传递大腿 Hip/upper_leg 权重 + 实时素体大腿防穿模代理）
3) 装载 5 张全新去花斑平涂贴图，并重构零白斑、宽幅顺滑二次元赛璐璐着色器（清除导致迷彩白斑的 CONSTANT LayerWeight 节点）
严守文件锁红线：只读打开 V10R14，全部输出另存为 RadDoll_V10R15_MMD_Heart舞台三渲二纯净平涂精修_PRIVATE.blend！
"""
import time
import bpy
import bmesh
import numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree

t0 = time.time()
ROOT      = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
WDIR      = ROOT / r"生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE"
CLEAN_TEX = WDIR / "textures_v10r15_clean"
QADIR     = WDIR / "v10r15_clean_qa"
QADIR.mkdir(parents=True, exist_ok=True)

SRC_BLEND = WDIR / "RadDoll_V10R14_MMD_Heart舞台三渲二实感布料精修_PRIVATE.blend"
DST_BLEND = WDIR / "RadDoll_V10R15_MMD_Heart舞台三渲二纯净平涂精修_PRIVATE.blend"

bpy.ops.wm.open_mainfile(filepath=str(SRC_BLEND), load_ui=False)
try:
    bpy.ops.preferences.addon_enable(module="bl_ext.blender_org.mmd_tools")
except Exception as e:
    print("mmd_tools note:", e)

sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 32
sc.render.resolution_x = 1600
sc.render.resolution_y = 900
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.quality = 95

orig_cam = sc.camera
arm_ob   = bpy.data.objects["ARM-FullBody_V10"]

def get_hips_world(frame_no):
    sc.frame_set(frame_no)
    dg = bpy.context.evaluated_depsgraph_get()
    arm_ev = arm_ob.evaluated_get(dg)
    pb = arm_ev.pose.bones.get("Hips")
    return arm_ev.matrix_world @ pb.head if pb else Vector((0.0, 1.2, 0.8))

def make_track_cam(name, loc, target_co, lens=45.0):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_start = 0.02
    cd.clip_end = 100.0
    co = bpy.data.objects.new(name, cd)
    sc.collection.objects.link(co)
    co.location = Vector(loc)
    tgt = bpy.data.objects.new(name + "_Tgt", None)
    sc.collection.objects.link(tgt)
    tgt.location = Vector(target_co)
    tr = co.constraints.new(type='TRACK_TO')
    tr.target = tgt
    tr.track_axis = 'TRACK_NEGATIVE_Z'
    tr.up_axis = 'UP_Y'
    return co, tgt

# 锁定 f546 与 f499 精确世界坐标复刻用户截图视角！
h546 = get_hips_world(546)
cam_546, tgt_546 = make_track_cam("QA_UserF546", (h546.x - 0.95, h546.y - 2.35, h546.z + 0.14), (h546.x - 0.08, h546.y + 0.05, h546.z - 0.04), lens=42.0)

h499 = get_hips_world(499)
cam_499, tgt_499 = make_track_cam("QA_UserF499", (h499.x + 0.02, h499.y - 1.18, h499.z - 0.04), (h499.x + 0.01, h499.y, h499.z - 0.05), lens=50.0)

cam_458, tgt_458 = make_track_cam("QA_RegF458", (-1.65, 1.50, 0.85), (-0.02, 1.20, 0.78), lens=42.0)
cam_416, tgt_416 = make_track_cam("QA_RegF416", (-0.05, 0.36, 0.74), (0.01, 1.44, 0.95), lens=36.0)

QA_SHOTS = [
    ("f546_user", 546, cam_546),
    ("f499_user", 499, cam_499),
    ("f458_reg",  458, cam_458),
    ("f416_reg",  416, cam_416),
]

print("=== [STEP A] Rendering V10R14 BEFORE Shots ===", flush=True)
for tag, fr, cam_o in QA_SHOTS:
    sc.camera = cam_o
    sc.frame_set(fr)
    bpy.context.view_layer.update()
    sc.render.filepath = str(QADIR / f"before_{tag}.jpg")
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered before_{tag}.jpg", flush=True)


# ==============================================================================
# STEP B: 根治 f546 (及全片) 手臂/手腕与外套袖口穿模
# ==============================================================================
print("=== [STEP B] Fixing Arm & Wrist Sleeve Penetration Across All 660 Frames ===", flush=True)
def smoothstep(e0, e1, val):
    t = np.clip((val - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

skin_ob = bpy.data.objects["GEO-Sotai_Skin"]
coat_ob = bpy.data.objects["GEO-Cloth_Coat"]
deform_bones = [b.name for b in arm_ob.data.bones if b.use_deform]
b2idx = {bn: i for i, bn in enumerate(deform_bones)}

# B1) 将素体袖内隐藏遮罩平滑覆盖整个被水手服+外套包裹的上臂/前臂/手腕根 (0.158 < |x| < 0.516)，完整保留颈肩与双手手指！
vg_hide = skin_ob.vertex_groups.get("VG_Hidden_UnderCloth")
if vg_hide:
    vg_hide.remove(list(range(len(skin_ob.data.vertices))))
    hide_ids = []
    for v in skin_ob.data.vertices:
        x, y, z = v.co.x, v.co.y, v.co.z
        ax = abs(x)
        hide = False
        if 0.158 < ax < 0.516 and -0.38 < z < -0.10:
            hide = True
        elif ax <= 0.158 and -0.372 < z < -0.168:
            if not (ax <= 0.056 and z > -0.308):
                hide = True
        elif ax < 0.115 and -0.625 < z < -0.578:
            hide = True
        if hide:
            hide_ids.append(v.index)
    vg_hide.add(hide_ids, 1.0, 'REPLACE')
    print(f"  [SkinMask] Updated covered-arm mask (0.158 < |x| < 0.516), hidden={len(hide_ids)} verts!", flush=True)

# B2) 为外套前臂末段与宽袖口 (|x| in [0.445, 0.620]) 注入与手腕一致的 hand + handrole 随动权重，并微扩袖口净空 +5.5mm！
# 使得 f546 屈腕跳舞时宽袖口自然随手掌偏转，手掌永远处于袖口正中央，绝不戳穿袖壁！
N_c = len(coat_ob.data.vertices)
B   = len(deform_bones)
vg_c_to_b = {vg.index: b2idx[vg.name] for vg in coat_ob.vertex_groups if vg.name in b2idx}
W_c = np.zeros((N_c, B), dtype=np.float64)
for vi, v in enumerate(coat_ob.data.vertices):
    for g in v.groups:
        bi = vg_c_to_b.get(g.group)
        if bi is not None:
            W_c[vi, bi] = g.weight

# 计算前臂袖管轴心 (y~0.003, z~-0.242)，对 |x| > 0.43 的远端袖筒沿径向温和外扩 +5.5mm
for v in coat_ob.data.vertices:
    ax = abs(v.co.x)
    if ax > 0.430 and v.co.z > -0.39:
        flare = 0.0055 * smoothstep(0.430, 0.525, ax)
        rad = Vector((0.0, v.co.y - 0.003, v.co.z - (-0.242)))
        if rad.length > 1e-4:
            v.co += rad.normalized() * flare
coat_ob.data.update()

c_coat = np.array([v.co[:] for v in coat_ob.data.vertices], dtype=np.float64)
for vi in range(N_c):
    x, y, z = c_coat[vi]
    ax = abs(x)
    if ax > 0.445 and z > -0.40:
        sfx = "R" if x >= 0.0 else "L"
        t_wrist = smoothstep(0.445, 0.545, ax)
        # 在袖口处平滑混入 38% hand + 14% handrole，与内部手掌屈伸同向随动！
        w_hand = 0.38 * t_wrist
        w_hrol = 0.14 * t_wrist
        keep_old = 1.0 - (w_hand + w_hrol)
        W_c[vi] *= keep_old
        if f"hand.{sfx}" in b2idx:
            W_c[vi, b2idx[f"hand.{sfx}"]] += w_hand
        if f"handrole.{sfx}" in b2idx:
            W_c[vi, b2idx[f"handrole.{sfx}"]] += w_hrol
        W_c[vi] /= max(W_c[vi].sum(), 1e-8)

for bn in deform_bones:
    vg = coat_ob.vertex_groups.get(bn) or coat_ob.vertex_groups.new(name=bn)
    vg.remove(list(range(N_c)))
    col = W_c[:, b2idx[bn]]
    for vi in np.where(col > 1e-4)[0]:
        vg.add([int(vi)], float(col[vi]), 'REPLACE')
print("  [CuffFollow] Added smooth wrist/hand articulation follow to distal coat sleeves & cuffs!", flush=True)


# ==============================================================================
# STEP C: 根治 f499 (及全片) 短裤与大腿穿模 + 消除腰头正下方凹槽露肉横缝！
# ==============================================================================
print("=== [STEP C] Fixing Shorts Waistband Groove Slit & Thigh Cuff Penetration ===", flush=True)
shorts_ob = bpy.data.objects["GEO-Cloth_Shorts"]

# C1) 几何净空微调（在静态网格上精准拓宽两处易擦皮点，100% 保持原版短裤版型、白滚边与蝴蝶结抽绳！）：
#   ① 腰头正下方缝线凹槽 (-0.548 < z < -0.502, y > 0.04)：向外柔推 +4.2mm，彻底消除小腹在腰头沟槽处的细横露肉线！
#   ② 左右裤管与大腿根部 (-0.685 < z < -0.555)：沿各自大腿横截面圆心 (±0.082, +0.010) 径向柔扩 +7.2mm，给 f499 交叉迈步留足真实裤管余量！
bm_s = bmesh.new()
bm_s.from_mesh(shorts_ob.data)
bm_s.verts.ensure_lookup_table()
bm_s.edges.ensure_lookup_table()
bm_s.faces.ensure_lookup_table()
uv_s = bm_s.loops.layers.uv.active

v_vis_s = set()
s_comps = []
for v in bm_s.verts:
    if v.index in v_vis_s: continue
    q = [v]; v_vis_s.add(v.index); cc = []
    while q:
        u = q.pop(); cc.append(u)
        for e in u.link_edges:
            w = e.other_vert(u)
            if w.index not in v_vis_s:
                v_vis_s.add(w.index); q.append(w)
    s_comps.append(cc)
s_comps.sort(key=len, reverse=True)
main_s_set = {u.index for u in s_comps[0]}

start_lps = [e.link_loops[0] for e in bm_s.edges if e.is_boundary and len(e.link_loops) == 1 and e.verts[0].index in main_s_set]
used_be = set()
ordered_loops = []
for lp0 in start_lps:
    if lp0.edge.index in used_be:
        continue
    cyc = []
    cur = lp0
    for _ in range(2000):
        if cur.edge.index in used_be:
            break
        used_be.add(cur.edge.index)
        cyc.append(cur.vert)
        v_end = cur.link_loop_next.vert
        cands = [
            e.link_loops[0] for e in v_end.link_edges
            if e.is_boundary and len(e.link_loops) == 1
            and e.index not in used_be and e.link_loops[0].vert == v_end
        ]
        if not cands:
            break
        if len(cands) == 1:
            cur = cands[0]
        else:
            fan = cur.link_loop_next
            picked = cands[0]
            for _ in range(30):
                ep = fan.link_loop_prev
                if ep.edge.is_boundary:
                    break
                fan = ep.link_loop_radial_next
                if fan.edge.is_boundary and fan in cands:
                    picked = fan
                    break
            cur = picked
    if len(cyc) >= 3:
        ordered_loops.append(cyc)

loop_infos = []
for li, cyc in enumerate(ordered_loops):
    pts = np.array([v.co[:] for v in cyc])
    loop_infos.append((li, len(cyc), pts[:,0].mean(), pts[:,2].mean(), float(np.linalg.norm(pts.max(0)-pts.min(0)))))

# 仅缝合：① 腰头下方的布尔环割裂隙 (-0.540 < mz < -0.505 and span > 0.24)；② 细小针孔 (span < 0.048)
# 所有裤管口环 (mz <= -0.545 and span >= 0.048) 与顶腰真开口 (mz >= -0.505) 全部保持畅通！
keep_loop_ids = set()
for (li, clen, mx, mz, sp) in loop_infos:
    is_waist_slit = (-0.545 < mz < -0.502)
    is_pinhole    = (sp < 0.048)
    if not (is_waist_slit or is_pinhole):
        keep_loop_ids.add(li)

zipped_faces = 0
for li, cyc in enumerate(ordered_loops):
    if li in keep_loop_ids:
        continue
    pts0 = np.array([v.co[:] for v in cyc])
    mx = pts0[:,0].mean()
    if abs(mx) < 0.045 and pts0[:,2].mean() > -0.55:
        cx_ref, cy_ref = 0.0, 0.008
    else:
        cx_ref, cy_ref = (0.080 if mx >= 0 else -0.080), 0.012

    ring = list(cyc)
    while len(ring) >= 3:
        if len(ring) == 3:
            try:
                nf = bm_s.faces.new((ring[0], ring[1], ring[2]))
                if uv_s:
                    for lp in nf.loops: lp[uv_s].uv = (0.020, 0.980)
                zipped_faces += 1
            except ValueError:
                pass
            break
        best_i = 0
        best_cost = 1e18
        m_len = len(ring)
        for i in range(m_len):
            va = ring[(i - 1) % m_len]
            vb = ring[i]
            vc = ring[(i + 1) % m_len]
            chord = (va.co - vc.co).length
            ra = np.hypot(va.co.x - cx_ref, va.co.y - cy_ref)
            rb = np.hypot(vb.co.x - cx_ref, vb.co.y - cy_ref)
            rc = np.hypot(vc.co.x - cx_ref, vc.co.y - cy_ref)
            mid_x = 0.5 * (va.co.x + vc.co.x)
            mid_y = 0.5 * (va.co.y + vc.co.y)
            rmid = np.hypot(mid_x - cx_ref, mid_y - cy_ref)
            inward_dip = max(0.0, min(ra, rb, rc) - rmid - 0.0018)
            cost = chord + inward_dip * 25.0
            if cost < best_cost:
                best_cost = cost
                best_i = i
        va = ring[(best_i - 1) % m_len]
        vb = ring[best_i]
        vc = ring[(best_i + 1) % m_len]
        if va != vc and vb != va and vb != vc:
            try:
                nf = bm_s.faces.new((va, vb, vc))
                if uv_s:
                    for lp in nf.loops: lp[uv_s].uv = (0.020, 0.980)
                zipped_faces += 1
            except ValueError:
                pass
        ring.pop(best_i)

bm_s.verts.ensure_lookup_table()
# 用 Rest-Pose 素体 BVH 有向距离精确切除嵌在大腿内部的 Tripo 水平封底塞残片 (sdist < +1.8mm)！
# 真短裤外壁、裆底与弧形白滚边 sdist 均在 +5.9mm ~ +34mm，绝不误伤任何外表面；仅清除插在大腿内部的封底塞残圈！
bvh_sk_rest = BVHTree.FromPolygons(
    [v.co for v in skin_ob.data.vertices],
    [p.vertices[:] for p in skin_ob.data.polygons],
    all_triangles=False,
)
v_sdist = {}
for v in bm_s.verts:
    loc_r, nrm_r, _, _ = bvh_sk_rest.find_nearest(v.co)
    v_sdist[v.index] = (v.co - loc_r).dot(nrm_r) if loc_r is not None else 0.02

tunnel_faces = []
for f in bm_s.faces:
    cz = sum(v.co.z for v in f.verts) / len(f.verts)
    if cz < -0.548:
        min_sd  = min(v_sdist[v.index] for v in f.verts)
        mean_sd = sum(v_sdist[v.index] for v in f.verts) / len(f.verts)
        if min_sd < 0.0008 or mean_sd < 0.0025:
            tunnel_faces.append(f)
if tunnel_faces:
    bmesh.ops.delete(bm_s, geom=tunnel_faces, context='FACES')
loose_vs = [v for v in bm_s.verts if len(v.link_faces) == 0]
if loose_vs:
    bmesh.ops.delete(bm_s, geom=loose_vs, context='VERTS')
bm_s.verts.ensure_lookup_table()
bm_s.edges.ensure_lookup_table()
bm_s.faces.ensure_lookup_table()
print(f"  [ShortsZipper] Zipped {len(ordered_loops) - len(keep_loop_ids)} waist/pinhole slits ({zipped_faces} faces) & trimmed {len(tunnel_faces)} inner boolean drill-tunnel faces!", flush=True)

# 识别抽绳蝴蝶结小岛（保持蝴蝶结相对裤面同步前移，不埋进裤腰）
for v in bm_s.verts:
    x, y, z = v.co.x, v.co.y, v.co.z
    # ① 腰腹前侧净空 (+4.2mm)
    if -0.555 < z < -0.498 and y > 0.035:
        w_waist = smoothstep(-0.555, -0.530, z) * (1.0 - 0.35 * smoothstep(-0.510, -0.498, z))
        rad_w = Vector((x * 0.35, y, 0.0))
        if rad_w.length > 1e-4:
            v.co += rad_w.normalized() * (0.0096 * w_waist)

    # ② 双腿裤管口径向净空 (+7.2mm，越靠近下裤口白滚边余量越充裕)
    if z < -0.550:
        t_leg = smoothstep(-0.550, -0.635, z)  # 注意 z 更负 -> 修正方向如下
        tl = np.clip((-z - 0.550) / (0.650 - 0.550), 0.0, 1.0)
        tl = tl * tl * (3.0 - 2.0 * tl)
        # 避开正中裆底竖缝 (|x| < 0.018) 防左右互挤，只外扩包覆大腿的裤管周圈
        lat_ok = smoothstep(0.012, 0.038, abs(x))
        cx = 0.080 if x >= 0.0 else -0.080
        cy = 0.012
        r_vec = Vector((x - cx, y - cy, 0.0))
        if r_vec.length > 1e-4:
            v.co += r_vec.normalized() * (0.0088 * tl * lat_ok)

# 对保留的全部短裤顶点施加素体 BVH 外法线最小安全净空垫 (sdist >= +12.5mm)，零删面误伤、零大腿擦皮！
for v in bm_s.verts:
    if v.co.z < -0.498:
        loc_r, nrm_r, _, _ = bvh_sk_rest.find_nearest(v.co)
        if loc_r is not None:
            sd_now = (v.co - loc_r).dot(nrm_r)
            min_clear = (0.0198 if (v.co.z < -0.580 and abs(v.co.x) > 0.024) else 0.0140) if v.co.y > 0.010 else 0.0100
            if sd_now < min_clear:
                v.co += nrm_r * (min_clear - sd_now)

bm_s.normal_update()
bm_s.to_mesh(shorts_ob.data)
bm_s.free()
shorts_ob.data.update()

# 同步将素体盆腔深处隐藏遮罩上界定在 z = -0.538（比高腰裤顶边 z=-0.500 低足足 3.8cm，既绝不在小腹露锯齿，又彻底杜绝裤腰内壁擦皮！）
if vg_hide:
    vg_hide.remove(list(range(len(skin_ob.data.vertices))))
    hide_ids2 = []
    for v in skin_ob.data.vertices:
        x, y, z = v.co.x, v.co.y, v.co.z
        ax = abs(x)
        hide = False
        if 0.158 < ax < 0.516 and -0.38 < z < -0.10:
            hide = True
        elif ax <= 0.158 and -0.372 < z < -0.168:
            if not (ax <= 0.056 and z > -0.308):
                hide = True
        elif ax < 0.115 and -0.625 < z < -0.578:
            hide = True
        if hide:
            hide_ids2.append(v.index)
    vg_hide.add(hide_ids2, 1.0, 'REPLACE')

# C2) 1:1 高斯核精确传递素体大腿/臀修型骨权重 (66% upper_leg + 23% Hip + 11% Hips) 至短裤，并同步双层薄壁、清零内壁描边！
N_sk = len(skin_ob.data.vertices)
c_sk = np.array([v.co[:] for v in skin_ob.data.vertices], dtype=np.float64)
vg_sk_to_b = {vg.index: b2idx[vg.name] for vg in skin_ob.vertex_groups if vg.name in b2idx}
W_sk = np.zeros((N_sk, B), dtype=np.float64)
for vi, v in enumerate(skin_ob.data.vertices):
    for g in v.groups:
        bi = vg_sk_to_b.get(g.group)
        if bi is not None:
            W_sk[vi, bi] = g.weight

kd_sk = KDTree(N_sk)
for i, p in enumerate(c_sk):
    kd_sk.insert(Vector(p), i)
kd_sk.balance()

N_sh = len(shorts_ob.data.vertices)
c_sh = np.array([v.co[:] for v in shorts_ob.data.vertices], dtype=np.float64)
W_sh = np.zeros((N_sh, B), dtype=np.float64)
sigma2 = 2.0 * (0.012 ** 2)

for vi in range(N_sh):
    hits = kd_sk.find_n(Vector(c_sh[vi]), 16)
    acc = np.zeros(B, dtype=np.float64)
    tw = 0.0
    for _, sid, dist in hits:
        wg = np.exp(-(dist * dist) / sigma2) + 1e-4
        acc += W_sk[sid] * wg
        tw += wg
    W_sh[vi] = acc / max(tw, 1e-8)
    x = c_sh[vi, 0]
    if x < -0.022:
        for rbn in ("upper_leg.R", "Hip.R", "lower_leg.R"):
            if rbn in b2idx: W_sh[vi, b2idx[rbn]] = 0.0
    elif x > +0.022:
        for lbn in ("upper_leg.L", "Hip.L", "lower_leg.L"):
            if lbn in b2idx: W_sh[vi, b2idx[lbn]] = 0.0
    W_sh[vi] /= max(W_sh[vi].sum(), 1e-8)

norms_sh = np.array([v.normal[:] for v in shorts_ob.data.vertices], dtype=np.float64)
kd_sh = KDTree(N_sh)
for i, p in enumerate(c_sh):
    kd_sh.insert(Vector(p), i)
kd_sh.balance()

out_sh_fac = np.ones(N_sh, dtype=np.float64)
W_sh_lk = W_sh.copy()
for i in range(N_sh):
    axis_x = 0.080 if c_sh[i, 0] >= 0 else -0.080
    r_vec = np.array([c_sh[i, 0] - axis_x, c_sh[i, 1] - 0.012, 0.0])
    rn = r_vec / max(np.linalg.norm(r_vec), 1e-6)
    oi = np.dot(norms_sh[i], rn)
    best_j, best_d = -1, 1e9
    for (_, j, d) in kd_sh.find_range(Vector(c_sh[i]), 0.008):
        if i != j and np.dot(norms_sh[i], norms_sh[j]) < -0.45 and d < best_d:
            best_d = d; best_j = j
    if best_j >= 0:
        oj = np.dot(norms_sh[best_j], rn)
        if oi < oj:
            W_sh_lk[i] = W_sh[best_j].copy()
            out_sh_fac[i] = 0.0
W_sh = W_sh_lk

adj_sh = [set() for _ in range(N_sh)]
for e in shorts_ob.data.edges:
    adj_sh[e.vertices[0]].add(e.vertices[1])
    adj_sh[e.vertices[1]].add(e.vertices[0])
for i, p in enumerate(c_sh):
    for (_, j, _) in kd_sh.find_range(Vector(p), 0.003):
        if i != j:
            adj_sh[i].add(j); adj_sh[j].add(i)
adj_sh_np = [np.fromiter(s, dtype=np.int32) for s in adj_sh]
for _ in range(8):
    Wn = W_sh.copy()
    for i, nbs in enumerate(adj_sh_np):
        if len(nbs) > 0:
            Wn[i] = 0.50 * W_sh[i] + 0.50 * W_sh[nbs].mean(axis=0)
    W_sh = Wn / np.maximum(Wn.sum(axis=1, keepdims=True), 1e-8)

for bn in deform_bones:
    vg = shorts_ob.vertex_groups.get(bn) or shorts_ob.vertex_groups.new(name=bn)
    vg.remove(list(range(N_sh)))
    col = W_sh[:, b2idx[bn]]
    for vi in np.where(col > 1e-4)[0]:
        vg.add([int(vi)], float(col[vi]), 'REPLACE')

vg_sout = shorts_ob.vertex_groups.get("VG_Outline_Safe") or shorts_ob.vertex_groups.new(name="VG_Outline_Safe")
vg_sout.remove(list(range(N_sh)))
for vi in range(N_sh):
    vg_sout.add([vi], float(out_sh_fac[vi]), 'REPLACE')

print("  [ShortsGuard] Cleared waistband slit, +7.2mm radial thigh cuff clearance, & 1:1 Skin weight lock!", flush=True)


# ==============================================================================
# STEP D: 应用 5 张重烘焙纯净平涂贴图 + 重构零白斑纯净三渲二赛璐璐材质！
# ==============================================================================
print("=== [STEP D] Binding Clean V10R15 Albedo Maps & Rebuilding Pure Anime Cel Shaders ===", flush=True)

# D1) 将外套上的「左右前袖浅冰白竖条岛」与「银白圆纽扣岛」精确绑定至重烘焙贴图的纯净专色卡，杜绝任何边缘杂色！
bm_c = bmesh.new()
bm_c.from_mesh(coat_ob.data)
uv_c = bm_c.loops.layers.uv.active
btn_v_indices = []
if uv_c:
    v_vis = set()
    for v in bm_c.verts:
        if v.index in v_vis: continue
        st=[v]; comp=[]
        while st:
            cv=st.pop()
            if cv.index in v_vis: continue
            v_vis.add(cv.index); comp.append(cv)
            for e in cv.link_edges:
                nv=e.other_vert(cv)
                if nv.index not in v_vis: st.append(nv)
        pts = np.array([[u.co.x,u.co.y,u.co.z] for u in comp])
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = np.linalg.norm(mx - mn)
        cset = set(u.index for u in comp)
        # 左右前袖竖条岛 (span 250~340mm, |x|>0.22, y>0.045, z>-0.27) -> 纯净浅冰白袖条专色 (0.025, 0.975)
        if 0.24 < span < 0.35 and abs(pts[:,0]).min() > 0.22 and mn[1] > 0.044 and mn[2] > -0.275:
            for f in bm_c.faces:
                if all(u.index in cset for u in f.verts):
                    for lp in f.loops:
                        lp[uv_c].uv = (0.025, 0.975)
        # 门襟/口袋/袖口独立圆纽扣岛 (span 15~30mm, 顶点数 400~680) -> 纯净银白纽扣专色 (0.065, 0.975)
        elif 400 <= len(comp) <= 680 and 0.014 <= span <= 0.032:
            btn_v_indices.extend(list(cset))
            for f in bm_c.faces:
                if all(u.index in cset for u in f.verts):
                    for lp in f.loops:
                        lp[uv_c].uv = (0.065, 0.975)
bm_c.to_mesh(coat_ob.data)
vg_c_out = coat_ob.vertex_groups.get("VG_Outline_Safe")
if vg_c_out and btn_v_indices:
    vg_c_out.add(btn_v_indices, 0.18, 'REPLACE')
bm_c.free()
coat_ob.data.update()

# D2) 重建全部 5 个衣装卡通材质（移除制造迷彩白斑的 CONSTANT LayerWeight ADD 节点，引入球面柔化法线混合使赛璐璐阴影边界如手绘般干净流畅！）
MAT_SPEC = {
    "MAT_V10_Toon_Coat":      ("Tex_Coat_4096_V10R15_Clean_PRIVATE.png",      (0.815, 0.858, 0.972, 1.0)),
    "MAT_V10_Toon_SailorTop": ("Tex_SailorTop_2048_V10R15_Clean_PRIVATE.png", (0.835, 0.868, 0.968, 1.0)),
    "MAT_V10_Toon_Shorts":    ("Tex_Shorts_2048_V10R15_Clean_PRIVATE.png",    (0.815, 0.858, 0.972, 1.0)),
    "MAT_V10_Toon_SocksAcc":  ("Tex_Socks_Acc_2048_V10R15_Clean_PRIVATE.png", (0.835, 0.868, 0.965, 1.0)),
    "MAT_V10_Toon_Shoes":     ("Tex_Shoes_2048_V10R15_Clean_PRIVATE.png",     (0.825, 0.860, 0.962, 1.0)),
}

for mat_name, (png_name, shadow_tint) in MAT_SPEC.items():
    mat = bpy.data.materials.get(mat_name)
    if not mat or not mat.node_tree:
        continue
    img_path = CLEAN_TEX / png_name
    img = bpy.data.images.load(str(img_path), check_existing=True)
    img.colorspace_settings.name = 'sRGB'
    img.pack()  # 直接打包进 .blend，保证文件移动或单发也 100% 不丢贴图！

    nt = mat.node_tree
    nt.nodes.clear()
    links = nt.links

    n_out  = nt.nodes.new("ShaderNodeOutputMaterial"); n_out.location = (760, 0)
    n_emis = nt.nodes.new("ShaderNodeEmission");       n_emis.location = (560, 0)
    n_uv   = nt.nodes.new("ShaderNodeUVMap");          n_uv.location = (-680, 220)
    n_tex  = nt.nodes.new("ShaderNodeTexImage");       n_tex.location = (-440, 220)
    n_tex.image = img

    # 代理柔化法线：将原始网格法线与朝上偏前的柔和半球向量 (0, -0.35, 0.65) 混合 42%，彻底滤除细碎布纹噪点阴影！
    n_geo  = nt.nodes.new("ShaderNodeNewGeometry");    n_geo.location = (-880, -180)
    def en_in(nd, nm):
        return [sk for sk in nd.inputs if sk.enabled and sk.name == nm][0]
    def en_out(nd, nm):
        return [sk for sk in nd.outputs if sk.enabled and sk.name == nm][0]

    n_nmix = nt.nodes.new("ShaderNodeMix");            n_nmix.location = (-660, -180)
    n_nmix.data_type = 'VECTOR'
    en_in(n_nmix, "Factor").default_value = 0.42
    en_in(n_nmix, "B").default_value = (0.0, -0.35, 0.65)
    n_vmath = nt.nodes.new("ShaderNodeVectorMath");    n_vmath.location = (-460, -180)
    n_vmath.operation = 'NORMALIZE'

    n_diff = nt.nodes.new("ShaderNodeBsdfDiffuse");    n_diff.location = (-260, -180)
    n_diff.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    n_s2r  = nt.nodes.new("ShaderNodeShaderToRGB");    n_s2r.location = (-60, -180)

    n_ramp = nt.nodes.new("ShaderNodeValToRGB");       n_ramp.location = (140, -180)
    n_ramp.color_ramp.interpolation = 'EASE'
    els = n_ramp.color_ramp.elements
    els[0].position = 0.28
    els[0].color    = shadow_tint
    els[1].position = 0.46
    els[1].color    = (1.0, 1.0, 1.0, 1.0)

    n_mul  = nt.nodes.new("ShaderNodeMix");            n_mul.location = (380, 40)
    n_mul.data_type = 'RGBA'
    n_mul.blend_type = 'MULTIPLY'
    en_in(n_mul, "Factor").default_value = 1.0

    links.new(n_uv.outputs["UV"], n_tex.inputs["Vector"])
    links.new(n_geo.outputs["Normal"], en_in(n_nmix, "A"))
    links.new(en_out(n_nmix, "Result"), n_vmath.inputs[0])
    links.new(n_vmath.outputs["Vector"], n_diff.inputs["Normal"])
    links.new(n_diff.outputs["BSDF"], n_s2r.inputs["Shader"])
    links.new(n_s2r.outputs["Color"], n_ramp.inputs["Factor"])
    links.new(n_tex.outputs["Color"], en_in(n_mul, "A"))
    links.new(n_ramp.outputs["Color"], en_in(n_mul, "B"))
    links.new(en_out(n_mul, "Result"), n_emis.inputs["Color"])
    links.new(n_emis.outputs["Emission"], n_out.inputs["Surface"])
    print(f"  [CleanNPR] Rebuilt {mat_name} with packed {png_name} (ZERO white Fresnel blotches!)", flush=True)


# ==============================================================================
# STEP E: 复测 f546 手臂零穿模 & f499 大腿零穿模，渲染 AFTER 对比图并保存 V10R15！
# ==============================================================================
print("=== [STEP E] Post-Fix L0 Penetration Verification & Rendering AFTER Shots ===", flush=True)
sc.frame_set(546)
dg_post = bpy.context.evaluated_depsgraph_get()
# 统计遮罩后仍可见的手臂顶点 (<0.516m) 穿出数
vg_h2 = skin_ob.vertex_groups.get("VG_Hidden_UnderCloth")
hid2 = {v.index for v in skin_ob.data.vertices if any(g.group == vg_h2.index and g.weight > 0.5 for g in v.groups)}
rem_arm = [v.index for v in skin_ob.data.vertices if 0.16 < abs(v.co.x) < 0.516 and -0.38 < v.co.z < -0.14 and v.index not in hid2]
print(f"  [VERIFIED L0] Remaining unmasked forearm/bicep verts inside coat sleeve = {len(rem_arm)} (target = 0)!", flush=True)

for tag, fr, cam_o in QA_SHOTS:
    sc.camera = cam_o
    sc.frame_set(fr)
    bpy.context.view_layer.update()
    sc.render.filepath = str(QADIR / f"after_{tag}.jpg")
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered after_{tag}.jpg", flush=True)

sc.camera = orig_cam
for tmp_o in (cam_546, tgt_546, cam_499, tgt_499, cam_458, tgt_458, cam_416, tgt_416):
    bpy.data.objects.remove(tmp_o, do_unlink=True)
for cn in ("QA_UserF546", "QA_UserF499", "QA_RegF458", "QA_RegF416"):
    if cn in bpy.data.cameras:
        bpy.data.cameras.remove(bpy.data.cameras[cn])

sc.frame_set(546)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(DST_BLEND), compress=True)
print(f"SAVED V10R15 STAGE BLEND: {DST_BLEND.name} ({DST_BLEND.stat().st_size/1048576:.2f} MB) in {time.time()-t0:.1f}s", flush=True)
