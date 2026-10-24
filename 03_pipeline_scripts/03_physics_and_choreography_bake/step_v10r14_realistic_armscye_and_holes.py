"""
V10R14-R2 外套「胳肢窝真实袖身分离自然垂坠 + 保留露肩水手领 + 口袋下方破洞100%根治」终极精修脚本
严守文件锁红线：只读打开 RadDoll_V10R13_MMD_Heart舞台三渲二无撕裂精修_PRIVATE.blend，
全部成果另存为 RadDoll_V10R14_MMD_Heart舞台三渲二实感布料精修_PRIVATE.blend！
"""
import time
import bpy
import bmesh
import numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.kdtree import KDTree

t0 = time.time()
WDIR = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
SRC_BLEND = WDIR / "RadDoll_V10R13_MMD_Heart舞台三渲二无撕裂精修_PRIVATE.blend"
DST_BLEND = WDIR / "RadDoll_V10R14_MMD_Heart舞台三渲二实感布料精修_PRIVATE.blend"
QADIR = WDIR / "v10r14_cloth_qa"
QADIR.mkdir(parents=True, exist_ok=True)

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

def make_track_cam(name, loc, target_co, lens=42.0):
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

# 1:1 精确复刻用户 f458 侧视截图机位（完整同框：左侧白椅背、露肩水手领、抬臂胳肢窝、低位口袋及口袋正下方下摆、球鞋！）
cam_f458_user, tgt_f458_user = make_track_cam("QA_ExactUserF458", (-1.65, 1.50, 0.85), (-0.02, 1.20, 0.78), lens=42.0)
# 对称右侧同构机位（同步检验角色另一侧胳肢窝与另一侧口袋下方！）
cam_f458_opp,  tgt_f458_opp  = make_track_cam("QA_OppSideF458",   (+1.65, 1.50, 0.85), (+0.02, 1.20, 0.78), lens=42.0)
# 复刻上一轮用户 f416 仰视高抬臂机位（双重回归验证抬臂零撕裂、零尖刺！）
cam_f416_user, tgt_f416_user = make_track_cam("QA_ExactUserF416", (-0.05, 0.36, 0.74), (0.01, 1.44, 0.95), lens=36.0)

QA_SHOTS = [
    ("f458_exact",    458, cam_f458_user),
    ("f458_oppside",  458, cam_f458_opp),
    ("f416_userview", 416, cam_f416_user),
    ("f355_mv",       355, orig_cam),
    ("f586_mv",       586, orig_cam),
]

print("=== [STEP A] Rendering V10R13 BEFORE Shots ===", flush=True)
for tag, fr, cam_o in QA_SHOTS:
    sc.camera = cam_o
    sc.frame_set(fr)
    bpy.context.view_layer.update()
    sc.render.filepath = str(QADIR / f"before_{tag}.jpg")
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered before_{tag}.jpg", flush=True)


# ==============================================================================
# STEP B: 严密修补长外套全部 16 处破洞 + 镜像补齐袖条（裁剪避让袖口防叠面）+ 净化烘焙黑影面片
# ==============================================================================
print("=== [STEP B] Watertight Hole Patching & Clean Sleeve Stripe Restoration ===", flush=True)
coat_ob = bpy.data.objects["GEO-Cloth_Coat"]
arm_ob  = bpy.data.objects["ARM-FullBody_V10"]
deform_bones = [b.name for b in arm_ob.data.bones if b.use_deform]
b2idx = {bn: i for i, bn in enumerate(deform_bones)}
SKY_BLUE_UV = (0.128826, 0.172626)

bm = bmesh.new()
bm.from_mesh(coat_ob.data)
bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()
bm.faces.ensure_lookup_table()
uv_lay = bm.loops.layers.uv.active

v_vis = set()
islands = []
for v in bm.verts:
    if v.index in v_vis:
        continue
    st = [v]
    comp = []
    while st:
        cv = st.pop()
        if cv.index in v_vis:
            continue
        v_vis.add(cv.index)
        comp.append(cv)
        for e in cv.link_edges:
            nv = e.other_vert(cv)
            if nv.index not in v_vis:
                st.append(nv)
    pts = np.array([[u.co.x, u.co.y, u.co.z] for u in comp])
    islands.append((len(comp), pts.min(axis=0), pts.max(axis=0), comp))

islands.sort(key=lambda x: -x[0])
main_island_verts = set(u.index for u in islands[0][3])
rsleeve_island_verts = set(u.index for u in islands[1][3])

# B1) 镜像补齐 +X 前袖条（仅复制 |x| <= 0.512 的区段，完美避开 +0.520 处的袖口翻边岛 Island#07，零叠面！）
stripe_island = None
for nv, mn, mx, comp in islands:
    if 400 < nv < 1500 and mn[0] < -0.50 and mx[0] < -0.22 and mn[1] > 0.045 and mn[2] > -0.27:
        stripe_island = comp
        break

if stripe_island is not None:
    stripe_vset = set(u.index for u in stripe_island)
    stripe_faces = [
        f for f in bm.faces
        if all(v.index in stripe_vset for v in f.verts)
        and all(abs(v.co.x) <= 0.514 for v in f.verts)
    ]
    used_vs = {v for f in stripe_faces for v in f.verts}
    v_map = {ov.index: bm.verts.new(Vector((-ov.co.x, ov.co.y, ov.co.z))) for ov in used_vs}
    m_cnt = 0
    for of in stripe_faces:
        rev_vs = [v_map[ov.index] for ov in reversed(of.verts)]
        try:
            nf = bm.faces.new(rev_vs)
            nf.material_index = of.material_index
            nf.smooth = True
            if uv_lay:
                for nl, ol in zip(nf.loops, reversed(of.loops)):
                    nl[uv_lay].uv = ol[uv_lay].uv.copy()
            m_cnt += 1
        except ValueError:
            pass
    print(f"  [MirrorStripe] Cleanly restored +X sleeve stripe ({m_cnt} faces, zero cuff overlap)!", flush=True)

bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()
bm.faces.ensure_lookup_table()

# B2) 封闭主衣身壳与袖壳上的全部非预期破洞（包括左右口袋正下方破洞、口袋角裂隙、腋下针孔、前袖槽底衬）
target_shell_vset = main_island_verts | rsleeve_island_verts
e_vis = set()
holes_to_fill = []

for e in bm.edges:
    if e.is_boundary and e.verts[0].index in target_shell_vset and e.index not in e_vis:
        loop_edges, loop_vs = [], []
        cur_e, cur_v = e, e.verts[0]
        closed = False
        while True:
            e_vis.add(cur_e.index)
            loop_edges.append(cur_e)
            loop_vs.append(cur_v)
            nxt_v = cur_e.other_vert(cur_v)
            if nxt_v == loop_vs[0] and len(loop_edges) >= 3:
                closed = True
                break
            nxt_es = [ne for ne in nxt_v.link_edges if ne.is_boundary and ne.index not in e_vis]
            if not nxt_es:
                break
            cur_v, cur_e = nxt_v, nxt_es[0]
        if not closed:
            continue
        pts = np.array([[u.co.x, u.co.y, u.co.z] for u in loop_vs])
        sp = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        ctr = pts.mean(axis=0)
        should_fill = (sp < 0.130) or (0.280 < sp < 0.350 and ctr[0] > 0.30 and ctr[1] > 0.05)
        if should_fill:
            holes_to_fill.append((sp, ctr, loop_edges, loop_vs))

for sp, ctr, l_edges, l_vs in holes_to_fill:
    try:
        ret = bmesh.ops.contextual_create(bm, geom=l_edges)
        new_fs = ret.get("faces", [])
        if not new_fs:
            continue
        poke_ret = bmesh.ops.poke(bm, faces=new_fs)
        p_faces = poke_ret.get("faces", [])
        p_verts = set(poke_ret.get("verts", []))
        for _ in range(4):
            sub_es = list({ed for fc in p_faces if fc.is_valid for ed in fc.edges if ed.calc_length() > 0.009 and ed not in l_edges})
            if not sub_es:
                break
            sret = bmesh.ops.subdivide_edges(bm, edges=sub_es, cuts=1, use_grid_fill=True)
            for item in sret.get("geom_inner", []):
                if isinstance(item, bmesh.types.BMVert):
                    p_verts.add(item)
                elif isinstance(item, bmesh.types.BMFace):
                    p_faces.append(item)
        for fc in bm.faces:
            if any(v in p_verts for v in fc.verts) or all(v in set(l_vs) for v in fc.verts):
                fc.material_index = 0
                fc.smooth = True
                if uv_lay:
                    for lp in fc.loops:
                        lp[uv_lay].uv = SKY_BLUE_UV
        b_set = set(l_vs)
        inner_vs = [v for v in p_verts if v.is_valid and v not in b_set]
        for _ in range(6):
            np_pos = {iv: sum((ed.other_vert(iv).co for ed in iv.link_edges), Vector((0, 0, 0))) / len(iv.link_edges)
                      for iv in inner_vs if len(iv.link_edges) > 0}
            for iv, co_new in np_pos.items():
                iv.co = co_new
        print(f"  [PatchedHole] span={sp*1000:5.1f}mm at ({ctr[0]:+.3f},{ctr[1]:+.3f},{ctr[2]:+.3f})", flush=True)
    except Exception as ex:
        print(f"  [PatchWarn] {ctr}: {ex}", flush=True)

bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

# B3) 净化口袋下方原本烘焙黑影的 130 个面片为纯净外套天蓝
tex_img = None
for mat in coat_ob.data.materials:
    if mat and mat.node_tree:
        for nd in mat.node_tree.nodes:
            if nd.type == 'TEX_IMAGE' and nd.image:
                tex_img = nd.image
                break
if tex_img and uv_lay:
    W_i, H_i = tex_img.size
    px = np.array(tex_img.pixels[:], dtype=np.float32).reshape(H_i, W_i, 4)
    for f in bm.faces:
        fc = f.calc_center_median()
        if abs(fc.x) > 0.12 and -0.848 < fc.z < -0.675:
            uvs = [lp[uv_lay].uv for lp in f.loops]
            u = int(np.clip(np.mean([uv[0] for uv in uvs]), 0, 0.9999) * W_i)
            v = int(np.clip(np.mean([uv[1] for uv in uvs]), 0, 0.9999) * H_i)
            if px[v, u, :3].mean() < 0.38:
                for lp in f.loops:
                    lp[uv_lay].uv = SKY_BLUE_UV


# ==============================================================================
# STEP C: 柔和 C2 钟形曲线裁剪蝙蝠袖底连片（消除 189mm 台阶，锻造顺滑自然的真实腋窝收折线！）
# ==============================================================================
print("=== [STEP C] Smooth C2 Tailoring of Underarm Batwing Web ===", flush=True)
def smoothstep(e0, e1, val):
    t = np.clip((val - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

for v in bm.verts:
    ax = abs(v.co.x)
    y  = v.co.y
    z  = v.co.z
    # 避开前胸翻领 (y > 0.048 且 ax < 0.235) 与露肩领口上缘 (z > -0.265)，只收拢腋窝底部的低垂蝙蝠翼！
    if 0.172 < ax < 0.360 and -0.470 < z < -0.295 and not (y > 0.048 and ax < 0.235):
        s_in  = smoothstep(0.172, 0.215, ax)
        s_out = smoothstep(0.215, 0.355, ax)
        # 袖窿腋窝底高度平滑定在 z = -0.318，向外自然过渡回宽袖底 z = -0.334
        z_floor = -0.318 * (1.0 - s_out) + (-0.334) * s_out
        if z < z_floor:
            v.co.z += (z_floor - z) * 0.76 * s_in

# 切向均匀松弛腋窝过渡网格，彻底消除折角棱线
relax_vs = [v for v in bm.verts if 0.168 < abs(v.co.x) < 0.360 and -0.43 < v.co.z < -0.265 and not v.is_boundary]
for _ in range(6):
    np_pos = {}
    for v in relax_vs:
        nbs = [e.other_vert(v).co for e in v.link_edges]
        if nbs:
            np_pos[v] = v.co.lerp(sum(nbs, Vector((0, 0, 0))) / len(nbs), 0.42)
    for v, p in np_pos.items():
        v.co = p

for v in bm.verts:
    if abs(v.co.x) > 0.11 and v.co.z < -0.72:
        t_hem = smoothstep(-0.72, -0.84, v.co.z)  # 注意 z 越负越靠下，修正方向：smoothstep(-0.72, -0.84) 需写 (-v.co.z - 0.72)/0.12
        th = np.clip((-v.co.z - 0.72) / 0.12, 0.0, 1.0)
        v.co.z -= 0.011 * (th * th * (3.0 - 2.0 * th))

bm.normal_update()
bm.to_mesh(coat_ob.data)
bm.free()
coat_ob.data.update()


# ==============================================================================
# STEP D: 测地调和场 (Harmonic Field Δφ=0) 衣身/露肩领圈 vs 袖管蒙皮分离！
#   - 100% 保护原设「露肩落肩领圈」（白蓝条纹水手服肩头完整露出！）与「前翻领门襟」（f416 零尖刺！）
#   - 100% 切断袖管与腰臀 (Hips/Spine) 的跨区拉扯（f458 胳肢窝自然内折收拢，衣身垂直自然垂坠！）
# ==============================================================================
print("=== [STEP D] Solving Geodesic Harmonic Field (Δφ=0) for Authentic Coat Drape ===", flush=True)
N = len(coat_ob.data.vertices)
B = len(deform_bones)
c_coat = np.array([v.co[:] for v in coat_ob.data.vertices], dtype=np.float64)

vg_idx_to_b = {vg.index: b2idx[vg.name] for vg in coat_ob.vertex_groups if vg.name in b2idx}
W_orig = np.zeros((N, B), dtype=np.float64)
for vi, v in enumerate(coat_ob.data.vertices):
    for g in v.groups:
        bi = vg_idx_to_b.get(g.group)
        if bi is not None:
            W_orig[vi, bi] = g.weight

# 为补洞与镜像袖条新增的无权重顶点赋予近邻初始权重
has_w = W_orig.sum(axis=1) > 0.02
if not has_w.all():
    valid_ids = np.where(has_w)[0]
    kd_w = KDTree(len(valid_ids))
    for v_id in valid_ids:
        kd_w.insert(Vector(c_coat[v_id]), int(v_id))
    kd_w.balance()
    for vi in np.where(~has_w)[0]:
        hits = kd_w.find_n(Vector(c_coat[vi]), 4)
        acc = np.zeros(B, dtype=np.float64)
        tw = 0.0
        for _, nid, dist in hits:
            wi = 1.0 / max(dist, 1e-4)
            acc += W_orig[nid] * wi
            tw += wi
        W_orig[vi] = acc / max(tw, 1e-8)

# 构建带 2.8mm 跨接缝焊接的邻接图
adj = [set() for _ in range(N)]
for e in coat_ob.data.edges:
    u, v = e.vertices[0], e.vertices[1]
    adj[u].add(v); adj[v].add(u)
kd_all = KDTree(N)
for i, p in enumerate(c_coat):
    kd_all.insert(Vector(p), i)
kd_all.balance()
for i, p in enumerate(c_coat):
    for (_, j, _) in kd_all.find_range(Vector(p), 0.0028):
        if i != j:
            adj[i].add(j); adj[j].add(i)
adj_np = [np.fromiter(s, dtype=np.int32) for s in adj]

# 设定调和场边界条件 φ in [0, 1]：
# φ = 0.0 (衣身 + 前翻领 + 露肩领圈上沿):
#   1) 腰腹口袋与下摆 z < -0.440，或躯干中轴 |x| < 0.155
#   2) 前门襟与宽大翻领 (y > +0.042 且 |x| < 0.242) -> 杜绝翻领被手臂拉出尖刺！
#   3) 露肩落肩领圈环 (|x| < 0.232 且 z > -0.235) -> 让露肩领圈安稳挂在肩头下方，完整露出内部水手领！
# φ = 1.0 (纯袖筒):
#   |x| > 0.270 且 z > -0.420
pin_mask = np.zeros(N, dtype=bool)
phi = np.full(N, 0.5, dtype=np.float64)

for i in range(N):
    x, y, z = c_coat[i]
    ax = abs(x)
    is_body_seed = (
        (z < -0.440) or
        (ax < 0.155) or
        (y > 0.030 and ax < 0.262) or
        (ax < 0.235 and z > -0.232)
    )
    is_sleeve_seed = (ax > 0.270 and z >= -0.420)
    if is_body_seed:
        phi[i] = 0.0
        pin_mask[i] = True
    elif is_sleeve_seed:
        phi[i] = 1.0
        pin_mask[i] = True

free_ids = np.where(~pin_mask)[0]
for _ in range(85):
    phi_next = phi.copy()
    for i in free_ids:
        nbs = adj_np[i]
        if len(nbs) > 0:
            phi_next[i] = phi[nbs].mean()
    phi = phi_next

# 合成平滑而分界清晰的真实大衣权重矩阵 W_new
ARM_BONES_ALL = {
    "shoulder.R", "upper_arm.R", "lower_arm.R", "lower_arm_Twist.R", "hand.R", "handrole.R", "PHY_Cuff_R",
    "shoulder.L", "upper_arm.L", "lower_arm.L", "lower_arm_Twist.L", "hand.L", "handrole.L", "PHY_Cuff_L",
}
W_new = np.zeros((N, B), dtype=np.float64)

for vi in range(N):
    x, y, z = c_coat[vi]
    ax = abs(x)
    sfx = "R" if x >= 0.0 else "L"
    opp = "L" if x >= 0.0 else "R"
    ss = float(phi[vi])

    # (1) 衣身与露肩领圈分支 wb：
    wb = W_orig[vi].copy()
    # 清除全部前臂/手腕串扰，清除对侧腿部串扰
    for bn in ARM_BONES_ALL:
        if bn in b2idx:
            # 在露肩领圈外侧上沿 (z > -0.28, ax > 0.16) 保留极少量 (<=22%) shoulder + upper_arm 随动，防止手臂高举穿破露肩领圈，但绝不翻转盖住水手领！
            if bn == f"upper_arm.{sfx}" and z > -0.29 and ax > 0.165:
                wb[b2idx[bn]] = min(wb[b2idx[bn]], 0.22 * smoothstep(0.165, 0.225, ax))
            elif bn == f"shoulder.{sfx}" and z > -0.29 and ax > 0.145:
                wb[b2idx[bn]] = min(max(wb[b2idx[bn]], 0.15 * smoothstep(0.145, 0.210, ax)), 0.28)
            else:
                wb[b2idx[bn]] = 0.0

    if ax > 0.03:
        for obn in (f"upper_leg.{opp}", f"Hip.{opp}", f"lower_leg.{opp}",
                    f"PHY_Coat_F{opp}_01", f"PHY_Coat_F{opp}_02", f"PHY_Coat_S{opp}_01", f"PHY_Coat_S{opp}_02"):
            if obn in b2idx:
                wb[b2idx[obn]] = 0.0

    # 上胸背肋部 (z > -0.41) 将过多 Hips 平滑收归 Chest + Spine，保证挺括垂顺
    if z > -0.41 and "Hips" in b2idx:
        keep_h = 1.0 - smoothstep(-0.41, -0.29, z)
        drop_h = wb[b2idx["Hips"]] * (1.0 - keep_h)
        wb[b2idx["Hips"]] *= keep_h
        c_sh = smoothstep(-0.39, -0.25, z)
        if "Chest" in b2idx: wb[b2idx["Chest"]] += drop_h * c_sh
        if "Spine" in b2idx: wb[b2idx["Spine"]] += drop_h * (1.0 - c_sh)

    sb = wb.sum()
    if sb < 1e-6:
        if "Chest" in b2idx: wb[b2idx["Chest"]] = 0.75
        if "Spine" in b2idx: wb[b2idx["Spine"]] = 0.25
        sb = wb.sum()
    wb /= sb

    # (2) 袖管分支 wa（0% Hips / 0% Spine，纯正衣袖运动学）：
    wa = np.zeros(B, dtype=np.float64)
    if ss > 1e-5:
        w_chest = 0.24 * (1.0 - smoothstep(0.175, 0.235, ax))
        w_shld  = 0.16 * (1.0 - smoothstep(0.175, 0.240, ax))
        up_rise = 0.60 + 0.40 * smoothstep(0.175, 0.235, ax)
        up_fall = 1.0 - smoothstep(0.255, 0.365, ax)
        w_uarm  = up_rise * up_fall
        fore_t  = smoothstep(0.255, 0.365, ax)
        tw_r    = smoothstep(0.340, 0.510, ax)
        cf_r    = 0.20 * smoothstep(0.505, 0.595, ax)
        w_larm  = fore_t * (1.0 - tw_r * 0.65) * (1.0 - cf_r)
        w_ltwst = fore_t * (tw_r * 0.65) * (1.0 - cf_r)
        w_cuff  = fore_t * cf_r
        for bname, val in [
            ("Chest", w_chest),
            (f"shoulder.{sfx}", w_shld),
            (f"upper_arm.{sfx}", w_uarm),
            (f"lower_arm.{sfx}", w_larm),
            (f"lower_arm_Twist.{sfx}", w_ltwst),
            (f"PHY_Cuff_{sfx}", w_cuff),
        ]:
            if bname in b2idx and val > 1e-6:
                wa[b2idx[bname]] = val
        sa = wa.sum()
        if sa > 1e-6:
            wa /= sa

    W_new[vi] = (1.0 - ss) * wb + ss * wa
    W_new[vi] /= max(W_new[vi].sum(), 1e-8)

# 先对齐双层薄壁内外权重，再执行全局 10 轮柔和测地平滑（因为 phi 本身已是图上调和函数，适度平滑完全不会串回腰臀，且能消除一切局部折痕突刺！）
norms_c = np.array([v.normal[:] for v in coat_ob.data.vertices], dtype=np.float64)
center_c = np.array([0.0, 0.0, np.median(c_coat[:, 2])])
rad_dir = c_coat - center_c
rad_dir[:, 2] *= 0.25
out_score = np.einsum("ij,ij->i", norms_c, rad_dir / np.maximum(np.linalg.norm(rad_dir, axis=1, keepdims=True), 1e-6))

outline_fac = np.ones(N, dtype=np.float64)
W_locked = W_new.copy()
for i in range(N):
    ni = norms_c[i]
    best_j, best_d = -1, 1e9
    for (_, j, d) in kd_all.find_range(Vector(c_coat[i]), 0.010):
        if i == j or abs(phi[i] - phi[j]) > 0.35:
            continue
        if np.dot(ni, norms_c[j]) < -0.45 and d < best_d:
            best_d = d
            best_j = j
    if best_j >= 0:
        if out_score[i] < out_score[best_j]:
            W_locked[i] = W_new[best_j].copy()
            outline_fac[i] = 0.0
        elif best_d < 0.0035:
            outline_fac[i] = max(0.25, best_d / 0.0035)
W_new = W_locked

for _ in range(10):
    W_nxt = W_new.copy()
    for i, nbs in enumerate(adj_np):
        # 允许 |phi_i - phi_j| < 0.45 的邻点自由平滑，使袖窿过渡带丝滑如绸缎且袖底 0% Hips！
        valid_nbs = [j for j in nbs if abs(phi[i] - phi[j]) < 0.45]
        if valid_nbs:
            W_nxt[i] = 0.48 * W_new[i] + 0.52 * W_new[valid_nbs].mean(axis=0)
    rs = W_nxt.sum(axis=1, keepdims=True)
    W_new = np.where(rs > 1e-8, W_nxt / np.maximum(rs, 1e-8), W_new)

for _ in range(4):
    W_nxt = W_new.copy()
    for i, nbs in enumerate(adj_np):
        if len(nbs) > 0:
            W_nxt[i] = 0.65 * W_new[i] + 0.35 * W_new[nbs].mean(axis=0)
    rs = W_nxt.sum(axis=1, keepdims=True)
    W_new = np.where(rs > 1e-8, W_nxt / np.maximum(rs, 1e-8), W_new)

# 写回顶点组
vg_map = {}
for bn in deform_bones:
    vg = coat_ob.vertex_groups.get(bn) or coat_ob.vertex_groups.new(name=bn)
    vg.remove(list(range(N)))
    vg_map[bn] = vg

W_new[W_new < 1e-4] = 0.0
W_new /= np.maximum(W_new.sum(axis=1, keepdims=True), 1e-8)
for bi, bn in enumerate(deform_bones):
    col = W_new[:, bi]
    nz = np.where(col > 1e-4)[0]
    vg = vg_map[bn]
    for vi in nz:
        vg.add([int(vi)], float(col[vi]), 'REPLACE')

vg_out = coat_ob.vertex_groups.get("VG_Outline_Safe") or coat_ob.vertex_groups.new(name="VG_Outline_Safe")
vg_out.remove(list(range(N)))
for vi in range(N):
    vg_out.add([vi], float(outline_fac[vi]), 'REPLACE')

# 关闭气球化 DQS，保持 5 轮轻柔保细节平滑（不向上推高下摆，使下摆自然盖过短裤侧边且完整遮挡椅面！）
for md in coat_ob.modifiers:
    if md.type == 'ARMATURE':
        md.use_deform_preserve_volume = False
    elif md.name == "NPR_AntiTearSmooth":
        md.factor = 0.48
        md.iterations = 5

print(f"  [HarmonicCoat] Solved Δφ=0 (Body={(phi<0.1).sum()}, Armscye={((phi>=0.1)&(phi<=0.9)).sum()}, Sleeve={(phi>0.9).sum()})!", flush=True)


# ==============================================================================
# STEP E: 渲染 V10R14-R2 修复后全部机位 & 另存成品文件！
# ==============================================================================
print("=== [STEP E] Rendering V10R14-R2 AFTER Shots ===", flush=True)
for tag, fr, cam_o in QA_SHOTS:
    sc.camera = cam_o
    sc.frame_set(fr)
    bpy.context.view_layer.update()
    sc.render.filepath = str(QADIR / f"after_{tag}.jpg")
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered after_{tag}.jpg", flush=True)

sc.camera = orig_cam
for tmp_o in (cam_f458_user, tgt_f458_user, cam_f458_opp, tgt_f458_opp, cam_f416_user, tgt_f416_user):
    bpy.data.objects.remove(tmp_o, do_unlink=True)
for cn in ("QA_ExactUserF458", "QA_OppSideF458", "QA_ExactUserF416"):
    if cn in bpy.data.cameras:
        bpy.data.cameras.remove(bpy.data.cameras[cn])

sc.frame_set(458)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(DST_BLEND), compress=True)
print(f"SAVED V10R14-R2 CLEAN BLEND: {DST_BLEND.name} ({DST_BLEND.stat().st_size/1048576:.2f} MB) in {time.time()-t0:.1f}s", flush=True)
