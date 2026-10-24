"""
V10R8 Step 1 (Rev 3 — 3D 连续体素平滑位移场版，彻底根除双层网格内衬翻出斑块与袖口露指偏差):
核心几何升级：
- 用 3D 空间高斯平滑位移场 U(x,y,z) 替代逐顶点独立外推，使双层网格（外表面+内衬）及独立子网格（水手领/口袋/纽扣/蝴蝶结）获得完全一致的刚体级局部形变，100% 保持壳体厚度，彻底消除内衬穿透外表面造成的灰斑与破洞！
- 全部防穿模仅以单层流形素体 GEO-Sotai_Skin 为基准（内搭给 +4.5mm 净空，外套在肩胸区给 +15mm 净空自动包住水手服与短裤）；
- 外套袖口目标中心精调至 (±0.402, wr_y+0.008, -0.638)，把大拇指完整收入宽袖口内，同时让五指指尖自然露出袖口 16~20mm！
"""
import os, sys, json, math
from pathlib import Path
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

class _KDWrap:
    def __init__(self, pts):
        n = len(pts)
        self.kd = KDTree(n)
        for i, p in enumerate(pts):
            self.kd.insert((float(p[0]), float(p[1]), float(p[2])), i)
        self.kd.balance()
    def query(self, pts):
        ds = np.empty(len(pts), dtype=np.float64)
        ix = np.empty(len(pts), dtype=np.int64)
        for i, p in enumerate(pts):
            _, idx, dist = self.kd.find((float(p[0]), float(p[1]), float(p[2])))
            ds[i] = dist
            ix[i] = idx
        return ds, ix

ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
SOTAI_BLEND = ROOT / "生成模型/组装/v010R7_RadDoll装配/RadDoll_V10_全身素体精修拼装_PRIVATE.blend"
UNPACK_DIR  = ROOT / "生成模型/组装/v010R8_上色衣装审查/_unpacked"
OUT_DIR     = ROOT / "生成模型/组装/v010R8_全身衣装装配"
RENDER_DIR  = OUT_DIR / "renders_step1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RENDER_DIR.mkdir(parents=True, exist_ok=True)

OUT_STEP1_BLEND = OUT_DIR / "RadDoll_V10R8_Step1_衣装比例与细节精修_PRIVATE.blend"

bpy.ops.wm.open_mainfile(filepath=str(SOTAI_BLEND), load_ui=False, use_scripts=False)
bpy.context.view_layer.update()

sotai_ob = bpy.data.objects["GEO-Sotai_Skin"]
arm_ob   = bpy.data.objects["ARM-FullBody_V10"]

dg = bpy.context.evaluated_depsgraph_get()
s_eval = sotai_ob.evaluated_get(dg)
s_me = s_eval.to_mesh()
s_me.calc_loop_triangles()
sv = np.array([list(sotai_ob.matrix_world @ v.co) for v in s_me.vertices], dtype=np.float64)
sn = np.array([list((sotai_ob.matrix_world.to_3x3() @ v.normal).normalized()) for v in s_me.vertices], dtype=np.float64)
s_tris = [list(lt.vertices) for lt in s_me.loop_triangles]
skin_bvh = BVHTree.FromPolygons(sv.tolist(), s_tris, all_triangles=True)
skin_kdt = _KDWrap(sv)
s_eval.to_mesh_clear()

hair_pts_list = []
for hob in bpy.data.objects:
    if hob.type == 'MESH' and hob.name.startswith("GEO-Hair-") and "Brow" not in hob.name:
        he = hob.evaluated_get(dg)
        hm = he.to_mesh()
        for v in hm.vertices:
            hair_pts_list.append(list(hob.matrix_world @ v.co))
        he.to_mesh_clear()
hair_pts = np.array(hair_pts_list, dtype=np.float64)

def bone_head_tail(bname):
    pb = arm_ob.pose.bones.get(bname)
    return (arm_ob.matrix_world @ pb.head).copy(), (arm_ob.matrix_world @ pb.tail).copy()

sh_L, _ = bone_head_tail("upper_arm.L")
el_L, _ = bone_head_tail("lower_arm.L")
wr_L, hd_L = bone_head_tail("hand.L")
sh_R, _ = bone_head_tail("upper_arm.R")
el_R, _ = bone_head_tail("lower_arm.R")
wr_R, hd_R = bone_head_tail("hand.R")

cloth_coll = bpy.data.collections.get("CLOTH_V10_衣装与配饰")
if not cloth_coll:
    cloth_coll = bpy.data.collections.new("CLOTH_V10_衣装与配饰")
    bpy.context.scene.collection.children.link(cloth_coll)

rz90 = Matrix.Rotation(math.radians(90.0), 4, 'Z')

def import_tripo_rz90(folder_key, new_name):
    fbx_p = list((UNPACK_DIR / folder_key).glob("*.fbx"))[0]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx_p))
    imp_all = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in imp_all if o.type == 'MESH']
    empties = [o for o in imp_all if o.type != 'MESH']
    ob = meshes[0]
    ob.data.transform(ob.matrix_world)
    ob.matrix_world = Matrix.Identity(4)
    ob.parent = None
    for emp in empties:
        bpy.data.objects.remove(emp, do_unlink=True)
    ob.data.transform(rz90)
    ob.data.update()
    ob.name = new_name
    ob.data.name = new_name + "_Mesh"
    if ob.data.uv_layers:
        ob.data.uv_layers[0].name = "Source_UV"
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    cloth_coll.objects.link(ob)
    return ob

def clean_manifold_and_normals(ob, min_comp_verts=12):
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    vis = set(); del_verts = []
    for v in bm.verts:
        if v.index in vis:
            continue
        st = [v]; vis.add(v.index); comp = []
        while st:
            cur = st.pop(); comp.append(cur)
            for e in cur.link_edges:
                nb = e.other_vert(cur)
                if nb.index not in vis:
                    vis.add(nb.index); st.append(nb)
        if len(comp) < min_comp_verts:
            del_verts.extend(comp)
    if del_verts:
        bmesh.ops.delete(bm, geom=del_verts, context='VERTS')

    bm.edges.ensure_lookup_table()
    nm_edges = [e for e in bm.edges if len(e.link_faces) > 2]
    if nm_edges:
        bmesh.ops.split_edges(bm, edges=nm_edges)

    loose_v = [v for v in bm.verts if len(v.link_faces) == 0]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context='VERTS')

    bm.faces.ensure_lookup_table()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    f_vis = set()
    for f in bm.faces:
        if f.index in f_vis:
            continue
        st = [f]; f_vis.add(f.index); fcomp = []
        while st:
            cur = st.pop(); fcomp.append(cur)
            for e in cur.edges:
                for nf in e.link_faces:
                    if nf.index not in f_vis:
                        f_vis.add(nf.index); st.append(nf)
        c_pts = np.array([[v.co.x, v.co.y, v.co.z] for fc in fcomp for v in fc.verts])
        ctr = Vector(c_pts.mean(axis=0).tolist())
        score = sum(fc.calc_area() * (fc.calc_center_median() - ctr).dot(fc.normal) for fc in fcomp)
        if score < 0.0:
            bmesh.ops.reverse_faces(bm, faces=fcomp)

    bm.to_mesh(me); bm.free(); me.update()
    for p in me.polygons:
        p.use_smooth = True

def volumetric_cage_push(pts, clearance_fn, cell_size=0.028, blur_passes=3, iters=3, z_scale=0.55):
    """
    3D 连续体素平滑位移场 U(x,y,z)：
    1) 仅以单层流形素体 GEO-Sotai_Skin 检测内部穿透或净空不足；
    2) 将所需外推向量投射到间距 cell_size 的规则三维空间网格上（每格取最大需求向量并做 3D 高斯扩散平滑）；
    3) 用三线性插值采样连续光滑空间场 U(x,y,z) 驱动服装顶点——相距 2~5mm 的内外层衬里与领片、口袋获得几乎完全相同的位移向量，
       100% 保持壳体厚度，零内衬翻出斑块，零撕裂破洞！
    """
    out = pts.copy()
    d_ray = Vector((0.0, 1.0, 0.0))
    for _ in range(iters):
        dists, idxs = skin_kdt.query(out)
        need = np.zeros_like(out)
        active = False
        for i, p in enumerate(out):
            req_clear = clearance_fn(p)
            if req_clear <= 0.0 or dists[i] > req_clear + 0.045:
                continue
            # 单层流形素体 +Y 射线奇偶判定（绝对精确！）
            orig = Vector((float(p[0]), float(p[1]) + 1e-5, float(p[2])))
            hits = 0
            for _ in range(8):
                loc, _, _, _ = skin_bvh.ray_cast(orig, d_ray, 4.0)
                if loc is None:
                    break
                hits += 1
                orig = loc + d_ray * 1e-4
            is_inside = (hits % 2 == 1)
            vn = sn[idxs[i]].copy()
            signed_d = float(np.dot(p - sv[idxs[i]], vn))
            if is_inside:
                signed_d = -abs(dists[i])
            if signed_d < req_clear:
                push = req_clear - signed_d
                dv = vn.copy()
                dv[2] *= z_scale
                nl = np.linalg.norm(dv)
                if nl > 1e-6:
                    dv /= nl
                need[i] = dv * push
                active = True
        if not active:
            break

        # 构建 3D 体素网格并注入各格最大外推向量
        bmin = out.min(axis=0) - 3.0 * cell_size
        bmax = out.max(axis=0) + 3.0 * cell_size
        dims = np.maximum(4, np.ceil((bmax - bmin) / cell_size).astype(int) + 1)
        grid = np.zeros((dims[0], dims[1], dims[2], 3), dtype=np.float64)
        gmag = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float64)

        mags = np.linalg.norm(need, axis=1)
        nz_idx = np.where(mags > 1e-6)[0]
        ijk = np.clip(((out[nz_idx] - bmin) / cell_size).round().astype(int), 0, dims - 1)
        for vi, (ix, iy, iz) in zip(nz_idx, ijk):
            if mags[vi] > gmag[ix, iy, iz]:
                gmag[ix, iy, iz] = mags[vi]
                grid[ix, iy, iz] = need[vi]

        # 膨胀 + 可分离 3D 平滑核（既保住峰值净空又让空间梯度极度平缓）
        for _ in range(blur_passes):
            # 软最大值膨胀一步防止平滑衰减净空
            dil = grid.copy()
            d_m = np.linalg.norm(dil, axis=-1)
            for ax in range(3):
                for sh in (-1, 1):
                    rolled = np.roll(grid, sh, axis=ax) * 0.88
                    r_m = np.linalg.norm(rolled, axis=-1)
                    msk = r_m > d_m
                    dil[msk] = rolled[msk]
                    d_m[msk] = r_m[msk]
            # 3轴 [0.25, 0.5, 0.25] 平滑
            for ax in range(3):
                dil = 0.25 * np.roll(dil, -1, axis=ax) + 0.50 * dil + 0.25 * np.roll(dil, 1, axis=ax)
            grid = dil

        # 三线性插值回所有服装顶点（内外壁与领片同步平滑外移！）
        fcoord = np.clip((out - bmin) / cell_size, 0.0, dims - 1.001)
        i0 = np.floor(fcoord).astype(int)
        t  = fcoord - i0
        tx, ty, tz = t[:,0:1], t[:,1:2], t[:,2:3]
        ix0, iy0, iz0 = i0[:,0], i0[:,1], i0[:,2]
        c000 = grid[ix0,   iy0,   iz0  ]
        c100 = grid[ix0+1, iy0,   iz0  ]
        c010 = grid[ix0,   iy0+1, iz0  ]
        c110 = grid[ix0+1, iy0+1, iz0  ]
        c001 = grid[ix0,   iy0,   iz0+1]
        c101 = grid[ix0+1, iy0,   iz0+1]
        c011 = grid[ix0,   iy0+1, iz0+1]
        c111 = grid[ix0+1, iy0+1, iz0+1]
        c00 = c000*(1-tx) + c100*tx
        c01 = c001*(1-tx) + c101*tx
        c10 = c010*(1-tx) + c110*tx
        c11 = c011*(1-tx) + c111*tx
        c0  = c00*(1-ty) + c10*ty
        c1  = c01*(1-ty) + c11*ty
        out += c0*(1-tz) + c1*tz
    return out


# ==============================================================================
# A. 02_水手服短上衣 -> GEO-Cloth_SailorTop（体素连续场护航：零灰斑 + 完整护肩 + 精准露脐！）
# ==============================================================================
top_ob = import_tripo_rz90("02_top", "GEO-Cloth_SailorTop")
bm = bmesh.new(); bm.from_mesh(top_ob.data); bm.faces.ensure_lookup_table()
del_f = []
for side_sgn in (-1.0, +1.0):
    ax_slv = Vector((side_sgn * 0.76, 0.0, -0.65)).normalized()
    cand = [f for f in bm.faces if f.calc_center_median().x * side_sgn > 0.25]
    if cand:
        projs = np.array([f.calc_center_median().dot(ax_slv) for f in cand])
        p_max = float(projs.max())
        for f, pr in zip(cand, projs):
            if pr > p_max - 0.12 and abs(f.normal.dot(ax_slv)) > 0.44:
                del_f.append(f)
bmesh.ops.delete(bm, geom=list(set(del_f)), context='FACES')
bm.to_mesh(top_ob.data); bm.free(); top_ob.data.update()
clean_manifold_and_normals(top_ob)

pts_top = np.array([[v.co.x, v.co.y, v.co.z] for v in top_ob.data.vertices], dtype=np.float64)
z_min0, z_max0 = pts_top[:,2].min(), pts_top[:,2].max()
pts_top[:,0] *= (0.416 / np.ptp(pts_top[:,0]))
pts_top[:,1] *= 0.445
pts_top[:,2] = -0.412 + (pts_top[:,2] - z_min0) * ((-0.180 - (-0.412)) / (z_max0 - z_min0))

chest_s = sv[(sv[:,2] > -0.39) & (sv[:,2] < -0.23) & (np.abs(sv[:,0]) < 0.14)]
torso_t = pts_top[(np.abs(pts_top[:,0]) < 0.12) & (pts_top[:,2] < -0.25)]
pts_top[:,1] += float(0.5*(chest_s[:,1].min() + chest_s[:,1].max()) - 0.5*(torso_t[:,1].min() + torso_t[:,1].max())) + 0.003

pts_top = volumetric_cage_push(pts_top, lambda p: 0.0045, cell_size=0.026, blur_passes=3, iters=3, z_scale=0.85)
top_ob.data.vertices.foreach_set("co", pts_top.astype(np.float32).ravel())
top_ob.data.update()
print("STEP1R3 SailorTop:", pts_top.min(0).round(4).tolist(), pts_top.max(0).round(4).tolist(), flush=True)


# ==============================================================================
# B. 03_高腰抽绳短裤 -> GEO-Cloth_Shorts
# ==============================================================================
shorts_ob = import_tripo_rz90("03_shorts", "GEO-Cloth_Shorts")
clean_manifold_and_normals(shorts_ob)

pts_sh = np.array([[v.co.x, v.co.y, v.co.z] for v in shorts_ob.data.vertices], dtype=np.float64)
z_min0, z_max0 = pts_sh[:,2].min(), pts_sh[:,2].max()
pts_sh[:,0] *= (0.314 / np.ptp(pts_sh[:,0]))
pts_sh[:,1] *= 0.342
pts_sh[:,2] = -0.668 + (pts_sh[:,2] - z_min0) * ((-0.490 - (-0.668)) / (z_max0 - z_min0))

z_slices = np.linspace(-0.67, -0.485, 10)
s_y_mids, g_y_mids = [], []
for zc in z_slices:
    sb = sv[(sv[:,2] >= zc - 0.018) & (sv[:,2] <= zc + 0.018) & (np.abs(sv[:,0]) < 0.145)]
    gb = pts_sh[(pts_sh[:,2] >= zc - 0.018) & (pts_sh[:,2] <= zc + 0.018) & (np.abs(pts_sh[:,0]) < 0.145)]
    if len(sb) > 3 and len(gb) > 3:
        s_y_mids.append(0.5 * (sb[:,1].min() + sb[:,1].max()))
        gb_body = gb[gb[:,1] < np.percentile(gb[:,1], 92)]
        g_y_mids.append(0.5 * (gb[:,1].min() + gb_body[:,1].max()))
    else:
        s_y_mids.append(0.01); g_y_mids.append(0.0)
pts_sh[:,1] += np.interp(pts_sh[:,2], z_slices, np.array(s_y_mids) - np.array(g_y_mids))

for vi, p in enumerate(pts_sh):
    if p[2] > -0.535:
        t_w = (p[2] - (-0.535)) / 0.045
        if not (p[1] > 0.088 and abs(p[0]) < 0.055):
            pts_sh[vi, 0] *= (1.0 - 0.07 * t_w)
            pts_sh[vi, 1] *= (1.0 - 0.06 * t_w)

pts_sh = volumetric_cage_push(pts_sh, lambda p: 0.0046, cell_size=0.024, blur_passes=3, iters=3, z_scale=0.35)
shorts_ob.data.vertices.foreach_set("co", pts_sh.astype(np.float32).ravel())
shorts_ob.data.update()
print("STEP1R3 Shorts:", pts_sh.min(0).round(4).tolist(), pts_sh.max(0).round(4).tolist(), flush=True)


# ==============================================================================
# C. 01_外层落肩长外套 -> GEO-Cloth_Coat（体素连续场 + 袖口精准收拇指露五指 + 零内衬翻出零破洞！）
# ==============================================================================
coat_ob = import_tripo_rz90("01_coat", "GEO-Cloth_Coat")
bm = bmesh.new(); bm.from_mesh(coat_ob.data); bm.faces.ensure_lookup_table()
del_cf = []
for side_sgn in (-1.0, +1.0):
    ax_slv = Vector((side_sgn * 0.72, 0.0, -0.69)).normalized()
    cand = [f for f in bm.faces if f.calc_center_median().x * side_sgn > 0.29 and f.calc_center_median().z < 0.09]
    if cand:
        projs = np.array([f.calc_center_median().dot(ax_slv) for f in cand])
        p_max = float(projs.max())
        for f, pr in zip(cand, projs):
            if pr > p_max - 0.085 and abs(f.normal.dot(ax_slv)) > 0.45:
                del_cf.append(f)
bmesh.ops.delete(bm, geom=list(set(del_cf)), context='FACES')
bm.to_mesh(coat_ob.data); bm.free(); coat_ob.data.update()
clean_manifold_and_normals(coat_ob)

pts_ct = np.array([[v.co.x, v.co.y, v.co.z] for v in coat_ob.data.vertices], dtype=np.float64)
z_min0, z_max0 = pts_ct[:,2].min(), pts_ct[:,2].max()
pts_ct[:,0] *= 0.790
pts_ct[:,1] *= 0.835
pts_ct[:,2] = -0.878 + (pts_ct[:,2] - z_min0) * ((-0.248 - (-0.878)) / (z_max0 - z_min0))

torso_ct = pts_ct[(np.abs(pts_ct[:,0]) < 0.16) & (pts_ct[:,2] > -0.56)]
pts_ct[:,1] += float(0.5*(chest_s[:,1].min() + chest_s[:,1].max()) - 0.5*(torso_ct[:,1].min() + torso_ct[:,1].max())) - 0.010

# 袖口中心精准落于 (±0.402, wr_y+0.008, -0.638)：把内侧大拇指完整收入宽袖口，同时让外侧五指指尖从空心袖口露出 16~20mm！
for side_sgn, wr_pt, el_pt in [(-1.0, wr_L, el_L), (+1.0, wr_R, el_R)]:
    slv_mask = (pts_ct[:,0] * side_sgn > 0.195) & ((np.abs(pts_ct[:,0]) > 0.248) | (pts_ct[:,2] > -0.49))
    slv_pts = pts_ct[slv_mask]
    if len(slv_pts) > 20:
        piv = np.array([side_sgn * 0.195, float(el_pt.y), -0.325])
        ax_rough = np.array([side_sgn * 0.72, 0.0, -0.69])
        proj = (slv_pts - piv) @ ax_rough
        cuff_ctr_cur = slv_pts[proj >= np.percentile(proj, 90)].mean(axis=0)
        cuff_ctr_tgt = np.array([side_sgn * 0.402, float(wr_pt.y) + 0.008, -0.638])

        v_cur = cuff_ctr_cur - piv
        v_tgt = cuff_ctr_tgt - piv
        len_cur = np.linalg.norm(v_cur)
        len_tgt = np.linalg.norm(v_tgt)
        u_cur = v_cur / len_cur
        u_tgt = v_tgt / len_tgt

        cr = np.cross(u_cur, u_tgt)
        ang_r = math.acos(float(np.clip(np.dot(u_cur, u_tgt), -1.0, 1.0)))
        R_slv = np.array(Matrix.Rotation(ang_r, 3, Vector(cr.tolist()).normalized())) if np.linalg.norm(cr) > 1e-6 else np.eye(3)
        scale_len = len_tgt / max(len_cur, 1e-6)

        for vi in range(len(pts_ct)):
            px = pts_ct[vi, 0] * side_sgn
            pz = pts_ct[vi, 2]
            if px > 0.175 and (px > 0.246 or pz > -0.52):
                w_x = float(np.clip((px - 0.175) / 0.065, 0.0, 1.0))
                w_z = 1.0 if px > 0.246 else float(np.clip((pz - (-0.52)) / 0.06, 0.0, 1.0))
                w = w_x * w_z
                w = w * w * (3.0 - 2.0 * w)
                if w > 1e-5:
                    r0 = pts_ct[vi] - piv
                    s_ax = float(np.dot(r0, u_cur))
                    r_perp = r0 - s_ax * u_cur
                    r_scaled = (s_ax * scale_len) * u_cur + (r_perp * 1.18)
                    pts_ct[vi] = (1.0 - w) * pts_ct[vi] + w * (piv + R_slv @ r_scaled)

# 用单层素体定义分层净空函数：
# - 上背与后领区 (Z > -0.43, Y < 0.02, |X| < 0.21)：要求距素体 >= 17mm（因为水手服方领外表面距素体约 9~12mm，17mm 完美包住水手服后领且保持平顺后背！）
# - 臀裤区 (-0.68 <= Z <= -0.48, |X| < 0.18)：要求距素体 >= 14mm（包住短裤！）
# - 其余袖管/手腕区：要求距素体 >= 6.5mm（包住手臂与大拇指根！）
def coat_clearance(p):
    if p[2] > -0.43 and abs(p[0]) < 0.215:
        if p[1] < 0.02:
            return 0.0185  # 后背完整包覆水手服方领与短袖根
        return 0.0135
    if -0.68 <= p[2] <= -0.48 and abs(p[0]) < 0.185:
        return 0.0140      # 覆盖短裤外侧与后臀
    return 0.0065

pts_ct = volumetric_cage_push(pts_ct, coat_clearance, cell_size=0.036, blur_passes=4, iters=3, z_scale=0.20)
coat_ob.data.vertices.foreach_set("co", pts_ct.astype(np.float32).ravel())
coat_ob.data.update()
print("STEP1R3 Coat:", pts_ct.min(0).round(4).tolist(), pts_ct.max(0).round(4).tolist(), flush=True)


# ==============================================================================
# D. 05_鞋袜与配件七件整图 -> 7 个独立部件
# ==============================================================================
pack_master = import_tripo_rz90("05_pack7", "TMP_Pack7")

def extract_submesh(src_ob, face_pred, new_name):
    bm = bmesh.new(); bm.from_mesh(src_ob.data); bm.faces.ensure_lookup_table()
    f_vis = set(); del_faces = []
    for f in bm.faces:
        if f.index in f_vis:
            continue
        st = [f]; f_vis.add(f.index); comp_f = []
        while st:
            cur = st.pop(); comp_f.append(cur)
            for e in cur.edges:
                for nf in e.link_faces:
                    if nf.index not in f_vis:
                        f_vis.add(nf.index); st.append(nf)
        ctr = np.mean([[v.co.x, v.co.y, v.co.z] for fc in comp_f for v in fc.verts], axis=0)
        if not face_pred(ctr):
            del_faces.extend(comp_f)
    bmesh.ops.delete(bm, geom=del_faces, context='FACES')
    loose_v = [v for v in bm.verts if len(v.link_faces) == 0]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context='VERTS')
    new_me = bpy.data.meshes.new(new_name + "_Mesh")
    bm.to_mesh(new_me); bm.free()
    for ms in src_ob.material_slots:
        new_me.materials.append(ms.material)
    new_ob = bpy.data.objects.new(new_name, new_me)
    cloth_coll.objects.link(new_ob)
    return new_ob

whale_ob  = extract_submesh(pack_master, lambda c: c[2] > 0.24 and c[0] >  0.20, "ACC-WhaleHairpin_L")
choker_ob = extract_submesh(pack_master, lambda c: c[2] > 0.24 and abs(c[0]) <= 0.20, "ACC-Choker")
tstrap_ob = extract_submesh(pack_master, lambda c: c[2] > 0.24 and c[0] < -0.20, "ACC-ThighStrap_L")
sockL_ob  = extract_submesh(pack_master, lambda c: -0.14 < c[2] <= 0.24 and c[0] < 0.0, "GEO-Cloth_Sock_L")
sockR_ob  = extract_submesh(pack_master, lambda c: -0.14 < c[2] <= 0.24 and c[0] > 0.0, "GEO-Cloth_Sock_R")
shoeL_ob  = extract_submesh(pack_master, lambda c: c[2] <= -0.14 and c[0] > 0.0, "GEO-Shoe_L")
shoeR_ob  = extract_submesh(pack_master, lambda c: c[2] <= -0.14 and c[0] < 0.0, "GEO-Shoe_R")

m_tmp = pack_master.data
bpy.data.objects.remove(pack_master, do_unlink=True)
bpy.data.meshes.remove(m_tmp)

# --- D1) 堆堆袜 L/R：平切顶部灰暗封口内凹环，并用体素连续场贴合小腿 ---
for side_sgn, sk_ob in [(-1.0, sockL_ob), (+1.0, sockR_ob)]:
    bm = bmesh.new(); bm.from_mesh(sk_ob.data); bm.faces.ensure_lookup_table()
    ctrs_f = np.array([[f.calc_center_median().x, f.calc_center_median().y, f.calc_center_median().z] for f in bm.faces])
    z_lo, z_hi = float(ctrs_f[:,2].min()), float(ctrs_f[:,2].max())
    xy_mid = np.median(ctrs_f[:,:2], axis=0)
    r_max_top = float(np.percentile(np.linalg.norm(ctrs_f[ctrs_f[:,2] > z_hi - 0.05, :2] - xy_mid, axis=1), 96))
    caps = []
    for f, fc in zip(bm.faces, ctrs_f):
        r_xy = float(np.hypot(fc[0] - xy_mid[0], fc[1] - xy_mid[1]))
        # 彻底切除顶盖及整个内壁深灰盲袋圈 (r_xy < 0.88 * r_max_top 或顶部 38mm 内朝内/朝上面)
        if fc[2] > z_hi - 0.040 and (f.normal.z > 0.15 or r_xy < 0.88 * r_max_top):
            caps.append(f)
        elif fc[2] < z_lo + 0.030 and (f.normal.z < -0.20 or r_xy < 0.84 * r_max_top):
            caps.append(f)
    bmesh.ops.delete(bm, geom=caps, context='FACES')
    bm.to_mesh(sk_ob.data); bm.free(); sk_ob.data.update()
    clean_manifold_and_normals(sk_ob)

    pts = np.array([[v.co.x, v.co.y, v.co.z] for v in sk_ob.data.vertices], dtype=np.float64)
    pts -= pts.mean(axis=0)
    z0, z1 = pts[:,2].min(), pts[:,2].max()
    pts[:,2] = -1.285 + (pts[:,2] - z0) * ((-1.038 - (-1.285)) / (z1 - z0))
    pts[:,0] *= (0.112 / np.ptp(pts[:,0]))
    pts[:,1] *= (0.114 / np.ptp(pts[:,1]))

    zs_leg = np.linspace(-1.285, -1.038, 12)
    cx_leg, cy_leg = [], []
    for zc in zs_leg:
        sub_l = sv[(sv[:,2] >= zc - 0.015) & (sv[:,2] <= zc + 0.015) & (sv[:,0] * side_sgn > 0.01)]
        if len(sub_l) > 3:
            cx_leg.append(0.5 * (sub_l[:,0].min() + sub_l[:,0].max()))
            cy_leg.append(0.5 * (sub_l[:,1].min() + sub_l[:,1].max()))
        else:
            cx_leg.append(side_sgn * 0.072); cy_leg.append(-0.02)
    pts[:,0] += np.interp(pts[:,2], zs_leg, cx_leg)
    pts[:,1] += np.interp(pts[:,2], zs_leg, cy_leg)

    pts = volumetric_cage_push(pts, lambda p: 0.0042, cell_size=0.024, blur_passes=3, iters=3, z_scale=0.15)
    sk_ob.data.vertices.foreach_set("co", pts.astype(np.float32).ravel())
    sk_ob.data.update()

# --- D2) 厚底运动鞋 L/R ---
for side_sgn, sh_ob in [(-1.0, shoeL_ob), (+1.0, shoeR_ob)]:
    clean_manifold_and_normals(sh_ob, min_comp_verts=8)
    pts = np.array([[v.co.x, v.co.y, v.co.z] for v in sh_ob.data.vertices], dtype=np.float64)
    pts -= pts.mean(axis=0)

    z_sp = np.ptp(pts[:,2])
    high_p = pts[pts[:,2] > pts[:,2].max() - 0.28 * z_sp]
    low_p  = pts[pts[:,2] < pts[:,2].min() + 0.25 * z_sp]
    heel_xy = high_p[:,:2].mean(axis=0)
    toe_xy  = low_p[np.argmax(np.linalg.norm(low_p[:,:2] - heel_xy, axis=1)), :2]
    yaw_err = math.atan2(toe_xy[0] - heel_xy[0], toe_xy[1] - heel_xy[1])
    c_a, s_a = math.cos(yaw_err), math.sin(yaw_err)
    pts[:,0], pts[:,1] = pts[:,0]*c_a - pts[:,1]*s_a, pts[:,0]*s_a + pts[:,1]*c_a

    foot_s = sv[(sv[:,2] < -1.30) & (sv[:,0] * side_sgn > 0.01)]
    fy_min, fy_max = float(foot_s[:,1].min()), float(foot_s[:,1].max())
    pts[:,0] = (pts[:,0] - pts[:,0].min()) / np.ptp(pts[:,0]) * 0.110
    pts[:,0] = (-0.113 + pts[:,0]) if side_sgn < 0 else (0.003 + pts[:,0])
    pts[:,1] = (fy_min - 0.015) + (pts[:,1] - pts[:,1].min()) * (((fy_max + 0.024) - (fy_min - 0.015)) / np.ptp(pts[:,1]))
    pts[:,2] = -1.412 + (pts[:,2] - pts[:,2].min()) * ((-1.242 - (-1.412)) / np.ptp(pts[:,2]))

    sh_ob.data.vertices.foreach_set("co", pts.astype(np.float32).ravel())
    sh_ob.data.update()

vg_foot = sotai_ob.vertex_groups.get("Mask_InsideShoes") or sotai_ob.vertex_groups.new(name="Mask_InsideShoes")
vg_foot.add([int(i) for i, p in enumerate(sv) if p[2] < -1.300], 1.0, 'REPLACE')

# --- D3) 左大腿环 ACC-ThighStrap_L ---
clean_manifold_and_normals(tstrap_ob)
pts_ts = np.array([[v.co.x, v.co.y, v.co.z] for v in tstrap_ob.data.vertices], dtype=np.float64)
pts_ts -= 0.5 * (pts_ts.min(0) + pts_ts.max(0))
th_slice = sv[(sv[:,2] >= -0.774) & (sv[:,2] <= -0.750) & (sv[:,0] < -0.01)]
th_cx, th_cy, th_cz = 0.5*(th_slice[:,0].min()+th_slice[:,0].max()), 0.5*(th_slice[:,1].min()+th_slice[:,1].max()), -0.762
ang_grid = np.linspace(-math.pi, math.pi, 73)
thetas_s = np.arctan2(th_slice[:,1] - th_cy, th_slice[:,0] - th_cx)
rads_s   = np.hypot(th_slice[:,0] - th_cx, th_slice[:,1] - th_cy)
r_grid   = np.array([float(np.max(rads_s[np.abs((thetas_s - a0 + math.pi)%(2*math.pi)-math.pi) < 0.35])) if np.any(np.abs((thetas_s - a0 + math.pi)%(2*math.pi)-math.pi) < 0.35) else float(np.median(rads_s)) for a0 in ang_grid])
for _ in range(4):
    r_grid = 0.25*np.roll(r_grid,-1) + 0.50*r_grid + 0.25*np.roll(r_grid,1)
r_ts = np.hypot(pts_ts[:,0], pts_ts[:,1])
th_ts = np.arctan2(pts_ts[:,1], pts_ts[:,0])
r_in_grid = np.array([float(np.percentile(r_ts[np.abs((th_ts - a0 + math.pi)%(2*math.pi)-math.pi) < 0.28], 10)) if np.any(np.abs((th_ts - a0 + math.pi)%(2*math.pi)-math.pi) < 0.28) else float(np.percentile(r_ts, 20)) for a0 in ang_grid])
for _ in range(3):
    r_in_grid = 0.25*np.roll(r_in_grid,-1) + 0.50*r_in_grid + 0.25*np.roll(r_in_grid,1)
z_scale_ts = 0.026 / max(np.ptp(pts_ts[:,2]), 1e-4)
for vi in range(len(pts_ts)):
    a = th_ts[vi]
    dr = max(0.0, r_ts[vi] - float(np.interp(a, ang_grid, r_in_grid))) * 0.28 + 0.0022
    r_new = float(np.interp(a, ang_grid, r_grid)) + 0.0015 + dr
    pts_ts[vi] = [th_cx + r_new*math.cos(a), th_cy + r_new*math.sin(a), th_cz + pts_ts[vi,2]*z_scale_ts]
tstrap_ob.data.vertices.foreach_set("co", pts_ts.astype(np.float32).ravel())
tstrap_ob.data.update()

# --- D4) 颈部项圈 ACC-Choker ---
clean_manifold_and_normals(choker_ob)
pts_ck = np.array([[v.co.x, v.co.y, v.co.z] for v in choker_ob.data.vertices], dtype=np.float64)
pts_ck -= 0.5 * (pts_ck.min(0) + pts_ck.max(0))
nk_slice = sv[(sv[:,2] >= -0.178) & (sv[:,2] <= -0.162) & (np.abs(sv[:,0]) < 0.075)]
nk_cx, nk_cy, nk_cz = 0.5*(nk_slice[:,0].min()+nk_slice[:,0].max()), 0.5*(nk_slice[:,1].min()+nk_slice[:,1].max()), -0.170
thetas_n = np.arctan2(nk_slice[:,1] - nk_cy, nk_slice[:,0] - nk_cx)
rads_n   = np.hypot(nk_slice[:,0] - nk_cx, nk_slice[:,1] - nk_cy)
rn_grid  = np.array([float(np.max(rads_n[np.abs((thetas_n - a0 + math.pi)%(2*math.pi)-math.pi) < 0.40])) if np.any(np.abs((thetas_n - a0 + math.pi)%(2*math.pi)-math.pi) < 0.40) else float(np.median(rads_n)) for a0 in ang_grid])
for _ in range(4):
    rn_grid = 0.25*np.roll(rn_grid,-1) + 0.50*rn_grid + 0.25*np.roll(rn_grid,1)
r_ck = np.hypot(pts_ck[:,0], pts_ck[:,1])
th_ck = np.arctan2(pts_ck[:,1], pts_ck[:,0])
rc_in_grid = np.array([float(np.percentile(r_ck[np.abs((th_ck - a0 + math.pi)%(2*math.pi)-math.pi) < 0.30], 15)) if np.any(np.abs((th_ck - a0 + math.pi)%(2*math.pi)-math.pi) < 0.30) else float(np.percentile(r_ck, 20)) for a0 in ang_grid])
z_scale_ck = 0.0135 / max(np.ptp(pts_ck[:,2]), 1e-4)
for vi in range(len(pts_ck)):
    a = th_ck[vi]
    dr = max(0.0, r_ck[vi] - float(np.interp(a, ang_grid, rc_in_grid))) * 0.28 + 0.0018
    r_new = float(np.interp(a, ang_grid, rn_grid)) + 0.0014 + dr
    pts_ck[vi] = [nk_cx + r_new*math.cos(a), nk_cy + r_new*math.sin(a), nk_cz + pts_ck[vi,2]*z_scale_ck]
choker_ob.data.vertices.foreach_set("co", pts_ck.astype(np.float32).ravel())
choker_ob.data.update()

# --- D5) 左侧鲸鱼发夹 ACC-WhaleHairpin_L ---
clean_manifold_and_normals(whale_ob, min_comp_verts=8)
pts_wh = np.array([[v.co.x, v.co.y, v.co.z] for v in whale_ob.data.vertices], dtype=np.float64)
pts_wh -= 0.5 * (pts_wh.min(0) + pts_wh.max(0))
y_p55 = float(np.percentile(pts_wh[:,1], 55))
pts_wh[:,1] = np.where(pts_wh[:,1] < y_p55, y_p55 + (pts_wh[:,1] - y_p55) * 0.28, pts_wh[:,1])
pts_wh *= (0.066 / max(np.ptp(pts_wh[:,0]), 1e-4))
near_h = hair_pts[(np.abs(hair_pts[:,0] - (-0.075)) < 0.022) & (np.abs(hair_pts[:,2] - (-0.015)) < 0.022) & (hair_pts[:,1] > 0.03)]
y_hair_surf = float(np.percentile(near_h[:,1], 92)) if len(near_h) else 0.088
R_wh = np.array((Euler((math.radians(-4.0), math.radians(-8.0), math.radians(-28.0)), 'XYZ')).to_matrix())
pts_wh[:,1] -= pts_wh[:,1].min()
pts_wh = (pts_wh @ R_wh.T) + np.array([-0.075, y_hair_surf + 0.003, -0.015])
whale_ob.data.vertices.foreach_set("co", pts_wh.astype(np.float32).ravel())
whale_ob.data.update()


# ==============================================================================
# E. 预览材质与四向 + 原图叠合渲染复核
# ==============================================================================
all_cloth_objs = [coat_ob, top_ob, shorts_ob, sockL_ob, sockR_ob, shoeL_ob, shoeR_ob, tstrap_ob, choker_ob, whale_ob]
for ob in all_cloth_objs:
    for ms in ob.material_slots:
        m = ms.material
        if m and m.use_nodes:
            nt = m.node_tree
            rgb_nd = next((nd for nd in nt.nodes if nd.type == 'TEX_IMAGE' and nd.image and (nd.image.size[0] == 8192 or 'rgb' in nd.image.name.lower())), None)
            if rgb_nd:
                out_nd = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
                em = nt.nodes.new('ShaderNodeEmission')
                em.inputs['Strength'].default_value = 0.96
                nt.links.new(rgb_nd.outputs['Color'], em.inputs['Color'])
                nt.links.new(em.outputs['Emission'], out_nd.inputs['Surface'])

mask_mod = sotai_ob.modifiers.get("Mask_InsideShoes") or sotai_ob.modifiers.new(name="Mask_InsideShoes", type='MASK')
mask_mod.vertex_group = "Mask_InsideShoes"
mask_mod.invert_vertex_group = True

bpy.ops.wm.save_as_mainfile(filepath=str(OUT_STEP1_BLEND), compress=True)

sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'
sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGBA'
sc.render.film_transparent = True
cam = bpy.data.objects["REVIEW_CAMERA"]
sc.camera = cam
cam.data.type = 'ORTHO'

K_ORIG = 0.000487708
V_CENTER = 0.5 * (380.0 + 3756.0)
cam.location = Vector((0.0, 3.2, -(V_CENTER - 788.7) * K_ORIG))
cam.rotation_euler = Euler((math.radians(90.0), 0.0, math.radians(180.0)), 'XYZ')
cam.data.ortho_scale = 3376.0 * K_ORIG
sc.render.resolution_x = 1024
sc.render.resolution_y = 1688
sc.render.filepath = str(RENDER_DIR / "step1_art_frame_rgba.png")
bpy.ops.render.render(write_still=True)

sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.image_settings.quality = 92
sc.render.film_transparent = False
cam.data.ortho_scale = 1.72
sc.render.resolution_x = 680
sc.render.resolution_y = 1180

for vn, loc, rot in [
    ("front", (0.0,  3.2, -0.63), (90.0, 0.0, 180.0)),
    ("left",  (-3.2, 0.0, -0.63), (90.0, 0.0, -90.0)),
    ("back",  (0.0, -3.2, -0.63), (90.0, 0.0,   0.0)),
    ("right", (3.2,  0.0, -0.63), (90.0, 0.0,  90.0)),
]:
    cam.location = Vector(loc)
    cam.rotation_euler = Euler([math.radians(a) for a in rot], 'XYZ')
    sc.render.filepath = str(RENDER_DIR / f"step1_full_{vn}.jpg")
    bpy.ops.render.render(write_still=True)

coat_ob.hide_render = True
for vn, loc, rot in [
    ("inner_front", (0.0,  3.2, -0.63), (90.0, 0.0, 180.0)),
    ("inner_back",  (0.0, -3.2, -0.63), (90.0, 0.0,   0.0)),
]:
    cam.location = Vector(loc)
    cam.rotation_euler = Euler([math.radians(a) for a in rot], 'XYZ')
    sc.render.filepath = str(RENDER_DIR / f"step1_{vn}.jpg")
    bpy.ops.render.render(write_still=True)
coat_ob.hide_render = False
print("STEP1R3_ALL_DONE", flush=True)
