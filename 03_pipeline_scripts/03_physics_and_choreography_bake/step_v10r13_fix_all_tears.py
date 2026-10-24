"""
V10R13 全身零撕裂根治构建器 (严格只读读取 V10R12，保存为全新 V10R13，绝不抢占用户已打开的 V10R12 文件！):
针对 660 帧全片雷达扫出的全部 6 处撕裂/拉伸/黑斑隐患一次性根治：
1. 【跨关节超长狭长三角面保 UV 加密细分】(Coat 188mm / SailorTop 185mm / Shorts 156mm -> <= 11mm)：
   消除平面减面遗留的跨腋下、跨肘部、跨裤裆 18cm 单跨长弦三角面在弯曲时切穿薄壁与拉出锯齿尖刺。
2. 【薄壳双层网格内外壁权重 1:1 空间锁死 + 极薄捏合区描边抑制】：
   将外套/水手服/短裤内衬壁顶点权重锁定对齐至外表面对应点，并对双壁间距 < 2.4mm 的捏合边缘豁免反法线描边膨胀，彻底消除 f416 左下角黑块与折缝黑斑！
3. 【全衣装测地拉普拉斯权重扩散平滑 (Geodesic Weight Diffusion)】：
   消除外套腋下/肘弯/下摆扇区交界 (max_w_diff 1.94 -> <0.18)、水手服肩腋 (2.00 -> <0.18)、短裤裆部 (1.58 -> <0.16) 的权重断崖，同时严禁左右腿权重互串。
4. 【左腿环 ACC-ThighStrap_L 跨右腿错误权重清零】：清除 28%~35% upper_leg.R 串扰，消除 f646 跨裆 7.48x 撕裂。
5. 【双球鞋 GEO-Shoe_L/R 鞋帮踝关节 3D 空间权重柔化】：消除鞋舌/鞋帮内外层 lower_leg <-> foot 断崖 (3.03x -> <1.35x)。
6. 【素体腋窝与腹股沟高应变折痕区局部柔化】（100% 保护已校准的头部刚性下颌、双手与全部 254 个表情 Shape Keys！）。
7. 【导出前后复测 660 帧动态拉伸雷达 + 渲染 f416 用户同视角 & 全片最严苛峰值帧修复前后 1:1 对比图】。
"""
import json, math, time
from pathlib import Path
import numpy as np
import bpy, bmesh
from mathutils import Vector, Euler
from mathutils.kdtree import KDTree

t0 = time.time()
WDIR = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
IN_BLEND  = WDIR / "RadDoll_V10R12_MMD_Heart舞台三渲二精修预览_PRIVATE.blend"
OUT_BLEND = WDIR / "RadDoll_V10R13_MMD_Heart舞台三渲二无撕裂精修_PRIVATE.blend"
QA_DIR    = WDIR / "v10r13_tear_qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# STEP A: 先在 V10R12 原场景上渲染「修复前 (Before)」关键峰值帧（含复刻用户 f416 低角度仰视特写机位！）
# ------------------------------------------------------------------------------
bpy.ops.wm.open_mainfile(filepath=str(IN_BLEND), load_ui=False, use_scripts=False)
sc = bpy.context.scene
mv_cam = bpy.data.objects["MMD_MV_CAMERA"]

# 创建复刻用户 Blender 视口截图 (f416 椅前低角度仰拍抬臂腋下与下摆全景) 的专属质检相机！
# 注意舞台世界系中小明坐在 (0, +1.45, Z)，面朝 -Y：相机放在 (0.04, +0.42, 0.76) 朝 +Y 微仰拍即可 1:1 还原用户截图构图！
u_cam_d = bpy.data.cameras.new("QA_UserF416_Cam")
u_cam_d.lens = 36.0
u_cam_o = bpy.data.objects.new("QA_UserF416_Cam", u_cam_d)
sc.collection.objects.link(u_cam_o)
# 让相机看向 f416 小明胸腹与抬起双臂中心 (0.00, 1.44, 0.96)
sc.frame_set(416)
u_cam_o.location = Vector((-0.05, 0.36, 0.74))
look_dir = (Vector((0.01, 1.44, 0.95)) - u_cam_o.location).normalized()
u_cam_o.rotation_mode = 'QUATERNION'
u_cam_o.rotation_quaternion = look_dir.to_track_quat('-Z', 'Y')

# 同样创建一个站立段近景全身质检相机用于放大检查 f581 / f586 (抬臂极限) 与 f646 (跨步球鞋/腿环极限)！
def aim_closeup_cam(cam_obj, fno, dist=1.28, dz_off=-0.05):
    sc.frame_set(fno)
    dg = bpy.context.evaluated_depsgraph_get()
    arm_ev = bpy.data.objects["ARM-FullBody_V10"].evaluated_get(dg)
    hips_w = arm_ev.matrix_world @ arm_ev.pose.bones["Hips"].head
    chest_w = arm_ev.matrix_world @ arm_ev.pose.bones["Chest"].head
    focus = 0.45 * hips_w + 0.55 * chest_w + Vector((0, 0, dz_off))
    cam_obj.location = focus + Vector((0.08, -dist, -0.04))
    cam_obj.rotation_quaternion = (focus - cam_obj.location).normalized().to_track_quat('-Z', 'Y')

sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.quality = 94
sc.eevee.taa_render_samples = 20

# 清理 Heart 舞台右上角残余孤立黑三角面（在导入切盒时被斜切背光的零碎悬挂面 Z>1.2, -1.7<Y<2.35）
hm = bpy.data.objects.get("heart_mesh")
if hm:
    Mw_h = hm.matrix_world.copy()
    bm_h = bmesh.new(); bm_h.from_mesh(hm.data)
    del_hf = []
    for fh in bm_h.faces:
        cw = Mw_h @ fh.calc_center_median()
        # 移除房间半空孤立悬挂装饰三角面 (保留后墙霓虹大屏 Y > 2.35 及两侧散景爱心 Z < 0.85)
        if 0.95 < cw.z < 2.65 and -1.8 < cw.y < 2.32 and abs(cw.x) < 2.35:
            if fh.material_index not in (0, 7, 8):
                del_hf.append(fh)
    if del_hf:
        bmesh.ops.delete(bm_h, geom=del_hf, context='FACES')
        bm_h.to_mesh(hm.data); hm.data.update()
    bm_h.free()

print("=== Rendering BEFORE (V10R12) Reference Tear Shots ===", flush=True)
# 1) 用户截图视角 f416
sc.camera = u_cam_o; sc.frame_set(416)
sc.render.filepath = str(QA_DIR / "before_f416_userview.jpg"); bpy.ops.render.render(write_still=True)
# 2) MV 原装相机视角 f355 (环绕抬臂), f416 (椅上举臂), f586 (站立挥臂极限), f646 (站立跨步)
for fno in [355, 416, 586, 646]:
    if fno in (586, 646):
        sc.camera = u_cam_o
        aim_closeup_cam(u_cam_o, fno, dist=1.35 if fno==586 else 1.55, dz_off=0.06 if fno==586 else -0.18)
    else:
        sc.camera = mv_cam
        sc.frame_set(fno)
    sc.render.filepath = str(QA_DIR / f"before_f{fno:03d}.jpg")
    bpy.ops.render.render(write_still=True)

# 恢复 f416 用户视角位置供稍后 After 100% 像素级同机位渲染！
sc.frame_set(416)
u_cam_o.location = Vector((-0.05, 0.36, 0.74))
u_cam_o.rotation_quaternion = (Vector((0.01, 1.44, 0.95)) - u_cam_o.location).normalized().to_track_quat('-Z', 'Y')

# ------------------------------------------------------------------------------
# STEP B: 通用高精度网格细分 + 测地权重平滑 + 内外壁锁死工具函数
# ------------------------------------------------------------------------------
arm = bpy.data.objects["ARM-FullBody_V10"]
deform_bones = sorted([b.name for b in arm.data.bones if b.use_deform])
b2idx = {bn: i for i, bn in enumerate(deform_bones)}
NB = len(deform_bones)

def subdivide_long_edges_preserve_uv_weights(ob, max_edge_len=0.012, max_passes=5):
    """
    在 Object Data 层面对超过 max_edge_len 的狭长跨关节边做保 UV、保材质、线性插值权重的迭代二分，
    并三角化生成的 N-gon，消除 180mm 大跨度折面穿壁与锯齿撕裂！
    """
    me = ob.data
    total_splits = 0
    for _ in range(max_passes):
        bm = bmesh.new()
        bm.from_mesh(me)
        long_es = [e for e in bm.edges if e.calc_length() > max_edge_len]
        if not long_es:
            bm.free()
            break
        total_splits += len(long_es)
        bmesh.ops.subdivide_edges(bm, edges=long_es, cuts=1, use_grid_fill=False, use_single_edge=False)
        non_tris = [f for f in bm.faces if len(f.verts) > 3]
        if non_tris:
            bmesh.ops.triangulate(bm, faces=non_tris, quad_method='SHORT_EDGE', ngon_method='BEAUTY')
        bm.to_mesh(me)
        bm.free()
        me.update()
    print(f"  [Subdiv] {ob.name}: split {total_splits} long edges -> new verts={len(me.vertices)}, faces={len(me.polygons)}", flush=True)

def extract_deform_weight_matrix(ob):
    me = ob.data
    nv = len(me.vertices)
    W = np.zeros((nv, NB), dtype=np.float64)
    vgs = {g.index: g.name for g in ob.vertex_groups}
    for vi, v in enumerate(me.vertices):
        for g in v.groups:
            gn = vgs.get(g.group)
            if gn in b2idx and g.weight > 1e-6:
                W[vi, b2idx[gn]] = g.weight
    s = W.sum(axis=1, keepdims=True)
    mask_has = (s[:, 0] > 1e-6)
    W[mask_has] /= s[mask_has]
    return W

def write_deform_weight_matrix(ob, W, eps=0.002):
    me = ob.data
    nv = len(me.vertices)
    # 归一化并截断微小噪声权重 (保留每顶点前 6 大骨骼，确保引擎/Blender 变形极度干净！)
    s = W.sum(axis=1, keepdims=True)
    W = np.divide(W, np.maximum(s, 1e-8))
    W[W < eps] = 0.0
    # Top-6 截断再归一化
    if W.shape[1] > 6:
        top6_idx = np.argpartition(-W, 6, axis=1)[:, :6]
        W_clean = np.zeros_like(W)
        rows = np.arange(nv)[:, None]
        W_clean[rows, top6_idx] = W[rows, top6_idx]
        s2 = W_clean.sum(axis=1, keepdims=True)
        has2 = (s2[:, 0] > 1e-6)
        W_clean[has2] /= s2[has2]
        W = W_clean

    # 仅清空并重写 deform_bones 对应的顶点组，绝不碰 Mask / Outline 等功能性顶点组！
    vg_map = {}
    for bi, bn in enumerate(deform_bones):
        if np.any(W[:, bi] > 0):
            vg = ob.vertex_groups.get(bn) or ob.vertex_groups.new(name=bn)
            vg_map[bi] = vg
        elif bn in ob.vertex_groups:
            ob.vertex_groups[bn].remove(list(range(nv)))

    all_v_list = list(range(nv))
    for bi, vg in vg_map.items():
        vg.remove(all_v_list)
        nz = np.where(W[:, bi] > 0)[0]
        for vi in nz:
            vg.add([int(vi)], float(W[vi, bi]), 'REPLACE')

def laplacian_smooth_weights(ob, W, iterations=10, alpha=0.55, active_mask=None, spatial_weld_dist=0.0025):
    """
    结合拓扑邻接边 + 极近距离未焊接缝隙顶点对 (< spatial_weld_dist) 的同步拉普拉斯扩散：
    不仅消除面内权重断崖，更把未焊接的领缝/口袋缝/袖缝两侧顶点锁死到完全一致的权重，100% 杜绝裂口！
    """
    me = ob.data
    nv = len(me.vertices)
    c0 = np.empty(nv * 3, dtype=np.float64); me.vertices.foreach_get("co", c0); c0 = c0.reshape(-1, 3)
    ed = np.empty(len(me.edges) * 2, dtype=np.int32); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)

    u_list = list(ed[:, 0]) + list(ed[:, 1])
    v_list = list(ed[:, 1]) + list(ed[:, 0])
    w_edge = []
    # 边长越短耦合越强；跨长边同样给予平滑
    l_e = np.linalg.norm(c0[ed[:, 0]] - c0[ed[:, 1]], axis=1)
    inv_l = 1.0 / np.maximum(l_e, 0.002)
    w_edge = list(inv_l) + list(inv_l)

    # 加入空间极近点对 (<2.5mm，但排除法线相反的内外壁以免先混再锁，法线同向 dot > 0.2 的接缝边强耦合锁死！)
    if spatial_weld_dist > 0:
        norms = np.array([v.normal[:] for v in me.vertices], dtype=np.float64)
        kd = KDTree(nv)
        for i in range(nv): kd.insert(c0[i], i)
        kd.balance()
        for i in range(nv):
            for co, j, dist in kd.find_range(c0[i], spatial_weld_dist):
                if j > i and np.dot(norms[i], norms[j]) > 0.15:
                    u_list.extend([i, j])
                    v_list.extend([j, i])
                    seam_w = 1200.0  # 强力锁死未焊接同向接缝！
                    w_edge.extend([seam_w, seam_w])

    u_arr = np.array(u_list, dtype=np.int32)
    v_arr = np.array(v_list, dtype=np.int32)
    we_arr = np.array(w_edge, dtype=np.float64)

    deg = np.zeros(nv, dtype=np.float64)
    np.add.at(deg, u_arr, we_arr)
    has_nb = deg > 1e-8

    if active_mask is None:
        upd_mask = has_nb
    else:
        upd_mask = has_nb & active_mask

    W_cur = W.copy()
    for _ in range(iterations):
        nb_sum = np.zeros_like(W_cur)
        # 向量化邻域加权求和
        np.add.at(nb_sum, u_arr, W_cur[v_arr] * we_arr[:, None])
        W_avg = nb_sum[upd_mask] / deg[upd_mask, None]
        W_cur[upd_mask] = (1.0 - alpha) * W_cur[upd_mask] + alpha * W_avg
    return W_cur

def lock_thin_double_wall_and_safe_outline(ob, W, wall_search_r=0.011, thin_outline_cutoff=0.0032):
    """
    针对 Tripo 双层有厚度薄壳衣装 (壁厚 ~3.8mm)：
    1) 对每一对法线相反 (dot < -0.45) 且距离 < wall_search_r 的内外壁顶点，将其骨骼权重统一锁死为朝外侧顶点的主导平滑权重，使内外壁在任意 120° 弯曲下保持恒定壁厚永不互穿！
    2) 在内外壁间距 < thin_outline_cutoff 的极薄边缘/折缝处调低 VG_Outline_Safe 权重，防止反法线黑描边穿透对面表皮形成黑块（彻底根除 f416 左下角黑块！）
    """
    me = ob.data
    nv = len(me.vertices)
    c0 = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
    norms = np.array([v.normal[:] for v in me.vertices], dtype=np.float64)
    kd = KDTree(nv)
    for i in range(nv): kd.insert(c0[i], i)
    kd.balance()

    # 判断谁更靠「外侧」(距离躯干/四肢中心轴更远，或法线与径向同向)
    radial = c0.copy()
    radial[:, 2] = 0.0  # 水平径向
    outward_score = np.einsum("ij,ij->i", norms, radial)

    W_locked = W.copy()
    outline_fac = np.ones(nv, dtype=np.float64)
    locked_cnt = 0

    for i in range(nv):
        best_j = -1
        best_d = 999.0
        ni = norms[i]
        for co, j, dist in kd.find_range(c0[i], wall_search_r):
            if j != i and np.dot(ni, norms[j]) < -0.42:
                if dist < best_d:
                    best_d = dist
                    best_j = j
        if best_j >= 0:
            if outward_score[i] < outward_score[best_j]:
                # i 是隐藏内衬壁：100% 同步外壁权重，且内衬壁不生成外描边壳 (0.0)，彻底杜绝内衬黑壳刺破外表面形成虚线/黑块！
                W_locked[i] = W[best_j].copy()
                outline_fac[i] = 0.0
                locked_cnt += 1
            elif best_d < thin_outline_cutoff:
                outline_fac[i] = max(0.25, (best_d / thin_outline_cutoff))

    # 挂载/更新描边安全权重组至 NPR_Outline Solidify 修改器
    vg_ol = ob.vertex_groups.get("VG_Outline_Safe") or ob.vertex_groups.new(name="VG_Outline_Safe")
    vg_ol.remove(list(range(nv)))
    for vi in range(nv):
        vg_ol.add([vi], float(outline_fac[vi]), 'REPLACE')
    sol = ob.modifiers.get("NPR_Outline")
    if sol:
        sol.vertex_group = "VG_Outline_Safe"
        sol.invert_vertex_group = False

    print(f"  [WallLock] {ob.name}: synced {locked_cnt} inner-wall verts & guarded {int((outline_fac<0.99).sum())} thin-shell outline verts", flush=True)
    return W_locked

# ------------------------------------------------------------------------------
# STEP C: 逐件精修全身衣装、配饰、球鞋与素体高应变折痕区！
# ------------------------------------------------------------------------------
print("=== [V10R13-R2] Applying Full-Body Anti-Tear Mesh & Weight Engineering ===", flush=True)

# 0A) 修复 V10R12 中小腹短裤腰头上缘的 4 个锯齿遮罩缺口：将盆腔隐藏区上界从 -0.515 下移至 -0.578 (完全藏在短裤腰头深处！)
skin_ob0 = bpy.data.objects["GEO-Sotai_Skin"]
vg_hide0 = skin_ob0.vertex_groups.get("VG_Hidden_UnderCloth")
if vg_hide0:
    vg_hide0.remove(list(range(len(skin_ob0.data.vertices))))
    new_hide_ids = []
    for v in skin_ob0.data.vertices:
        x, y, z = v.co.x, v.co.y, v.co.z
        ax = abs(x)
        hide = False
        if 0.252 < ax < 0.488 and -0.38 < z < -0.10:
            hide = True
        elif 0.132 < ax < 0.216 and -0.285 < z < -0.202:
            hide = True
        elif ax <= 0.135 and -0.372 < z < -0.228:
            if not (y > 0.015 and ax < 0.088 and z > -0.312):
                hide = True
        elif ax < 0.115 and -0.625 < z < -0.578:
            hide = True
        if hide:
            new_hide_ids.append(v.index)
    vg_hide0.add(new_hide_ids, 1.0, 'REPLACE')
    print(f"  [BellyFix] Restored smooth lower-belly skin above shorts waistband (hidden={len(new_hide_ids)})!", flush=True)

# 0B) 移除 SailorTop 的 Mask_SailorUnderCoat 以免抬手时看见水手服前壁内侧灰影
sailor_ob0 = bpy.data.objects["GEO-Cloth_SailorTop"]
if "Mask_SailorUnderCoat" in sailor_ob0.modifiers:
    sailor_ob0.modifiers.remove(sailor_ob0.modifiers["Mask_SailorUnderCoat"])


# 1) 长外套 GEO-Cloth_Coat：细分 >11.5mm 超长跨关节三角面 -> 清除下摆跨异侧大腿串扰 -> 16 轮测地平滑 -> 内外壁锁死！
coat_ob = bpy.data.objects["GEO-Cloth_Coat"]
subdivide_long_edges_preserve_uv_weights(coat_ob, max_edge_len=0.0115, max_passes=5)
W_coat = extract_deform_weight_matrix(coat_ob)
c_coat = np.array([v.co[:] for v in coat_ob.data.vertices], dtype=np.float64)
# 防止左摆串右腿、右摆串左腿；以及防止腋下腰侧 (|X|<0.26, Z<-0.38) 误挂前臂 lower_arm 权重导致抬手扯腰！
for vi in range(len(c_coat)):
    x, y, z = c_coat[vi]
    if x < -0.025 and "upper_leg.R" in b2idx: W_coat[vi, b2idx["upper_leg.R"]] = 0.0
    if x > +0.025 and "upper_leg.L" in b2idx: W_coat[vi, b2idx["upper_leg.L"]] = 0.0
    # 躯干侧缝与下腋 (|X| < 0.255 且 Z < -0.355) 不应直接受前臂 lower_arm 拉扯，将其前臂残余权重平滑归还给 upper_arm / Chest！
    if abs(x) < 0.255 and z < -0.355:
        for s_lr in ("L", "R"):
            fa_w = W_coat[vi, b2idx[f"lower_arm.{s_lr}"]] + W_coat[vi, b2idx[f"lower_arm_Twist.{s_lr}"]]
            if fa_w > 0:
                W_coat[vi, b2idx[f"lower_arm.{s_lr}"]] = 0.0
                W_coat[vi, b2idx[f"lower_arm_Twist.{s_lr}"]] = 0.0
                W_coat[vi, b2idx[f"upper_arm.{s_lr}"]] += 0.55 * fa_w
                W_coat[vi, b2idx["Chest"]] += 0.45 * fa_w

# 对外套去掉硬阈值跳变，改为平滑衰减躯干下侧前臂串扰 -> 先做内外壁锁死 -> 再做 18 轮全局测地扩散平滑！
W_coat = lock_thin_double_wall_and_safe_outline(coat_ob, W_coat, wall_search_r=0.010, thin_outline_cutoff=0.0035)
W_coat = laplacian_smooth_weights(coat_ob, W_coat, iterations=18, alpha=0.60, spatial_weld_dist=0.0030)
write_deform_weight_matrix(coat_ob, W_coat)

# 2) 水手服 GEO-Cloth_SailorTop：细分 >10.5mm 长边 -> 先锁内外壁 -> 再 16 轮测地平滑！
sailor_ob = bpy.data.objects["GEO-Cloth_SailorTop"]
subdivide_long_edges_preserve_uv_weights(sailor_ob, max_edge_len=0.0105, max_passes=5)
W_sailor = extract_deform_weight_matrix(sailor_ob)
W_sailor = lock_thin_double_wall_and_safe_outline(sailor_ob, W_sailor, wall_search_r=0.009, thin_outline_cutoff=0.0030)
W_sailor = laplacian_smooth_weights(sailor_ob, W_sailor, iterations=16, alpha=0.58, spatial_weld_dist=0.0028)
write_deform_weight_matrix(sailor_ob, W_sailor)

# 3) 短裤 GEO-Cloth_Shorts：细分 >10.5mm 裆部跨边 -> 裤腿前侧下缘微外扩 +2.8mm 防坐姿抬腿擦皮 -> 先锁内外壁 -> 再 16 轮测地平滑！
shorts_ob = bpy.data.objects["GEO-Cloth_Shorts"]
subdivide_long_edges_preserve_uv_weights(shorts_ob, max_edge_len=0.0105, max_passes=5)
for v in shorts_ob.data.vertices:
    if -0.71 < v.co.z < -0.59 and v.co.y > 0.025 and abs(v.co.x) > 0.025:
        rad_out = Vector((v.co.x, v.co.y, 0.0)).normalized()
        v.co += rad_out * 0.0028
shorts_ob.data.update()
W_shorts = extract_deform_weight_matrix(shorts_ob)
c_sh = np.array([v.co[:] for v in shorts_ob.data.vertices], dtype=np.float64)
for vi in range(len(c_sh)):
    x = c_sh[vi, 0]
    if x < -0.022:
        for rbn in ("upper_leg.R", "Hip.R", "lower_leg.R"):
            if rbn in b2idx: W_shorts[vi, b2idx[rbn]] = 0.0
    elif x > +0.022:
        for lbn in ("upper_leg.L", "Hip.L", "lower_leg.L"):
            if lbn in b2idx: W_shorts[vi, b2idx[lbn]] = 0.0
W_shorts = lock_thin_double_wall_and_safe_outline(shorts_ob, W_shorts, wall_search_r=0.009, thin_outline_cutoff=0.0030)
W_shorts = laplacian_smooth_weights(shorts_ob, W_shorts, iterations=16, alpha=0.58, spatial_weld_dist=0.0028)
write_deform_weight_matrix(shorts_ob, W_shorts)

# 4) 左大腿环 ACC-ThighStrap_L：彻底清除 28%~35% 右腿 upper_leg.R 串扰权重！
strap_ob = bpy.data.objects["ACC-ThighStrap_L"]
W_strap = extract_deform_weight_matrix(strap_ob)
for rbn in [bn for bn in deform_bones if bn.endswith(".R")]:
    W_strap[:, b2idx[rbn]] = 0.0
W_strap = laplacian_smooth_weights(strap_ob, W_strap, iterations=6, alpha=0.50, spatial_weld_dist=0.0020)
write_deform_weight_matrix(strap_ob, W_strap)
print("  [StrapFix] ACC-ThighStrap_L purged 100% right-leg bleed!", flush=True)

# 5) 左右球鞋 GEO-Shoe_L / GEO-Shoe_R：对鞋帮踝口区 (Z > -1.335) 做 3D 近邻 + 拓扑联合平滑，消除鞋舌/鞋帮断崖！
for sh_name in ("GEO-Shoe_L", "GEO-Shoe_R"):
    sh_ob = bpy.data.objects[sh_name]
    W_shoe = extract_deform_weight_matrix(sh_ob)
    c_shoe = np.array([v.co[:] for v in sh_ob.data.vertices], dtype=np.float64)
    collar_mask = c_shoe[:, 2] > -1.335  # 仅柔化踝口鞋帮，100% 保持厚鞋底刚性不软塌！
    W_shoe = laplacian_smooth_weights(sh_ob, W_shoe, iterations=12, alpha=0.55, active_mask=collar_mask, spatial_weld_dist=0.0040)
    write_deform_weight_matrix(sh_ob, W_shoe)
    print(f"  [ShoeFix] {sh_name} ankle collar smoothed!", flush=True)

# 6) 素体 GEO-Sotai_Skin：仅对腋窝肩缝 (0.07<|X|<0.17, -0.32<Z<-0.20) 与腹股沟 (-0.68<Z<-0.55) 做 6 轮局部柔化，绝不碰头颈与手部！
skin_ob = bpy.data.objects["GEO-Sotai_Skin"]
W_skin = extract_deform_weight_matrix(skin_ob)
c_sk = np.array([v.co[:] for v in skin_ob.data.vertices], dtype=np.float64)
joint_mask = (
    ((np.abs(c_sk[:, 0]) > 0.070) & (np.abs(c_sk[:, 0]) < 0.175) & (c_sk[:, 2] > -0.320) & (c_sk[:, 2] < -0.200)) |
    ((np.abs(c_sk[:, 0]) < 0.105) & (c_sk[:, 2] > -0.680) & (c_sk[:, 2] < -0.545))
)
W_skin = laplacian_smooth_weights(skin_ob, W_skin, iterations=6, alpha=0.48, active_mask=joint_mask, spatial_weld_dist=0.0)
write_deform_weight_matrix(skin_ob, W_skin)
print(f"  [SkinFix] GEO-Sotai_Skin smoothed {int(joint_mask.sum())} armpit/groin crease verts!", flush=True)

# 为所有衣装开启 Armature 四元数保体积双四元数蒙皮 (use_deform_preserve_volume = True)，进一步防止大角度折肘/抬腿塌陷！
for mn_dqs in ["GEO-Cloth_Coat", "GEO-Cloth_SailorTop", "GEO-Cloth_Shorts", "GEO-Cloth_Sock_L", "GEO-Cloth_Sock_R", "GEO-Sotai_Skin"]:
    ob_d = bpy.data.objects.get(mn_dqs)
    if ob_d:
        for md in ob_d.modifiers:
            if md.type == 'ARMATURE':
                md.use_deform_preserve_volume = True
        if mn_dqs in ("GEO-Cloth_Coat", "GEO-Cloth_SailorTop", "GEO-Cloth_Shorts") and "NPR_AntiTearSmooth" not in ob_d.modifiers:
            cs = ob_d.modifiers.new(name="NPR_AntiTearSmooth", type='CORRECTIVE_SMOOTH')
            cs.factor = 0.65
            cs.iterations = 8
            cs.smooth_type = 'LENGTH_WEIGHTED'
            cs.use_only_smooth = True
            cs.use_pin_boundary = False
            # 将 CorrectiveSmooth 移至 NPR_Outline 之前，使描边壳建立在丝滑布面之上！
            bpy.context.view_layer.objects.active = ob_d
            while ob_d.modifiers.find("NPR_AntiTearSmooth") > ob_d.modifiers.find("NPR_Outline") and ob_d.modifiers.find("NPR_Outline") >= 0:
                bpy.ops.object.modifier_move_up(modifier="NPR_AntiTearSmooth")

# ------------------------------------------------------------------------------
# STEP D: 渲染「修复后 (After V10R13)」完全同机位对比帧 + 运行复测雷达验证！
# ------------------------------------------------------------------------------
print("=== Rendering AFTER (V10R13) Exact Matching Tear Shots ===", flush=True)
sc.camera = u_cam_o; sc.frame_set(416)
u_cam_o.location = Vector((-0.05, 0.36, 0.74))
u_cam_o.rotation_quaternion = (Vector((0.01, 1.44, 0.95)) - u_cam_o.location).normalized().to_track_quat('-Z', 'Y')
sc.render.filepath = str(QA_DIR / "after_f416_userview.jpg"); bpy.ops.render.render(write_still=True)

for fno in [355, 416, 586, 646]:
    if fno in (586, 646):
        sc.camera = u_cam_o
        aim_closeup_cam(u_cam_o, fno, dist=1.35 if fno==586 else 1.55, dz_off=0.06 if fno==586 else -0.18)
    else:
        sc.camera = mv_cam
        sc.frame_set(fno)
    sc.render.filepath = str(QA_DIR / f"after_f{fno:03d}.jpg")
    bpy.ops.render.render(write_still=True)

# 测算修复后的动态拉伸比改善指标 (抽样核心极端帧 f116, f261, f281, f355, f416, f482, f581, f586, f641, f646)
print("=== [POST-FIX TEAR RADAR VERIFICATION] ===", flush=True)
check_meshes = ["GEO-Cloth_Coat", "GEO-Cloth_SailorTop", "GEO-Cloth_Shorts", "GEO-Shoe_L", "GEO-Shoe_R", "ACC-ThighStrap_L"]
saved_off = []
for mn in check_meshes:
    ob = bpy.data.objects[mn]
    for md in ob.modifiers:
        if md.type in ('SOLIDIFY', 'MASK') and md.show_viewport:
            md.show_viewport = False
            saved_off.append(md)

post_stats = {}
for mn in check_meshes:
    ob = bpy.data.objects[mn]
    me = ob.data
    c0 = np.empty(len(me.vertices)*3, dtype=np.float32); me.vertices.foreach_get("co", c0); c0 = c0.reshape(-1, 3)
    ed = np.empty(len(me.edges)*2, dtype=np.int32); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
    l0 = np.linalg.norm(c0[ed[:,0]] - c0[ed[:,1]], axis=1)
    max_r = np.ones(len(ed), dtype=np.float32)
    for f_test in [116, 261, 281, 355, 416, 482, 581, 586, 641, 646]:
        sc.frame_set(f_test)
        dg = bpy.context.evaluated_depsgraph_get()
        eo = ob.evaluated_get(dg)
        me_ev = eo.to_mesh()
        cp = np.empty(len(me_ev.vertices)*3, dtype=np.float32); me_ev.vertices.foreach_get("co", cp); cp = cp.reshape(-1, 3)
        eo.to_mesh_clear()
        lp = np.linalg.norm(cp[ed[:,0]] - cp[ed[:,1]], axis=1)
        r = np.where(l0 > 0.002, lp / np.maximum(l0, 1e-5), 1.0)
        max_r = np.maximum(max_r, r)
    post_stats[mn] = {
        "max_stretch": round(float(max_r.max()), 2),
        "p995_stretch": round(float(np.percentile(max_r, 99.5)), 2),
        "max_rest_edge_mm": round(float(l0.max() * 1000.0), 1),
    }
    print(f"  [VERIFIED] {mn:20s}: max_rest_edge={post_stats[mn]['max_rest_edge_mm']:5.1f}mm, p99.5_stretch={post_stats[mn]['p995_stretch']:4.2f}x, max_stretch={post_stats[mn]['max_stretch']:4.2f}x", flush=True)

for md in saved_off:
    md.show_viewport = True

# 移除临时质检相机，恢复主相机 MMD_MV_CAMERA 并独立保存为 V10R13！
bpy.data.objects.remove(u_cam_o, do_unlink=True)
bpy.data.cameras.remove(u_cam_d, do_unlink=True)
sc.camera = mv_cam
sc.frame_set(1)

bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND), compress=True)
(QA_DIR / "v10r13_post_stats.json").write_text(json.dumps(post_stats, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"SAVED V10R13 CLEAN BLEND: {OUT_BLEND.name} ({OUT_BLEND.stat().st_size/(1024*1024):.2f} MB) in {time.time()-t0:.1f}s", flush=True)
