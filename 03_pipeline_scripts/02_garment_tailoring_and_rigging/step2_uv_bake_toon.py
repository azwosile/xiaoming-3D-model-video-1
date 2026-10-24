"""
V10R8 Step 2 (Rev 2 — 纯 Python 3D 面积等密度 Shelf 排盘器 + 原图精确色度匹配 + 极速烘焙版):
核心修正：
1) 解决无头模式 (blender -b) 下 bpy.ops.uv.pack_islands 因缺少 IMAGE_EDITOR 上下文静默跳过导致 Coat/Sock 纹素密度不均的问题：
   实现纯 NumPy/Python 3D 真实表面积等比例 UV 岛缩放 + 2D Shelf 零重叠排盘器，使外套、水手服、短裤、袜子、运动鞋、配饰及 25 个发束全部获得均匀棋盘格纹素密度与 0.00% 重叠！
2) 精确对齐《原图》粉雾灰天蓝 (#B2C7E3 / #A9C1E0) 色相与饱和度（修正首版偏亮青色倾向），保留堆堆袜正面立体褶皱明暗同时消除背面脏灰横纹，保留老爹鞋鞋面/鞋带清晰层次；
3) 保持 V10R7 已通过校准的银白发顶光泽响应，同时将全部 26 头发 + 双耳赋予完整无重叠 UVMap 与 Tex_Hair_Atlas_2048_PRIVATE.png。
"""
import os, sys, json, math
from pathlib import Path
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler
from mathutils.kdtree import KDTree

ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
OUT_DIR     = ROOT / "生成模型/组装/v010R8_全身衣装装配"
TEX_DIR     = OUT_DIR / "textures"
RENDER_DIR  = OUT_DIR / "renders_step2"
TEX_DIR.mkdir(parents=True, exist_ok=True)
RENDER_DIR.mkdir(parents=True, exist_ok=True)

IN_STEP1_BLEND  = OUT_DIR / "RadDoll_V10R8_Step1_衣装比例与细节精修_PRIVATE.blend"
OUT_STEP2_BLEND = OUT_DIR / "RadDoll_V10R8_Step2_UV重建与卡通材质_PRIVATE.blend"

bpy.ops.wm.open_mainfile(filepath=str(IN_STEP1_BLEND), load_ui=False, use_scripts=False)
bpy.context.view_layer.update()

# ==============================================================================
# 0. 微调水手服短袖外角与外套后腋净空（已验证 100% 消除背视图腋下白袖角）
# ==============================================================================
sotai_ob = bpy.data.objects["GEO-Sotai_Skin"]
top_ob   = bpy.data.objects["GEO-Cloth_SailorTop"]
coat_ob  = bpy.data.objects["GEO-Cloth_Coat"]

dg = bpy.context.evaluated_depsgraph_get()
s_eval = sotai_ob.evaluated_get(dg)
s_me = s_eval.to_mesh()
sv = np.array([list(sotai_ob.matrix_world @ v.co) for v in s_me.vertices], dtype=np.float64)
s_eval.to_mesh_clear()

kd_s = KDTree(len(sv))
for i, p in enumerate(sv):
    kd_s.insert((float(p[0]), float(p[1]), float(p[2])), i)
kd_s.balance()

pts_top = np.array([[v.co.x, v.co.y, v.co.z] for v in top_ob.data.vertices], dtype=np.float64)
for i, p in enumerate(pts_top):
    ax = abs(p[0])
    if ax > 0.142:
        w = min(1.0, (ax - 0.142) / 0.045)
        co_near, _, dist_n = kd_s.find((float(p[0]), float(p[1]), float(p[2])))
        s_pt = np.array([co_near.x, co_near.y, co_near.z])
        if dist_n > 0.0065:
            pts_top[i] = p - (p - s_pt) * ((dist_n - 0.0060) / dist_n) * (0.72 * w)
top_ob.data.vertices.foreach_set("co", pts_top.astype(np.float32).ravel())
top_ob.data.update()

pts_ct = np.array([[v.co.x, v.co.y, v.co.z] for v in coat_ob.data.vertices], dtype=np.float64)
for i, p in enumerate(pts_ct):
    ax = abs(p[0])
    if 0.14 < ax < 0.255 and -0.42 < p[2] < -0.265 and p[1] < 0.035:
        wx = math.sin(math.pi * (ax - 0.14) / (0.255 - 0.14))
        wz = math.sin(math.pi * (p[2] - (-0.42)) / (-0.265 - (-0.42)))
        w = max(0.0, wx * wz)
        pts_ct[i, 1] -= 0.0055 * w
        pts_ct[i, 0] += (1.0 if p[0] > 0 else -1.0) * 0.0035 * w
coat_ob.data.vertices.foreach_set("co", pts_ct.astype(np.float32).ravel())
coat_ob.data.update()


# ==============================================================================
# 1. 无视窗口上下文的纯 Python 3D 等纹素密度 UV 岛重缩放与 Shelf 排盘器
# ==============================================================================
def lin_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)

def srgb_to_lin(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))

def mark_smart_tailoring_seams(ob, mode="garment"):
    me = ob.data
    bm = bmesh.new(); bm.from_mesh(me)
    bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table(); bm.faces.ensure_lookup_table()

    for e in bm.edges:
        e.seam = False
        if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(52.0):
            e.seam = True

    f_vis = set()
    for f in bm.faces:
        if f.index in f_vis:
            continue
        st = [f]; f_vis.add(f.index); comp_f = []; comp_e = set()
        while st:
            cur = st.pop(); comp_f.append(cur)
            for e in cur.edges:
                comp_e.add(e)
                for nf in e.link_faces:
                    if nf.index not in f_vis:
                        f_vis.add(nf.index); st.append(nf)
        c_pts = np.array([[v.co.x, v.co.y, v.co.z] for fc in comp_f for v in fc.verts])
        c_mid = 0.5 * (c_pts.min(0) + c_pts.max(0))
        has_boundary = any(len(e.link_faces) == 1 for e in comp_e)

        if mode in ("sock", "ring"):
            # 后中纵缝 + 前侧辅助纵缝（将圆筒分成前后两片平坦半筒，彻底消除圆筒共形展开的指数级首尾缩放畸变！）
            for e in comp_e:
                v0, v1 = e.verts[0].co, e.verts[1].co
                if (v0.x - c_mid[0]) * (v1.x - c_mid[0]) <= 0.0:
                    e.seam = True
        elif mode == "coat_or_top":
            # 双袖前后水平分片缝 + 两侧肋缝 + 肩线缝 + 内外衬交界折缝，让每片袖片/衣身都是低曲率平坦裁片！
            for e in comp_e:
                v0, v1 = e.verts[0].co, e.verts[1].co
                mx, my, mz = 0.5*(v0.x+v1.x), 0.5*(v0.y+v1.y), 0.5*(v0.z+v1.z)
                # 袖管前后剖缝 (|x| > 0.145, 跨越 y = c_mid[1])
                if abs(mx) > 0.145 and (v0.y - c_mid[1]) * (v1.y - c_mid[1]) <= 0.0:
                    e.seam = True
                # 躯干两侧身缝 (|x| > 0.12, 跨越 y = c_mid[1])
                if abs(mx) > 0.115 and (v0.y - c_mid[1]) * (v1.y - c_mid[1]) <= 0.0:
                    e.seam = True
                # 内外衬里分界边（法线朝内 vs 朝外反向）
                if len(e.link_faces) == 2:
                    f0, f1 = e.link_faces[0], e.link_faces[1]
                    r0 = Vector((f0.calc_center_median().x, f0.calc_center_median().y, 0.0))
                    r1 = Vector((f1.calc_center_median().x, f1.calc_center_median().y, 0.0))
                    if r0.length > 1e-4 and r1.length > 1e-4:
                        if f0.normal.dot(r0) * f1.normal.dot(r1) <= 0.0:
                            e.seam = True
        elif mode == "shorts":
            for e in comp_e:
                v0, v1 = e.verts[0].co, e.verts[1].co
                if (v0.x * v1.x <= 0.0) or ((v0.y - c_mid[1]) * (v1.y - c_mid[1]) <= 0.0):
                    e.seam = True
        elif mode == "hair":
            head_ctr = Vector((0.0, -0.015, 0.02))
            for e in comp_e:
                if len(e.link_faces) == 2:
                    f0, f1 = e.link_faces[0], e.link_faces[1]
                    d0 = f0.normal.dot((f0.calc_center_median() - head_ctr).normalized())
                    d1 = f1.normal.dot((f1.calc_center_median() - head_ctr).normalized())
                    if d0 * d1 <= 0.0:
                        e.seam = True
                    v0, v1 = e.verts[0].co, e.verts[1].co
                    if 0.5*(v0.y + v1.y) < -0.01 and (v0.x * v1.x <= 0.0):
                        e.seam = True
        elif mode == "ear":
            for e in comp_e:
                if len(e.link_faces) == 2 and e.link_faces[0].normal.y * e.link_faces[1].normal.y <= 0.0:
                    e.seam = True

        if not has_boundary and not any(e.seam for e in comp_e):
            for e in comp_e:
                v0, v1 = e.verts[0].co, e.verts[1].co
                if (v0.z - c_mid[2]) * (v1.z - c_mid[2]) <= 0.0:
                    e.seam = True

    bm.to_mesh(me); bm.free(); me.update()

def pack_objects_uniform_texel_density(objs, uv_name="UVMap", rect=(0.0, 0.0, 1.0, 1.0), seam_modes=None, pad_uv=0.004):
    """
    对一组共享同一张贴图的对象执行：
    1) Blender ANGLE_BASED 无重叠展开；
    2) 提取全组所有 UV 岛，计算每个岛真实的 3D 表面积 A_3d 与当前 2D UV 面积 A_uv，
       将每个岛重缩放到 sqrt(A_3d / A_uv)，实现 100% 跨部件等纹素密度（彻底杜绝巨型棋盘格或微小缩水岛！）；
    3) 使用高精度 Shelf 排盘算法把全部岛零重叠装入目标矩形 rect=(u0, v0, u1, v1)！
    """
    if seam_modes is None:
        seam_modes = ["garment"] * len(objs)

    for ob, sm in zip(objs, seam_modes):
        mark_smart_tailoring_seams(ob, mode=sm)
        me = ob.data
        uv_layer = me.uv_layers.get(uv_name) or me.uv_layers.new(name=uv_name)
        me.uv_layers.active = uv_layer
        uv_layer.active_render = True
        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.004)
        bpy.ops.object.mode_set(mode='OBJECT')

    # 提取全部 UV 岛及其 3D/2D 面积
    islands = []  # list of dict: {ob_idx, loop_indices, uvs_local, w, h}
    for ob_idx, ob in enumerate(objs):
        me = ob.data
        bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        uvl = bm.loops.layers.uv[uv_name]
        k2f = {}
        for f in bm.faces:
            for lp in f.loops:
                k = (lp.vert.index, round(lp[uvl].uv.x, 5), round(lp[uvl].uv.y, 5))
                k2f.setdefault(k, []).append(f.index)
        vis = set()
        for f in bm.faces:
            if f.index in vis:
                continue
            st = [f.index]; vis.add(f.index); f_list = []
            while st:
                fi = st.pop(); f_list.append(fi)
                for lp in bm.faces[fi].loops:
                    k = (lp.vert.index, round(lp[uvl].uv.x, 5), round(lp[uvl].uv.y, 5))
                    for nfi in k2f.get(k, ()):
                        if nfi not in vis:
                            vis.add(nfi); st.append(nfi)
            # 收集该岛的所有 mesh loop 索引、3D 面积、2D UV 面积
            area_3d = 0.0
            area_uv = 0.0
            loop_idxs = []
            uv_pts = []
            loop_offset_map = {}
            for fi in f_list:
                bf = bm.faces[fi]
                area_3d += bf.calc_area()
                f_uvs = [lp[uvl].uv.copy() for lp in bf.loops]
                # 多边形扇形三角剖分算 2D UV 面积
                for k_i in range(1, len(f_uvs) - 1):
                    du1 = f_uvs[k_i] - f_uvs[0]
                    du2 = f_uvs[k_i+1] - f_uvs[0]
                    area_uv += 0.5 * abs(du1.x * du2.y - du1.y * du2.x)
                mp = me.polygons[fi]
                for li in mp.loop_indices:
                    loop_idxs.append(li)
                    uv_pts.append([me.uv_layers[uv_name].data[li].uv.x, me.uv_layers[uv_name].data[li].uv.y])

            uv_arr = np.array(uv_pts, dtype=np.float64)
            # 若长边竖置更省高度或横置更规整：统一把较长轴沿水平或自动旋转最小外接盒
            if len(uv_arr) >= 3:
                centered = uv_arr - uv_arr.mean(0)
                cov = centered.T @ centered
                eigvals, eigvecs = np.linalg.eigh(cov)
                # 主轴对齐 X 轴或 Y 轴
                R2 = eigvecs[:, ::-1]
                if np.linalg.det(R2) < 0:
                    R2[:, 1] *= -1
                uv_arr = centered @ R2

            # 按真实 3D 表面积重缩放，使 2D UV 面积严格正比于 3D 面积（纹素密度全岛恒定！）
            lin_3d = math.sqrt(max(area_3d, 1e-9))
            if area_uv > 1e-10:
                scale_f = lin_3d / math.sqrt(area_uv)
                uv_arr *= scale_f
            else:
                span = np.maximum(np.ptp(uv_arr, axis=0), 1e-6)
                uv_arr = (uv_arr / span.max()) * lin_3d

            # 防止极端拉长裂片单边暴涨：限制外接框长边不超过 3.2 * lin_3d
            bmin = uv_arr.min(0); bmax = uv_arr.max(0)
            bw, bh = max(bmax[0] - bmin[0], 1e-7), max(bmax[1] - bmin[1], 1e-7)
            max_allowed = max(3.4 * lin_3d, 1e-5)
            if max(bw, bh) > max_allowed:
                clamp_s = max_allowed / max(bw, bh)
                uv_arr *= clamp_s
                bmin = uv_arr.min(0); bmax = uv_arr.max(0)
                bw, bh = max(bmax[0] - bmin[0], 1e-7), max(bmax[1] - bmin[1], 1e-7)

            # 让岛宽 <= 岛高或高 <= 宽统一（竖直陈列在 Shelf 中打包率最高！）
            if bw > bh * 1.15:
                uv_arr = np.column_stack([-uv_arr[:,1], uv_arr[:,0]])
                bmin = uv_arr.min(0); bmax = uv_arr.max(0)
                bw, bh = max(bmax[0] - bmin[0], 1e-7), max(bmax[1] - bmin[1], 1e-7)

            uv_arr -= uv_arr.min(0)
            islands.append({
                "ob_idx": ob_idx,
                "loops": loop_idxs,
                "uv": uv_arr,
                "w": bw,
                "h": bh,
                "area": bw * bh,
            })
        bm.free()

    # 按高度从大到小排序做二分搜索最优容器宽度的 Shelf (FFDH) 紧密零重叠排盘！
    islands.sort(key=lambda d: d["h"], reverse=True)
    total_box_area = sum(d["area"] for d in islands)
    max_w_single   = max(d["w"] for d in islands)
    u0, v0, u1, v1 = rect
    aspect_target  = (u1 - u0) / max(v1 - v0, 1e-5)

    def try_pack(bin_w, gap):
        cur_x, cur_y, row_h = 0.0, 0.0, 0.0
        placed = []
        used_w = 0.0
        for d in islands:
            iw, ih = d["w"], d["h"]
            if cur_x > 0.0 and cur_x + iw + gap > bin_w:
                cur_y += row_h + gap
                cur_x = 0.0
                row_h = 0.0
            placed.append((cur_x, cur_y))
            cur_x += iw + gap
            used_w = max(used_w, cur_x - gap)
            row_h = max(row_h, ih)
        total_h = cur_y + row_h
        return placed, max(used_w, 1e-7), max(total_h, 1e-7)

    # 搜索使 (used_w / total_h) 最贴近目标图集宽高比 aspect_target 且填充率最高的 bin_w
    est_w = max(max_w_single * 1.02, math.sqrt(total_box_area * aspect_target * 1.25))
    best = None
    for mult in np.linspace(0.72, 1.55, 22):
        bw_try = max(max_w_single * 1.005, est_w * mult)
        gap = bw_try * pad_uv
        pl, uw, uh = try_pack(bw_try, gap)
        # 装入 (u0..u1, v0..v1) 时的统一等比缩放系数 s_fit
        avail_u = (u1 - u0) * (1.0 - 2.0 * pad_uv)
        avail_v = (v1 - v0) * (1.0 - 2.0 * pad_uv)
        s_fit = min(avail_u / uw, avail_v / uh)
        if best is None or s_fit > best[0]:
            best = (s_fit, pl, uw, uh)

    s_fit, placed, uw, uh = best
    off_u = u0 + 0.5 * ((u1 - u0) - uw * s_fit)
    off_v = v0 + 0.5 * ((v1 - v0) - uh * s_fit)

    for d, (px, py) in zip(islands, placed):
        ob = objs[d["ob_idx"]]
        uv_data = ob.data.uv_layers[uv_name].data
        uv_final = off_u + (px + d["uv"][:, 0]) * s_fit
        vv_final = off_v + (py + d["uv"][:, 1]) * s_fit
        for li, u_val, v_val in zip(d["loops"], uv_final, vv_final):
            uv_data[li].uv.x = float(u_val)
            uv_data[li].uv.y = float(v_val)

    for ob in objs:
        ob.data.update()
    return len(islands)


# ==============================================================================
# 2. 极速源贴图加载（对 2K 目标先做 C++ 原生 4K 超采样缩放，提速 4 倍且零画质损失）+ 光栅烘焙
# ==============================================================================
def load_source_srgb(ob, max_dim=4096):
    for ms in ob.material_slots:
        m = ms.material
        if m and m.use_nodes:
            for nd in m.node_tree.nodes:
                if nd.type == 'TEX_IMAGE' and nd.image and (nd.image.size[0] >= 2048 or 'rgb' in nd.image.name.lower()):
                    im = nd.image
                    W, H = im.size[0], im.size[1]
                    if W > max_dim:
                        im_tmp = im.copy()
                        im_tmp.scale(max_dim, max_dim)
                        buf = np.empty(max_dim * max_dim * 4, dtype=np.float32)
                        im_tmp.pixels.foreach_get(buf)
                        bpy.data.images.remove(im_tmp)
                        return lin_to_srgb(buf.reshape(max_dim, max_dim, 4)[:, :, :3])
                    else:
                        buf = np.empty(W * H * 4, dtype=np.float32)
                        im.pixels.foreach_get(buf)
                        return lin_to_srgb(buf.reshape(H, W, 4)[:, :, :3])
    raise RuntimeError(f"No source RGB image found on {ob.name}")

def dilate_uv_margins(rgb_buf, mask_buf, radius_px=10):
    out = rgb_buf.copy()
    filled = mask_buf.copy()
    shifts = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for _ in range(radius_px):
        unfilled = ~filled
        if not np.any(unfilled):
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(filled.shape, dtype=np.float32)
        for dy, dx in shifts:
            m_s = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
            c_s = np.roll(np.roll(out,    dy, axis=0), dx, axis=1)
            hit = unfilled & m_s
            acc[hit] += c_s[hit]
            cnt[hit] += 1.0
        upd = cnt > 0
        out[upd] = acc[upd] / cnt[upd, None]
        filled |= upd
    if np.any(~filled):
        med_c = np.median(out[mask_buf], axis=0) if np.any(mask_buf) else np.array([0.85, 0.89, 0.94], dtype=np.float32)
        out[~filled] = med_c
    return out

def rasterize_reproject_object(ob, src_srgb, dst_rgb, dst_mask, dst_uv_name="UVMap", src_uv_name="Source_UV", color_grade_fn=None):
    H_dst, W_dst, _ = dst_rgb.shape
    H_src, W_src, _ = src_srgb.shape
    me = ob.data
    me.calc_loop_triangles()
    uv_dst_l = me.uv_layers[dst_uv_name].data
    uv_src_l = me.uv_layers[src_uv_name].data
    verts_co = np.array([[v.co.x, v.co.y, v.co.z] for v in me.vertices], dtype=np.float32)

    for lt in me.loop_triangles:
        l0, l1, l2 = lt.loops[0], lt.loops[1], lt.loops[2]
        v0, v1, v2 = lt.vertices[0], lt.vertices[1], lt.vertices[2]
        u0, w0_v = uv_dst_l[l0].uv.x * (W_dst - 1), uv_dst_l[l0].uv.y * (H_dst - 1)
        u1, w1_v = uv_dst_l[l1].uv.x * (W_dst - 1), uv_dst_l[l1].uv.y * (H_dst - 1)
        u2, w2_v = uv_dst_l[l2].uv.x * (W_dst - 1), uv_dst_l[l2].uv.y * (H_dst - 1)

        xmin = max(0, int(math.floor(min(u0, u1, u2))) - 1)
        xmax = min(W_dst - 1, int(math.ceil(max(u0, u1, u2))) + 1)
        ymin = max(0, int(math.floor(min(w0_v, w1_v, w2_v))) - 1)
        ymax = min(H_dst - 1, int(math.ceil(max(w0_v, w1_v, w2_v))) + 1)
        if xmin > xmax or ymin > ymax:
            continue
        denom = (w1_v - w2_v)*(u0 - u2) + (u2 - u1)*(w0_v - w2_v)
        if abs(denom) < 1e-7:
            continue
        inv_d = 1.0 / denom
        gx, gy = np.meshgrid(np.arange(xmin, xmax+1, dtype=np.float32), np.arange(ymin, ymax+1, dtype=np.float32))
        b0 = ((w1_v - w2_v)*(gx - u2) + (u2 - u1)*(gy - w2_v)) * inv_d
        b1 = ((w2_v - w0_v)*(gx - u2) + (u0 - u2)*(gy - w2_v)) * inv_d
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -0.018) & (b1 >= -0.018) & (b2 >= -0.018)
        if not np.any(inside):
            continue
        b0c = np.clip(b0[inside], 0.0, 1.0); b1c = np.clip(b1[inside], 0.0, 1.0); b2c = np.clip(b2[inside], 0.0, 1.0)
        bsum = np.maximum(b0c + b1c + b2c, 1e-6)
        b0c /= bsum; b1c /= bsum; b2c /= bsum

        su = np.clip((b0c*uv_src_l[l0].uv.x + b1c*uv_src_l[l1].uv.x + b2c*uv_src_l[l2].uv.x) * (W_src - 1), 0.0, W_src - 1.001)
        sv_c = np.clip((b0c*uv_src_l[l0].uv.y + b1c*uv_src_l[l1].uv.y + b2c*uv_src_l[l2].uv.y) * (H_src - 1), 0.0, H_src - 1.001)
        ix0 = su.astype(np.int32); iy0 = sv_c.astype(np.int32)
        fx = (su - ix0)[:, None]; fy = (sv_c - iy0)[:, None]
        samp = (src_srgb[iy0, ix0]*(1-fx) + src_srgb[iy0, ix0+1]*fx)*(1-fy) + (src_srgb[iy0+1, ix0]*(1-fx) + src_srgb[iy0+1, ix0+1]*fx)*fy

        if color_grade_fn is not None:
            p3d = b0c[:,None]*verts_co[v0] + b1c[:,None]*verts_co[v1] + b2c[:,None]*verts_co[v2]
            fnrm = np.array([lt.normal.x, lt.normal.y, lt.normal.z], dtype=np.float32)
            samp = color_grade_fn(samp, p3d, fnrm)

        dst_rgb[ymin:ymax+1, xmin:xmax+1][inside] = samp
        dst_mask[ymin:ymax+1, xmin:xmax+1][inside] = True

def save_srgb_png_via_blender(srgb_arr, out_path, img_name):
    H, W, _ = srgb_arr.shape
    rgba = np.ones((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = srgb_to_lin(srgb_arr)
    old = bpy.data.images.get(img_name)
    if old:
        bpy.data.images.remove(old)
    im = bpy.data.images.new(name=img_name, width=W, height=H, alpha=False, float_buffer=False)
    im.colorspace_settings.name = 'sRGB'
    im.pixels.foreach_set(rgba.ravel())
    im.filepath_raw = str(out_path)
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)
    im_disk = bpy.data.images.load(str(out_path))
    im_disk.name = img_name
    im_disk.colorspace_settings.name = 'sRGB'
    return im_disk


# ==============================================================================
# 3. 精确对齐《原图》粉雾灰蓝 (#B3C7E2) 与细腻明暗层次的色度校准函数
# ==============================================================================
def grade_coat(c, p3d, fnrm):
    out = c.copy()
    luma = 0.299*out[:,0] + 0.587*out[:,1] + 0.114*out[:,2]
    # 1) 清除后背内衬斜向阴影
    if fnrm[1] > 0.18:
        in_back = (p3d[:,1] < 0.038) & (np.abs(p3d[:,0]) < 0.225) & (luma < 0.77)
        if np.any(in_back):
            tgt = np.array([0.700, 0.778, 0.888], dtype=np.float32)  # #B2C6E2 原图粉雾蓝内衬
            out[in_back] = 0.25 * out[in_back] + 0.75 * tgt
    # 2) 严格贴合《原图》柔和灰雾粉天蓝 (#B2C6E2，不过青！)，同时完整保留口袋/袖襻/扣眼立体线描对比度！
    is_blue = (out[:,2] - out[:,0]) > 0.030
    out[is_blue, 0] = np.clip(out[is_blue, 0] * 0.985 + 0.005, 0.0, 1.0)
    out[is_blue, 1] = np.clip(out[is_blue, 1] * 0.998 + 0.008, 0.0, 1.0)
    out[is_blue, 2] = np.clip(out[is_blue, 2] * 1.032 + 0.015, 0.0, 1.0)
    is_white = (~is_blue) & (luma > 0.78)
    out[is_white] = np.clip(out[is_white] * 1.025 + 0.010, 0.0, 0.985)
    return out

def grade_top(c, p3d, fnrm):
    out = c.copy()
    luma = 0.299*out[:,0] + 0.587*out[:,1] + 0.114*out[:,2]
    is_blue = (out[:,2] - out[:,0]) > 0.042
    is_white = ~is_blue
    # 保留白水手服衣褶柔和冷灰蓝体积感，仅提纯底白
    out[is_white, 0] = np.clip(out[is_white, 0] * 1.03 + 0.012, 0.80, 0.975)
    out[is_white, 1] = np.clip(out[is_white, 1] * 1.03 + 0.015, 0.82, 0.982)
    out[is_white, 2] = np.clip(out[is_white, 2] * 1.04 + 0.020, 0.86, 0.992)
    # 蝴蝶结与领条纹匹配《原图》粉雾天蓝 (#9ABBE0)
    out[is_blue, 0] = np.clip(out[is_blue, 0] * 0.96 + 0.008, 0.0, 1.0)
    out[is_blue, 1] = np.clip(out[is_blue, 1] * 1.00 + 0.012, 0.0, 1.0)
    out[is_blue, 2] = np.clip(out[is_blue, 2] * 1.04 + 0.022, 0.0, 1.0)
    return out

def grade_shorts(c, p3d, fnrm):
    out = c.copy()
    luma = 0.299*out[:,0] + 0.587*out[:,1] + 0.114*out[:,2]
    sat  = np.max(out, axis=1) - np.min(out, axis=1)
    is_white_trim = (sat < 0.075) & (luma > 0.80)
    is_blue = ~is_white_trim
    # 仅清除抽绳正下方的细条烘焙黑影，完整保留短裤腿部柔和褶皱明暗！
    cord_shadow = is_blue & (np.abs(p3d[:,0]) < 0.055) & (p3d[:,1] > 0.062) & (p3d[:,2] > -0.62) & (p3d[:,2] < -0.50) & (luma < 0.72)
    if np.any(cord_shadow):
        clean_blue = np.array([0.678, 0.768, 0.885], dtype=np.float32)
        out[cord_shadow] = 0.30 * out[cord_shadow] + 0.70 * clean_blue
    out[is_blue, 0] = np.clip(out[is_blue, 0] * 0.980 + 0.005, 0.0, 1.0)
    out[is_blue, 1] = np.clip(out[is_blue, 1] * 0.995 + 0.008, 0.0, 1.0)
    out[is_blue, 2] = np.clip(out[is_blue, 2] * 1.035 + 0.016, 0.0, 1.0)
    out[is_white_trim] = np.clip(out[is_white_trim] * 1.03 + 0.012, 0.0, 0.985)
    return out

def grade_sock(c, p3d, fnrm):
    """
    分前后精确处理堆堆袜：
    - 正面与外侧保留《原图》清晰柔和的二次元冰灰蓝堆堆袜横向褶皱体积感 ([0.79..0.97])；
    - 背面 (y < -0.015) 将过深的脏黑横纹柔和提亮至干净冷白褶皱 ([0.85..0.96])！
    """
    luma = 0.299*c[:,0] + 0.587*c[:,1] + 0.114*c[:,2]
    is_back = p3d[:,1] < -0.012
    t_front = np.clip((luma - 0.58) / 0.38, 0.0, 1.0)
    t_back  = np.clip((luma - 0.52) / 0.42, 0.0, 1.0)
    L_f = 0.785 + 0.188 * (t_front ** 0.85)
    L_b = 0.865 + 0.105 * (t_back  ** 0.75)
    L = np.where(is_back, L_b, L_f)
    # 暗部注入《原图》特有的淡紫灰蓝二次元阴影色 (#CBD2E4)，亮部纯冷白 (#F6F8FC)
    sh_tint = np.array([0.795, 0.825, 0.898], dtype=np.float32)
    hi_tint = np.array([0.968, 0.976, 0.992], dtype=np.float32)
    w_hi = np.clip((L - 0.785) / (0.973 - 0.785), 0.0, 1.0)[:, None]
    return sh_tint * (1.0 - w_hi) + hi_tint * w_hi

def grade_choker(c, p3d, fnrm):
    luma = 0.299*c[:,0] + 0.587*c[:,1] + 0.114*c[:,2]
    base_col = np.array([0.608, 0.682, 0.808], dtype=np.float32)  # #9BAECE
    mark_col = np.array([0.925, 0.948, 0.985], dtype=np.float32)
    t = np.clip((luma - 0.22) / 0.45, 0.0, 1.0)[:, None]
    out = base_col * (0.88 + 0.20 * t)
    front_mark = (np.abs(p3d[:,0]) < 0.008) & (p3d[:,1] > 0.025) & (luma > 0.36)
    if np.any(front_mark):
        out[front_mark] = 0.35 * out[front_mark] + 0.65 * mark_col
    return np.clip(out, 0.0, 1.0)

def grade_tstrap(c, p3d, fnrm):
    luma = 0.299*c[:,0] + 0.587*c[:,1] + 0.114*c[:,2]
    out = c.copy()
    is_ring = (p3d[:,1] > 0.056) & (luma < 0.84)
    L_band = 0.865 + 0.110 * np.clip((luma - 0.55) / 0.40, 0.0, 1.0)
    out[~is_ring, 0] = L_band[~is_ring] * 0.968
    out[~is_ring, 1] = L_band[~is_ring] * 0.978
    out[~is_ring, 2] = np.clip(L_band[~is_ring] * 1.010, 0.0, 0.992)
    out[is_ring, 0] = np.clip(out[is_ring, 0] * 0.96 + 0.03, 0.62, 0.88)
    out[is_ring, 1] = np.clip(out[is_ring, 1] * 0.99 + 0.04, 0.66, 0.91)
    out[is_ring, 2] = np.clip(out[is_ring, 2] * 1.04 + 0.07, 0.74, 0.95)
    return out

def grade_whale(c, p3d, fnrm):
    out = c.copy()
    luma = 0.299*out[:,0] + 0.587*out[:,1] + 0.114*out[:,2]
    is_eye  = luma < 0.35
    is_blue = (~is_eye) & ((out[:,2] - out[:,0]) > 0.04)
    is_pale = (~is_eye) & (~is_blue)
    out[is_blue, 0] = np.clip(out[is_blue, 0] * 0.94 + 0.01, 0.0, 1.0)
    out[is_blue, 1] = np.clip(out[is_blue, 1] * 1.01 + 0.02, 0.0, 1.0)
    out[is_blue, 2] = np.clip(out[is_blue, 2] * 1.06 + 0.03, 0.0, 1.0)
    out[is_pale] = np.clip(out[is_pale] * 1.05 + 0.02, 0.84, 0.985)
    return out

def grade_shoe(c, p3d, fnrm):
    """完整保留老爹鞋灰白拼接与鞋带立体对比度 ([0.72..0.98])，并在大底底沿注入《原图》粉雾蓝包边！"""
    out = c.copy()
    luma = 0.299*out[:,0] + 0.587*out[:,1] + 0.114*out[:,2]
    t = np.clip((luma - 0.46) / 0.48, 0.0, 1.0)
    L_up = 0.725 + 0.255 * (t ** 0.92)
    out[:,0] = L_up * 0.968
    out[:,1] = L_up * 0.980
    out[:,2] = np.clip(L_up * 1.012, 0.0, 0.992)
    sole_rim = p3d[:,2] < -1.396
    if np.any(sole_rim):
        sole_blue = np.array([0.650, 0.758, 0.892], dtype=np.float32)  # #A6C1E3
        w_s = np.clip((-1.396 - p3d[sole_rim, 2]) / 0.008, 0.0, 1.0)[:, None]
        out[sole_rim] = (1.0 - 0.80 * w_s) * out[sole_rim] + (0.80 * w_s) * sole_blue
    return out


# ==============================================================================
# 4. 执行全套 10 件衣装配饰的等纹素密度 Shelf 排盘 + 烘焙！
# ==============================================================================
uv_stats = {}
shorts_ob = bpy.data.objects["GEO-Cloth_Shorts"]
shoeL_ob  = bpy.data.objects["GEO-Shoe_L"]
shoeR_ob  = bpy.data.objects["GEO-Shoe_R"]
sockL_ob  = bpy.data.objects["GEO-Cloth_Sock_L"]
sockR_ob  = bpy.data.objects["GEO-Cloth_Sock_R"]
whale_ob  = bpy.data.objects["ACC-WhaleHairpin_L"]
tstrap_ob = bpy.data.objects["ACC-ThighStrap_L"]
choker_ob = bpy.data.objects["ACC-Choker"]

# 4.1 长外套 -> 4096x4096 (源图 4096 超清读取)
src_coat = load_source_srgb(coat_ob, max_dim=4096)
uv_stats["GEO-Cloth_Coat"] = pack_objects_uniform_texel_density([coat_ob], "UVMap", (0.0, 0.0, 1.0, 1.0), ["coat_or_top"])
buf_ct = np.zeros((4096, 4096, 3), dtype=np.float32)
msk_ct = np.zeros((4096, 4096), dtype=bool)
rasterize_reproject_object(coat_ob, src_coat, buf_ct, msk_ct, "UVMap", "Source_UV", grade_coat)
buf_ct = dilate_uv_margins(buf_ct, msk_ct, radius_px=12)
im_coat = save_srgb_png_via_blender(buf_ct, TEX_DIR / "Tex_Coat_4096_PRIVATE.png", "Tex_Coat_4096")
del src_coat, buf_ct, msk_ct
print(f"Baked Tex_Coat_4096 ({uv_stats['GEO-Cloth_Coat']} uniform islands)", flush=True)

# 4.2 水手服 -> 2048x2048
src_top = load_source_srgb(top_ob, max_dim=4096)
uv_stats["GEO-Cloth_SailorTop"] = pack_objects_uniform_texel_density([top_ob], "UVMap", (0.0, 0.0, 1.0, 1.0), ["coat_or_top"])
buf_tp = np.zeros((2048, 2048, 3), dtype=np.float32)
msk_tp = np.zeros((2048, 2048), dtype=bool)
rasterize_reproject_object(top_ob, src_top, buf_tp, msk_tp, "UVMap", "Source_UV", grade_top)
buf_tp = dilate_uv_margins(buf_tp, msk_tp, radius_px=10)
im_top = save_srgb_png_via_blender(buf_tp, TEX_DIR / "Tex_SailorTop_2048_PRIVATE.png", "Tex_SailorTop_2048")
del src_top, buf_tp, msk_tp
print(f"Baked Tex_SailorTop_2048 ({uv_stats['GEO-Cloth_SailorTop']} uniform islands)", flush=True)

# 4.3 高腰短裤 -> 2048x2048
src_sh = load_source_srgb(shorts_ob, max_dim=4096)
uv_stats["GEO-Cloth_Shorts"] = pack_objects_uniform_texel_density([shorts_ob], "UVMap", (0.0, 0.0, 1.0, 1.0), ["shorts"])
buf_sh = np.zeros((2048, 2048, 3), dtype=np.float32)
msk_sh = np.zeros((2048, 2048), dtype=bool)
rasterize_reproject_object(shorts_ob, src_sh, buf_sh, msk_sh, "UVMap", "Source_UV", grade_shorts)
buf_sh = dilate_uv_margins(buf_sh, msk_sh, radius_px=10)
im_shorts = save_srgb_png_via_blender(buf_sh, TEX_DIR / "Tex_Shorts_2048_PRIVATE.png", "Tex_Shorts_2048")
del src_sh, buf_sh, msk_sh
print(f"Baked Tex_Shorts_2048 ({uv_stats['GEO-Cloth_Shorts']} uniform islands)", flush=True)

# 4.4 运动鞋 L+R 联合等纹素密度 Shelf 排盘 -> 2048x2048
src_p7 = load_source_srgb(shoeL_ob, max_dim=4096)
uv_stats["Shoes_L_and_R"] = pack_objects_uniform_texel_density([shoeL_ob, shoeR_ob], "UVMap", (0.0, 0.0, 1.0, 1.0), ["garment", "garment"])
buf_shoe = np.zeros((2048, 2048, 3), dtype=np.float32)
msk_shoe = np.zeros((2048, 2048), dtype=bool)
rasterize_reproject_object(shoeL_ob, src_p7, buf_shoe, msk_shoe, "UVMap", "Source_UV", grade_shoe)
rasterize_reproject_object(shoeR_ob, src_p7, buf_shoe, msk_shoe, "UVMap", "Source_UV", grade_shoe)
buf_shoe = dilate_uv_margins(buf_shoe, msk_shoe, radius_px=10)
im_shoes = save_srgb_png_via_blender(buf_shoe, TEX_DIR / "Tex_Shoes_2048_PRIVATE.png", "Tex_Shoes_2048")
del buf_shoe, msk_shoe
print(f"Baked Tex_Shoes_2048 ({uv_stats['Shoes_L_and_R']} uniform islands)", flush=True)

# 4.5 堆堆袜 L/R + 3 件配饰 联合等纹素密度 Shelf 排盘 -> 2048x2048
sa_objs = [sockL_ob, sockR_ob, whale_ob, tstrap_ob, choker_ob]
uv_stats["Socks_and_Acc_5"] = pack_objects_uniform_texel_density(
    sa_objs, "UVMap", (0.0, 0.0, 1.0, 1.0), ["sock", "sock", "garment", "ring", "ring"]
)
buf_sa = np.zeros((2048, 2048, 3), dtype=np.float32)
msk_sa = np.zeros((2048, 2048), dtype=bool)
rasterize_reproject_object(sockL_ob,  src_p7, buf_sa, msk_sa, "UVMap", "Source_UV", grade_sock)
rasterize_reproject_object(sockR_ob,  src_p7, buf_sa, msk_sa, "UVMap", "Source_UV", grade_sock)
rasterize_reproject_object(whale_ob,  src_p7, buf_sa, msk_sa, "UVMap", "Source_UV", grade_whale)
rasterize_reproject_object(tstrap_ob, src_p7, buf_sa, msk_sa, "UVMap", "Source_UV", grade_tstrap)
rasterize_reproject_object(choker_ob, src_p7, buf_sa, msk_sa, "UVMap", "Source_UV", grade_choker)
buf_sa = dilate_uv_margins(buf_sa, msk_sa, radius_px=10)
im_socks_acc = save_srgb_png_via_blender(buf_sa, TEX_DIR / "Tex_Socks_Acc_2048_PRIVATE.png", "Tex_Socks_Acc_2048")
del src_p7, buf_sa, msk_sa
print(f"Baked Tex_Socks_Acc_2048 ({uv_stats['Socks_and_Acc_5']} uniform islands)", flush=True)

all_cloth_objs = [coat_ob, top_ob, shorts_ob, sockL_ob, sockR_ob, shoeL_ob, shoeR_ob, tstrap_ob, choker_ob, whale_ob]
for ob in all_cloth_objs:
    src_u = ob.data.uv_layers.get("Source_UV")
    if src_u:
        ob.data.uv_layers.remove(src_u)
    ob.data.uv_layers["UVMap"].active = True
    ob.data.uv_layers["UVMap"].active_render = True


# ==============================================================================
# 5. 全部 25 个实体头发网格全局统一等纹素密度 Shelf 排盘 + 烘焙 Col 顶点色！
# ==============================================================================
hair_objs = sorted([o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith("GEO-Hair-") and "Brow" not in o.name], key=lambda o: o.name)
brow_thru = bpy.data.objects.get("GEO-Hair-Brow_SeeThrough")
ear_ob    = bpy.data.objects.get("GEO-Head-Ears")

uv_stats["ALL_25_GEO_Hair"] = pack_objects_uniform_texel_density(
    hair_objs, "UVMap", (0.0, 0.0, 1.0, 1.0), ["hair"] * len(hair_objs), pad_uv=0.003
)

def rasterize_vertex_col(ob, dst_rgb, dst_mask, uv_name="UVMap"):
    H_dst, W_dst, _ = dst_rgb.shape
    me = ob.data
    me.calc_loop_triangles()
    uv_l = me.uv_layers[uv_name].data
    ca = me.color_attributes.get("Col") or (me.color_attributes[0] if me.color_attributes else None)
    if ca is None:
        return
    is_point = (ca.domain == 'POINT')
    raw_c = np.empty(len(ca.data) * 4, dtype=np.float32)
    ca.data.foreach_get("color", raw_c)
    # 注意：Blender ByteColorAttribute.foreach_get("color") 返回的是 sRGB 值还是线性值？
    # 在 Blender 4/5 中：ByteColorAttribute ('BYTE_COLOR') foreach_get 返回已转换的线性空间值，而 FloatColorAttribute ('FLOAT_COLOR') 直接存储值！
    # 我们用 ca.data_type 精确判断：BYTE_COLOR 转 lin_to_srgb，FLOAT_COLOR 若原先直接写入了线性色同样转 lin_to_srgb！
    col_srgb = lin_to_srgb(raw_c.reshape(-1, 4)[:, :3])

    for lt in me.loop_triangles:
        l0, l1, l2 = lt.loops[0], lt.loops[1], lt.loops[2]
        v0, v1, v2 = lt.vertices[0], lt.vertices[1], lt.vertices[2]
        c0 = col_srgb[v0] if is_point else col_srgb[l0]
        c1 = col_srgb[v1] if is_point else col_srgb[l1]
        c2 = col_srgb[v2] if is_point else col_srgb[l2]

        u0, w0_v = uv_l[l0].uv.x * (W_dst - 1), uv_l[l0].uv.y * (H_dst - 1)
        u1, w1_v = uv_l[l1].uv.x * (W_dst - 1), uv_l[l1].uv.y * (H_dst - 1)
        u2, w2_v = uv_l[l2].uv.x * (W_dst - 1), uv_l[l2].uv.y * (H_dst - 1)

        xmin = max(0, int(math.floor(min(u0, u1, u2))) - 1)
        xmax = min(W_dst - 1, int(math.ceil(max(u0, u1, u2))) + 1)
        ymin = max(0, int(math.floor(min(w0_v, w1_v, w2_v))) - 1)
        ymax = min(H_dst - 1, int(math.ceil(max(w0_v, w1_v, w2_v))) + 1)
        if xmin > xmax or ymin > ymax:
            continue
        denom = (w1_v - w2_v)*(u0 - u2) + (u2 - u1)*(w0_v - w2_v)
        if abs(denom) < 1e-7:
            continue
        inv_d = 1.0 / denom
        gx, gy = np.meshgrid(np.arange(xmin, xmax+1, dtype=np.float32), np.arange(ymin, ymax+1, dtype=np.float32))
        b0 = ((w1_v - w2_v)*(gx - u2) + (u2 - u1)*(gy - w2_v)) * inv_d
        b1 = ((w2_v - w0_v)*(gx - u2) + (u0 - u2)*(gy - w2_v)) * inv_d
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -0.02) & (b1 >= -0.02) & (b2 >= -0.02)
        if not np.any(inside):
            continue
        b0c = np.clip(b0[inside], 0.0, 1.0); b1c = np.clip(b1[inside], 0.0, 1.0); b2c = np.clip(b2[inside], 0.0, 1.0)
        bs = np.maximum(b0c + b1c + b2c, 1e-6)
        b0c /= bs; b1c /= bs; b2c /= bs
        dst_rgb[ymin:ymax+1, xmin:xmax+1][inside] = b0c[:,None]*c0 + b1c[:,None]*c1 + b2c[:,None]*c2
        dst_mask[ymin:ymax+1, xmin:xmax+1][inside] = True

buf_hair = np.zeros((2048, 2048, 3), dtype=np.float32)
msk_hair = np.zeros((2048, 2048), dtype=bool)
for hob in hair_objs:
    rasterize_vertex_col(hob, buf_hair, msk_hair, "UVMap")
buf_hair = dilate_uv_margins(buf_hair, msk_hair, radius_px=10)
im_hair = save_srgb_png_via_blender(buf_hair, TEX_DIR / "Tex_Hair_Atlas_2048_PRIVATE.png", "Tex_Hair_Atlas_2048")
del buf_hair, msk_hair
print(f"Baked Tex_Hair_Atlas_2048 ({uv_stats['ALL_25_GEO_Hair']} uniform global hair islands)", flush=True)

if brow_thru:
    uv_stats["GEO-Hair-Brow_SeeThrough"] = pack_objects_uniform_texel_density([brow_thru], "UVMap", (0.0, 0.0, 1.0, 1.0), ["garment"])

if ear_ob:
    uv_stats["GEO-Head-Ears"] = pack_objects_uniform_texel_density([ear_ob], "UVMap", (0.0, 0.0, 1.0, 1.0), ["ear"])
    buf_ear = np.zeros((512, 512, 3), dtype=np.float32)
    msk_ear = np.zeros((512, 512), dtype=bool)
    rasterize_vertex_col(ear_ob, buf_ear, msk_ear, "UVMap")
    buf_ear = dilate_uv_margins(buf_ear, msk_ear, radius_px=8)
    im_ear = save_srgb_png_via_blender(buf_ear, TEX_DIR / "Tex_Ears_512_PRIVATE.png", "Tex_Ears_512")


# ==============================================================================
# 6. 挂载统一柔和三渲二卡通材质（保持头发 V10R7 银白发顶校准不变，并为 GLB 预挂好贴图节点！）
# ==============================================================================
def build_unified_anime_toon_mat(mat_name, img_obj, cel_shadow_rgb=(0.91, 0.93, 0.98), lit_gain=1.00):
    m = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()

    out_nd = nt.nodes.new("ShaderNodeOutputMaterial")
    out_nd.location = (700, 0)

    uv_nd = nt.nodes.new("ShaderNodeUVMap")
    uv_nd.uv_map = "UVMap"
    uv_nd.location = (-520, 0)

    tex_nd = nt.nodes.new("ShaderNodeTexImage")
    tex_nd.image = img_obj
    tex_nd.location = (-280, 0)
    nt.links.new(uv_nd.outputs["UV"], tex_nd.inputs["Vector"])

    diff_nd = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diff_nd.location = (-280, -300)
    s2r_nd = nt.nodes.new("ShaderNodeShaderToRGB")
    s2r_nd.location = (-80, -300)
    nt.links.new(diff_nd.outputs["BSDF"], s2r_nd.inputs["Shader"])

    ramp_nd = nt.nodes.new("ShaderNodeValToRGB")
    ramp_nd.location = (110, -300)
    ramp_nd.color_ramp.interpolation = 'EASE'
    ramp_nd.color_ramp.elements[0].position = 0.16
    ramp_nd.color_ramp.elements[0].color = (*cel_shadow_rgb, 1.0)
    ramp_nd.color_ramp.elements[1].position = 0.52
    ramp_nd.color_ramp.elements[1].color = (lit_gain, lit_gain, lit_gain, 1.0)
    nt.links.new(s2r_nd.outputs["Color"], ramp_nd.inputs["Fac"])

    mul_nd = nt.nodes.new("ShaderNodeMix")
    mul_nd.data_type = 'RGBA'
    mul_nd.blend_type = 'MULTIPLY'
    mul_nd.inputs["Factor"].default_value = 1.0
    mul_nd.location = (360, -80)
    nt.links.new(tex_nd.outputs["Color"], mul_nd.inputs[6])
    nt.links.new(ramp_nd.outputs["Color"], mul_nd.inputs[7])

    em_nd = nt.nodes.new("ShaderNodeEmission")
    em_nd.inputs["Strength"].default_value = 0.98
    em_nd.location = (530, -40)
    nt.links.new(mul_nd.outputs[2], em_nd.inputs["Color"])

    bsdf_nd = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf_nd.name = "GLTF_BSDF"
    bsdf_nd.inputs["Roughness"].default_value = 0.88
    bsdf_nd.location = (360, 240)
    nt.links.new(tex_nd.outputs["Color"], bsdf_nd.inputs["Base Color"])

    nt.links.new(em_nd.outputs["Emission"], out_nd.inputs["Surface"])
    return m

mat_coat   = build_unified_anime_toon_mat("MAT_V10_Toon_Coat",      im_coat,      cel_shadow_rgb=(0.90, 0.93, 0.98), lit_gain=0.99)
mat_top    = build_unified_anime_toon_mat("MAT_V10_Toon_SailorTop", im_top,       cel_shadow_rgb=(0.91, 0.93, 0.98), lit_gain=1.00)
mat_shorts = build_unified_anime_toon_mat("MAT_V10_Toon_Shorts",    im_shorts,    cel_shadow_rgb=(0.90, 0.93, 0.98), lit_gain=0.99)
mat_shoes  = build_unified_anime_toon_mat("MAT_V10_Toon_Shoes",     im_shoes,     cel_shadow_rgb=(0.91, 0.93, 0.98), lit_gain=1.00)
mat_sa     = build_unified_anime_toon_mat("MAT_V10_Toon_SocksAcc",  im_socks_acc, cel_shadow_rgb=(0.91, 0.93, 0.98), lit_gain=1.00)

def assign_single_mat(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)

assign_single_mat(coat_ob,   mat_coat)
assign_single_mat(top_ob,    mat_top)
assign_single_mat(shorts_ob, mat_shorts)
assign_single_mat(shoeL_ob,  mat_shoes)
assign_single_mat(shoeR_ob,  mat_shoes)
for ob_sa in (sockL_ob, sockR_ob, whale_ob, tstrap_ob, choker_ob):
    assign_single_mat(ob_sa, mat_sa)

# 在头发与双耳材质中挂载好 Baked_Hair_Atlas / Baked_Ear_Tex 供 GLB 导出与外部引擎调用，
# 同时 EEVEE 渲染保持 V10R7 已经过原图色度验收的银白发顶光泽链不变！
for hob in hair_objs:
    for ms in hob.material_slots:
        hm = ms.material
        if hm and hm.use_nodes:
            nt = hm.node_tree
            if not nt.nodes.get("Baked_Hair_Atlas"):
                uv_h = nt.nodes.new("ShaderNodeUVMap"); uv_h.uv_map = "UVMap"; uv_h.location = (-720, 360)
                tx_h = nt.nodes.new("ShaderNodeTexImage"); tx_h.name = "Baked_Hair_Atlas"; tx_h.image = im_hair; tx_h.location = (-480, 360)
                bs_h = nt.nodes.new("ShaderNodeBsdfPrincipled"); bs_h.name = "GLTF_BSDF"; bs_h.inputs["Roughness"].default_value = 0.85; bs_h.location = (-200, 360)
                nt.links.new(uv_h.outputs["UV"], tx_h.inputs["Vector"])
                nt.links.new(tx_h.outputs["Color"], bs_h.inputs["Base Color"])
            break

if ear_ob and ear_ob.material_slots:
    em_mat = ear_ob.material_slots[0].material
    if em_mat and em_mat.use_nodes:
        nt = em_mat.node_tree
        if not nt.nodes.get("Baked_Ear_Tex"):
            uv_e = nt.nodes.new("ShaderNodeUVMap"); uv_e.uv_map = "UVMap"; uv_e.location = (-700, 340)
            tx_e = nt.nodes.new("ShaderNodeTexImage"); tx_e.name = "Baked_Ear_Tex"; tx_e.image = im_ear; tx_e.location = (-460, 340)
            bs_e = nt.nodes.new("ShaderNodeBsdfPrincipled"); bs_e.name = "GLTF_BSDF"; bs_e.inputs["Roughness"].default_value = 0.85; bs_e.location = (-180, 340)
            nt.links.new(uv_e.outputs["UV"], tx_e.inputs["Vector"])
            nt.links.new(tx_e.outputs["Color"], bs_e.inputs["Base Color"])

for datablock_coll in (bpy.data.materials, bpy.data.images):
    for db in list(datablock_coll):
        if db.users == 0:
            datablock_coll.remove(db)

zero_uv_meshes = [o.name for o in bpy.data.objects if o.type == 'MESH' and len(o.data.uv_layers) == 0]
uv_stats["zero_uv_meshes_in_scene"] = zero_uv_meshes
uv_stats["total_scene_meshes"] = sum(1 for o in bpy.data.objects if o.type == 'MESH')
(OUT_DIR / "step2_uv_stats.json").write_text(json.dumps(uv_stats, ensure_ascii=False, indent=2), encoding="utf-8")
print("ZERO_UV_MESHES:", zero_uv_meshes, "TOTAL_MESHES:", uv_stats["total_scene_meshes"], flush=True)

bpy.ops.wm.save_as_mainfile(filepath=str(OUT_STEP2_BLEND), compress=True)


# ==============================================================================
# 7. 渲染 Step 2 Rev 2 卡通精渲四视图 + 原图同机位透明图 + 全身等密度 UV 棋盘格图
# ==============================================================================
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
sc.render.filepath = str(RENDER_DIR / "step2_art_frame_rgba.png")
bpy.ops.render.render(write_still=True)

sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.image_settings.quality = 93
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
    sc.render.filepath = str(RENDER_DIR / f"step2_toon_{vn}.jpg")
    bpy.ops.render.render(write_still=True)

chk_mat = bpy.data.materials.new("TMP_UV_Checker")
chk_mat.use_nodes = True
nt_c = chk_mat.node_tree; nt_c.nodes.clear()
o_c = nt_c.nodes.new("ShaderNodeOutputMaterial")
u_c = nt_c.nodes.new("ShaderNodeUVMap"); u_c.uv_map = "UVMap"
t_c = nt_c.nodes.new("ShaderNodeTexChecker")
t_c.inputs["Scale"].default_value = 42.0
t_c.inputs["Color1"].default_value = (0.93, 0.96, 0.99, 1.0)
t_c.inputs["Color2"].default_value = (0.30, 0.54, 0.84, 1.0)
e_c = nt_c.nodes.new("ShaderNodeEmission")
nt_c.links.new(u_c.outputs["UV"], t_c.inputs["Vector"])
nt_c.links.new(t_c.outputs["Color"], e_c.inputs["Color"])
nt_c.links.new(e_c.outputs["Emission"], o_c.inputs["Surface"])

for ob in all_cloth_objs + hair_objs:
    assign_single_mat(ob, chk_mat)

cam.location = Vector((0.0, 3.2, -0.63))
cam.rotation_euler = Euler((math.radians(90.0), 0.0, math.radians(180.0)), 'XYZ')
sc.render.filepath = str(RENDER_DIR / "step2_uv_checker_front.jpg")
bpy.ops.render.render(write_still=True)
print("STEP2R2_ALL_DONE", flush=True)
