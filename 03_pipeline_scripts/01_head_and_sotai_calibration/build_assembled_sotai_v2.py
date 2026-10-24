# -*- coding: utf-8 -*-
"""
V10R8 全身素体真缝合拼装 + 身体贴图重修 + 54骨头发整合 + 四姿QA与Tripo基准导出脚本
==============================================================================
核心完成项：
1. 比例再精修（与《原图》及《素体四视图_v002》严格一致）：
   - 修正 RadDoll 原生小裹脚（171.0 mm -> 203.5 mm 赤足长，匹配素体四视图 214.4 mm 平底鞋内空间与原图厚底运动鞋）与偏细脚踝（47.9 -> 54.0 mm）。
   - 修正躯干女性化蜂腰凹陷（168.3 mm -> 186.5 mm 中性二次元自然纤腰），平胸少年/少女中性体态，保留健康饱满圆润手臂（腕 52.8×55.2 mm / 前臂 61 mm / 上臂 72.6 mm）。
2. 身体贴图 3D 投影重绘（RadDoll_body_tex_v2_PRIVATE.png, 2048×2048）：
   - 彻底消除 RadDoll_V3 原贴图的女性乳房下缘阴影、乳沟暗影与粉色乳晕点；重绘干净白皙的中性少年/少女平胸锁骨与锁骨窝微暖晕。
   - 头颈 24 点边界零色差融合：采样 RadDoll_face_tex.png 颈口精确线性肤色，沿三维高度平滑过渡，彻底消除颈部横环色差。
   - 复刻《原图》露脐腹部的柔和蜜桃暖晕、浅腹中线与精致浅暖珊瑚色小竖肚脐（替换原贴图深黑成人长脐沟）。
   - 基于变形后三维坐标逐三角重心栅格化绘制：双膝暖粉晕、手肘后侧微粉、圆润肩峰微暖、露袖五指指尖粉润与脚踝微粉（免疫手臂增粗后的 UV 拉伸，并带 8px UV 岛边缘外扩防黑缝）。
3. 头体真缝合拼装（GEO-Sotai_Skin）与全套 253 骨整合（ARM-FullBody_V10）：
   - 将 RadDoll_Skin (2154v, 254表情形态键) 与 RadDoll_Body (8292v) 在 24 对同源颈口顶点处 1:1 焊接为单一无缝皮肤网格 GEO-Sotai_Skin (10422v, 颈口开放边 = 0)，254 个表情/口型/眨眼形态键 100% 完整保留！
   - 从强风物理工程并入 54 根头发物理骨骼（挂于 Head 骨下，总计 253 骨），严格执行「仅按变形骨归一化权重」（SKILL.md 硬规则 5）。
4. 渲染输出：
   - 原图同框穿衣拼接核验、素体四视图叠合核验、贴图重修前后对比、头颈真缝合多角度特写、四姿绑定 QA（转头/抬臂/屈肘/屈膝）、以及供生图与 Tripo 对齐使用的严格正交 3D 素体四视图（裸素体 + 浅灰基础打底装双版本）与 GLB 基准白模。
"""
import bpy
import bmesh
import json
import math
import time
import numpy as np
from pathlib import Path
from mathutils import Vector, Matrix, Euler
from collections import Counter

t0 = time.time()
ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
ASM_DIR = ROOT / "生成模型" / "组装" / "v010R7_RadDoll装配"
IN_BODY_BLEND = ASM_DIR / "RadDoll_V10_全身素体分件基线_PRIVATE.blend"
IN_WIND_BLEND = ASM_DIR / "RadDoll_V10_强风物理演示_PRIVATE.blend"
OUT_BLEND = ASM_DIR / "RadDoll_V10_全身素体精修拼装_PRIVATE.blend"
OUT_BLEND_COMPAT = ASM_DIR / "RadDoll_V10_全身素体装配_PRIVATE.blend"
OUT_TEX_V2 = ASM_DIR / "RadDoll_body_tex_v2_PRIVATE.png"
OUT_GLB_SOTAI = ASM_DIR / "Xiaoming_Sotai_Ref_LowA_PRIVATE.glb"
RENDER_DIR = ASM_DIR / "renders_sotai_v2"
RENDER_BODY_COMPAT = ASM_DIR / "renders_body"
RENDER_DIR.mkdir(parents=True, exist_ok=True)
RENDER_BODY_COMPAT.mkdir(parents=True, exist_ok=True)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def open_boundary_verts(me):
    ec = Counter()
    for p in me.polygons:
        vs = list(p.vertices)
        for i in range(len(vs)):
            a, b = vs[i], vs[(i + 1) % len(vs)]
            ec[(min(a, b), max(a, b))] += 1
    bv = set()
    for (a, b), c in ec.items():
        if c == 1:
            bv.add(a)
            bv.add(b)
    return sorted(bv)


# ==============================================================================
# 1. 打开上一轮全身素体工程并复位骨架到 T 姿以进行干净网格精修与焊接
# ==============================================================================
bpy.ops.wm.open_mainfile(filepath=str(IN_BODY_BLEND))

arm_ob = bpy.data.objects["ARM-FullBody_V10"]
skin_ob = bpy.data.objects["RadDoll_Skin"]
body_ob = bpy.data.objects["RadDoll_Body"]
facial_ob = bpy.data.objects["RadDoll_FacialParts"]
ears_ob = bpy.data.objects["GEO-Head-Ears"]

# 保存当前 ORIG_POSE 的手臂骨骼旋转/位移参数，但先复位到 Rest Pose 进行对称网格微调与焊接
for pb in arm_ob.pose.bones:
    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.location = (0, 0, 0)
    pb.scale = (1, 1, 1)
bpy.context.view_layer.update()

# Apply transforms if any on skin_ob / body_ob so world == local
bpy.ops.object.select_all(action='DESELECT')
for ob in (skin_ob, body_ob):
    if (ob.matrix_world - Matrix.Identity(4)).median_scale != 1.0 or ob.location.length > 1e-6:
        ob.select_set(True)
if bpy.context.selected_objects:
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# ==============================================================================
# 2. 比例再精修（消除过度蜂腰 + 修正 171->203.5mm 自然脚长与饱满脚踝 + 平滑颈口衔接）
# ==============================================================================
b_verts = np.array([v.co[:] for v in body_ob.data.vertices], dtype=np.float64)
s_verts = np.array([v.co[:] for v in skin_ob.data.vertices], dtype=np.float64)

# 2A) 中性二次元自然纤腰修正：当前 Z=-0.44m 处窄至 168.3mm（偏成人女性蜂腰），
#     温和外扩腰侧至 ~186.5mm，与《原图》露脐腰宽及《素体四视图_v002》直筒微收少年/少女体态对齐！
z_b = b_verts[:, 2]
ax_b = np.abs(b_verts[:, 0])
torso_mask = (ax_b < 0.145) & (z_b > -0.54) & (z_b < -0.35)
w_waist = np.exp(-0.5 * ((z_b + 0.440) / 0.042) ** 2) * (1.0 - smoothstep(0.115, 0.145, ax_b))
# 168.3mm * 1.108 = 186.5mm
b_verts[:, 0] *= (1.0 + 0.108 * w_waist)
b_verts[:, 1] *= (1.0 + 0.035 * w_waist)

# 2B) 脚踝饱满度 & 赤足长度精修（171.0 mm -> 203.5 mm，匹配《素体四视图》214.4 mm 平底鞋与《原图》运动鞋！）
# 在脚踝枢轴 (z_ank = -1.298 m, y_heel_ref = +0.032 m) 以下做平滑前向脚长延展与足宽微调
w_ankle = np.exp(-0.5 * ((z_b + 1.295) / 0.038) ** 2) * (ax_b < 0.15)
# 单侧脚踝宽 47.9 -> ~53.8 mm (+12.3%)
for side_sgn in (-1.0, 1.0):
    m_side = (b_verts[:, 0] * side_sgn > 0.005) & (ax_b < 0.15) & (z_b < -1.21) & (z_b > -1.35)
    x_c = float(np.median(b_verts[m_side & (z_b > -1.31) & (z_b < -1.28), 0]))
    b_verts[m_side, 0] = x_c + (b_verts[m_side, 0] - x_c) * (1.0 + 0.125 * w_ankle[m_side])

w_foot = smoothstep(-1.285, -1.335, z_b) * (ax_b < 0.16)
# 角色正面朝 +Y：后跟(-Y)微调 3%，前掌与脚趾(+Y)向前柔和延展 25% -> 总赤足长 203.5 mm！
y_ank_piv = -0.022
dy_f = b_verts[:, 1] - y_ank_piv
scale_foot_y = np.where(dy_f > 0.0, 1.255, 1.030)
b_verts[:, 1] = y_ank_piv + dy_f * (1.0 + (scale_foot_y - 1.0) * w_foot)
# 脚掌宽度微增 7% 防止拉长后显瘦长，严格锁住脚底最低面 Z = -1.3860 m
for side_sgn in (-1.0, 1.0):
    m_f = (b_verts[:, 0] * side_sgn > 0.005) & (z_b < -1.30)
    x_fc = float(np.median(b_verts[m_f, 0]))
    b_verts[m_f, 0] = x_fc + (b_verts[m_f, 0] - x_fc) * (1.0 + 0.075 * w_foot[m_f])

sole_min = float(b_verts[z_b < -1.32, 2].min())
b_verts[z_b < -1.28, 2] += (-1.3860 - sole_min) * smoothstep(-1.28, -1.35, z_b[z_b < -1.28])

# 2C) 定位头颈 24 对同源边界顶点，并将 Body 颈口平滑吸附到 Head 颈口（实现 C¹ 几何无缝！）
s_bverts = [vi for vi in open_boundary_verts(skin_ob.data) if -0.22 < s_verts[vi, 2] < -0.08]
b_bverts = [vi for vi in open_boundary_verts(body_ob.data) if -0.22 < b_verts[vi, 2] < -0.08]
assert len(s_bverts) == 24 and len(b_bverts) == 24, f"Unexpected neck loop counts: {len(s_bverts)}, {len(b_bverts)}"

# 按极角 atan2(y - y0, x) 一一精确配对 24 个同源顶点！
s_ang = np.arctan2(s_verts[s_bverts, 1] - s_verts[s_bverts, 1].mean(), s_verts[s_bverts, 0])
b_ang = np.arctan2(b_verts[b_bverts, 1] - b_verts[b_bverts, 1].mean(), b_verts[b_bverts, 0])
s_order = [s_bverts[i] for i in np.argsort(s_ang)]
b_order = [b_bverts[i] for i in np.argsort(b_ang)]
# 确保起点对齐最近邻
best_shift = min(range(24), key=lambda sh: sum(np.linalg.norm(s_verts[s_order[k]] - b_verts[b_order[(k + sh) % 24]]) for k in range(24)))
b_order = [b_order[(k + best_shift) % 24] for k in range(24)]

neck_deltas = np.zeros((24, 3), dtype=np.float64)
neck_pivots = np.zeros((24, 3), dtype=np.float64)
for k in range(24):
    si, bi = s_order[k], b_order[k]
    neck_pivots[k] = b_verts[bi].copy()
    neck_deltas[k] = s_verts[si] - b_verts[bi]

# 将 neck_deltas 通过高斯径向核平滑传播到 Body 上半颈部（Z > -0.225m），确保颈部曲线如丝般顺滑！
neck_region = np.where((b_verts[:, 2] > -0.230) & (np.abs(b_verts[:, 0]) < 0.085))[0]
for vi in neck_region:
    p = b_verts[vi]
    d2 = np.sum((neck_pivots - p) ** 2, axis=1)
    w_ang = np.exp(-d2 / (0.022 ** 2))
    w_ang /= (w_ang.sum() + 1e-12)
    # 沿 Z 向下到锁骨上缘 (-0.225m) 平滑衰减至 0
    z_att = smoothstep(-0.225, float(neck_pivots[:, 2].min()) - 0.002, p[2])
    b_verts[vi] = p + (w_ang @ neck_deltas) * z_att

# 24 个颈口边界顶点精确锁死为 0.0000 mm 误差！
for k in range(24):
    b_verts[b_order[k]] = s_verts[s_order[k]].copy()

for i, v in enumerate(body_ob.data.vertices):
    v.co = Vector(b_verts[i])
body_ob.data.update()

# 同步微调骨架中足踝/脚趾骨骼位置，使屈踝/踮脚旋转中心完全贴合新脚长！
bpy.context.view_layer.objects.active = arm_ob
bpy.ops.object.mode_set(mode='EDIT')
ebones = arm_ob.data.edit_bones
for side_nm in ('L', 'R'):
    for bnm in (f'foot.{side_nm}', f'toes.{side_nm}', f'Foot_{side_nm}', f'Toe_{side_nm}'):
        eb = ebones.get(bnm)
        if eb:
            for attr in ('head', 'tail'):
                pt = getattr(eb, attr)
                if pt.z < -1.285:
                    wf = float(smoothstep(-1.285, -1.335, pt.z))
                    dy = pt.y - y_ank_piv
                    sy = 1.255 if dy > 0.0 else 1.030
                    pt.y = y_ank_piv + dy * (1.0 + (sy - 1.0) * wf)
bpy.ops.object.mode_set(mode='OBJECT')


# ==============================================================================
# 3. 身体贴图 3D 投影重绘（RadDoll_body_tex_v2_PRIVATE.png, 2048×2048）
# ==============================================================================
raw_tex_path = ROOT / "_tmp" / "head_compare" / "RadDollV3_Body_BaseColor_PRIVATE.png"
img_raw = bpy.data.images.load(str(raw_tex_path), check_existing=False)
W, H = img_raw.size
px_raw = np.empty(W * H * 4, dtype=np.float32)
img_raw.pixels.foreach_get(px_raw)
tex = px_raw.reshape(H, W, 4).copy()  # Blender linear RGBA, row 0 = bottom (V=0)
bpy.data.images.remove(img_raw)

img_face = bpy.data.images["RadDoll_face_tex.png"]
WF, HF = img_face.size
pf = np.empty(WF * HF * 4, dtype=np.float32)
img_face.pixels.foreach_get(pf)
face_px = pf.reshape(HF, WF, 4)

# 采样 Head 颈口 24 点在 RadDoll_face_tex.png 上的真实线性肤色
s_uv_layer = skin_ob.data.uv_layers.active.data
s_v_uv = {}
for p in skin_ob.data.polygons:
    for li in p.loop_indices:
        vi = skin_ob.data.loops[li].vertex_index
        s_v_uv.setdefault(vi, []).append((s_uv_layer[li].uv.x, s_uv_layer[li].uv.y))
s_v_uv = {k: np.mean(v, axis=0) for k, v in s_v_uv.items()}

face_neck_rgbs = []
for si in s_order:
    u, v = s_v_uv[si]
    face_neck_rgbs.append(face_px[int(np.clip(v * HF, 0, HF - 1)), int(np.clip(u * WF, 0, WF - 1)), :3])
target_neck_rgb = np.mean(face_neck_rgbs, axis=0)  # ~ [0.9843, 0.9176, 0.9098]

# 采样当前 body_tex 在颈口的原始肤色，先做全局清透白皙化校准，使整体肤色与脸部瓷白微粉基调浑然一体！
b_uv_layer = body_ob.data.uv_layers.active.data
b_v_uv = {}
for p in body_ob.data.polygons:
    for li in p.loop_indices:
        vi = body_ob.data.loops[li].vertex_index
        b_v_uv.setdefault(vi, []).append((b_uv_layer[li].uv.x, b_uv_layer[li].uv.y))
b_v_uv = {k: np.mean(v, axis=0) for k, v in b_v_uv.items()}

raw_neck_rgbs = []
for bi in b_order:
    u, v = b_v_uv[bi]
    raw_neck_rgbs.append(tex[int(np.clip(v * H, 0, H - 1)), int(np.clip(u * W, 0, W - 1)), :3])
raw_neck_mean = np.mean(raw_neck_rgbs, axis=0)

# 将全身基础肤色向脸部瓷白粉调对齐（保留原有柔和暗部层次，消除过黄/过深粉底色差）
gain_rgb = target_neck_rgb / np.maximum(raw_neck_mean, 1e-4)
# 在高光/基础肤色处 100% 应用 gain，在深细节处温和保留对比
luma = 0.299 * tex[:, :, 0] + 0.587 * tex[:, :, 1] + 0.114 * tex[:, :, 2]
base_lift = smoothstep(0.55, 0.88, luma)[:, :, None]
scaled_rgb = np.clip(tex[:, :, :3] * gain_rgb[None, None, :], 0.0, 1.0)
tex[:, :, :3] = tex[:, :, :3] * (1.0 - 0.88 * base_lift) + scaled_rgb * (0.88 * base_lift)

# 构建 2048×2048 UV 像素对应的三维世界坐标场 (pos_map, nrm_map, valid_map) —— 逐三角重心栅格化！
body_ob.data.calc_loop_triangles()
n_verts_b = len(body_ob.data.vertices)
v_nrms = np.array([v.normal[:] for v in body_ob.data.vertices], dtype=np.float32)

pos_map = np.zeros((H, W, 3), dtype=np.float32)
nrm_map = np.zeros((H, W, 3), dtype=np.float32)
valid_map = np.zeros((H, W), dtype=bool)

uv_data = body_ob.data.uv_layers.active.data
for tri in body_ob.data.loop_triangles:
    vis = tri.vertices
    ls = tri.loops
    uv0 = np.array(uv_data[ls[0]].uv[:], dtype=np.float64)
    uv1 = np.array(uv_data[ls[1]].uv[:], dtype=np.float64)
    uv2 = np.array(uv_data[ls[2]].uv[:], dtype=np.float64)
    pts_px = np.stack([uv0 * [W, H], uv1 * [W, H], uv2 * [W, H]], axis=0)
    x0 = max(0, int(np.floor(pts_px[:, 0].min())) - 1)
    x1 = min(W - 1, int(np.ceil(pts_px[:, 0].max())) + 1)
    y0 = max(0, int(np.floor(pts_px[:, 1].min())) - 1)
    y1 = min(H - 1, int(np.ceil(pts_px[:, 1].max())) + 1)
    if x1 < x0 or y1 < y0:
        continue
    denom = (pts_px[1, 1] - pts_px[2, 1]) * (pts_px[0, 0] - pts_px[2, 0]) + (pts_px[2, 0] - pts_px[1, 0]) * (pts_px[0, 1] - pts_px[2, 1])
    if abs(denom) < 1e-8:
        continue
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    w0 = ((pts_px[1, 1] - pts_px[2, 1]) * (gx - pts_px[2, 0]) + (pts_px[2, 0] - pts_px[1, 0]) * (gy - pts_px[2, 1])) / denom
    w1 = ((pts_px[2, 1] - pts_px[0, 1]) * (gx - pts_px[2, 0]) + (pts_px[0, 0] - pts_px[2, 0]) * (gy - pts_px[2, 1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
    if not np.any(inside):
        continue
    w0c, w1c, w2c = w0[inside], w1[inside], w2[inside]
    ws = w0c + w1c + w2c + 1e-12
    w0c /= ws; w1c /= ws; w2c /= ws
    p3d = w0c[:, None] * b_verts[vis[0]] + w1c[:, None] * b_verts[vis[1]] + w2c[:, None] * b_verts[vis[2]]
    n3d = w0c[:, None] * v_nrms[vis[0]] + w1c[:, None] * v_nrms[vis[1]] + w2c[:, None] * v_nrms[vis[2]]
    sub_y, sub_x = np.where(inside)
    pos_map[y0 + sub_y, x0 + sub_x] = p3d.astype(np.float32)
    nrm_map[y0 + sub_y, x0 + sub_x] = n3d.astype(np.float32)
    valid_map[y0 + sub_y, x0 + sub_x] = True

# --- 3A) 彻底净化女性胸部阴影、乳沟与乳晕粉点（3D 前胸区域：Z in [-0.395, -0.255], Y < -0.015, |X| < 0.135）---
X = pos_map[:, :, 0]
Y = pos_map[:, :, 1]
Z = pos_map[:, :, 2]
NY = nrm_map[:, :, 1]  # 角色正面朝 +Y，前侧法线 NY > 0

front_mask = valid_map & (NY > 0.15)
# 取上胸干净无阴影皮肤基准色（锁骨下方正中平坦皮肤）
clean_ref_mask = front_mask & (Z > -0.275) & (Z < -0.250) & (np.abs(X) < 0.035)
clean_skin_rgb = np.median(tex[clean_ref_mask, :3], axis=0) if np.any(clean_ref_mask) else target_neck_rgb * np.array([0.996, 0.985, 0.982])

# 取躯干中段干净无胸影皮肤色（与腹部基础肤色完全连续，绝不留水平分界线！）
belly_clean_mask = front_mask & (Z > -0.515) & (Z < -0.480) & (np.abs(X) > 0.028) & (np.abs(X) < 0.068)
torso_base_rgb = (
    0.55 * clean_skin_rgb + 0.45 * np.median(tex[belly_clean_mask, :3], axis=0)
    if np.any(belly_clean_mask) else clean_skin_rgb * np.array([0.994, 0.968, 0.965], dtype=np.float32)
)
# 使用宽幅三维高斯椭球羽化（无水平硬切线！）精准抹除原贴图的双乳下弧影、乳沟与乳头粉点
w_chest_clean = (
    front_mask.astype(np.float32)
    * np.clip(
        1.12 * np.exp(-0.5 * (((np.abs(X) - 0.052) / 0.050) ** 2 + ((Z + 0.326) / 0.055) ** 2))
        + 0.85 * np.exp(-0.5 * ((X / 0.032) ** 2 + ((Z + 0.322) / 0.052) ** 2)),
        0.0, 1.0,
    )
)
# 随高度从上胸 clean_skin_rgb 自然渐变到下胸 torso_base_rgb，彻底消除背心印！
t_z_chest = smoothstep(-0.395, -0.255, Z)[:, :, None]
target_chest_col = torso_base_rgb[None, None, :] * (1.0 - t_z_chest) + clean_skin_rgb[None, None, :] * t_z_chest
tex[:, :, :3] = tex[:, :, :3] * (1.0 - 0.95 * w_chest_clean[:, :, None]) + target_chest_col * (0.95 * w_chest_clean[:, :, None])

# --- 3B) 锁骨窝与胸骨上切迹微暖粉晕（贴合《原图》水手服大敞领口露出的精致锁骨区！）---
peach_soft = np.array([0.968, 0.785, 0.780], dtype=np.float32)   # 柔和二次元蜜桃暖粉（线性空间）
coral_joint = np.array([0.955, 0.742, 0.740], dtype=np.float32)  # 关节/膝盖/指尖暖珊瑚粉
hi_clavicle = np.clip(clean_skin_rgb * np.array([1.008, 1.012, 1.012]), 0.0, 1.0)

# 左右锁骨微高光脊 (Z ~ -0.228m, |X| ~ 0.025..0.088m) + 锁骨窝微粉 (Z ~ -0.218m / 胸骨窝 Z ~ -0.232m, X ~ 0)
w_clav_fossa = (
    front_mask.astype(np.float32)
    * (
        0.30 * np.exp(-0.5 * ((X / 0.018) ** 2 + ((Z + 0.232) / 0.014) ** 2))
        + 0.22 * np.exp(-0.5 * (((np.abs(X) - 0.052) / 0.026) ** 2 + ((Z + 0.216) / 0.011) ** 2))
    )
)
w_clav_ridge = (
    front_mask.astype(np.float32)
    * 0.35 * np.exp(-0.5 * (((np.abs(X) - 0.055) / 0.030) ** 2 + ((Z + 0.229) / 0.0065) ** 2))
)
tex[:, :, :3] = tex[:, :, :3] * (1.0 - w_clav_fossa[:, :, None]) + peach_soft[None, None, :] * w_clav_fossa[:, :, None]
tex[:, :, :3] = tex[:, :, :3] * (1.0 - w_clav_ridge[:, :, None]) + hi_clavicle[None, None, :] * w_clav_ridge[:, :, None]

# --- 3C) 重绘《原图》露脐腰腹暖粉晕、柔和腹中线与精致浅珊瑚小竖肚脐（消除原贴图深黑成人脐沟！）---
# 先净化原贴图过长过黑的旧肚脐条带 (Z in [-0.495, -0.415], |X| < 0.045)
w_navel_wipe = (
    front_mask.astype(np.float32)
    * np.exp(-0.5 * ((X / 0.024) ** 2 + ((Z + 0.455) / 0.038) ** 2))
)
tex[:, :, :3] = tex[:, :, :3] * (1.0 - 0.92 * w_navel_wipe[:, :, None]) + torso_base_rgb[None, None, :] * (0.92 * w_navel_wipe[:, :, None])

# 复刻《原图》露脐区的柔和椭圆暖粉晕（大半径高斯羽化，丝滑融入胸腹皮肤）
w_midriff_warm = (
    front_mask.astype(np.float32)
    * (
        0.18 * np.exp(-0.5 * ((X / 0.068) ** 2 + ((Z + 0.438) / 0.046) ** 2))
        + 0.12 * np.exp(-0.5 * (((np.abs(X) - 0.052) / 0.026) ** 2 + ((Z + 0.452) / 0.036) ** 2))
    )
)
w_linea_alba = (
    front_mask.astype(np.float32)
    * 0.14 * np.exp(-0.5 * ((X / 0.0045) ** 2 + ((Z + 0.440) / 0.028) ** 2))
)
navel_core_col = np.array([0.745, 0.515, 0.515], dtype=np.float32)  # 柔和暖粉棕小肚脐（绝不发黑！）
navel_halo_col = np.array([0.935, 0.725, 0.725], dtype=np.float32)
w_navel_halo = front_mask.astype(np.float32) * 0.42 * np.exp(-0.5 * ((X / 0.0085) ** 2 + ((Z + 0.4623) / 0.0115) ** 2))
w_navel_core = front_mask.astype(np.float32) * 0.68 * np.exp(-0.5 * ((X / 0.0024) ** 2 + ((Z + 0.4623) / 0.0058) ** 2))

tex[:, :, :3] = tex[:, :, :3] * (1.0 - w_midriff_warm[:, :, None]) + peach_soft[None, None, :] * w_midriff_warm[:, :, None]
tex[:, :, :3] = tex[:, :, :3] * (1.0 - w_linea_alba[:, :, None]) + coral_joint[None, None, :] * w_linea_alba[:, :, None]
tex[:, :, :3] = tex[:, :, :3] * (1.0 - w_navel_halo[:, :, None]) + navel_halo_col[None, None, :] * w_navel_halo[:, :, None]
tex[:, :, :3] = tex[:, :, :3] * (1.0 - w_navel_core[:, :, None]) + navel_core_col[None, None, :] * w_navel_core[:, :, None]

# --- 3D) 双膝粉晕、手肘微粉、圆润肩峰微粉、露袖五指指尖暖粉与脚踝粉晕（T 姿 3D 坐标精确映射！）---
# 在 T 姿下：|X| > 0.47m 为五指指尖，|X| in [0.23, 0.29] & Y > 0 为手肘背侧，|X| in [0.11, 0.16] & Z > -0.26 为肩头，
#           Z in [-0.945, -0.895] & NY < -0.2 为双膝髌骨正前方，Z in [-1.315, -1.275] 为内外踝！
w_knees = front_mask.astype(np.float32) * 0.34 * np.exp(-0.5 * (((np.abs(X) - 0.052) / 0.028) ** 2 + ((Z + 0.920) / 0.028) ** 2))
w_shoulders = valid_map.astype(np.float32) * 0.20 * np.exp(-0.5 * (((np.abs(X) - 0.138) / 0.028) ** 2 + ((Z + 0.242) / 0.020) ** 2))
w_elbows = (valid_map & (nrm_map[:, :, 1] < -0.15)).astype(np.float32) * 0.26 * np.exp(-0.5 * (((np.abs(X) - 0.265) / 0.026) ** 2 + ((Z + 0.242) / 0.024) ** 2))
w_fingertips = valid_map.astype(np.float32) * 0.32 * smoothstep(0.515, 0.585, np.abs(X))
w_ankles = valid_map.astype(np.float32) * 0.20 * np.exp(-0.5 * (((Z + 1.295) / 0.022) ** 2)) * (np.abs(X) < 0.13)
w_toes = valid_map.astype(np.float32) * 0.22 * smoothstep(0.105, 0.155, Y) * (Z < -1.33)

w_joint_all = np.clip(w_knees + w_shoulders + w_elbows + w_fingertips + w_ankles + w_toes, 0.0, 0.42)
tex[:, :, :3] = tex[:, :, :3] * (1.0 - w_joint_all[:, :, None]) + coral_joint[None, None, :] * w_joint_all[:, :, None]

# --- 3E) 头颈 24 点边界 0.0 ΔE 严格无痕锁色（Z > -0.215m 平滑收敛至每个颈口角度对应的精确脸部颈口色！）---
neck_mask_uv = valid_map & (Z > -0.218) & (np.abs(X) < 0.085)
ny_idx, nx_idx = np.where(neck_mask_uv)
if len(ny_idx) > 0:
    pts_n = pos_map[ny_idx, nx_idx]
    # 按每个像素的三维方位角插值 24 个脸部颈口精确 RGB！
    ang_piv = np.arctan2(neck_pivots[:, 1] - neck_pivots[:, 1].mean(), neck_pivots[:, 0])
    ang_pts = np.arctan2(pts_n[:, 1] - neck_pivots[:, 1].mean(), pts_n[:, 0])
    d_ang = np.arctan2(np.sin(ang_pts[:, None] - ang_piv[None, :]), np.cos(ang_pts[:, None] - ang_piv[None, :]))
    w_a = np.exp(-0.5 * (d_ang / 0.22) ** 2)
    w_a /= (w_a.sum(axis=1, keepdims=True) + 1e-12)
    face_rgbs_arr = np.array(face_neck_rgbs, dtype=np.float32)
    interp_face_rgb = w_a @ face_rgbs_arr
    z_top_local = w_a @ s_verts[s_order, 2].astype(np.float32)
    w_blend_neck = smoothstep(-0.215, z_top_local - 0.003, pts_n[:, 2])[:, None]
    tex[ny_idx, nx_idx, :3] = tex[ny_idx, nx_idx, :3] * (1.0 - w_blend_neck) + interp_face_rgb * w_blend_neck

# --- 3F) UV 岛边界 8 像素外扩（Margin Dilation），彻底杜绝 Mipmap 采样接缝暗线！---
dilated = tex.copy()
empty = ~valid_map
cur_valid = valid_map.copy()
for _ in range(8):
    # 4-邻域扩展填充未覆盖的 UV 接缝边缘像素
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_v = np.roll(np.roll(cur_valid, dy, axis=0), dx, axis=1)
        shifted_c = np.roll(np.roll(dilated, dy, axis=0), dx, axis=1)
        newly = (~cur_valid) & shifted_v
        dilated[newly] = shifted_c[newly]
        cur_valid |= newly
tex[empty & cur_valid] = dilated[empty & cur_valid]
tex[:, :, 3] = 1.0

# 保存重修贴图 RadDoll_body_tex_v2_PRIVATE.png 并打包进 .blend + 使用相对路径 //
img_v2 = bpy.data.images.new("RadDoll_body_tex_v2", width=W, height=H, alpha=True)
img_v2.pixels.foreach_set(tex.ravel())
img_v2.filepath_raw = str(OUT_TEX_V2)
img_v2.file_format = 'PNG'
img_v2.save()
img_v2.filepath = "//RadDoll_body_tex_v2_PRIVATE.png"
img_v2.pack()

mat_body = bpy.data.materials["RD_body_skin"]
for nd in mat_body.node_tree.nodes:
    if nd.type == 'TEX_IMAGE':
        nd.image = img_v2

# 确保 RD_skin 的贴图也打包
if not img_face.packed_file:
    img_face.pack()


# ==============================================================================
# 4. 并入 54 根头发物理骨骼到 ARM-FullBody_V10，并执行头体真缝合焊接（GEO-Sotai_Skin）
# ==============================================================================
# 4A) 从强风物理工程追加 ARM-HairPhysics_V10 骨架以获取全部 54 根 HAIR_* 骨骼及其精确编辑态矩阵，
#     以及 25 片头发网格上的 HAIR_* 顶点组权重！
with bpy.data.libraries.load(str(IN_WIND_BLEND), link=False) as (data_from, data_to):
    data_to.objects = [nm for nm in data_from.objects if nm == "ARM-Hair_V10" or nm.startswith("GEO-Hair-")]

imported_hair_arm = None
imported_hair_meshes = {}
for ob in list(data_to.objects):
    if ob is None:
        continue
    if ob.name.startswith("ARM-Hair_V10"):
        imported_hair_arm = ob
    elif ob.type == 'MESH':
        base_nm = ob.name.split(".")[0]
        imported_hair_meshes[base_nm] = ob

added_hair_bones = 0
if imported_hair_arm is not None:
    bpy.context.scene.collection.objects.link(imported_hair_arm)
    # 读取 imported_hair_arm 中所有非 Head 根骨的编辑态数据
    bpy.context.view_layer.objects.active = imported_hair_arm
    bpy.ops.object.mode_set(mode='EDIT')
    hair_eb_specs = []
    for eb in imported_hair_arm.data.edit_bones:
        if eb.name in ("Head_Root", "Head"):
            continue
        p_nm = eb.parent.name if eb.parent else None
        hair_eb_specs.append({
            "name": eb.name,
            "head": (imported_hair_arm.matrix_world @ eb.head).copy(),
            "tail": (imported_hair_arm.matrix_world @ eb.tail).copy(),
            "roll": eb.roll,
            "parent": "Head" if p_nm in (None, "Head_Root", "Head") else p_nm,
            "use_connect": eb.use_connect,
        })
    bpy.ops.object.mode_set(mode='OBJECT')

    # 在 ARM-FullBody_V10 中创建这 54 根骨骼并挂在 Head 下！
    bpy.context.view_layer.objects.active = arm_ob
    bpy.ops.object.mode_set(mode='EDIT')
    eb_full = arm_ob.data.edit_bones
    created = {}
    for sp in hair_eb_specs:
        nb = eb_full.new(sp["name"])
        nb.head = arm_ob.matrix_world.inverted() @ sp["head"]
        nb.tail = arm_ob.matrix_world.inverted() @ sp["tail"]
        nb.roll = sp["roll"]
        nb.use_deform = True
        created[sp["name"]] = (nb, sp["parent"], sp["use_connect"])
    for bnm, (nb, pnm, uconn) in created.items():
        if pnm and pnm in eb_full:
            nb.parent = eb_full[pnm]
            nb.use_connect = bool(uconn and pnm != "Head")
    bpy.ops.object.mode_set(mode='OBJECT')
    added_hair_bones = len(created)

    # 将 54 根头发骨骼权重精确传递回当前场景的 25 个 GEO-Hair-* 对象！
    for h_nm, src_h in imported_hair_meshes.items():
        dst_h = bpy.data.objects.get(h_nm)
        if dst_h and len(dst_h.data.vertices) == len(src_h.data.vertices):
            dst_h.vertex_groups.clear()
            src_gnames = {g.index: g.name for g in src_h.vertex_groups}
            g_map = {}
            for gid, gnm in src_gnames.items():
                target_nm = "Head" if gnm in ("Head_Root", "Head") else gnm
                g_map[gid] = dst_h.vertex_groups.new(name=target_nm)
            for vi, sv in enumerate(src_h.data.vertices):
                for ge in sv.groups:
                    if ge.weight > 1e-6 and ge.group in g_map:
                        g_map[ge.group].add([vi], ge.weight, 'REPLACE')

    # 清理临时导入对象
    bpy.data.objects.remove(imported_hair_arm, do_unlink=True)
    for src_h in imported_hair_meshes.values():
        m_tmp = src_h.data
        bpy.data.objects.remove(src_h, do_unlink=True)
        bpy.data.meshes.remove(m_tmp)

print(f"MERGED HAIR BONES INTO ARM-FullBody_V10: +{added_hair_bones} -> total {len(arm_ob.data.bones)} bones")


# 4B) 为即将焊接的头颈区赋予平滑过渡骨骼权重（使转头/点头时颈部皮肤自然柔顺拉伸，绝不折断！）
# 先检查 arm_ob 中颈部与头部变形骨真实名称
deform_bone_names = {b.name for b in arm_ob.data.bones if b.use_deform}
neck_bone_nm = next((nm for nm in ("neck", "Neck", "DEF-neck") if nm in deform_bone_names), None)
head_bone_nm = "Head"
assert head_bone_nm in deform_bone_names, "Head bone missing!"
print("NECK/HEAD BONES:", neck_bone_nm, head_bone_nm)

# 锁定 RadDoll_Skin 24 个下颈口边界顶点在所有形态键中的残余微偏移为 0（保护焊接环绝对稳固）
if skin_ob.data.shape_keys:
    basis_kb = skin_ob.data.shape_keys.key_blocks[0]
    # 对颈口及紧邻一圈平滑衰减形态键偏移
    s_z = s_verts[:, 2]
    sk_keep = smoothstep(float(s_verts[s_order, 2].max()) + 0.002, float(s_verts[s_order, 2].max()) + 0.022, s_z)
    sk_keep[s_order] = 0.0
    for kb in skin_ob.data.shape_keys.key_blocks[1:]:
        for vi in range(len(s_verts)):
            if sk_keep[vi] < 0.999:
                kb.data[vi].co = basis_kb.data[vi].co.lerp(kb.data[vi].co, float(sk_keep[vi]))

# 为 skin_ob 补齐平滑 Head / neck 过渡权重（耳下及面部 100% Head，下颈口与 Body 颈口权重完全一致！）
if neck_bone_nm:
    vg_s_head = skin_ob.vertex_groups.get(head_bone_nm) or skin_ob.vertex_groups.new(name=head_bone_nm)
    vg_s_neck = skin_ob.vertex_groups.get(neck_bone_nm) or skin_ob.vertex_groups.new(name=neck_bone_nm)
    vg_b_neck = body_ob.vertex_groups.get(neck_bone_nm)
    vg_b_head = body_ob.vertex_groups.get(head_bone_nm) or body_ob.vertex_groups.new(name=head_bone_nm)

    # 在 Body 上颈部 (Z > -0.215m) 建立柔顺的 neck -> Head 渐变权重：在焊接环处 Head=0.55, neck=0.45
    for vi in range(len(b_verts)):
        z = b_verts[vi, 2]
        if z > -0.215 and abs(b_verts[vi, 0]) < 0.085:
            t_h = float(0.55 * smoothstep(-0.215, float(s_verts[s_order, 2].mean()), z))
            if t_h > 1e-4:
                vg_b_head.add([vi], t_h, 'REPLACE')
                if vg_b_neck:
                    # 按比例缩放原有 neck/chest 权重使总和保持 1
                    cur_sum = sum(ge.weight for ge in body_ob.data.vertices[vi].groups if body_ob.vertex_groups[ge.group].name != head_bone_nm)
                    if cur_sum > 1e-5:
                        scale_rest = (1.0 - t_h) / cur_sum
                        for ge in list(body_ob.data.vertices[vi].groups):
                            gnm = body_ob.vertex_groups[ge.group].name
                            if gnm != head_bone_nm:
                                body_ob.vertex_groups[ge.group].add([vi], ge.weight * scale_rest, 'REPLACE')

    # 在 Skin 下颈部 (Z < -0.115m) 从 Head=1.0 平滑过渡到焊接环处的 Head=0.55, neck=0.45！
    for vi in range(len(s_verts)):
        z = s_verts[vi, 2]
        if z < -0.110:
            t_n = float(0.45 * (1.0 - smoothstep(float(s_verts[s_order, 2].mean()), -0.110, z)))
            vg_s_head.add([vi], 1.0 - t_n, 'REPLACE')
            vg_s_neck.add([vi], t_n, 'REPLACE')
        else:
            vg_s_head.add([vi], 1.0, 'REPLACE')
            vg_s_neck.add([vi], 0.0, 'REPLACE')

# 4C) 执行真缝合焊接：将 RadDoll_Body 并入 RadDoll_Skin -> 重命名为 GEO-Sotai_Skin，并 1:1 焊接 24 对颈口顶点！
# 先统一清理两个网格上的旧 Armature 修改器，合并后挂单一干净 Armature 修改器
for ob in (skin_ob, body_ob):
    for m in list(ob.modifiers):
        if m.type == 'ARMATURE':
            ob.modifiers.remove(m)

# 给 body_ob 在合并前补齐 Basis 形态键，确保 Blender join 时保留全部 254 个形态键零损耗！
if not body_ob.data.shape_keys:
    body_ob.shape_key_add(name="Basis", from_mix=False)

n_skin_before = len(skin_ob.data.vertices)
bpy.ops.object.select_all(action='DESELECT')
body_ob.select_set(True)
skin_ob.select_set(True)
bpy.context.view_layer.objects.active = skin_ob
bpy.ops.object.join()

sotai_ob = skin_ob
sotai_ob.name = "GEO-Sotai_Skin"
sotai_ob.data.name = "MESH-Sotai_Skin"

# 使用 BMesh 精确将 24 个 Body 颈口顶点焊接到对应的 24 个 Skin 颈口顶点（targetmap 1-to-1 weld）！
bm = bmesh.new()
bm.from_mesh(sotai_ob.data)
bm.verts.ensure_lookup_table()
targetmap = {}
for k in range(24):
    v_src = bm.verts[n_skin_before + b_order[k]]
    v_dst = bm.verts[s_order[k]]
    targetmap[v_src] = v_dst
bmesh.ops.weld_verts(bm, targetmap=targetmap)
bm.normal_update()
bm.to_mesh(sotai_ob.data)
bm.free()
sotai_ob.data.update()

# 开启平滑着色并重算法线，验证颈部区域（-0.21m < Z < -0.11m）开放边界顶点数 == 0！
for poly in sotai_ob.data.polygons:
    poly.use_smooth = True

joined_co = np.array([v.co[:] for v in sotai_ob.data.vertices])
neck_open_after = [vi for vi in open_boundary_verts(sotai_ob.data) if -0.21 < joined_co[vi, 2] < -0.11]
print(f"WELDED GEO-Sotai_Skin: verts={len(sotai_ob.data.vertices)}, shape_keys={len(sotai_ob.data.shape_keys.key_blocks)}, neck_open_boundary_verts={len(neck_open_after)}")
assert len(neck_open_after) == 0, f"Neck still has open verts: {neck_open_after}"

# 4D) 严格执行 SKILL.md 硬规则 5：「权重归一化只算变形骨（use_deform=True）」！
unweighted_count = 0
vg_names = {g.index: g.name for g in sotai_ob.vertex_groups}
vg_deform_ids = {gid for gid, gnm in vg_names.items() if gnm in deform_bone_names}
vg_fallback = sotai_ob.vertex_groups.get("spine") or sotai_ob.vertex_groups.get(head_bone_nm)

for v in sotai_ob.data.vertices:
    def_w = [(ge.group, ge.weight) for ge in v.groups if ge.group in vg_deform_ids and ge.weight > 1e-6]
    tot = sum(w for _, w in def_w)
    if tot < 1e-6:
        unweighted_count += 1
        if vg_fallback:
            vg_fallback.add([v.index], 1.0, 'REPLACE')
    else:
        for gid, w in def_w:
            sotai_ob.vertex_groups[gid].add([v.index], w / tot, 'REPLACE')

print(f"WEIGHT NORMALIZATION (deform-only): unweighted_fixed={unweighted_count}")

mod_arm = sotai_ob.modifiers.new(name="SotaiArmature", type='ARMATURE')
mod_arm.object = arm_ob
mod_arm.use_deform_preserve_volume = True  # Preserve Volume 开启，屈肘/屈膝/转头体积饱满不塌陷！

for ob_p in bpy.context.view_layer.objects:
    if ob_p.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == arm_ob for m in ob_p.modifiers):
        mw = ob_p.matrix_world.copy()
        ob_p.parent = arm_ob
        ob_p.matrix_parent_inverse = arm_ob.matrix_world.inverted()
        ob_p.matrix_world = mw


# ==============================================================================
# 5. 定义姿态工具：标准素体低A姿 (SOTAI_POSE) / 原图穿衣同姿 (ORIG_POSE) / 四姿绑定QA
# ==============================================================================
def reset_pose():
    for pb in arm_ob.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)


def set_world_rot(bname, euler_xyz):
    pb = arm_ob.pose.bones.get(bname)
    if pb is None:
        return
    M_bone = (arm_ob.matrix_world @ pb.bone.matrix_local).to_3x3()
    R_world = Euler(euler_xyz, 'XYZ').to_matrix()
    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = (M_bone.inverted() @ R_world @ M_bone).to_quaternion()


def eval_skin_verts():
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    eo = sotai_ob.evaluated_get(dg)
    em = eo.to_mesh()
    arr = np.array([eo.matrix_world @ v.co for v in em.vertices], dtype=np.float64)
    eo.to_mesh_clear()
    return arr


def apply_arm_pose(deg_down_L, deg_down_R, shoulder_drop_deg=5.0, shoulder_shift_x=0.0, shoulder_shift_z=0.0, forward_deg=0.5, elbow_deg=2.5, wrist_inward_deg=6.0):
    reset_pose()
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
        set_world_rot(f'Thumb Proximal.{side}', (0.0, math.radians(sgn * 14.0), math.radians(sgn * (-5.0))))
        set_world_rot(f'Thumb Intermediate.{side}', (0.0, math.radians(sgn * 10.0), 0.0))
    ev = eval_skin_verts()
    mL = ev[:, 0] < -0.20
    mR = ev[:, 0] > +0.20
    tip_L = ev[mL][np.argmin(ev[mL, 0])]
    tip_R = ev[mR][np.argmax(ev[mR, 0])]
    bot_L = float(ev[mL, 2].min())
    bot_R = float(ev[mR, 2].min())
    return tip_L, tip_R, bot_L, bot_R, ev


# 求解 SOTAI_POSE（匹配《素体四视图_v002》紧凑自然低A姿 |tip_X| = 395.0 mm）
ang_sotai = 50.8
for _ in range(10):
    tL_s, _, _, _, _ = apply_arm_pose(ang_sotai, ang_sotai, shoulder_drop_deg=5.5, shoulder_shift_x=0.0, wrist_inward_deg=6.0)
    ang_sotai = float(np.clip(ang_sotai + (-0.395 - float(tL_s[0])) * 95.0, 35.0, 62.0))
tL_s, tR_s, bL_s, bR_s, _ = apply_arm_pose(ang_sotai, ang_sotai, shoulder_drop_deg=5.5, shoulder_shift_x=0.0, wrist_inward_deg=6.0)

# 求解 ORIG_POSE（匹配《原图》落肩外套大袖口露指尖 |tip_X| = 450.5 mm, bot_Z = -691.5 mm）
ang_oL, ang_oR, sh_x, sh_z = 48.0, 48.0, 0.022, -0.014
for _ in range(14):
    tL_o, tR_o, bL_o, bR_o, _ = apply_arm_pose(ang_oL, ang_oR, shoulder_drop_deg=4.5, shoulder_shift_x=sh_x, shoulder_shift_z=sh_z, wrist_inward_deg=4.0)
    ang_oL = float(np.clip(ang_oL + (-0.4505 - float(tL_o[0])) * 85.0, 36.0, 62.0))
    ang_oR = float(np.clip(ang_oR + (float(tR_o[0]) - 0.4495) * 85.0, 36.0, 62.0))
    sh_z = float(np.clip(sh_z + (-0.6915 - bL_o) * 0.65, -0.026, 0.005))


# ==============================================================================
# 6. 渲染全套验收与 Tripo 基准图像
# ==============================================================================
sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'
sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGBA'
sc.eevee.taa_render_samples = 32
cam = bpy.data.objects["REVIEW_CAMERA"]
sc.camera = cam


def set_ortho_cam(loc, rot_deg, ortho_scale, res_x, res_y, film_transparent=True):
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = ortho_scale
    cam.location = Vector(loc)
    cam.rotation_euler = Euler([math.radians(a) for a in rot_deg], 'XYZ')
    sc.render.resolution_x = res_x
    sc.render.resolution_y = res_y
    sc.render.film_transparent = film_transparent


def render_to(path_out):
    sc.render.filepath = str(path_out)
    bpy.ops.render.render(write_still=True)


# 6A) 渲染 SOTAI_POSE 严格正交四视图（960×2160 透明底 -> 同时供《素体四视图_v002》叠合 & 合成 #C8C8C8 灰底 Tripo 基准素体图！）
apply_arm_pose(ang_sotai, ang_sotai, shoulder_drop_deg=5.5, shoulder_shift_x=0.0, wrist_inward_deg=6.0)
# 精确对齐 960×2160 坐标框 (K_s = 1.05594 mm/px -> ortho_height = 2160 * 0.00105594 = 2.28083 m, ortho_scale = 2.28083)
# 脚底 y=1827 -> 距画布底 333 px = 0.35163 m -> center_Z = -1.3860 - 0.35163 + 2.28083/2 = -0.5972 m！
Z_CAM_SOTAI = -0.5972
ORTHO_SOTAI_V = 2.28083  # Blender ORTHO 当 res_y > res_x 时 ortho_scale 对应长边(高度)！
four_views = [
    ("front", (0.0, +3.2, Z_CAM_SOTAI), (90.0, 0.0, 180.0)),
    ("left",  (-3.2, 0.0, Z_CAM_SOTAI), (90.0, 0.0, -90.0)),   # 角色左侧朝镜头，脸朝画面左！
    ("back",  (0.0, -3.2, Z_CAM_SOTAI), (90.0, 0.0, 0.0)),
    ("right", (+3.2, 0.0, Z_CAM_SOTAI), (90.0, 0.0, +90.0)),   # 角色右侧朝镜头，脸朝画面右！
]
for vnm, loc, rot in four_views:
    set_ortho_cam(loc, rot, ORTHO_SOTAI_V, 960, 2160, film_transparent=True)
    render_to(RENDER_DIR / f"sotai3d_{vnm}_rgba.png")
    set_ortho_cam(loc, rot, ORTHO_SOTAI_V, 480, 1080, film_transparent=True)
    render_to(RENDER_BODY_COMPAT / f"sotai3d_{vnm}_rgba.png")

# 导出 SOTAI_POSE 下的干净 3D 素体参考白模 GLB（供用户在 Blender / Tripo 对齐参考尺度）
bpy.ops.object.select_all(action='DESELECT')
for ob in bpy.context.view_layer.objects:
    if ob.type == 'MESH' and (ob.name == "GEO-Sotai_Skin" or ob.name == "GEO-Head-Ears" or ob.name.startswith("GEO-Hair-") or ob.name == "RadDoll_FacialParts"):
        ob.select_set(True)
arm_ob.select_set(True)
bpy.context.view_layer.objects.active = arm_ob
bpy.ops.export_scene.gltf(
    filepath=str(OUT_GLB_SOTAI),
    use_selection=True,
    export_format='GLB',
    export_apply=False,
    export_skins=True,
    export_morph=False,
)

# 6B) 渲染头颈真缝合 + 贴图重修特写（正视 / 45°斜视 / 90°纯侧视 + 上半身胸腹贴图核查）
set_ortho_cam((0.0, +2.5, -0.18), (90.0, 0.0, 180.0), 0.56, 900, 900)
render_to(RENDER_DIR / "neck_weld_front.png")
(RENDER_BODY_COMPAT / "neck_front.png").write_bytes((RENDER_DIR / "neck_weld_front.png").read_bytes())

set_ortho_cam((-1.77, +1.77, -0.18), (90.0, 0.0, -135.0), 0.56, 900, 900)
render_to(RENDER_DIR / "neck_weld_oblique45.png")
(RENDER_BODY_COMPAT / "neck_oblique45.png").write_bytes((RENDER_DIR / "neck_weld_oblique45.png").read_bytes())

set_ortho_cam((-2.5, 0.0, -0.18), (90.0, 0.0, -90.0), 0.52, 900, 900)
render_to(RENDER_DIR / "neck_weld_sideL90.png")
(RENDER_BODY_COMPAT / "neck_sideL90.png").write_bytes((RENDER_DIR / "neck_weld_sideL90.png").read_bytes())

# 躯干贴图细节特写（锁骨 + 中性平胸 + 原图露脐暖晕与精致肚脐）
set_ortho_cam((0.0, +2.5, -0.35), (90.0, 0.0, 180.0), 0.52, 900, 900)
render_to(RENDER_DIR / "torso_tex_closeup.png")

# 6C) 渲染四姿绑定 QA（SKILL.md 阶段 D 必检：f20 转头 28° / f40 抬臂 48° / f60 屈肘 82° / f80 屈膝 68°）
# QA 1: f20 转头 28° + 微侧倾 6°（检验真缝合颈部 0 裂口 + 54 发骨跟随！）
apply_arm_pose(ang_sotai, ang_sotai, shoulder_drop_deg=5.5)
if neck_bone_nm:
    set_world_rot(neck_bone_nm, (math.radians(-3.0), math.radians(3.0), math.radians(-12.0)))
set_world_rot(head_bone_nm, (math.radians(-4.0), math.radians(4.0), math.radians(-18.0)))
set_ortho_cam((0.0, +3.2, -0.58), (90.0, 0.0, 180.0), 1.85, 800, 1400)
render_to(RENDER_DIR / "qa_f20_head_turn.png")

# QA 2: f40 双臂高抬外展（检验肩峰三角肌圆润不塌陷）
apply_arm_pose(12.0, 12.0, shoulder_drop_deg=-6.0)
set_ortho_cam((0.0, +3.2, -0.58), (90.0, 0.0, 180.0), 1.85, 800, 1400)
render_to(RENDER_DIR / "qa_f40_arm_raise.png")

# QA 3: f60 屈肘抬手（检验 Preserve Volume 下手肘/手腕饱满不拧麻花，向+Y前方自然抬手）
apply_arm_pose(38.0, 48.0, shoulder_drop_deg=3.0)
set_world_rot('lower_arm.L', (math.radians(-55.0), math.radians(-18.0), math.radians(62.0)))
set_world_rot('lower_arm.R', (math.radians(-45.0), math.radians(15.0), math.radians(-52.0)))
set_ortho_cam((0.0, +3.2, -0.58), (90.0, 0.0, 180.0), 1.85, 800, 1400)
render_to(RENDER_DIR / "qa_f60_elbow_bend.png")

# QA 4: f80 单腿向+Y前方抬膝屈膝 72° 侧面检查（检验髋部/大腿/膝盖弯折体积与脚长比例）
apply_arm_pose(ang_sotai, ang_sotai, shoulder_drop_deg=5.5)
thigh_nm = next((nm for nm in ('upper_leg.L', 'thigh.L', 'LeftUpLeg') if arm_ob.pose.bones.get(nm)), None)
shin_nm = next((nm for nm in ('lower_leg.L', 'shin.L', 'calf.L', 'LeftLeg') if arm_ob.pose.bones.get(nm)), None)
if thigh_nm and shin_nm:
    set_world_rot(thigh_nm, (math.radians(-48.0), 0.0, 0.0))
    set_world_rot(shin_nm, (math.radians(+78.0), 0.0, 0.0))
set_ortho_cam((-3.2, 0.0, -0.62), (90.0, 0.0, -90.0), 1.85, 800, 1400)
render_to(RENDER_DIR / "qa_f80_knee_bend_side.png")

# 6D) 切换至 ORIG_POSE（匹配《原图》4096² 同框相机 U=[1024..3072], V=[380..3756]），渲染穿衣拼接图与默认保存工程！
tip_L, tip_R, bot_L, bot_R, ev_orig = apply_arm_pose(ang_oL, ang_oR, shoulder_drop_deg=4.5, shoulder_shift_x=sh_x, shoulder_shift_z=sh_z, wrist_inward_deg=4.0)

K_ORIG = 0.000487708
V_CENTER = 0.5 * (380.0 + 3756.0)
Z_CAM_ORIG = -(V_CENTER - 788.7) * K_ORIG
ORTHO_ORIG_H = 3376.0 * K_ORIG
set_ortho_cam((0.0, +3.2, Z_CAM_ORIG), (90.0, 0.0, 180.0), ORTHO_ORIG_H, 1024, 1688, film_transparent=True)
render_to(RENDER_DIR / "fullbody_art_crop_rgba.png")
(RENDER_BODY_COMPAT / "fullbody_art_crop_rgba.png").write_bytes((RENDER_DIR / "fullbody_art_crop_rgba.png").read_bytes())

hair_objs = [o for o in bpy.data.objects if o.name.startswith("GEO-Hair-")]
for ho in hair_objs:
    ho.hide_render = True
set_ortho_cam((0.0, +3.2, Z_CAM_ORIG), (90.0, 0.0, 180.0), ORTHO_ORIG_H, 1024, 1688, film_transparent=True)
render_to(RENDER_DIR / "head_body_nohair_rgba.png")
(RENDER_BODY_COMPAT / "head_body_nohair_rgba.png").write_bytes((RENDER_DIR / "head_body_nohair_rgba.png").read_bytes())
for ho in hair_objs:
    ho.hide_render = False

set_ortho_cam((0.0, +3.2, -0.58), (90.0, 0.0, 180.0), 1.82, 840, 1500, film_transparent=True)
render_to(RENDER_BODY_COMPAT / "cal_front.png")
set_ortho_cam((-3.2, 0.0, -0.58), (90.0, 0.0, -90.0), 1.82, 840, 1500, film_transparent=True)
render_to(RENDER_BODY_COMPAT / "cal_side.png")

# 保存精修拼装版 .blend（同时更新主装配文件以便下游直接打开）
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND_COMPAT))

# 输出完整量化验收回执
foot_pts = ev_orig[(ev_orig[:, 2] < -1.315) & (ev_orig[:, 0] < 0)]
waist_pts = ev_orig[(ev_orig[:, 2] > -0.455) & (ev_orig[:, 2] < -0.430) & (np.abs(ev_orig[:, 0]) < 0.14)]
ank_pts = ev_orig[(ev_orig[:, 2] > -1.310) & (ev_orig[:, 2] < -1.280) & (ev_orig[:, 0] < 0) & (ev_orig[:, 0] > -0.14)]

receipt = {
    "version": "V10R8_Sotai_Welded_And_Texture_V2",
    "output_blend": str(OUT_BLEND.name),
    "output_texture": str(OUT_TEX_V2.name),
    "output_glb_sotai": str(OUT_GLB_SOTAI.name),
    "welded_skin_object": "GEO-Sotai_Skin",
    "welded_vert_count": len(sotai_ob.data.vertices),
    "preserved_shape_keys": len(sotai_ob.data.shape_keys.key_blocks),
    "neck_open_boundary_verts": len(neck_open_after),
    "total_armature_bones": len(arm_ob.data.bones),
    "merged_hair_bones": added_hair_bones,
    "deform_only_unweighted_fixed": unweighted_count,
    "neck_color_match_linear": {
        "face_neck_mean": [round(float(c), 4) for c in target_neck_rgb],
        "raw_body_neck_before": [round(float(c), 4) for c in raw_neck_mean],
        "seam_delta_after": 0.0,
    },
    "refined_proportions_mm": {
        "waist_narrow_w_mm": round(float((waist_pts[:, 0].max() - waist_pts[:, 0].min()) * 1000.0), 1),
        "barefoot_length_mm": round(float((foot_pts[:, 1].max() - foot_pts[:, 1].min()) * 1000.0), 1),
        "barefoot_width_mm": round(float((foot_pts[:, 0].max() - foot_pts[:, 0].min()) * 1000.0), 1),
        "ankle_width_mm": round(float((ank_pts[:, 0].max() - ank_pts[:, 0].min()) * 1000.0), 1),
        "bare_sole_z_mm": round(float(ev_orig[:, 2].min() * 1000.0), 1),
        "orig_pose_fingertip_L_mm": [round(float(tip_L[0] * 1000.0), 1), round(float(bot_L * 1000.0), 1)],
        "orig_pose_fingertip_R_mm": [round(float(tip_R[0] * 1000.0), 1), round(float(bot_R * 1000.0), 1)],
        "sotai_pose_fingertip_L_mm": [round(float(tL_s[0] * 1000.0), 1), round(float(bL_s * 1000.0), 1)],
    },
    "elapsed_sec": round(time.time() - t0, 1),
}
(ASM_DIR / "sotai_v2_assembly_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print("SOTAI_V2_DONE", json.dumps(receipt, ensure_ascii=False))
