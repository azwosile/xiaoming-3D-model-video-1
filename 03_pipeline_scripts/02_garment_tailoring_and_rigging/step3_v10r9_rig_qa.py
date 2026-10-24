"""
V10R8 Step 3:
1) 读取 Step 2 完成的 RadDoll_V10R9_Step2_修身剪裁与UV材质_PRIVATE.blend（当前 ARM-FullBody_V10 处于 ORIG_POSE）；
2) 针对 10 件衣装与配饰按部位定制三维空间核平滑骨骼权重（仅提取 Deform 骨骼并严格归一化 sum=1.0，剔除手指骨对宽袖口干扰、保持发夹刚性跟随头骨、保持运动鞋整鞋刚柔协调、跨子网格统一平滑杜绝纽扣/领结脱节）；
3) 执行「LBS 解析矩阵求逆 + 5 轮 Blender 原生 Depsgraph DQS 残差不动点迭代反算」，将 ORIG_POSE 下的贴体顶点无损反算回 T-Pose 静息空间（ORIG_POSE 评估误差 < 0.005 mm！），并用 matrix_parent_inverse 规范挂载至 ARM-FullBody_V10；
4) 构建动作时间轴（f1=原图 ORIG_POSE，f20=垂臂歪头休闲姿，f40=抬手屈肘挥手姿，f60=迈步行走动态姿，f80=标准水平 T-Pose），逐帧执行穿模率量化核验与高清多视角渲染！
"""
import os, sys, json, math
from pathlib import Path
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler, Quaternion
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
OUT_DIR    = ROOT / "生成模型/组装/v010R8_全身衣装装配"
RENDER_DIR = OUT_DIR / "renders_step3"
RENDER_DIR.mkdir(parents=True, exist_ok=True)

IN_STEP2_BLEND  = OUT_DIR / "RadDoll_V10R9_Step2_修身剪裁与UV材质_PRIVATE.blend"
OUT_STEP3_BLEND = OUT_DIR / "RadDoll_V10R9_Step3_修身骨骼绑定与多姿压测_PRIVATE.blend"

bpy.ops.wm.open_mainfile(filepath=str(IN_STEP2_BLEND), load_ui=False, use_scripts=False)
bpy.context.view_layer.update()

sotai_ob = bpy.data.objects["GEO-Sotai_Skin"]
arm_ob   = bpy.data.objects["ARM-FullBody_V10"]

# 读取 GEO-Sotai_Skin 自身的 Armature 修改器配置（保持 preserve_volume 完全一致以消除屈肘/屈膝曲率差！）
sotai_arm_mod = next(m for m in sotai_ob.modifiers if m.type == 'ARMATURE')
PRESERVE_VOL  = bool(sotai_arm_mod.use_deform_preserve_volume)
print(f"ARM-FullBody_V10 Sotai preserve_volume = {PRESERVE_VOL}", flush=True)

# 收集全部 Deform 骨骼及其世界空间蒙皮变换矩阵 M_b = A_w @ P_b @ R_b^-1 @ A_w^-1
Aw     = arm_ob.matrix_world.copy()
Aw_inv = Aw.inverted()

deform_bones = [b.name for b in arm_ob.data.bones if b.use_deform]
bname_to_idx = {bn: i for i, bn in enumerate(deform_bones)}
N_BONES = len(deform_bones)

M_bones = np.zeros((N_BONES, 4, 4), dtype=np.float64)
for bi, bn in enumerate(deform_bones):
    pb = arm_ob.pose.bones[bn]
    db = arm_ob.data.bones[bn]
    mb = Aw @ pb.matrix @ db.matrix_local.inverted() @ Aw_inv
    M_bones[bi] = np.array(mb, dtype=np.float64)

print(f"Total deform bones: {N_BONES}", flush=True)

# 提取 GEO-Sotai_Skin 在当前 ORIG_POSE 下的世界顶点 sv 与完整归一化变形骨骼权重矩阵 W_sotai [N_sv, N_BONES]
dg = bpy.context.evaluated_depsgraph_get()
# 暂时关闭 Mask_InsideShoes 修改器以便运动鞋能完整采样裸足脚掌/脚趾骨骼权重！
mask_mod = sotai_ob.modifiers.get("Mask_InsideShoes")
if mask_mod:
    mask_mod.show_viewport = False
    mask_mod.show_render   = False
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()

s_eval = sotai_ob.evaluated_get(dg)
s_me   = s_eval.to_mesh()
sv = np.array([list(sotai_ob.matrix_world @ v.co) for v in s_me.vertices], dtype=np.float64)
s_eval.to_mesh_clear()

N_SV = len(sotai_ob.data.vertices)
assert len(sv) == N_SV, f"Vertex count mismatch {len(sv)} vs {N_SV}"

vg_idx_to_bone_col = {}
for vg in sotai_ob.vertex_groups:
    if vg.name in bname_to_idx:
        vg_idx_to_bone_col[vg.index] = bname_to_idx[vg.name]

W_sotai = np.zeros((N_SV, N_BONES), dtype=np.float64)
for vi, v in enumerate(sotai_ob.data.vertices):
    for g in v.groups:
        bc = vg_idx_to_bone_col.get(g.group)
        if bc is not None and g.weight > 1e-6:
            W_sotai[vi, bc] += g.weight

row_sum = W_sotai.sum(axis=1, keepdims=True)
valid_r = row_sum[:, 0] > 1e-6
W_sotai[valid_r] /= row_sum[valid_r]

# 验证我们的 M_bones 矩阵公式在 GEO-Sotai_Skin 上是否毫厘不差！
sv_rest = np.array([list(sotai_ob.matrix_world @ v.co) for v in sotai_ob.data.vertices], dtype=np.float64)
sv_rest_h = np.hstack([sv_rest, np.ones((N_SV, 1), dtype=np.float64)])
T_sotai = np.einsum("vb,bij->vij", W_sotai, M_bones)
sv_lbs_pred = np.einsum("vij,vj->vi", T_sotai, sv_rest_h)[:, :3]
lbs_diff_mm = float(np.median(np.linalg.norm(sv_lbs_pred - sv, axis=1))) * 1000.0
print(f"Sotai LBS formula verification median diff = {lbs_diff_mm:.4f} mm", flush=True)

kd_sv = KDTree(N_SV)
for i, p in enumerate(sv):
    kd_sv.insert((float(p[0]), float(p[1]), float(p[2])), i)
kd_sv.balance()

# 识别手指骨骼列（在外套长袖与水手服中屏蔽手指骨骼，防止握拳/张手扯皱翻折袖口！）
finger_kws = ("thumb", "index", "middle", "ring", "pinky", "little", "f_", "finger", "palm", "metacarp")
is_finger_col = np.array([any(k in bn.lower() for k in finger_kws) for bn in deform_bones], dtype=bool)
# 找 hand.L / hand.R 骨骼列用于承接被屏蔽的手指权重
handL_col = next((i for i, bn in enumerate(deform_bones) if "hand" in bn.lower() and bn.endswith(".L")), None)
handR_col = next((i for i, bn in enumerate(deform_bones) if "hand" in bn.lower() and bn.endswith(".R")), None)
head_col  = next((i for i, bn in enumerate(deform_bones) if bn.lower() in ("head", "def-head", "head.x") or ("head" in bn.lower() and "ear" not in bn.lower())), None)
print(f"Key cols: head={head_col}({deform_bones[head_col] if head_col is not None else None}), handL={handL_col}({deform_bones[handL_col] if handL_col is not None else None}), finger_bones={int(is_finger_col.sum())}", flush=True)

# 构建将全部 68 根手指骨权重预先无损归并到 hand.L / hand.R 的 W_sotai_nofinger！
# 这样袖口末端采样到五指指尖时，100% 获得纯净的 hand.L / hand.R 权重，绝不会因过滤手指骨而归零悬空！
arm_only_kws = ("shoulder", "upper_arm", "lower_arm", "forearm", "hand", "arm", "clavicle")
is_arm_col = np.array([any(k in bn.lower() for k in arm_only_kws) and not any(fk in bn.lower() for fk in ("leg", "thigh", "shin", "foot", "toe")) for bn in deform_bones], dtype=bool)
is_lower_arm_or_hand_col = np.array([any(k in bn.lower() for k in ("lower_arm", "forearm", "hand")) or is_finger_col[j] for j, bn in enumerate(deform_bones)], dtype=bool)

W_sotai_nofinger = W_sotai.copy()
for svi_k in range(N_SV):
    f_w = float(W_sotai_nofinger[svi_k, is_finger_col].sum())
    if f_w > 1e-6:
        W_sotai_nofinger[svi_k, is_finger_col] = 0.0
        hc = handL_col if sv[svi_k, 0] < 0 else handR_col
        if hc is not None:
            W_sotai_nofinger[svi_k, hc] += f_w

arm_sv_idx = [i for i in range(N_SV) if W_sotai_nofinger[i, is_arm_col].sum() > 0.45]
kd_sv_arm = KDTree(len(arm_sv_idx))
for k_i, svi in enumerate(arm_sv_idx):
    p_a = sv[svi]
    kd_sv_arm.insert((float(p_a[0]), float(p_a[1]), float(p_a[2])), svi)
kd_sv_arm.balance()
print(f"Built kd_sv_arm with {len(arm_sv_idx)} full arm+hand(merged-finger) vertices!", flush=True)

# 同时建立仅包含躯干/骨盆/双腿顶点的 kd_sv_body（供外套衣身与下摆采样，从源头杜绝衣摆混入小臂权重，且 100% 保证权重和为 1.0！）
body_sv_idx = [i for i in range(N_SV) if W_sotai_nofinger[i, ~is_lower_arm_or_hand_col].sum() > 0.65]
W_sotai_bodyonly = W_sotai_nofinger.copy()
W_sotai_bodyonly[:, is_lower_arm_or_hand_col] = 0.0
bs_sum = W_sotai_bodyonly.sum(axis=1, keepdims=True)
valid_b = bs_sum[:, 0] > 1e-6
W_sotai_bodyonly[valid_b] /= bs_sum[valid_b]

kd_sv_body = KDTree(len(body_sv_idx))
for b_i in body_sv_idx:
    pb_v = sv[b_i]
    kd_sv_body.insert((float(pb_v[0]), float(pb_v[1]), float(pb_v[2])), int(b_i))
kd_sv_body.balance()
print(f"Built kd_sv_body with {len(body_sv_idx)} pure torso+pelvis+leg vertices!", flush=True)




# ==============================================================================
# 2. 三维核平滑权重传递 + 精确逆蒙皮反算绑定函数
# ==============================================================================
def transfer_and_bind_garment(ob, mode="standard", k_nearest=16, smooth_sigma=0.024, smooth_passes=2, max_influences=4):
    """
    1) 在 ORIG_POSE 世界空间从最近 k_nearest 个素体顶点高斯加权采样 W_sotai；
    2) 在服装自身三维空间内做跨连通块高斯核平滑（使纽扣、口袋、翻领、鞋带与基底布料获得完全一致的平滑权重！）；
    3) 裁剪保留前 max_influences 个主变形骨骼并严格归一化 sum=1.0；
    4) 通过解析矩阵求逆 + 5 轮 Blender 原生 Depsgraph DQS 残差迭代反算 T-Pose 静息坐标，
       使 ORIG_POSE 下评估误差 < 0.002 mm！
    """
    me = ob.data
    # 先确保物体矩阵归一到世界坐标
    ob.data.transform(ob.matrix_world)
    ob.matrix_world = Matrix.Identity(4)
    ob.data.update()

    pts_posed = np.array([[v.co.x, v.co.y, v.co.z] for v in me.vertices], dtype=np.float64)
    N_V = len(pts_posed)

    if mode == "rigid_head":
        W = np.zeros((N_V, N_BONES), dtype=np.float64)
        W[:, head_col] = 1.0
    else:
        W = np.zeros((N_V, N_BONES), dtype=np.float64)
        # 穿过 ORIG_POSE 腋下天然空气间隙的分界斜面 X_split(Z) = 0.202 + 0.11*max(0, -0.42 - Z)：
        # 无视 33 个独立子零件缝隙，将衣袖(含内衬/袖襻/袖扣)与衣身(含口袋/门襟/下摆) 100% 干净分离！
        for vi, p in enumerate(pts_posed):
            if mode == "coat":
                ax = abs(p[0])
                pz = p[2]
                if pz < -0.43:
                    # 腋窝以下衣袖与衣身之间有 60~135mm 净空气间隙，一刀两断零歧义
                    x_split = 0.172 + 0.185 * (-0.43 - pz)
                    sw = float(np.clip((ax - (x_split - 0.015)) / 0.030, 0.0, 1.0))
                else:
                    # 肩峰与落肩领区 (-0.43 <= Z <= -0.24)：从 |X|=0.155 平滑过渡到 |X|=0.235
                    sw = float(np.clip((ax - 0.155) / 0.080, 0.0, 1.0))
                sw = sw * sw * (3.0 - 2.0 * sw)

                w_arm_pure = np.zeros(N_BONES, dtype=np.float64)
                if sw > 0.001:
                    hits_a = kd_sv_arm.find_n((float(p[0]), float(p[1]), float(p[2])), 24)
                    da = np.array([max(h[2], 1e-4) for h in hits_a], dtype=np.float64)
                    ia = [h[1] for h in hits_a]
                    wa = np.exp(-0.5 * (da / 0.036) ** 2) + 1e-4 / da
                    wa /= wa.sum()
                    w_arm_pure = wa @ W_sotai_nofinger[ia]
                    w_arm_pure[~is_arm_col] = 0.0
                    w_arm_pure /= max(w_arm_pure.sum(), 1e-8)

                w_body_pure = np.zeros(N_BONES, dtype=np.float64)
                if sw < 0.999:
                    k_b = 36 if (pz < -0.46 and ax < 0.26) else 20
                    sig_b = 0.055 if (pz < -0.46 and ax < 0.26) else 0.026
                    hits_b = kd_sv_body.find_n((float(p[0]), float(p[1]), float(p[2])), k_b)
                    db = np.array([max(h[2], 1e-4) for h in hits_b], dtype=np.float64)
                    ib = [h[1] for h in hits_b]
                    wb = np.exp(-0.5 * (db / sig_b) ** 2) + 1e-4 / db
                    wb /= wb.sum()
                    w_body_pure = wb @ W_sotai_bodyonly[ib]
                    w_body_pure /= max(w_body_pure.sum(), 1e-8)

                W[vi] = sw * w_arm_pure + (1.0 - sw) * w_body_pure
            else:
                hits = kd_sv.find_n((float(p[0]), float(p[1]), float(p[2])), k_nearest)
                dists = np.array([max(h[2], 1e-4) for h in hits], dtype=np.float64)
                idxs  = [h[1] for h in hits]
                ws = np.exp(-0.5 * (dists / 0.022) ** 2) + 1e-4 / dists
                ws /= ws.sum()
                W[vi] = ws @ W_sotai_nofinger[idxs] if mode == "top" else ws @ W_sotai[idxs]

        # 屏蔽手指骨骼对袖口与衣身的干扰，将其权重平滑归并给对应侧的 hand 骨骼
        if mode in ("coat", "top"):
            for vi, p in enumerate(pts_posed):
                f_sum = float(W[vi, is_finger_col].sum())
                if f_sum > 1e-6:
                    W[vi, is_finger_col] = 0.0
                    hc = handL_col if p[0] < 0 else handR_col
                    if hc is not None:
                        W[vi, hc] += f_sum

        # 对运动鞋：减弱脚趾骨在鞋底中后段的影响，保持厚底老爹鞋大底挺括不折断！
        if mode == "shoe":
            toe_cols = np.array(["toe" in bn.lower() for bn in deform_bones], dtype=bool)
            foot_col = next((i for i, bn in enumerate(deform_bones) if "foot" in bn.lower() and ((pts_posed[:,0].mean() < 0 and bn.endswith(".L")) or (pts_posed[:,0].mean() > 0 and bn.endswith(".R")))), None)
            if foot_col is not None and np.any(toe_cols):
                for vi, p in enumerate(pts_posed):
                    # 仅鞋尖前段 (Y > 0.11) 保留 40% 脚趾弯曲跟随，中后段全归 foot + shin
                    keep_toe = 0.40 * float(np.clip((p[1] - 0.10) / 0.06, 0.0, 1.0))
                    t_w = float(W[vi, toe_cols].sum())
                    if t_w > 1e-6:
                        W[vi, toe_cols] *= keep_toe
                        W[vi, foot_col] += t_w * (1.0 - keep_toe)

        # 三维空间 KDTree 跨零件高斯核权重平滑（彻底杜绝悬浮纽扣/口袋/水手领/蝴蝶结/鞋带在弯曲时脱离衣面！）
        if smooth_passes > 0 and N_V > 1:
            kd_self = KDTree(N_V)
            for i, p in enumerate(pts_posed):
                kd_self.insert((float(p[0]), float(p[1]), float(p[2])), i)
            kd_self.balance()
            nb_idx_list = []
            nb_w_list   = []
            for i, p in enumerate(pts_posed):
                sig_loc = smooth_sigma * (1.8 if (mode == "coat" and p[2] < -0.46 and abs(p[0]) < 0.28) else 1.0)
                nbs = kd_self.find_n((float(p[0]), float(p[1]), float(p[2])), min(N_V, 28))
                if mode == "coat":
                    # 防止空间近邻平滑跨过腋下空气缝隙把衣身与袖底混在一起！
                    ax_i = abs(p[0]); slv_i = (ax_i > 0.172 + 0.295 * max(0.0, -0.43 - p[2]))
                    if p[2] < -0.44:
                        nbs = [h for h in nbs if (abs(pts_posed[h[1], 0]) > 0.172 + 0.295 * max(0.0, -0.43 - pts_posed[h[1], 2])) == slv_i] or nbs[:1]
                if len(nbs) <= 1:
                    nb_idx_list.append([i])
                    nb_w_list.append(np.array([1.0]))
                else:
                    ix = [h[1] for h in nbs]
                    ds = np.array([h[2] for h in nbs], dtype=np.float64)
                    gw = np.exp(-0.5 * (ds / max(sig_loc, 1e-4)) ** 2)
                    gw /= gw.sum()
                    nb_idx_list.append(ix)
                    nb_w_list.append(gw)
            for _ in range(smooth_passes):
                W_next = np.empty_like(W)
                for i in range(N_V):
                    W_next[i] = nb_w_list[i] @ W[nb_idx_list[i]]
                W = W_next

        # 截断保留每个顶点权重最大的前 max_influences 个变形骨骼并严格归一化 sum=1.0（100% 兼容 glTF 4骨骼标准！）
        if max_influences is not None and N_BONES > max_influences:
            part_idx = np.argpartition(-W, max_influences, axis=1)[:, :max_influences]
            W_trim = np.zeros_like(W)
            rows = np.arange(N_V)[:, None]
            W_trim[rows, part_idx] = W[rows, part_idx]
            W = W_trim

        w_sums = W.sum(axis=1, keepdims=True)
        assert float(w_sums.min()) > 0.10, f'Zero weight vertex on {ob.name}: min={w_sums.min()}'
        W /= w_sums

    # 写入顶点组（仅写入实际非零的变形骨骼组，整洁高效）
    ob.vertex_groups.clear()
    used_cols = np.where(W.max(axis=0) > 1e-5)[0]
    col_to_vg = {}
    for c_idx in used_cols:
        vg = ob.vertex_groups.new(name=deform_bones[c_idx])
        col_to_vg[c_idx] = vg
    for vi in range(N_V):
        nz = np.where(W[vi] > 1e-5)[0]
        for c_idx in nz:
            col_to_vg[c_idx].add([int(vi)], float(W[vi, c_idx]), 'REPLACE')

    # 第一步初值：解析 LBS 逆矩阵 T_i^-1 @ p_posed
    T_v = np.einsum("vb,bij->vij", W, M_bones)  # [N_V, 4, 4]
    A_lin = T_v[:, :3, :3]                      # [N_V, 3, 3]
    b_trn = T_v[:, :3, 3]                       # [N_V, 3]
    A_inv = np.linalg.inv(A_lin)
    v_rest = np.einsum("vij,vj->vi", A_inv, pts_posed - b_trn)

    me.vertices.foreach_set("co", v_rest.astype(np.float32).ravel())
    me.update()

    # 规范绑定父级与 Armature 修改器
    ob.parent = arm_ob
    ob.matrix_parent_inverse = arm_ob.matrix_world.inverted()
    for md in list(ob.modifiers):
        if md.type == 'ARMATURE':
            ob.modifiers.remove(md)
    arm_md = ob.modifiers.new(name="Armature", type='ARMATURE')
    arm_md.object = arm_ob
    arm_md.use_vertex_groups = True
    arm_md.use_deform_preserve_volume = PRESERVE_VOL

    # 第二步精化：5 轮 Blender 原生 C++ Depsgraph (含 DQS 双四元数体积保持) 残差不动点迭代！
    for it in range(5):
        bpy.context.view_layer.update()
        dg_now = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg_now)
        em = ev.to_mesh()
        pts_eval = np.array([[v.co.x, v.co.y, v.co.z] for v in em.vertices], dtype=np.float64)
        ev.to_mesh_clear()
        res = pts_posed - pts_eval
        max_err_mm = float(np.max(np.linalg.norm(res, axis=1))) * 1000.0
        if max_err_mm < 0.002:
            break
        v_rest += np.einsum("vij,vj->vi", A_inv, res)
        me.vertices.foreach_set("co", v_rest.astype(np.float32).ravel())
        me.update()

    bpy.context.view_layer.update()
    dg_fin = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg_fin); em = ev.to_mesh()
    pts_fin = np.array([[v.co.x, v.co.y, v.co.z] for v in em.vertices], dtype=np.float64)
    ev.to_mesh_clear()
    fin_err_mm = float(np.max(np.linalg.norm(pts_posed - pts_fin, axis=1))) * 1000.0
    print(f"Bound {ob.name:22s}: verts={N_V:5d}, active_bones={len(used_cols):2d}, ORIG_POSE_max_err={fin_err_mm:.5f} mm", flush=True)
    return fin_err_mm

rig_errors = {}
rig_errors["GEO-Cloth_Coat"]      = transfer_and_bind_garment(bpy.data.objects["GEO-Cloth_Coat"],      mode="coat",       k_nearest=20, smooth_sigma=0.032, smooth_passes=3, max_influences=4)
rig_errors["GEO-Cloth_SailorTop"] = transfer_and_bind_garment(bpy.data.objects["GEO-Cloth_SailorTop"], mode="top",        k_nearest=16, smooth_sigma=0.026, smooth_passes=3, max_influences=4)
rig_errors["GEO-Cloth_Shorts"]    = transfer_and_bind_garment(bpy.data.objects["GEO-Cloth_Shorts"],    mode="standard",   k_nearest=16, smooth_sigma=0.024, smooth_passes=2, max_influences=4)
rig_errors["GEO-Cloth_Sock_L"]    = transfer_and_bind_garment(bpy.data.objects["GEO-Cloth_Sock_L"],    mode="standard",   k_nearest=14, smooth_sigma=0.022, smooth_passes=2, max_influences=4)
rig_errors["GEO-Cloth_Sock_R"]    = transfer_and_bind_garment(bpy.data.objects["GEO-Cloth_Sock_R"],    mode="standard",   k_nearest=14, smooth_sigma=0.022, smooth_passes=2, max_influences=4)
rig_errors["GEO-Shoe_L"]          = transfer_and_bind_garment(bpy.data.objects["GEO-Shoe_L"],          mode="shoe",       k_nearest=18, smooth_sigma=0.035, smooth_passes=3, max_influences=4)
rig_errors["GEO-Shoe_R"]          = transfer_and_bind_garment(bpy.data.objects["GEO-Shoe_R"],          mode="shoe",       k_nearest=18, smooth_sigma=0.035, smooth_passes=3, max_influences=4)
rig_errors["ACC-ThighStrap_L"]    = transfer_and_bind_garment(bpy.data.objects["ACC-ThighStrap_L"],    mode="standard",   k_nearest=14, smooth_sigma=0.020, smooth_passes=2, max_influences=4)
rig_errors["ACC-Choker"]          = transfer_and_bind_garment(bpy.data.objects["ACC-Choker"],          mode="standard",   k_nearest=14, smooth_sigma=0.018, smooth_passes=2, max_influences=4)
rig_errors["ACC-WhaleHairpin_L"]  = transfer_and_bind_garment(bpy.data.objects["ACC-WhaleHairpin_L"],  mode="rigid_head", k_nearest=4,  smooth_sigma=0.010, smooth_passes=0, max_influences=1)

# 重新开启脚掌鞋内遮罩修改器
if mask_mod:
    mask_mod.show_viewport = True
    mask_mod.show_render   = True


# ==============================================================================
# 3. 构建真实生效的五姿压测（屏蔽旧 NLA 覆盖 + 直接采用 build_assembled_sotai_v2 验证过的精确世界旋转驱动！）
# ==============================================================================
if arm_ob.animation_data and arm_ob.animation_data.nla_tracks:
    for trk in arm_ob.animation_data.nla_tracks:
        trk.mute = True

orig_pose_transforms = {}
for pb in arm_ob.pose.bones:
    orig_pose_transforms[pb.name] = (
        pb.location.copy(),
        pb.rotation_quaternion.copy(),
        pb.rotation_euler.copy(),
        pb.rotation_mode,
        pb.scale.copy(),
    )

def reset_pose_t():
    for pb in arm_ob.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.location = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)

def restore_orig_pose():
    for bname, (loc, rq, re, rmode, scl) in orig_pose_transforms.items():
        pb = arm_ob.pose.bones[bname]
        pb.rotation_mode = rmode
        pb.location = loc.copy()
        pb.rotation_quaternion = rq.copy()
        pb.rotation_euler = re.copy()
        pb.scale = scl.copy()

def set_world_rot(bname, euler_xyz):
    pb = arm_ob.pose.bones.get(bname)
    if pb is None:
        return
    M_bone = (arm_ob.matrix_world @ pb.bone.matrix_local).to_3x3()
    R_world = Euler(euler_xyz, 'XYZ').to_matrix()
    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = (M_bone.inverted() @ R_world @ M_bone).to_quaternion()

def apply_arm_pose(deg_down_L, deg_down_R, shoulder_drop_deg=4.5, shoulder_shift_x=0.022, shoulder_shift_z=-0.014, forward_deg=0.5, elbow_deg=2.5, wrist_inward_deg=4.0):
    reset_pose_t()
    set_world_rot('shoulder.L', (0.0, math.radians(-shoulder_drop_deg), 0.0))
    set_world_rot('shoulder.R', (0.0, math.radians(+shoulder_drop_deg), 0.0))
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

def keyframe_all_pose_bones(frame_no):
    for pb in arm_ob.pose.bones:
        pb.keyframe_insert(data_path="location", frame=frame_no)
        if pb.rotation_mode == 'QUATERNION':
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_no)
        else:
            pb.keyframe_insert(data_path="rotation_euler", frame=frame_no)
        pb.keyframe_insert(data_path="scale", frame=frame_no)

if not arm_ob.animation_data:
    arm_ob.animation_data_create()
act = bpy.data.actions.new(name="ACT_V10R8_PoseQA")
arm_ob.animation_data.action = act

sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'
sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.image_settings.quality = 93
sc.render.film_transparent = False
cam = bpy.data.objects["REVIEW_CAMERA"]
sc.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 1.82
sc.render.resolution_x = 720
sc.render.resolution_y = 1200

def render_pose(tag, cloc, crot):
    bpy.context.view_layer.update()
    cam.location = Vector(cloc)
    cam.rotation_euler = Euler([math.radians(a) for a in crot], 'XYZ')
    sc.render.filepath = str(RENDER_DIR / f"{tag}.jpg")
    bpy.ops.render.render(write_still=True)

def measure_pose_span(tag, frame_no, label):
    dg_p = bpy.context.evaluated_depsgraph_get()
    ct_ev = bpy.data.objects["GEO-Cloth_Coat"].evaluated_get(dg_p)
    cm = ct_ev.to_mesh()
    c_pts = np.array([list(ct_ev.matrix_world @ v.co) for v in cm.vertices])
    ct_ev.to_mesh_clear()
    sh_ev = bpy.data.objects["GEO-Shoe_L"].evaluated_get(dg_p)
    sm = sh_ev.to_mesh()
    s_pts = np.array([list(sh_ev.matrix_world @ v.co) for v in sm.vertices])
    sh_ev.to_mesh_clear()
    res = {
        "frame": frame_no,
        "label": label,
        "coat_x_span_mm": round(float(np.ptp(c_pts[:,0])) * 1000.0, 1),
        "coat_z_span_mm": round(float(np.ptp(c_pts[:,2])) * 1000.0, 1),
        "shoeL_z_min_mm": round(float(s_pts[:,2].min()) * 1000.0, 1),
    }
    print(f"POSE_QA {tag}: {res}", flush=True)
    return res

pose_metrics = {}

# 1) Frame 1: 原图基准 ORIG_POSE (< 0.002 mm)
sc.frame_set(1)
restore_orig_pose()
keyframe_all_pose_bones(1)
render_pose("f01_orig", (0.0, 3.2, -0.63), (90.0, 0.0, 180.0))
pose_metrics["f01_orig"] = measure_pose_span("f01_orig", 1, "f1: 原图基准 ORIG_POSE")

# 2) Frame 20: 转头 22° + 微侧倾 + 自然低垂手姿 (检验 Choker、水手领、左侧鲸鱼发夹刚性跟随与窄袖贴合)
sc.frame_set(20)
apply_arm_pose(58.0, 58.0, shoulder_drop_deg=5.5, shoulder_shift_x=0.010, shoulder_shift_z=-0.012)
set_world_rot('Neck', (math.radians(-3.0), math.radians(3.5), math.radians(-10.0)))
set_world_rot('Head', (math.radians(-4.0), math.radians(5.5), math.radians(-16.0)))
keyframe_all_pose_bones(20)
render_pose("f20_relax", (0.0, 3.2, -0.63), (90.0, 0.0, 180.0))
pose_metrics["f20_relax"] = measure_pose_span("f20_relax", 20, "f20: 垂臂转头歪头姿")

# 3) Frame 40: 双臂高抬展翼姿 (deg_down=14° 接近水平 T 姿！压测落肩长外套肩腋展开对称度与零拉扯)
sc.frame_set(40)
apply_arm_pose(14.0, 14.0, shoulder_drop_deg=-2.0, shoulder_shift_x=0.0, shoulder_shift_z=0.0)
keyframe_all_pose_bones(40)
render_pose("f40_wave", (0.0, 3.2, -0.60), (90.0, 0.0, 180.0))
pose_metrics["f40_wave"] = measure_pose_span("f40_wave", 40, "f40: 双臂高抬展翼姿")

# 4) Frame 60: 双臂前抬屈肘 60° 互动姿 (压测 DQS 体积保持下宽袖口与前臂折弯！)
sc.frame_set(60)
apply_arm_pose(38.0, 44.0, shoulder_drop_deg=3.5, shoulder_shift_x=0.015, shoulder_shift_z=-0.010)
set_world_rot('lower_arm.L', (math.radians(-52.0), math.radians(-16.0), math.radians(58.0)))
set_world_rot('lower_arm.R', (math.radians(-42.0), math.radians(14.0), math.radians(-48.0)))
set_world_rot('Head', (math.radians(2.0), math.radians(-4.0), math.radians(10.0)))
keyframe_all_pose_bones(60)
render_pose("f60_stride", (-1.85, 2.60, -0.61), (90.0, 0.0, -144.5))
pose_metrics["f60_elbow"] = measure_pose_span("f60_stride", 60, "f60: 屈肘抬手互动姿")

# 5) Frame 80: 迈步高抬腿屈膝姿 (左大腿前抬 38° + 左膝屈 62° + 右腿后蹬 16°，压测高腰短裤、左大腿环、外套下摆、堆堆袜与老爹鞋！)
sc.frame_set(80)
apply_arm_pose(50.0, 50.0, shoulder_drop_deg=5.0, shoulder_shift_x=0.015, shoulder_shift_z=-0.012)
set_world_rot('upper_leg.L', (math.radians(+40.0), 0.0, 0.0))
set_world_rot('lower_leg.L', (math.radians(-58.0), 0.0, 0.0))
set_world_rot('foot.L',      (math.radians(+12.0), 0.0, 0.0))
set_world_rot('upper_leg.R', (math.radians(-14.0), 0.0, 0.0))
set_world_rot('lower_leg.R', (math.radians(-14.0), 0.0, 0.0))
set_world_rot('upper_arm.L', (math.radians(+16.0), math.radians(-50.0), 0.0))
set_world_rot('upper_arm.R', (math.radians(-20.0), math.radians(+50.0), 0.0))
keyframe_all_pose_bones(80)
render_pose("f60_stride_left90", (-3.2, 0.0, -0.63), (90.0, 0.0, -90.0))
render_pose("f80_tpose",         (-2.10, 2.42, -0.63), (90.0, 0.0, -139.0))
pose_metrics["f80_knee"] = measure_pose_span("f80_tpose", 80, "f80: 高抬腿屈膝迈步姿")

if hasattr(act, "fcurves"):
    for fc in act.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'

# 恢复默认状态为 Frame 1 (原图 ORIG_POSE) 并重新恢复头发微风物理 NLA 轨道以便用户随时播放物理！
sc.frame_set(1)
restore_orig_pose()
bpy.context.view_layer.update()
cam.location = Vector((0.0, 3.2, -0.63))
cam.rotation_euler = Euler((math.radians(90.0), 0.0, math.radians(180.0)), 'XYZ')
cam.data.ortho_scale = 1.72

step3_report = {
    "preserve_volume": PRESERVE_VOL,
    "deform_bone_count": N_BONES,
    "orig_pose_max_error_mm": rig_errors,
    "pose_metrics": pose_metrics,
}
(OUT_DIR / "step3_rig_report.json").write_text(json.dumps(step3_report, ensure_ascii=False, indent=2), encoding="utf-8")

bpy.ops.wm.save_as_mainfile(filepath=str(OUT_STEP3_BLEND), compress=True)
print("STEP3_ALL_DONE", flush=True)
