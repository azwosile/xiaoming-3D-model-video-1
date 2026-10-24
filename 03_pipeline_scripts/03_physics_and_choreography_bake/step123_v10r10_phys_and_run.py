"""
V10R10 一体化构建脚本：
Step 1: 在 ARM-FullBody_V10 中新增 14 根衣装物理飘骨 (风衣五路双节下摆10 + 宽袖口2 + 水手蝴蝶结飘带2)，平滑分配防脱扣权重；
Step 2: 构建 24 帧 (@30fps) 无缝循环二次元轻快跑步动作 ACT_V10R10_AnimeRun_Loop (含手指自然微握、躯干前倾反向扭转、鞋底精确触地零滑步)；
Step 3: 5周期 RK4 稳态积分求解 14 根衣装飘骨(含大腿抬起防穿模护盾) + 54 根头发空间笼骨/呆毛迎风弹跳物理，实现 f1==f25 零跳帧闭合；
        并渲染 8 帧关键相位 × 多视角连拍看板进行严格防穿模视觉质检。
"""
import json, math, time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector, Euler, Matrix, Quaternion

ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
ODIR = ROOT / "生成模型/组装/v010R8_全身衣装装配"
RRUN = ODIR / "renders_v10r10_run"
RRUN.mkdir(parents=True, exist_ok=True)

IN_BLEND  = ODIR / "RadDoll_V10R9_全身衣装修身总装成品_PRIVATE.blend"
OUT_STAGE = ODIR / "RadDoll_V10R10_Step3_物理与跑步压测_PRIVATE.blend"

bpy.ops.wm.open_mainfile(filepath=str(IN_BLEND), load_ui=False, use_scripts=False)
sc = bpy.context.scene

arm_ob   = bpy.data.objects["ARM-FullBody_V10"]
arm_data = arm_ob.data
coat_ob  = bpy.data.objects["GEO-Cloth_Coat"]
top_ob   = bpy.data.objects["GEO-Cloth_SailorTop"]
shoeL_ob = bpy.data.objects["GEO-Shoe_L"]
shoeR_ob = bpy.data.objects["GEO-Shoe_R"]
sotai_ob = bpy.data.objects["GEO-Sotai_Skin"]

# 保留原图基准 Action 名字以便随时切回静态原图姿
orig_act = bpy.data.actions.get("ACT_V10R8_PoseQA")
if orig_act:
    orig_act.name = "ACT_V10R9_OrigAndPoses"
    orig_act.use_fake_user = True

# 静音任何旧 NLA 轨道
if arm_ob.animation_data and arm_ob.animation_data.nla_tracks:
    for trk in arm_ob.animation_data.nla_tracks:
        trk.mute = True

def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# ==============================================================================
# STEP 1: 在编辑态 (Rest T-Pose) 新增 14 根衣装物理飘骨，并平滑分配权重
# ==============================================================================
M_arm = arm_ob.matrix_world
M_arm_inv = M_arm.inverted()

# 定义 14 根衣装物理骨规格 (世界坐标下，全部骨骼朝向自然下垂或沿袖延伸，roll=0)
# 5 条风衣下摆链 (挂在 Hips 下)：FL(左前襟,-X,+Y), FR(右前襟,+X,+Y), SL(左后侧,-X,-Y), SR(右后侧,+X,-Y), BK(正后,0,-Y)
skirt_specs = [
    ("FL", -0.105, +0.082, math.atan2(+0.082, -0.105)),
    ("FR", +0.105, +0.082, math.atan2(+0.082, +0.105)),
    ("SL", -0.148, -0.048, math.atan2(-0.048, -0.148)),
    ("SR", +0.148, -0.048, math.atan2(-0.048, +0.148)),
    ("BK",  0.000, -0.088, -0.5 * math.pi),
]

new_bones_spec = []
for tag, bx, by, _ in skirt_specs:
    b1 = f"PHY_Coat_{tag}_01"
    b2 = f"PHY_Coat_{tag}_02"
    new_bones_spec.append((b1, "Hips", False, (bx, by, -0.525), (bx*1.04, by*1.03, -0.695)))
    new_bones_spec.append((b2, b1,     True,  (bx*1.04, by*1.03, -0.695), (bx*1.08, by*1.06, -0.865)))

# 左右宽大翻折袖口微摆骨 (挂在 lower_arm.L/R 下)
new_bones_spec.append(("PHY_Cuff_L", "lower_arm.L", False, (-0.445, 0.002, -0.242), (-0.605, 0.002, -0.242)))
new_bones_spec.append(("PHY_Cuff_R", "lower_arm.R", False, (+0.445, 0.002, -0.242), (+0.605, 0.002, -0.242)))

# 水手服胸前浅蓝蝴蝶结双飘带骨 (挂在 Chest 下)
new_bones_spec.append(("PHY_Bow_L", "Chest", False, (-0.022, 0.110, -0.302), (-0.036, 0.116, -0.388)))
new_bones_spec.append(("PHY_Bow_R", "Chest", False, (+0.022, 0.110, -0.302), (+0.036, 0.116, -0.388)))

bpy.context.view_layer.objects.active = arm_ob
bpy.ops.object.mode_set(mode='EDIT')
eb_all = arm_data.edit_bones
for bnm, _, _, _, _ in new_bones_spec:
    if bnm in eb_all:
        eb_all.remove(eb_all[bnm])

for bnm, pnm, uconn, h_w, t_w in new_bones_spec:
    eb = eb_all.new(bnm)
    eb.head = M_arm_inv @ Vector(h_w)
    eb.tail = M_arm_inv @ Vector(t_w)
    eb.roll = 0.0
    eb.use_deform = True
    eb.parent = eb_all[pnm]
    eb.use_connect = uconn
bpy.ops.object.mode_set(mode='OBJECT')

# 切换骨架到 REST 态读取精确 T-Pose 世界坐标以分配光滑连续物理权重（天然跨过33个子网格接缝，绝不掉扣子！）
arm_data.pose_position = 'REST'
bpy.context.view_layer.update()

def get_rest_world_co(obj):
    me = obj.data
    n = len(me.vertices)
    c = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", c)
    Mw = np.array(obj.matrix_world, dtype=np.float64)
    return (c.reshape(n, 3) @ Mw[:3, :3].T) + Mw[:3, 3]

def blend_in_phys_weights(obj, phys_w_dict):
    """
    phys_w_dict: {bone_name: np.ndarray(n_verts)}，其逐点总和 w_new <= 1.0。
    将顶点原有变形骨权重按 (1 - w_new) 等比缩放后叠入新物理骨权重，保持总和恒等于 1.000000！
    """
    me = obj.data
    n = len(me.vertices)
    deform_names = {b.name for b in arm_data.bones if b.use_deform}
    for bnm in phys_w_dict:
        if bnm not in obj.vertex_groups:
            obj.vertex_groups.new(name=bnm)

    phys_gidxs = {obj.vertex_groups[bnm].index for bnm in phys_w_dict}
    def_gidxs  = {vg.index for vg in obj.vertex_groups if vg.name in deform_names and vg.index not in phys_gidxs}

    w_add_sum = np.zeros(n, dtype=np.float64)
    for arr in phys_w_dict.values():
        w_add_sum += arr
    w_add_sum = np.clip(w_add_sum, 0.0, 0.96)

    for vi, v in enumerate(me.vertices):
        wn = float(w_add_sum[vi])
        if wn < 1e-5:
            continue
        keep = 1.0 - wn
        # 缩放旧变形骨权重
        old_pairs = [(g.group, g.weight) for g in v.groups if g.group in def_gidxs and g.weight > 1e-6]
        old_sum = sum(w for _, w in old_pairs)
        for gid, w in old_pairs:
            obj.vertex_groups[gid].add([vi], (w / max(old_sum, 1e-8)) * keep, 'REPLACE')
        # 写入新物理骨权重
        for bnm, arr in phys_w_dict.items():
            wv = float(arr[vi])
            gid = obj.vertex_groups[bnm].index
            if wv > 1e-5:
                obj.vertex_groups[gid].add([vi], wv, 'REPLACE')
            else:
                obj.vertex_groups[gid].remove([vi])

# 0) 为落肩外套上袖筒外前壁（Rest T-Pose 中上臂二头肌正对 +Y/+Z 方向，垂臂前摆时对应外侧与前侧！）
#    平滑释放 12mm 摆臂动态余量，并让袖筒中上段更顺畅跟随上臂前后摆动，彻底根除 f13/f16 前摆露大臂！
coat_r0 = get_rest_world_co(coat_ob)
# 注意：肘关节 Rest 位置在 |X| = 0.391m，腕关节在 |X| = 0.585m！
# 跑步曲肘 55° 时，肘尖 (|X| ≈ 0.36..0.46) 在垂臂姿态下朝外侧 (Rest +Z) 与后外侧 (Rest -Y/+Z) 顶出！
# 因此对整个大臂中段至前臂中段 (0.23 < |X| < 0.51，避开领口 |X|<0.20 与翻折袖口 |X|>0.54) 的外侧(+Z)/后侧(-Y) 平滑释放 16mm 屈肘余量！
slv_ax_z = -0.252
slv_ax_y = -0.012
dy_s = coat_r0[:, 1] - slv_ax_y
dz_s = coat_r0[:, 2] - slv_ax_z
rad_s = np.maximum(np.sqrt(dy_s**2 + dz_s**2), 1e-6)
gate_slv_x = sstep(0.215, 0.275, np.abs(coat_r0[:, 0])) * (1.0 - sstep(0.475, 0.535, np.abs(coat_r0[:, 0])))
elbow_peak = np.exp(-((np.abs(coat_r0[:, 0]) - 0.395) / 0.075) ** 2)
# 均匀扩张外侧 (+Z) 与后外侧 (-Y) 袖壁，彻底包容屈肘肘尖！
w_outer = 0.45 + 0.75 * sstep(-0.025, +0.035, dz_s) + 0.45 * sstep(+0.020, -0.030, dy_s)
pad_amt = (0.0085 + 0.0095 * elbow_peak) * gate_slv_x * np.clip(w_outer, 0.0, 1.45)
coat_r0[:, 1] += (dy_s / rad_s) * pad_amt - 0.0045 * elbow_peak * sstep(+0.02, -0.03, dy_s)
coat_r0[:, 2] += (dz_s / rad_s) * pad_amt + 0.0055 * elbow_peak * sstep(-0.02, +0.03, dz_s)
Mw_c_inv = np.linalg.inv(np.array(coat_ob.matrix_world, dtype=np.float64))
coat_loc_new = (coat_r0 @ Mw_c_inv[:3, :3].T) + Mw_c_inv[:3, 3]
coat_ob.data.vertices.foreach_set("co", coat_loc_new.astype(np.float32).ravel())
coat_ob.data.update()

# 同时微调落肩上袖筒权重：对 0.24 < |X| < 0.38 的纯袖管区，把少量滞留的 Chest 权重平滑移交给 upper_arm，使摆臂时袖管与二头肌同相运动！
for side, sgn in [("L", -1.0), ("R", +1.0)]:
    vg_ch = coat_ob.vertex_groups.get("Chest")
    vg_sh = coat_ob.vertex_groups.get(f"shoulder.{side}")
    vg_ua = coat_ob.vertex_groups.get(f"upper_arm.{side}")
    if vg_ch and vg_ua:
        idx_ch, idx_sh, idx_ua = vg_ch.index, (vg_sh.index if vg_sh else -1), vg_ua.index
        for vi, v in enumerate(coat_ob.data.vertices):
            rx = coat_r0[vi, 0] * sgn
            if 0.225 < rx < 0.40:
                tr = float(sstep(0.225, 0.295, rx)) * 0.78
                wmap = {g.group: g.weight for g in v.groups}
                w_from_ch = wmap.get(idx_ch, 0.0) * tr
                w_from_sh = wmap.get(idx_sh, 0.0) * (tr * 0.65) if idx_sh >= 0 else 0.0
                if w_from_ch + w_from_sh > 1e-4:
                    vg_ch.add([vi], max(0.0, wmap.get(idx_ch, 0.0) - w_from_ch), 'REPLACE')
                    if idx_sh >= 0 and w_from_sh > 0:
                        vg_sh.add([vi], max(0.0, wmap.get(idx_sh, 0.0) - w_from_sh), 'REPLACE')
                    vg_ua.add([vi], wmap.get(idx_ua, 0.0) + w_from_ch + w_from_sh, 'REPLACE')

# 1A) GEO-Cloth_Coat 权重混合 (T-Pose 下 Z < -0.49m 100% 是纯衣摆，|X| > 0.44m 100% 是宽袖口！)
coat_rest = get_rest_world_co(coat_ob)
n_coat = len(coat_rest)
cx, cy, cz = coat_rest[:, 0], coat_rest[:, 1], coat_rest[:, 2]

# 衣摆垂直总物理占比：Z=-0.50m 处为 0（腰部平滑衔接躯干），向下到 Z=-0.62m 升至 0.90
w_skirt_tot = 0.92 * sstep(-0.505, -0.625, cz) * (1.0 - sstep(0.32, 0.38, np.abs(cx)))
# 分节占比：seg01 主导上中段，seg02 主导下摆段 (-0.67..-0.87)
f_seg2 = sstep(-0.65, -0.78, cz)
f_seg1 = 1.0 - f_seg2

# 5 扇区方位权重（关键：前胸正中敞开，左前襟 cx<0 绝不跨到 FR，右前襟 cx>0 绝不跨到 FL！）
phi_c = np.arctan2(cy, cx)
sec_phis = {tag: ang for tag, _, _, ang in skirt_specs}
raw_sec = {}
for tag, p0 in sec_phis.items():
    dphi = np.angle(np.exp(1j * (phi_c - p0)))
    # 半张角 78° 余弦窗
    w_ang = 0.5 * (1.0 + np.cos(np.clip(np.abs(dphi) / math.radians(78.0), 0.0, 1.0) * math.pi))
    if tag == "FL":
        w_ang *= sstep(+0.015, -0.025, cx)  # 仅限 X < 0 左侧门襟
    elif tag == "FR":
        w_ang *= sstep(-0.015, +0.025, cx)  # 仅限 X > 0 右侧门襟
    raw_sec[tag] = w_ang + 1e-6

sec_sum = sum(raw_sec.values())
coat_phys_w = {}
for tag in raw_sec:
    wn = raw_sec[tag] / sec_sum
    coat_phys_w[f"PHY_Coat_{tag}_01"] = w_skirt_tot * wn * f_seg1
    coat_phys_w[f"PHY_Coat_{tag}_02"] = w_skirt_tot * wn * f_seg2

# 左右宽袖口下垂折边微摆权重 (|X| > 0.44m 且 Z > -0.42m)
in_sleeve_zone = sstep(-0.42, -0.35, cz)
cuff_rad = sstep(0.44, 0.56, np.abs(cx)) * in_sleeve_zone
# 袖口下方悬垂部分摆动稍大 (0.68)，袖口上方较稳 (0.35)
cuff_gain = 0.35 + 0.35 * sstep(-0.22, -0.31, cz)
coat_phys_w["PHY_Cuff_L"] = cuff_rad * cuff_gain * (cx < 0).astype(np.float64)
coat_phys_w["PHY_Cuff_R"] = cuff_rad * cuff_gain * (cx > 0).astype(np.float64)

blend_in_phys_weights(coat_ob, coat_phys_w)

# 1B) GEO-Cloth_SailorTop 胸前蝴蝶结双飘带权重 (仅限 Y > 0.101m 且 Z < -0.302m 的凸出飘带，绝不动贴身白衣身！)
top_rest = get_rest_world_co(top_ob)
tx, ty, tz = top_rest[:, 0], top_rest[:, 1], top_rest[:, 2]
w_bow_tot = 0.82 * sstep(0.100, 0.112, ty) * sstep(-0.300, -0.345, tz) * (1.0 - sstep(0.065, 0.095, np.abs(tx)))
bow_L_share = sstep(+0.012, -0.012, tx)
bow_R_share = 1.0 - bow_L_share
blend_in_phys_weights(top_ob, {
    "PHY_Bow_L": w_bow_tot * bow_L_share,
    "PHY_Bow_R": w_bow_tot * bow_R_share,
})

arm_data.pose_position = 'POSE'
bpy.context.view_layer.update()
print("STEP 1 CLOTH PHYSICS BONES & WEIGHTS READY!", flush=True)

# ==============================================================================
# STEP 2 & 3: 构建 24 帧无缝循环二次元轻快跑步 + 14衣装骨 & 54头发骨 RK4 稳态物理求解
# ==============================================================================
N_FRAMES = 24
FPS = 30.0
Period = N_FRAMES / FPS   # 0.80 秒一完整左右双步周期 (步频 150 BPM，非常轻快元气的二次元慢跑节奏！)
OMEGA = 2.0 * math.pi / Period

# 创建全新跑步循环 Action
run_act = bpy.data.actions.new(name="ACT_V10R10_AnimeRun_Loop")
run_act.use_fake_user = True
arm_ob.animation_data_create()
arm_ob.animation_data.action = run_act

sc.render.fps = int(FPS)
sc.frame_start = 1
sc.frame_end = N_FRAMES

# 预先记录所有骨骼在编辑态(Rest)的世界旋转矩阵 R_rest_w[bname]，实现任意骨精确世界增量旋转！
R_arm_w = arm_ob.matrix_world.to_3x3()
R_rest_local = {b.name: b.matrix_local.to_3x3().copy() for b in arm_data.bones}
R_rest_world = {bn: (R_arm_w @ R_loc) for bn, R_loc in R_rest_local.items()}

for pb in arm_ob.pose.bones:
    pb.rotation_mode = 'QUATERNION'
    pb.location = (0.0, 0.0, 0.0)
    pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    pb.scale = (1.0, 1.0, 1.0)
bpy.context.view_layer.update()

def set_bone_world_delta(pb, R_world_delta, deps_update=False):
    """
    对当前已评估父级姿态下的骨骼 pb 施加世界空间旋转增量 R_world_delta
    （使得该骨从其自身 Rest 方向在世界空间旋转 R_world_delta，完全消除父级旋转耦合与局部轴差异！）
    """
    bn = pb.name
    R_target_arm = R_arm_w.inverted() @ R_world_delta @ R_rest_world[bn]
    if pb.parent:
        # 使用父骨当前 pose.matrix 计算精确局部矩阵
        R_par_pose = pb.parent.matrix.to_3x3()
        R_par_rest = R_rest_local[pb.parent.name]
        R_rest_rel = R_par_rest.inverted() @ R_rest_local[bn]
        R_basis = (R_par_pose @ R_rest_rel).inverted() @ R_target_arm
    else:
        R_basis = R_rest_local[bn].inverted() @ R_target_arm
    pb.rotation_quaternion = R_basis.to_quaternion()
    if deps_update:
        bpy.context.view_layer.update()

# ------------------------------------------------------------------------------
# 定义身体主运动学曲线 (严格 2π 周期连续 C∞ 傅里叶级数 -> 首尾帧速度/加速度 100% 连续无跳帧！)
# 相位定义：ph = 0 对应【左脚前触地 Contact / 右脚后蹬离】；ph = π 对应【右脚前触地 / 左脚后蹬离】
# ------------------------------------------------------------------------------
def leg_kinematics(ph):
    """
    返回单腿在相位 ph 下的世界俯仰角 (deg, Rx>0 向前抬，Rx<0 向后摆):
      thigh_rx, knee_bend (<=0 向后屈膝), ankle_rx
    关键相位验证:
      ph = 0.00 (f1  触地 Contact):   thigh = +27°, knee = -14°, ankle = +12° (脚跟微勾迎接落地)
      ph = 0.78 (f4  缓冲 Down):      thigh = +12°, knee = -28°, ankle =  -2° (屈膝缓冲，全掌踩实)
      ph = 1.57 (f7  支撑过中 Pass):  thigh =  -8°, knee = -13°, ankle =  +4°
      ph = 2.36 (f10 后蹬腾空 Push):  thigh = -25°, knee = -18°, ankle = -24° (漂亮的后蹬伸展绷脚背！)
      ph = 3.60 (f15 后摆折叠 Fold):  thigh =  -6°, knee = -74°, ankle = -12° (小腿轻快向后折叠收腿)
      ph = 4.90 (f19 前摆高抬 Swing): thigh = +32°, knee = -52°, ankle =  +8° (迈腿向前跨越)
    """
    thigh_rx = (
        +3.5
        + 28.0 * math.cos(ph - 0.32)
        + 3.5  * math.cos(2.0 * ph - 1.10)
    )
    # 屈膝双峰：支撑期小缓冲峰 (-28° @ ph≈0.8) + 摆动期高折腿大峰 (-75° @ ph≈3.7)
    knee_bend = -(
        35.5
        - 19.5 * math.cos(ph - 0.55)
        + 18.5 * math.cos(2.0 * ph - 1.05)
        - 4.5  * math.cos(3.0 * ph + 0.40)
    )
    knee_bend = min(-8.0, knee_bend)
    ankle_rx = (
        -3.0
        + 15.0 * math.cos(ph + 0.35)
        + 7.5  * math.sin(2.0 * ph + 0.30)
    )
    toe_rx = -12.0 * max(0.0, math.sin(ph - 1.55))  # 后蹬期脚趾顺势微压地面
    return thigh_rx, knee_bend, ankle_rx, toe_rx

def body_kinematics(t_sec):
    ph = OMEGA * t_sec
    # 骨盆双峰弹跳：在 ph≈0.75 与 ph≈3.89 (缓冲帧 f4/f16) 最低 (-21mm)，在 ph≈2.32 与 ph≈5.46 (腾空帧 f10/f22) 最高 (+23mm)！
    hips_dz = 0.022 * math.cos(2.0 * ph - 2.32)
    hips_dx = -0.009 * math.cos(ph - 0.65)          # 支撑侧重心自然微移 ±9mm (左脚支撑向 -X 微移)
    hips_dy = 0.005 * math.sin(2.0 * ph)            # 前后轻盈呼吸脉动

    hips_rx = -7.5 + 1.8 * math.sin(2.0 * ph - 0.4) # 跑步重心自然前倾 (Rx<0 使 +Z 朝 +Y 前倾！)
    hips_ry = +3.0 * math.cos(ph - 0.65)            # 摆动侧骨盆自然微沉
    hips_rz = -5.5 * math.cos(ph - 0.25)            # 左腿在前时骨盆轻微右旋平衡

    spine_rx = -8.5 + 1.8 * math.sin(2.0 * ph - 0.8)  # 叠加 Rest 本身的 +9.5° 前倾 = ~18° 流线前倾
    spine_ry = -1.6 * math.cos(ph - 0.65)
    spine_rz = +2.5 * math.cos(ph - 0.25)

    chest_rx = -10.5 + 2.0 * math.sin(2.0 * ph - 1.0)
    chest_ry = -2.2 * math.cos(ph - 0.65)
    chest_rz = +6.5 * math.cos(ph - 0.20)             # 胸胯反向扭转（左腿前时右肩微前！）

    # 头颈视线稳定：抬起视线直视正前方并带双峰轻盈弹跳！
    neck_rx = -3.5 + 1.4 * math.sin(2.0 * ph - 1.3)
    neck_ry = +0.8 * math.cos(ph)
    neck_rz = +1.8 * math.cos(ph - 0.20)

    head_rx = +1.5 + 2.0 * math.sin(2.0 * ph - 1.65)  # 下颌微收平视正前方
    head_ry = +1.8 * math.sin(ph - 0.40)              # 随步伐轻微灵动侧倾 ±1.8°
    head_rz = -1.5 * math.cos(ph - 0.20)

    return {
        "ph": ph,
        "hips_loc": (hips_dx, hips_dy, hips_dz),
        "Hips":  (hips_rx,  hips_ry,  hips_rz),
        "Spine": (spine_rx, spine_ry, spine_rz),
        "Chest": (chest_rx, chest_ry, chest_rz),
        "Neck":  (neck_rx,  neck_ry,  neck_rz),
        "Head":  (head_rx,  head_ry,  head_rz),
    }

# ------------------------------------------------------------------------------
# RK4 5周期稳态求解：14 根衣装物理飘骨 + 54 根头发笼骨/呆毛物理！
# ------------------------------------------------------------------------------
N_AZ = 12
az_phis = np.linspace(-math.pi, math.pi, N_AZ, endpoint=False)
cage_bone_names = [[f"DEF-CageAz{ai:02d}_{bi+1:02d}" for bi in range(4)] for ai in range(N_AZ)]
ah_bns = [f"DEF-Ahoge_{bi+1:02d}" for bi in range(5)]

SUB = 16
dt = Period / (N_FRAMES * SUB)
N_WARM_CYCLES = 6

# 状态向量：
# 头发：q_hc(12, 2), v_hc(12, 2); 呆毛：q_ah(2), v_ah(2)
# 风衣 5 路下摆俯仰/外展角 (rad)：q_sk(5, 2) [0=outward_flare, 1=tangential_sway], v_sk(5, 2)
# 袖口 2 + 蝴蝶结 2 (rad)：q_aux(4) [cuffL, cuffR, bowL, bowR], v_aux(4)
q_hc  = np.zeros((N_AZ, 2), np.float64); v_hc  = np.zeros((N_AZ, 2), np.float64)
q_ah  = np.zeros(2, np.float64);         v_ah  = np.zeros(2, np.float64)
q_sk  = np.zeros((5, 2), np.float64);    v_sk  = np.zeros((5, 2), np.float64)
q_aux = np.zeros(4, np.float64);         v_aux = np.zeros(4, np.float64)

w_hair = 2.0 * math.pi * 2.35; z_hair = 0.46
w_ahg  = 2.0 * math.pi * 2.60; z_ahg  = 0.32
w_skt  = 2.0 * math.pi * 2.10; z_skt  = 0.48
w_aux  = 2.0 * math.pi * 2.50; z_aux  = 0.38

def rhs_coupled(t_s, qhc, vhc, qah, vah, qsk, vsk, qax, vax):
    ph = OMEGA * t_s
    # 跑步垂直惯性力：骨盆 z(t) = 0.022*cos(2*ph - 2.32) -> 向上加速度峰值在缓冲谷底(把头发衣摆向下压)，腾空顶峰把头发衣摆向上抛起！
    vert_pump = math.cos(2.0 * ph - 2.32)   # +1 = 腾空最高点(衣摆发梢轻扬)，-1 = 缓冲最低点
    lat_sway  = math.sin(ph - 0.65)

    # 1) 迎面跑步气流场 (角色向 +Y 奔跑 -> 相对气流吹向 -Y 后方，叠加步伐上下脉动)
    W_run = np.array([
        0.10 * lat_sway,
        -(0.52 + 0.22 * math.sin(2.0 * ph - 0.4)),   # 稳定向后 (-Y) 迎面轻风！
        0.24 * (0.35 + 0.65 * vert_pump),            # 步伐上下起伏气动升力！
    ], np.float64)

    # 头发 12 方位笼骨目标角
    targ_hc = np.zeros((N_AZ, 2), np.float64)
    for ai in range(N_AZ):
        p0 = az_phis[ai]
        e_r = np.array([math.cos(p0), math.sin(p0), 0.0])
        e_t = np.array([-math.sin(p0), math.cos(p0), 0.0])
        w_r = float(np.dot(W_run, e_r))
        w_t = float(np.dot(W_run, e_t))
        d_front = abs(float(np.angle(np.exp(1j * (p0 - 0.5 * math.pi)))))
        sec_gain = 0.35 + 0.65 * float(sstep(0.0, math.radians(85.0), d_front))
        # 切向沿头骨顺流向后掠，迎风前刘海严禁向内压穿额头 (pos_part >= 0 + 轻微伯努利浮起)！
        pos_wr = 0.5 * (w_r + math.sqrt(w_r * w_r + 0.02))
        lift   = 0.085 * abs(w_t) + 0.11 * max(0.0, W_run[2] + 0.15)
        targ_hc[ai, 0] = sec_gain * (0.20 * w_t)
        targ_hc[ai, 1] = sec_gain * (0.28 * pos_wr + lift)

    # 头顶呆毛欢快弹跳 (向后微仰 + 随步伐双峰上下Q弹 + 左右微摆)
    targ_ah = np.array([
        0.16 * lat_sway + 0.12 * math.sin(2.0 * ph + 0.3),
        0.26 + 0.22 * vert_pump,
    ], np.float64)

    # 2) 风衣 5 路下摆目标外扬角 (rad)：结合左右大腿抬起角度防穿模护盾 + 迎风后飘！
    thL_deg, knL_deg, _, _ = leg_kinematics(ph)
    thR_deg, knR_deg, _, _ = leg_kinematics(ph + math.pi)

    # 大腿前抬护盾：当左大腿向前抬起 (thL_deg > -2°) 时，左前襟 FL 主动向外前方扬起让路，绝不穿短裤/大腿！
    shield_FL = math.radians(max(0.0, 0.85 * thL_deg + 9.0) + 5.0 + 4.5 * max(0.0, vert_pump))
    shield_FR = math.radians(max(0.0, 0.85 * thR_deg + 9.0) + 5.0 + 4.5 * max(0.0, vert_pump))
    # 后摆与侧后摆：迎风向后优雅飘起 (13°~22°) + 腿后摆折叠防穿让路！
    back_kick_L = max(0.0, -0.45 * thL_deg + 0.12 * abs(knL_deg) - 4.0)
    back_kick_R = max(0.0, -0.45 * thR_deg + 0.12 * abs(knR_deg) - 4.0)
    flare_SL = math.radians(11.0 + 0.55 * back_kick_L + 5.5 * (0.5 + 0.5 * vert_pump))
    flare_SR = math.radians(11.0 + 0.55 * back_kick_R + 5.5 * (0.5 + 0.5 * vert_pump))
    flare_BK = math.radians(14.5 + 0.45 * max(back_kick_L, back_kick_R) + 6.5 * (0.5 + 0.5 * vert_pump))

    targ_sk = np.array([
        [shield_FL,  math.radians(3.5 * lat_sway)],
        [shield_FR,  math.radians(3.5 * lat_sway)],
        [flare_SL,   math.radians(4.0 * lat_sway)],
        [flare_SR,   math.radians(4.0 * lat_sway)],
        [flare_BK,   math.radians(3.0 * lat_sway)],
    ], np.float64)

    # 3) 袖口滞后摆动 + 胸前蝴蝶结双峰弹跳
    # 左臂在 ph=0 时向后摆、ph=π 时向前摆 (-cos(ph))
    armL_vel = +math.sin(ph)
    armR_vel = -math.sin(ph)
    targ_aux = np.array([
        math.radians(-9.5 * armL_vel + 4.0 * vert_pump),   # Cuff L
        math.radians(-9.5 * armR_vel + 4.0 * vert_pump),   # Cuff R
        math.radians(10.0 + 9.5 * vert_pump + 3.5 * lat_sway), # Bow L (向前上方欢快轻弹！)
        math.radians(10.0 + 9.5 * vert_pump - 3.5 * lat_sway), # Bow R
    ], np.float64)

    acc_hc  = (w_hair**2) * (targ_hc  - qhc) - (2.0 * z_hair * w_hair) * vhc
    acc_ah  = (w_ahg**2)  * (targ_ah  - qah) - (2.0 * z_ahg  * w_ahg)  * vah
    acc_sk  = (w_skt**2)  * (targ_sk  - qsk) - (2.0 * z_skt  * w_skt)  * vsk
    acc_aux = (w_aux**2)  * (targ_aux - qax) - (2.0 * z_aux  * w_aux)  * vax
    return vhc, acc_hc, vah, acc_ah, vsk, acc_sk, vax, acc_aux

sampled_phys = []
total_steps = N_WARM_CYCLES * N_FRAMES * SUB
for si in range(total_steps + 1):
    t_now = si * dt
    if si >= (N_WARM_CYCLES - 1) * N_FRAMES * SUB and (si % SUB == 0):
        sampled_phys.append((t_now, q_hc.copy(), q_ah.copy(), q_sk.copy(), q_aux.copy()))
    if si == total_steps:
        break
    st0 = (q_hc, v_hc, q_ah, v_ah, q_sk, v_sk, q_aux, v_aux)
    d1 = rhs_coupled(t_now, *st0)
    st2 = [a + 0.5*dt*da for a, da in zip(st0, d1)]
    d2 = rhs_coupled(t_now + 0.5*dt, *st2)
    st3 = [a + 0.5*dt*da for a, da in zip(st0, d2)]
    d3 = rhs_coupled(t_now + 0.5*dt, *st3)
    st4 = [a + dt*da for a, da in zip(st0, d3)]
    d4 = rhs_coupled(t_now + dt, *st4)
    q_hc, v_hc, q_ah, v_ah, q_sk, v_sk, q_aux, v_aux = [
        a + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        for a, k1, k2, k3, k4 in zip(st0, d1, d2, d3, d4)
    ]

assert len(sampled_phys) == N_FRAMES + 1, f"Expected 25 frames (1..25), got {len(sampled_phys)}"
loop_seam_err = float(np.max(np.abs(sampled_phys[0][3] - sampled_phys[-1][3])))
print(f"RK4 PERIODICITY SEAM ERR (f1 vs f25): {loop_seam_err:.2e} rad", flush=True)

# 预先准备头发笼骨静态局部信息 (与 build_wind_physics.py 完全一致的零穿模四元数算法)
neck_pivot = np.array([-0.0030, 0.0150, -0.1420], np.float64)
hair_chains = []
for ai, p0 in enumerate(az_phis):
    d_front = abs(float(np.angle(np.exp(1j * (p0 - 0.5 * math.pi)))))
    grp = 'fringe' if d_front < math.radians(50.0) else ('side' if d_front < math.radians(125.0) else 'back')
    bns = cage_bone_names[ai]
    bones_info = []
    for bn in bns:
        b = arm_data.bones[bn]
        bones_info.append((bn, np.array(M_arm @ b.head_local, np.float64), np.array(M_arm @ b.tail_local, np.float64)))
    hair_chains.append({'phi': float(p0), 'group': grp, 'bones': bones_info})

ah_bones_info = []
for bn in ah_bns:
    b = arm_data.bones[bn]
    ah_bones_info.append((bn, np.array(M_arm @ b.head_local, np.float64), np.array(M_arm @ b.tail_local, np.float64)))
hair_chains.append({'phi': 0.0, 'group': 'ahoge', 'bones': ah_bones_info})

Y_AXIS = Vector((0.0, 1.0, 0.0))

# 手指微握自然跑姿设定 (排除大拇指单独柔和内扣)
finger_curl_bones = []
for b in arm_data.bones:
    bn = b.name
    low = bn.lower()
    if any(k in low for k in ("index", "middle", "ring", "pinky", "little")):
        deg = 36.0 if "proximal" in low else (44.0 if "intermediate" in low else 28.0)
        sgn = -1.0 if bn.endswith(".L") else +1.0
        finger_curl_bones.append((bn, sgn * deg, "finger"))
    elif "thumb" in low:
        deg = 16.0 if "proximal" in low else 22.0
        sgn = -1.0 if bn.endswith(".L") else +1.0
        finger_curl_bones.append((bn, sgn * deg, "thumb"))

# ------------------------------------------------------------------------------
# 逐帧写入 1..25 帧全身运动学 + 衣装物理 + 头发物理，并自动校准离地高度使支撑鞋底紧贴 Z_FLOOR！
# ------------------------------------------------------------------------------
Z_FLOOR = -1.4120
pbs = arm_ob.pose.bones

# 先做第 1 遍姿态求解测量全周期最低鞋底轨迹，精确校准骨盆基准高度偏移，保证支撑脚刚好踩在地面 Z_FLOOR 上绝不悬空或陷地！
def apply_full_frame_pose(fi, z_offset=0.0):
    t_cur, qhc_f, qah_f, qsk_f, qax_f = sampled_phys[fi]
    bk = body_kinematics(t_cur)
    ph = bk["ph"]

    # 1) 骨盆位移与躯干链旋转 (按父子层级顺序设置并刷新矩阵)
    pb_hips = pbs["Hips"]
    # Hips local X=+World_X, Y=+World_Z (up!), Z=-World_Y (back!)
    hx, hy, hz = bk["hips_loc"]
    pb_hips.location = Vector((hx, hz + z_offset, -hy))

    for bnm in ["Hips", "Spine", "Chest", "Neck", "Head"]:
        rx, ry, rz = bk[bnm]
        Rw = Euler((math.radians(rx), math.radians(ry), math.radians(rz)), 'XYZ').to_matrix()
        set_bone_world_delta(pbs[bnm], Rw, deps_update=True)

    # DEF-Head 保持与 Head 完全同步刚体跟随
    pbs["DEF-Head"].rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))

    # 2) 双腿运动学 (左腿相位 ph，右腿相位 ph + π)
    for side, sgn, ph_s in [("L", -1.0, ph), ("R", +1.0, ph + math.pi)]:
        th_rx, kn_rx, ak_rx, to_rx = leg_kinematics(ph_s)
        # 大腿轻微内收 1.5° 使双足落点沿跑道中线自然秀气，不劈叉！
        Rw_ul = Euler((math.radians(th_rx), math.radians(-sgn * 1.5), 0.0), 'XYZ').to_matrix()
        set_bone_world_delta(pbs[f"upper_leg.{side}"], Rw_ul, deps_update=True)

        Rw_ll = Euler((math.radians(th_rx + kn_rx), math.radians(-sgn * 1.2), 0.0), 'XYZ').to_matrix()
        set_bone_world_delta(pbs[f"lower_leg.{side}"], Rw_ll, deps_update=True)

        # 脚掌世界俯仰 = 小腿俯仰的一半过渡 + 踝关节主动勾/绷，保证触地期鞋底水平贴地！
        foot_world_rx = 0.35 * (th_rx + kn_rx) + ak_rx + 8.0
        Rw_ft = Euler((math.radians(foot_world_rx), 0.0, math.radians(-sgn * 2.5)), 'XYZ').to_matrix()
        set_bone_world_delta(pbs[f"foot.{side}"], Rw_ft, deps_update=True)

        Rw_to = Euler((math.radians(foot_world_rx + to_rx), 0.0, math.radians(-sgn * 2.5)), 'XYZ').to_matrix()
        set_bone_world_delta(pbs[f"toe.{side}"], Rw_to, deps_update=True)

    # 3) 双臂运动学（精确 3D 方向向量驱动：上臂自然贴肋垂落外展 18°、前臂适度曲肘 46~54° 朝正前微内收，100% 零炸袖零穿肘！）
    for side, sgn, ph_arm in [("L", -1.0, ph + math.pi), ("R", +1.0, ph)]:
        # sgn = -1.0 for L (-X side), +1.0 for R (+X side)
        swing = math.cos(ph_arm - 0.25)   # +1 前摆最高，-1 后摆最高
        Rw_sh = Euler((0.0, math.radians(sgn * 4.0), math.radians(-sgn * 3.0 * swing)), 'XYZ').to_matrix()
        set_bone_world_delta(pbs[f"shoulder.{side}"], Rw_sh, deps_update=True)

        # Rest 方向：L 臂指向 (-1, 0, 0)，R 臂指向 (+1, 0, 0)
        v_rest_arm = Vector((sgn * 1.0, 0.0, 0.0))

        # 上臂目标世界方向：外展仅 ~18° (x = sgn*0.31)，前后摆动 (-20° ~ +22°)，向下为主 (z ≈ -0.90)！
        ua_sag_deg = -1.5 + 17.0 * swing   # >0 朝 +Y 前摆，<0 朝 -Y 后摆
        ua_abd_deg = 21.5 + 2.0 * (1.0 - swing)
        v_ua = Vector((
            sgn * math.sin(math.radians(ua_abd_deg)),
            math.cos(math.radians(ua_abd_deg)) * math.sin(math.radians(ua_sag_deg)),
            -math.cos(math.radians(ua_abd_deg)) * math.cos(math.radians(ua_sag_deg)),
        )).normalized()
        q_ua_sw = v_rest_arm.rotation_difference(v_ua)
        R_ua = (Quaternion(v_ua, math.radians(sgn * 12.0)) @ q_ua_sw).to_matrix()
        set_bone_world_delta(pbs[f"upper_arm.{side}"], R_ua, deps_update=True)

        # 前臂目标世界方向：在已验证零穿模的 46°~54° 舒适曲肘区间内朝前上方抬起，且横向微向内收（不向外炸翅膀！）
        elbow_flex_deg = 50.0 + 8.0 * swing   # 前摆曲肘 54°，后摆舒展 42° -> 宽袖自然折弯绝对零穿肘！
        fa_sag_deg = ua_sag_deg + elbow_flex_deg
        # 前摆时手腕向身前微内收 (x ≈ sgn*0.06)，后摆时自然顺垂 (x ≈ sgn*0.20)
        fa_lat = sgn * (0.13 - 0.07 * swing)
        v_fa = Vector((
            fa_lat,
            math.sin(math.radians(fa_sag_deg)),
            -math.cos(math.radians(fa_sag_deg)),
        )).normalized()
        # 叠加轻微旋前使手背朝外侧、拇指朝上（标准慢跑手姿！）
        q_fa_swing = v_rest_arm.rotation_difference(v_fa)
        q_fa_pron  = Quaternion(v_fa, math.radians(sgn * 24.0))
        set_bone_world_delta(pbs[f"lower_arm.{side}"], (q_fa_pron @ q_fa_swing).to_matrix(), deps_update=True)

        # 手掌顺延前臂并带 3.5° 柔和惯性轻摆
        wr_sag_deg = fa_sag_deg + 3.5 * math.sin(ph_arm)
        v_wr = Vector((
            fa_lat * 0.92,
            math.sin(math.radians(wr_sag_deg)),
            -math.cos(math.radians(wr_sag_deg)),
        )).normalized()
        q_wr_swing = v_rest_arm.rotation_difference(v_wr)
        q_wr_pron  = Quaternion(v_wr, math.radians(sgn * 32.0))
        set_bone_world_delta(pbs[f"hand.{side}"], (q_wr_pron @ q_wr_swing).to_matrix(), deps_update=True)

    # 手指自然微握拳 (局部四元数直接优雅卷曲)
    for fbn, fdeg, fkind in finger_curl_bones:
        pb_f = pbs.get(fbn)
        if pb_f:
            if fkind == "finger":
                pb_f.rotation_quaternion = Euler((math.radians(fdeg), 0.0, 0.0), 'XYZ').to_quaternion()
            else:
                pb_f.rotation_quaternion = Euler((0.0, 0.0, math.radians(-fdeg)), 'XYZ').to_quaternion()

    # 4) 14 根衣装物理飘骨旋转 (风衣五路下摆 + 袖口 + 蝴蝶结飘带)
    for si_idx, (tag, _, _, p_ang) in enumerate(skirt_specs):
        flare_rad = float(qsk_f[si_idx, 0])
        sway_rad  = float(qsk_f[si_idx, 1])
        # 沿该扇区径向外法向 (cos p_ang, sin p_ang, 0) 向外掀起 flare_rad：
        # 旋转轴为切向 t_axis = (-sin p_ang, cos p_ang, 0)？（验证：下垂向量 (0,0,-1) 绕 (-sin p, cos p, 0) 旋转 +θ：
        # axis × (0,0,-1) = (-cos p, -sin p, 0)，故绕 (+sin p, -cos p, 0) 旋转 +flare_rad 可使 (0,0,-1) 朝 (+cos p, +sin p) 外扬！）
        out_axis = Vector((math.sin(p_ang), -math.cos(p_ang), 0.0))
        for seg_i, b_suf in enumerate(["01", "02"]):
            bn = f"PHY_Coat_{tag}_{b_suf}"
            # 第 2 节带鞭梢放大与行波滞后
            gain = 0.68 if seg_i == 0 else 1.22
            wave = 1.0 + (0.18 * math.sin(2.0 * ph - 0.65) if seg_i == 1 else 0.0)
            R_fl = Quaternion(out_axis, flare_rad * gain * wave).to_matrix()
            R_sw = Euler((0.0, sway_rad * gain, 0.0), 'XYZ').to_matrix()
            set_bone_world_delta(pbs[bn], R_sw @ R_fl, deps_update=(seg_i == 0))

    # 左右宽袖口微摆骨 (局部绕 Z 前后滞后摆 + 绕 Y 上下微弹)
    pbs["PHY_Cuff_L"].rotation_quaternion = Euler((0.0, 0.0, float(qax_f[0]))).to_quaternion()
    pbs["PHY_Cuff_R"].rotation_quaternion = Euler((0.0, 0.0, float(-qax_f[1]))).to_quaternion()

    # 胸前蝴蝶结双飘带骨 (向前 +Y 扬起 = Rx > 0，并微向外分叉)
    R_bowL = Euler((float(qax_f[2]), math.radians(-5.0), 0.0), 'XYZ').to_matrix()
    R_bowR = Euler((float(qax_f[3]), math.radians(+5.0), 0.0), 'XYZ').to_matrix()
    set_bone_world_delta(pbs["PHY_Bow_L"], R_bowL, deps_update=False)
    set_bone_world_delta(pbs["PHY_Bow_R"], R_bowR, deps_update=False)

    # 5) 54 根头发空间笼骨与头顶呆毛物理写入 (叠加在当前 Head 姿态之上)
    R_head_cur = (M_arm.to_3x3() @ pbs["Head"].matrix.to_3x3()) @ R_rest_world["Head"].inverted()
    for ci, ch in enumerate(hair_chains):
        for bi, (bn, h_rest_w, t_rest_w) in enumerate(ch["bones"]):
            seg_gain = 0.48 + 0.38 * bi
            wave = 1.0 + 0.16 * math.sin(2.0 * ph - 0.50 * bi)
            if ch["group"] == "ahoge":
                d_ang = Euler((
                    float(qah_f[1]) * seg_gain * wave,       # Rx>0? Check: ahoge points UP (+Z), Rx>0 tilts +Z toward -Y (backward!) -> exact headwind blowback!
                    float(qah_f[0]) * seg_gain * wave,
                    0.0,
                ), 'XYZ').to_matrix()
            else:
                p0 = ch["phi"]
                ai_idx = ci
                th_tan   = float(qhc_f[ai_idx, 0]) * seg_gain * wave
                th_flare = float(qhc_f[ai_idx, 1]) * seg_gain * wave
                # 切向绕 Z 旋转 th_tan + 径向外扬绕 (+sin p0, -cos p0, 0) 旋转 th_flare (下垂发束朝外飘起！)
                out_ax_h = Vector((math.sin(p0), -math.cos(p0), 0.0))
                d_ang = Euler((0.0, 0.0, th_tan), 'XYZ').to_matrix() @ Quaternion(out_ax_h, th_flare).to_matrix()
            set_bone_world_delta(pbs[bn], d_ang @ R_head_cur, deps_update=(bi < len(ch["bones"]) - 1))

    bpy.context.view_layer.update()

# 先探测触地最低高度以自动对齐 Z_FLOOR = -1.4120m！
min_shoe_zs = []
for test_fi in [0, 3, 6, 9, 12, 15, 18, 21]:
    apply_full_frame_pose(test_fi, z_offset=0.0)
    deps_t = bpy.context.evaluated_depsgraph_get()
    z_low = 999.0
    for sh_ob in (shoeL_ob, shoeR_ob):
        eo = sh_ob.evaluated_get(deps_t)
        me = eo.to_mesh()
        c = np.empty(len(me.vertices)*3, dtype=np.float32)
        me.vertices.foreach_get("co", c)
        Mw = np.array(eo.matrix_world, dtype=np.float32)
        eo.to_mesh_clear()
        wz = ((c.reshape(-1, 3) @ Mw[:3, :3].T) + Mw[:3, 3])[:, 2]
        z_low = min(z_low, float(wz.min()))
    min_shoe_zs.append(z_low)

stance_min_z = min(min_shoe_zs)
z_calib = Z_FLOOR - stance_min_z
print(f"HEAD_VS_HIPS_DY={(M_arm @ pbs['Head'].head).y - (M_arm @ pbs['Hips'].head).y:+.4f}m", flush=True)
print(f"SHOE FLOOR CALIBRATION: raw_min={stance_min_z:.4f}m -> z_calib={z_calib*1000:+.1f}mm (flight peak clearance={(max(min_shoe_zs)+z_calib-Z_FLOOR)*1000:+.1f}mm)", flush=True)

# 正式逐帧 (f=1..25) 烘焙全部关键帧！
key_bone_list = (
    ["Hips", "Spine", "Chest", "Neck", "Head", "DEF-Head",
     "shoulder.L", "upper_arm.L", "lower_arm.L", "hand.L",
     "shoulder.R", "upper_arm.R", "lower_arm.R", "hand.R",
     "upper_leg.L", "lower_leg.L", "foot.L", "toe.L",
     "upper_leg.R", "lower_leg.R", "foot.R", "toe.R"]
    + [b for b, _, _ in finger_curl_bones]
    + [b for b, _, _, _, _ in new_bones_spec]
    + [bn for row in cage_bone_names for bn in row]
    + ah_bns
)

for fi in range(N_FRAMES + 1):
    fno = fi + 1
    sc.frame_set(fno)
    apply_full_frame_pose(fi, z_offset=z_calib)
    pbs["Hips"].keyframe_insert(data_path="location", frame=fno)
    for bnm in key_bone_list:
        pb = pbs.get(bnm)
        if pb:
            pb.keyframe_insert(data_path="rotation_quaternion", frame=fno)

sc.frame_set(1)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_STAGE), compress=True)

# 渲染 8 个关键相位 (f1, f4, f7, f10, f13, f16, f19, f22) × 3 视角 (斜侧35° / 正侧90° / 正面0°) 连拍图用于视觉质检！
sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 20
sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.image_settings.quality = 92
sc.render.film_transparent = False

cam = bpy.data.objects["REVIEW_CAMERA"]
sc.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 1.76
sc.render.resolution_x = 520
sc.render.resolution_y = 920

qa_frames = [1, 4, 7, 10, 13, 16, 19, 22]
views = [
    ("obliq35", (-1.84, 2.62, -0.63), (90.0, 0.0, -145.0)),
    ("side90",  (-3.20, 0.00, -0.63), (90.0, 0.0,  -90.0)),
    ("front0",  ( 0.00, 3.20, -0.63), (90.0, 0.0,  180.0)),
]
for vnm, loc, rot in views:
    cam.location = Vector(loc)
    cam.rotation_euler = Euler([math.radians(a) for a in rot], 'XYZ')
    for fno in qa_frames:
        sc.frame_set(fno)
        bpy.context.view_layer.update()
        sc.render.filepath = str(RRUN / f"qa_{vnm}_f{fno:02d}.jpg")
        bpy.ops.render.render(write_still=True)

print("V10R10_STEP123_DONE", flush=True)
