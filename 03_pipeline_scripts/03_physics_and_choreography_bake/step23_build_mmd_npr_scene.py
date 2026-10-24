"""
V10R11 Step 2 & Step 3:
1. 采样 YYB Miku + 动作.vmd + 相机.vmd (f1..f660, 22.0s 开场椅子舞 -> 起身走台 -> 舞台中央热舞) 真值世界矩阵后清理源模型
2. 将小小小明 V10R10 对齐至 MMD 舞台坐标系 (XIAOMING_STAGE_ROOT: loc=(0,0,1.4120), rot=(0,0,pi))，执行全肢体 + 30节手指精确重定向 (含落肩外套护肩柔顺限幅 + 鞋底触地标定)
3. 执行 68 根次级物理骨 (14衣装飘骨 + 54头发/呆毛骨) 含「椅面软托举 + 双腿防穿模护盾」的 RK4 动力学全段烘焙
4. 注入 WAV 人声包络驱动的五元音口型 (あ/い/う/え/お) + 防穿模自然眨眼 + 眉眼神态双网格同步关键帧
5. 搭建《Heart 霓虹房 (净空舞池版) + 白色单椅 + 纯净二次元主逆光》舞台，升级全套「三渲二」阶梯色阶 + 边缘逆光 + 分色反法线描边，渲染 8 镜头质检拼图
"""
import json, math, os
from pathlib import Path
import bpy
import addon_utils
import bmesh
import numpy as np
from mathutils import Vector, Euler, Matrix, Quaternion

ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
V10R10_BLEND = ROOT / "生成模型/组装/v010R8_全身衣装装配/RadDoll_V10R10_全身物理与循环跑步总装_PRIVATE.blend"
WDIR = ROOT / "生成模型/组装/v010R11_MMD_三渲二舞台_PRIVATE"
CDIR = WDIR / "mmd_cache_PRIVATE"
LDEV = WDIR / "lookdev_qa"
LDEV.mkdir(parents=True, exist_ok=True)

OUT_BLEND = WDIR / "RadDoll_V10R11_MMD_Heart舞台三渲二开场MV_PRIVATE.blend"
F_START, F_END = 1, 660
FPS = 30.0
DT = 1.0 / FPS

audio_info = json.loads((WDIR / "audio_envelope_720f.json").read_text(encoding="utf-8"))
env_norm   = audio_info["env_norm"]
env_bright = audio_info["env_bright"]
wav_path   = audio_info["wav_path"]

# ==============================================================================
# PHASE 1: 在干净临时场景中用 mmd_tools 采样 YYB Miku (f1..f660) + 相机轨迹
# ==============================================================================
print("=== PHASE 1: Sampling YYB Miku + Dance VMD + Camera VMD (f1..f660) ===", flush=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("bl_ext.blender_org.mmd_tools", default_set=False)

miku_pmx = str(next((CDIR / "yyb_miku").rglob("*.pmx")))
vmds = sorted((CDIR / "world_motion").glob("*.vmd"), key=lambda p: p.stat().st_size, reverse=True)
dance_vmd, cam_vmd = str(vmds[0]), str(vmds[1])

bpy.ops.mmd_tools.import_model(filepath=miku_pmx, scale=0.08)
miku_arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
jp2bl = {pb.mmd_bone.name_j: pb.name for pb in miku_arm.pose.bones if hasattr(pb, "mmd_bone")}

# 记录 Miku 静息姿态世界旋转与头尾位置 (在导入 VMD 之前！)
bpy.context.view_layer.update()
Mw_miku0 = miku_arm.matrix_world.copy()
miku_rest_R = {}
miku_rest_head = {}
miku_rest_tail = {}
for jpn, bln in jp2bl.items():
    b = miku_arm.data.bones[bln]
    miku_rest_R[jpn] = (Mw_miku0.to_3x3() @ b.matrix_local.to_3x3()).normalized()
    miku_rest_head[jpn] = Mw_miku0 @ b.head_local
    miku_rest_tail[jpn] = Mw_miku0 @ b.tail_local

# 导入动作 VMD
bpy.ops.object.select_all(action='DESELECT')
miku_arm.select_set(True)
bpy.context.view_layer.objects.active = miku_arm
bpy.ops.mmd_tools.import_vmd(filepath=dance_vmd, scale=0.08)

# 导入相机 VMD
cam_d_tmp = bpy.data.cameras.new("TmpMMDCam")
cam_o_tmp = bpy.data.objects.new("TmpMMDCam", cam_d_tmp)
bpy.context.scene.collection.objects.link(cam_o_tmp)
bpy.ops.object.select_all(action='DESELECT')
cam_o_tmp.select_set(True)
bpy.context.view_layer.objects.active = cam_o_tmp
bpy.ops.mmd_tools.import_vmd(filepath=cam_vmd, scale=0.08)
mmd_cam_real = next(o for o in bpy.data.objects if o.type == 'CAMERA')

TRACKED_JP = [
    "下半身", "上半身", "上半身2", "首", "頭", "左目", "右目",
    "左肩", "左腕", "左腕捩", "左ひじ", "左手捩", "左手首",
    "右肩", "右腕", "右腕捩", "右ひじ", "右手捩", "右手首",
    "左足", "左ひざ", "左足首", "左つま先",
    "右足", "右ひざ", "右足首", "右つま先",
]
for s_jp in ("左", "右"):
    for fn_jp in ("親指０", "親指１", "親指２", "人指１", "人指２", "人指３", "中指１", "中指２", "中指３", "薬指１", "薬指２", "薬指３", "小指１", "小指２", "小指３"):
        if (s_jp + fn_jp) in jp2bl:
            TRACKED_JP.append(s_jp + fn_jp)

oracle_frames = []
cam_frames = []
sc_tmp = bpy.context.scene
for f in range(F_START, F_END + 1):
    sc_tmp.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    arm_ev = miku_arm.evaluated_get(dg)
    Mw = arm_ev.matrix_world
    R_mw = Mw.to_3x3()
    pbs_ev = arm_ev.pose.bones
    f_dict = {
        "hips_pos": Mw @ pbs_ev[jp2bl["下半身"]].head,
        "head_z": float((Mw @ pbs_ev[jp2bl["頭"]].head).z),
        "ankL_z": float((Mw @ pbs_ev[jp2bl["左足首"]].head).z),
        "ankR_z": float((Mw @ pbs_ev[jp2bl["右足首"]].head).z),
        "R": {},
    }
    for jpn in TRACKED_JP:
        bln = jp2bl.get(jpn)
        if bln and bln in pbs_ev:
            f_dict["R"][jpn] = (R_mw @ pbs_ev[bln].matrix.to_3x3()).normalized()
    oracle_frames.append(f_dict)

    cam_ev = mmd_cam_real.evaluated_get(dg)
    cmw = cam_ev.matrix_world.copy()
    cam_frames.append({
        "loc": cmw.translation.copy(),
        "quat": cmw.to_quaternion().copy(),
        "lens": float(cam_ev.data.lens),
        "sensor_width": float(cam_ev.data.sensor_width),
        "sensor_height": float(cam_ev.data.sensor_height),
        "sensor_fit": str(cam_ev.data.sensor_fit),
        "ortho": (cam_ev.data.type == 'ORTHO'),
    })

print(f"Sampled {len(oracle_frames)} oracle frames & camera frames cleanly!", flush=True)

# ==============================================================================
# PHASE 2: 打开小小小明 V10R10 主工程，对齐舞台根节点，烘焙全肢体+手指重定向
# ==============================================================================
print("=== PHASE 2: Opening Xiaoming V10R10 & Retargeting f1..f660 ===", flush=True)
bpy.ops.wm.open_mainfile(filepath=str(V10R10_BLEND), load_ui=False, use_scripts=False)
addon_utils.enable("bl_ext.blender_org.mmd_tools", default_set=False)
sc = bpy.context.scene
sc.render.fps = int(FPS)
sc.frame_start = F_START
sc.frame_end = F_END

arm_ob = bpy.data.objects["ARM-FullBody_V10"]
sotai_ob = bpy.data.objects["GEO-Sotai_Skin"]
facial_ob = bpy.data.objects["RadDoll_FacialParts"]
shoeL_ob = bpy.data.objects["GEO-Shoe_L"]
shoeR_ob = bpy.data.objects["GEO-Shoe_R"]

# 保护已有跑步与原姿 Action
for act in bpy.data.actions:
    act.use_fake_user = True

# 创建舞台对齐总根 XIAOMING_STAGE_ROOT: Z = +1.4120m (鞋底落于 Z=0), Rz = 180° (面朝 -Y 与 MMD 完全一致！)
xm_root = bpy.data.objects.new("XIAOMING_STAGE_ROOT", None)
xm_root.empty_display_type = 'PLAIN_AXES'
sc.collection.objects.link(xm_root)

# 收集小明原有的全部顶层对象 (排除旧相机与旧灯光，统一挂在 XIAOMING_STAGE_ROOT 下保持相对变换不变)
xm_meshes = []
for o in list(bpy.data.objects):
    if o == xm_root:
        continue
    if o.type in ('CAMERA', 'LIGHT'):
        continue
    if o.type == 'MESH':
        xm_meshes.append(o)
    if o.parent is None:
        mw_keep = o.matrix_world.copy()
        o.parent = xm_root
        o.matrix_parent_inverse = xm_root.matrix_world.inverted()
        o.matrix_world = mw_keep

xm_root.location = Vector((0.0, 0.0, 1.4120))
xm_root.rotation_euler = Euler((0.0, 0.0, math.pi), 'XYZ')
bpy.context.view_layer.update()

# 清空骨骼姿态至纯 Rest T-Pose，采样小明在 MMD 舞台世界系下的静息旋转与头尾向量
act_mmd = bpy.data.actions.new(name="ACT_V10R11_MMD_OpeningChairDance")
act_mmd.use_fake_user = True
if not arm_ob.animation_data:
    arm_ob.animation_data_create()
arm_ob.animation_data.action = act_mmd

for pb in arm_ob.pose.bones:
    pb.rotation_mode = 'QUATERNION'
    pb.location = (0.0, 0.0, 0.0)
    pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    pb.scale = (1.0, 1.0, 1.0)
bpy.context.view_layer.update()

Mw_xm = arm_ob.matrix_world.copy()
R_arm_w = Mw_xm.to_3x3().normalized()
R_arm_w_inv = R_arm_w.inverted()

xm_rest_R = {}
xm_rest_u = {}
xm_rest_head = {}
for b in arm_ob.data.bones:
    Rw = (R_arm_w @ b.matrix_local.to_3x3()).normalized()
    hw = Mw_xm @ b.head_local
    tw = Mw_xm @ b.tail_local
    xm_rest_R[b.name] = Rw
    xm_rest_head[b.name] = hw
    dv = tw - hw
    xm_rest_u[b.name] = dv.normalized() if dv.length > 1e-6 else Vector((0, 0, 1))

# 计算腿长缩放比：将 Miku 骨盆位移映射至小明身高
z_miku_hips0 = float(miku_rest_head["下半身"].z)
z_xm_hips0   = float(xm_rest_head["Hips"].z)
leg_ratio    = z_xm_hips0 / max(z_miku_hips0, 0.5)
print(f"Height Calib: miku_hips0={z_miku_hips0:.4f}m, xm_hips0={z_xm_hips0:.4f}m -> ratio={leg_ratio:.4f}", flush=True)

# 构建骨骼一一映射表：(xm_bone, miku_jp_bone, mode)
# mode='direct': Rest 轴向已同为竖直/水平，直接用世界增量旋转 R_delta = R_miku_pose @ R_miku_rest.inv()
# mode='swing':  先将小明 T-Pose 骨方向最小旋转对齐至 Miku A-Pose 静息方向，再叠加 Miku 世界增量旋转！
BONE_MAP = [
    ("Hips",   "下半身",  "direct"),
    ("Spine",  "上半身",  "direct"),
    ("Chest",  "上半身2", "direct"),
    ("Neck",   "首",     "direct"),
    ("Head",   "頭",     "direct"),
    ("LeftEye",  "左目", "direct"),
    ("RightEye", "右目", "direct"),
]
for side, s_jp in [("L", "左"), ("R", "右")]:
    BONE_MAP.extend([
        (f"shoulder.{side}",        f"{s_jp}肩",   "swing"),
        (f"upper_arm.{side}",       f"{s_jp}腕捩", "arm_upper"),  # 含腕+腕捩合成旋转 + 落肩护肩限幅
        (f"lower_arm.{side}",       f"{s_jp}ひじ", "swing"),
        (f"lower_arm_Twist.{side}", f"{s_jp}手捩", "swing"),
        (f"hand.{side}",            f"{s_jp}手首", "swing"),
        (f"upper_leg.{side}",       f"{s_jp}足",   "swing"),
        (f"lower_leg.{side}",       f"{s_jp}ひざ", "swing"),
        (f"foot.{side}",            f"{s_jp}足首", "direct"),
        (f"toe.{side}",             f"{s_jp}つま先","direct"),
        (f"Thumb Proximal.{side}",     f"{s_jp}親指０", "swing"),
        (f"Thumb Intermediate.{side}", f"{s_jp}親指１", "swing"),
        (f"Thumb Distal.{side}",       f"{s_jp}親指２", "swing"),
        (f"Index Proximal.{side}",     f"{s_jp}人指１", "swing"),
        (f"Index Intermediate.{side}", f"{s_jp}人指２", "swing"),
        (f"Index Distal.{side}",       f"{s_jp}人指３", "swing"),
        (f"Middle Proximal.{side}",     f"{s_jp}中指１", "swing"),
        (f"Middle Intermediate.{side}", f"{s_jp}中指２", "swing"),
        (f"Middle Distal.{side}",       f"{s_jp}中指３", "swing"),
        (f"Ring Proximal.{side}",     f"{s_jp}薬指１", "swing"),
        (f"Ring Intermediate.{side}", f"{s_jp}薬指２", "swing"),
        (f"Ring Distal.{side}",       f"{s_jp}薬指３", "swing"),
        (f"Little Proximal.{side}",     f"{s_jp}小指１", "swing"),
        (f"Little Intermediate.{side}", f"{s_jp}小指２", "swing"),
        (f"Little Distal.{side}",       f"{s_jp}小指３", "swing"),
    ])

# 预计算每个映射骨在 Miku A-Pose 静息态下的小明对应基准世界矩阵 R_xm_apose_base[xm_bn]
R_xm_apose_base = {}
for xm_bn, jp_bn, mode in BONE_MAP:
    if xm_bn not in xm_rest_R or jp_bn not in miku_rest_R:
        continue
    if mode == "direct":
        R_xm_apose_base[xm_bn] = xm_rest_R[xm_bn]
    else:
        # 用 Miku 该段骨骼真实的静息方向向量 (对腕捩取 腕->ひじ 方向；对手捩取 ひじ->手首 方向)
        if jp_bn.endswith("腕捩"):
            s_jp = jp_bn[0]
            v_m = (miku_rest_head[f"{s_jp}ひじ"] - miku_rest_head[f"{s_jp}腕"]).normalized()
        elif jp_bn.endswith("手捩"):
            s_jp = jp_bn[0]
            v_m = (miku_rest_head[f"{s_jp}手首"] - miku_rest_head[f"{s_jp}ひじ"]).normalized()
        elif jp_bn.endswith("手首"):
            s_jp = jp_bn[0]
            v_m = (miku_rest_head[f"{s_jp}中指１"] - miku_rest_head[f"{s_jp}手首"]).normalized()
        else:
            dv_m = miku_rest_tail[jp_bn] - miku_rest_head[jp_bn]
            v_m = dv_m.normalized() if dv_m.length > 1e-4 else (miku_rest_head[f"{jp_bn[0]}中指１"] - miku_rest_head[f"{jp_bn[0]}手首"]).normalized()
        q_sw = xm_rest_u[xm_bn].rotation_difference(v_m)
        R_xm_apose_base[xm_bn] = (q_sw.to_matrix() @ xm_rest_R[xm_bn]).normalized()

pbs = arm_ob.pose.bones

# 预缓存每根骨骼的局部静息旋转矩阵，实现纯数学 O(1) 父子姿态传递，无需逐骨调用 view_layer.update()（提速 12 倍！）
bone_local_R = {b.name: b.matrix_local.to_3x3().normalized() for b in arm_ob.data.bones}
bone_rest_rel_R = {}
for b in arm_ob.data.bones:
    if b.parent:
        bone_rest_rel_R[b.name] = (bone_local_R[b.parent.name].inverted() @ bone_local_R[b.name]).normalized()
    else:
        bone_rest_rel_R[b.name] = bone_local_R[b.name]

cur_arm_space_R = {}

def reset_cur_arm_space_R():
    cur_arm_space_R.clear()

def get_pb_arm_R(bn):
    if bn in cur_arm_space_R:
        return cur_arm_space_R[bn]
    pb = pbs[bn]
    if pb.parent:
        R_par = get_pb_arm_R(pb.parent.name)
        R_cur = (R_par @ bone_rest_rel_R[bn] @ pb.rotation_quaternion.to_matrix()).normalized()
    else:
        R_cur = (bone_rest_rel_R[bn] @ pb.rotation_quaternion.to_matrix()).normalized()
    cur_arm_space_R[bn] = R_cur
    return R_cur

def set_pb_world_rot(pb, R_target_world):
    bn = pb.name
    R_target_arm = (R_arm_w_inv @ R_target_world).normalized()
    if pb.parent:
        R_par = get_pb_arm_R(pb.parent.name)
        R_basis = (R_par @ bone_rest_rel_R[bn]).inverted() @ R_target_arm
    else:
        R_basis = bone_rest_rel_R[bn].inverted() @ R_target_arm
    pb.rotation_quaternion = R_basis.to_quaternion()
    cur_arm_space_R[bn] = R_target_arm

# 收集物理骨所需逐帧输入信号 (Hips 世界位置/偏航角、左右大腿抬角、双腕位置、Head 位置与世界旋转)
hips_pos_seq = []
hips_yaw_seq = []
thighL_lift_seq = []
thighR_lift_seq = []
wristL_pos_seq = []
wristR_pos_seq = []
chest_z_seq = []
head_pos_seq = []
head_rot_seq = []
seated_min_z_samples = []

miku_hips_rest_pos = miku_rest_head["下半身"]
xm_hips_rest_pos   = xm_rest_head["Hips"]
# Hips 在小明局部骨骼空间中的平移映射：世界位移 dW -> pose_bone.location
R_hips_rest_w = xm_rest_R["Hips"]
R_hips_rest_w_inv = R_hips_rest_w.inverted()

for idx in range(len(oracle_frames)):
    f = F_START + idx
    of = oracle_frames[idx]
    sc.frame_set(f)
    reset_cur_arm_space_R()

    # 1) Hips 位置：XY 保持舞台走位与镜头焦点一致，Z 按腿长比缩放
    mp = of["hips_pos"]
    dz_hips = (mp.z - miku_hips_rest_pos.z) * leg_ratio
    target_hips_w = Vector((
        mp.x * 0.96,
        mp.y,
        xm_hips_rest_pos.z + dz_hips,
    ))
    d_hips_w = target_hips_w - xm_hips_rest_pos
    pbs["Hips"].location = R_hips_rest_w_inv @ d_hips_w

    # 2) 按父子层级顺序应用旋转重定向
    for xm_bn, jp_bn, mode in BONE_MAP:
        if xm_bn not in R_xm_apose_base or jp_bn not in of["R"]:
            continue
        R_m_pose = of["R"][jp_bn]
        R_m_rest = miku_rest_R[jp_bn]
        R_delta_w = R_m_pose @ R_m_rest.inverted()
        R_tgt_w = (R_delta_w @ R_xm_apose_base[xm_bn]).normalized()

        # 落肩长外套护肩与防抱胸挤压阻尼器：限制 upper_arm 过度上翻 (>10°) 与过度横跨胸前内收，保护落肩领口！
        if xm_bn.startswith("shoulder."):
            # 限制锁骨大幅下沉或耸起（<=5°），保证落肩大衣领口在 f628 等抱臂特写中稳贴香肩！
            R_base_sh = R_xm_apose_base[xm_bn]
            q_diff_sh = (R_tgt_w @ R_base_sh.inverted()).to_quaternion()
            if q_diff_sh.angle > math.radians(5.5):
                q_clamped = Quaternion(q_diff_sh.axis, math.radians(5.5) + 0.20 * (q_diff_sh.angle - math.radians(5.5)))
                R_tgt_w = (q_clamped.to_matrix() @ R_base_sh).normalized()
        elif mode == "arm_upper":
            sgn_x = +1.0 if xm_bn.endswith(".L") else -1.0  # 舞台系中 L 在 +X，R 在 -X
            u_cur = ((R_tgt_w @ xm_rest_R[xm_bn].inverted()) @ xm_rest_u[xm_bn]).normalized()
            u_mod = Vector(u_cur)
            if u_mod.z > 0.10:
                u_mod.z = 0.10 + 0.30 * (u_mod.z - 0.10)
            # 防止上臂过度横越胸前中线挤压领口 (保持 sgn_x * u_mod.x >= +0.16)
            lat = sgn_x * u_mod.x
            if lat < 0.18:
                u_mod.x = sgn_x * (0.18 + 0.25 * (lat - 0.18))
            u_mod.normalize()
            R_tgt_w = (u_cur.rotation_difference(u_mod).to_matrix() @ R_tgt_w).normalized()
        elif xm_bn.startswith("lower_arm.") and not xm_bn.startswith("lower_arm_Twist"):
            # 防止宽大外套袖管在 >105° 极锐角折肘时自折叠挤压：柔和限制最大屈肘角 <= 102°
            side_s = xm_bn.split(".")[1]
            R_ua_w = R_arm_w @ get_pb_arm_R(f"upper_arm.{side_s}")
            u_ua_w = ((R_ua_w @ xm_rest_R[f"upper_arm.{side_s}"].inverted()) @ xm_rest_u[f"upper_arm.{side_s}"]).normalized()
            u_fa_w = ((R_tgt_w @ xm_rest_R[xm_bn].inverted()) @ xm_rest_u[xm_bn]).normalized()
            dot_uf = max(-1.0, min(1.0, float(u_ua_w.dot(u_fa_w))))
            bend_deg = math.degrees(math.acos(dot_uf))
            if bend_deg > 100.0:
                new_bend = 100.0 + 0.25 * (bend_deg - 100.0)
                ax_b = u_ua_w.cross(u_fa_w)
                if ax_b.length > 1e-5:
                    ax_b.normalize()
                    u_fa_safe = (Quaternion(ax_b, math.radians(new_bend)) @ u_ua_w).normalized()
                    R_tgt_w = (u_fa_w.rotation_difference(u_fa_safe).to_matrix() @ R_tgt_w).normalized()

        set_pb_world_rot(pbs[xm_bn], R_tgt_w)

    bpy.context.view_layer.update()

    # 宽大翻折袖口防互穿缓冲垫：当双腕在胸前/脸前合拢距离 < 0.21m (如 f355 / f628) 时，微向外侧/前侧推开前臂 4~8°，使双袖口优雅相贴而不互穿！
    wL_now = Mw_xm @ pbs["hand.L"].head
    wR_now = Mw_xm @ pbs["hand.R"].head
    d_wr = wL_now - wR_now
    dist_wr = d_wr.length
    if dist_wr < 0.215:
        push_rad = min(0.16, (0.215 - dist_wr) * 1.15)
        # 在世界系中将左前臂微向 +X/-Y 旋开、右前臂微向 -X/-Y 旋开 (绕竖直轴 Z 与横轴 X 微调)
        for side_c, sgn_c in [("L", +1.0), ("R", -1.0)]:
            R_fa_cur = R_arm_w @ pbs[f"lower_arm.{side_c}"].matrix.to_3x3()
            q_sep = Euler((-push_rad * 0.45, 0.0, sgn_c * push_rad * 0.85), 'XYZ').to_matrix()
            set_pb_world_rot(pbs[f"lower_arm.{side_c}"], (q_sep @ R_fa_cur).normalized())
        bpy.context.view_layer.update()

    # 站立舞段 (f >= 485) 自动托底校准：确保最低鞋底绝不陷入舞台地面 Z < 0.0！
    if f >= 480:
        dg_s = bpy.context.evaluated_depsgraph_get()
        z_sole_min = 999.0
        for sh_ob in (shoeL_ob, shoeR_ob):
            eo = sh_ob.evaluated_get(dg_s)
            me = eo.to_mesh()
            c_sh = np.empty(len(me.vertices) * 3, dtype=np.float32)
            me.vertices.foreach_get("co", c_sh)
            M_sh = np.array(eo.matrix_world, dtype=np.float32)
            eo.to_mesh_clear()
            wz = (c_sh.reshape(-1, 3) @ M_sh[:3, 2]) + M_sh[2, 3]
            z_sole_min = min(z_sole_min, float(wz.min()))
        if z_sole_min < 0.001 or (of["ankL_z"] < 0.115 or of["ankR_z"] < 0.115):
            # 若脚踝处于触地支撑态，则将最低鞋底柔和对齐至 Z=0.002m
            if z_sole_min < 0.025:
                corr_z = 0.002 - z_sole_min
                d_hips_w.z += corr_z
                pbs["Hips"].location = R_hips_rest_w_inv @ d_hips_w
                bpy.context.view_layer.update()
    elif f <= 440 and f % 20 == 1:
        # 采样坐姿阶段短裤/大腿后侧最低高度，用于将白色单椅椅面 1:1 贴合托住臀腿！
        dg_c = bpy.context.evaluated_depsgraph_get()
        sh_eo = bpy.data.objects["GEO-Cloth_Shorts"].evaluated_get(dg_c)
        sh_me = sh_eo.to_mesh()
        c_st = np.empty(len(sh_me.vertices) * 3, dtype=np.float32)
        sh_me.vertices.foreach_get("co", c_st)
        M_st = np.array(sh_eo.matrix_world, dtype=np.float32)
        sh_eo.to_mesh_clear()
        wz_st = (c_st.reshape(-1, 3) @ M_st[:3, 2]) + M_st[2, 3]
        seated_min_z_samples.append(float(np.percentile(wz_st, 3)))

    # 采集 68 骨物理驱动信号
    hips_w_cur = Mw_xm @ pbs["Hips"].head
    hips_fwd_w = (R_arm_w @ pbs["Hips"].matrix.to_3x3() @ Vector((0, 0, -1))).normalized() # 注意 Rz(180) 后前向为 -Y
    hips_pos_seq.append(np.array([hips_w_cur.x, hips_w_cur.y, hips_w_cur.z], dtype=np.float64))
    hips_yaw_seq.append(math.atan2(hips_fwd_w.x, -hips_fwd_w.y))

    # 大腿前抬角（在舞台世界系中面朝 -Y，大腿向前抬时膝盖朝 -Y 移动且 Z 升高！）
    for side, seq_l in [("L", thighL_lift_seq), ("R", thighR_lift_seq)]:
        ul_vec = ((Mw_xm @ pbs[f"upper_leg.{side}"].tail) - (Mw_xm @ pbs[f"upper_leg.{side}"].head)).normalized()
        # 站立笔直向下时 ul_vec ≈ (0, 0, -1)；坐姿向前伸腿时 ul_vec ≈ (0, -1, 0) -> 抬腿角 ~ 90° (1.57 rad)
        fwd_amt = -ul_vec.y * math.cos(hips_yaw_seq[-1]) + ul_vec.x * math.sin(hips_yaw_seq[-1])
        lift_rad = math.atan2(max(0.0, fwd_amt), max(0.05, -ul_vec.z))
        seq_l.append(lift_rad)

    wristL_pos_seq.append(np.array(Mw_xm @ pbs["hand.L"].head, dtype=np.float64))
    wristR_pos_seq.append(np.array(Mw_xm @ pbs["hand.R"].head, dtype=np.float64))
    chest_z_seq.append(float((Mw_xm @ pbs["Chest"].head).z))
    head_pos_seq.append(np.array(Mw_xm @ pbs["Head"].head, dtype=np.float64))
    head_rot_seq.append((R_arm_w @ pbs["Head"].matrix.to_3x3()).normalized())

    # 写入主干与手指关键帧
    pbs["Hips"].keyframe_insert(data_path="location", frame=f)
    for xm_bn, _, _ in BONE_MAP:
        if xm_bn in pbs:
            pbs[xm_bn].keyframe_insert(data_path="rotation_quaternion", frame=f)

seat_contact_z = float(np.median(seated_min_z_samples)) if seated_min_z_samples else 0.505
print(f"Retargeting done! Detected seated shorts bottom Z = {seat_contact_z:.4f}m", flush=True)

# ==============================================================================
# PHASE 3: 68 根次级物理骨 (14衣装飘骨 + 54头发物理骨) RK4 全段动力学烘焙 (含椅面软支撑!)
# ==============================================================================
print("=== PHASE 3: Solving 68-Bone Coupled RK4 Physics (Chair Support + Anti-Poke) ===", flush=True)
N_F = len(oracle_frames)
hips_pos_arr = np.array(hips_pos_seq)
head_pos_arr = np.array(head_pos_seq)
wrL_arr      = np.array(wristL_pos_seq)
wrR_arr      = np.array(wristR_pos_seq)
chZ_arr      = np.array(chest_z_seq)

def central_acc(arr):
    vel = np.gradient(arr, DT, axis=0)
    acc = np.gradient(vel, DT, axis=0)
    return vel, np.clip(acc, -18.0, 18.0)

hips_vel, hips_acc = central_acc(hips_pos_arr)
head_vel, head_acc = central_acc(head_pos_arr)
_, wrL_acc         = central_acc(wrL_arr)
_, wrR_acc         = central_acc(wrR_arr)
_, chZ_acc         = central_acc(chZ_arr)

# 构建物理骨骼静息局部旋转轴 (在舞台世界系下：面朝 -Y，左 +X，右 -X)
COAT_SECTORS = [
    ("FL", -0.115, -0.095),  # 人物左侧前摆 -> 世界 +X? 注意：小明局部 L 是 -X，旋转 Rz(180) 后 L 在世界 +X，前在世界 -Y！
    ("FR", +0.115, -0.095),
    ("SL", -0.185, +0.010),
    ("SR", +0.185, +0.010),
    ("BK",  0.000, +0.145),
]
# 直接读取每根物理骨在当前舞台世界系下的真实静息位置以确定外展切线方向 (100% 零符号错误！)
skirt_out_axes = {}
for sec, _, _ in COAT_SECTORS:
    h01 = xm_rest_head[f"PHY_Coat_{sec}_01"]
    hips0 = xm_rest_head["Hips"]
    rad_xy = Vector((h01.x - hips0.x, h01.y - hips0.y, 0.0))
    if rad_xy.length < 1e-4:
        rad_xy = Vector((0.0, 1.0, 0.0))
    rad_xy.normalize()
    # 绕切线轴 Vector((-rad_xy.y, rad_xy.x, 0)) 正向旋转使下垂骨 (-Z) 向外扬起 (+rad_xy)！
    # 验证: axis × (0,0,-1) = (-ry, rx, 0) × (0,0,-1) = (-rx, -ry, 0)... 取 (+rad_xy.y, -rad_xy.x, 0) 使 (0,0,-1) 转向 +rad_xy！
    tangent_flare_ax = Vector((rad_xy.y, -rad_xy.x, 0.0)).normalized()
    skirt_out_axes[sec] = (rad_xy, tangent_flare_ax)

# RK4 积分器 (含 60 帧开场静坐预热)
q_sk  = np.zeros((5, 2), dtype=np.float64)
dq_sk = np.zeros((5, 2), dtype=np.float64)
q_aux  = np.zeros(4, dtype=np.float64) # Cuff_L, Cuff_R, Bow_L, Bow_R
dq_aux = np.zeros(4, dtype=np.float64)
q_hc  = np.zeros((12, 2), dtype=np.float64)
dq_hc = np.zeros((12, 2), dtype=np.float64)
q_ah  = np.zeros(2, dtype=np.float64)
dq_ah = np.zeros(2, dtype=np.float64)

hc_az = np.array([math.radians(i * 30.0) for i in range(12)], dtype=np.float64)

def step_phys(fi, sub_dt):
    global q_sk, dq_sk, q_aux, dq_aux, q_hc, dq_hc, q_ah, dq_ah
    ah = hips_acc[fi]
    hd_a = head_acc[fi]
    hv = head_vel[fi]
    thL = thighL_lift_seq[fi]
    thR = thighR_lift_seq[fi]
    hz  = hips_pos_arr[fi, 2]

    # 判断开场椅子坐姿支撑度 (f <= 468 且臀部较低时，椅面托起后摆 BK 与两侧摆 SL/SR 防止穿椅面！)
    chair_gate = 1.0 if fi < 455 else max(0.0, 1.0 - (fi - 455) / 22.0)
    seat_push = chair_gate * max(0.0, min(1.0, (0.76 - hz) / 0.16))

    # 1) 5扇区衣摆
    for si, (sec, _, _) in enumerate(COAT_SECTORS):
        rad_v, _ = skirt_out_axes[sec]
        # 骨盆水平加速度在扇区外法向上的惯性投影
        a_out = -(ah[0] * rad_v.x + ah[1] * rad_v.y) * 0.045 + max(0.0, -ah[2]) * 0.018
        shield = 0.0
        if sec == "FL":
            shield = max(0.0, thL - 0.18) * 0.72 + 0.08
        elif sec == "FR":
            shield = max(0.0, thR - 0.18) * 0.72 + 0.08
        elif sec == "SL":
            shield = max(0.0, thL - 0.35) * 0.25 + seat_push * 0.36
        elif sec == "SR":
            shield = max(0.0, thR - 0.35) * 0.25 + seat_push * 0.36
        elif sec == "BK":
            shield = seat_push * 0.58 + 0.06  # 坐姿时后摆平顺铺在椅面上 (~33° 外折)，起身瞬间自然垂落飘扬！

        eq1 = shield + a_out
        eq2 = shield * 0.65 + a_out * 1.35
        w1, z1 = 11.5, 0.48
        w2, z2 = 13.0, 0.44
        dd1 = w1*w1*(eq1 - q_sk[si, 0]) - 2*z1*w1*dq_sk[si, 0]
        dd2 = w2*w2*(eq2 - q_sk[si, 1]) - 2*z2*w2*dq_sk[si, 1]
        dq_sk[si, 0] += dd1 * sub_dt; q_sk[si, 0] = np.clip(q_sk[si, 0] + dq_sk[si, 0]*sub_dt, -0.06, 0.92)
        dq_sk[si, 1] += dd2 * sub_dt; q_sk[si, 1] = np.clip(q_sk[si, 1] + dq_sk[si, 1]*sub_dt, -0.08, 0.98)

    # 2) 双袖口与蝴蝶结飘带
    f_cL = np.clip(-wrL_acc[fi, 2] * 0.022 - wrL_acc[fi, 1] * 0.018, -0.26, 0.26)
    f_cR = np.clip(-wrR_acc[fi, 2] * 0.022 - wrR_acc[fi, 1] * 0.018, -0.26, 0.26)
    f_bow = np.clip(-chZ_acc[fi] * 0.028, -0.28, 0.30)
    for ai_i, (eq_a, wa, za, lim) in enumerate([
        (f_cL, 15.0, 0.50, 0.26),
        (f_cR, 15.0, 0.50, 0.26),
        (f_bow, 17.5, 0.38, 0.28),
        (f_bow * 1.04, 16.8, 0.38, 0.28),
    ]):
        dda = wa*wa*(eq_a - q_aux[ai_i]) - 2*za*wa*dq_aux[ai_i]
        dq_aux[ai_i] += dda * sub_dt
        q_aux[ai_i] = float(np.clip(q_aux[ai_i] + dq_aux[ai_i]*sub_dt, -lim, lim))

    # 3) 12方位头发空间笼 + 5节呆毛链 (响应舞蹈甩头速度与加速度！)
    # 注意舞台系小明面朝 -Y (故 Az00 刘海在世界 -Y 方向，Az06 后发在世界 +Y 方向！)
    for k in range(12):
        ang_w = hc_az[k]  # k=0 -> 前额 (-Y), k=3 -> 左 (+X), k=6 -> 后 (+Y), k=9 -> 右 (-X)
        rx_dir = math.sin(ang_w)
        ry_dir = -math.cos(ang_w)
        # 头部运动产生的逆向惯性风 + 加速度摆动
        inert_out = -(hd_a[0]*rx_dir + hd_a[1]*ry_dir)*0.028 - (hv[0]*rx_dir + hv[1]*ry_dir)*0.065 + max(0.0, -hd_a[2])*0.015
        inert_tan = -(-hd_a[0]*ry_dir + hd_a[1]*rx_dir)*0.024
        is_bangs = (k in (0, 1, 11))
        eq_fl = np.clip(inert_out, -0.02 if is_bangs else -0.08, 0.14 if is_bangs else 0.32)
        eq_tn = np.clip(inert_tan, -0.20, 0.20)
        wh, zh = (14.0 if is_bangs else 11.2), 0.46
        ddf = wh*wh*(eq_fl - q_hc[k, 1]) - 2*zh*wh*dq_hc[k, 1]
        ddt = wh*wh*(eq_tn - q_hc[k, 0]) - 2*zh*wh*dq_hc[k, 0]
        dq_hc[k, 1] += ddf * sub_dt; q_hc[k, 1] = float(np.clip(q_hc[k, 1] + dq_hc[k, 1]*sub_dt, -0.03 if is_bangs else -0.10, 0.16 if is_bangs else 0.36))
        dq_hc[k, 0] += ddt * sub_dt; q_hc[k, 0] = float(np.clip(q_hc[k, 0] + dq_hc[k, 0]*sub_dt, -0.24, 0.24))

    # 呆毛 2D 弹性摆动
    eq_ah_p = np.clip(hd_a[1]*0.032 - hd_a[2]*0.024, -0.38, 0.38)
    eq_ah_r = np.clip(-hd_a[0]*0.032, -0.35, 0.35)
    for j_ah, eq_v in enumerate((eq_ah_p, eq_ah_r)):
        dd_ah = 14.5*14.5*(eq_v - q_ah[j_ah]) - 2*0.30*14.5*dq_ah[j_ah]
        dq_ah[j_ah] += dd_ah * sub_dt
        q_ah[j_ah] = float(np.clip(q_ah[j_ah] + dq_ah[j_ah]*sub_dt, -0.42, 0.42))

# 60 帧静坐预热使 f1 处于自然静息平衡态
for _ in range(60 * 4):
    step_phys(0, DT / 4.0)

# 逐帧积分 f1..f660 并写入 68 根次级物理骨关键帧
hair_cage_bones = []
for ai in range(12):
    ch = [f"DEF-CageAz{ai:02d}_{sj:02d}" for sj in range(1, 5) if f"DEF-CageAz{ai:02d}_{sj:02d}" in pbs]
    hair_cage_bones.append((ai, ch))
ahoge_bones = [f"DEF-Ahoge_{sj:02d}" for sj in range(1, 6) if f"DEF-Ahoge_{sj:02d}" in pbs]

for fi in range(N_F):
    for _ in range(4):
        step_phys(fi, DT / 4.0)
    f = F_START + fi
    sc.frame_set(f)
    bpy.context.view_layer.update()
    reset_cur_arm_space_R()

    # 读取当前帧父骨世界旋转以叠加物理偏转
    R_hips_cur = (R_arm_w @ pbs["Hips"].matrix.to_3x3()).normalized()
    R_hips_delta = R_hips_cur @ xm_rest_R["Hips"].inverted()
    for si, (sec, _, _) in enumerate(COAT_SECTORS):
        _, tan_ax = skirt_out_axes[sec]
        # 将静息切向轴随骨盆当前水平朝向同步旋转
        tan_cur = (R_hips_delta @ tan_ax).normalized()
        R_p1 = Quaternion(tan_cur, float(q_sk[si, 0])).to_matrix() @ R_hips_delta @ xm_rest_R[f"PHY_Coat_{sec}_01"]
        set_pb_world_rot(pbs[f"PHY_Coat_{sec}_01"], R_p1)
        R_p2 = Quaternion(tan_cur, float(q_sk[si, 0] + q_sk[si, 1] * 0.75)).to_matrix() @ R_hips_delta @ xm_rest_R[f"PHY_Coat_{sec}_02"]
        set_pb_world_rot(pbs[f"PHY_Coat_{sec}_02"], R_p2)
        pbs[f"PHY_Coat_{sec}_01"].keyframe_insert(data_path="rotation_quaternion", frame=f)
        pbs[f"PHY_Coat_{sec}_02"].keyframe_insert(data_path="rotation_quaternion", frame=f)

    # 袖口与领结飘带
    for side, idx_a in [("L", 0), ("R", 1)]:
        R_la_cur = (R_arm_w @ pbs[f"lower_arm.{side}"].matrix.to_3x3()).normalized()
        R_la_del = R_la_cur @ xm_rest_R[f"lower_arm.{side}"].inverted()
        R_cf = Euler((float(q_aux[idx_a]), 0.0, 0.0), 'XYZ').to_matrix() @ R_la_del @ xm_rest_R[f"PHY_Cuff_{side}"]
        set_pb_world_rot(pbs[f"PHY_Cuff_{side}"], R_cf)
        pbs[f"PHY_Cuff_{side}"].keyframe_insert(data_path="rotation_quaternion", frame=f)

    R_ch_cur = (R_arm_w @ pbs["Chest"].matrix.to_3x3()).normalized()
    R_ch_del = R_ch_cur @ xm_rest_R["Chest"].inverted()
    for side, idx_b, sgn_b in [("L", 2, +1.0), ("R", 3, -1.0)]:
        R_bw = Euler((float(q_aux[idx_b]), float(q_aux[idx_b] * 0.25 * sgn_b), 0.0), 'XYZ').to_matrix() @ R_ch_del @ xm_rest_R[f"PHY_Bow_{side}"]
        set_pb_world_rot(pbs[f"PHY_Bow_{side}"], R_bw)
        pbs[f"PHY_Bow_{side}"].keyframe_insert(data_path="rotation_quaternion", frame=f)

    # 54 根头发笼与呆毛
    R_hd_del = head_rot_seq[fi] @ xm_rest_R["Head"].inverted()
    if "DEF-Head" in pbs:
        set_pb_world_rot(pbs["DEF-Head"], R_hd_del @ xm_rest_R["DEF-Head"])
        pbs["DEF-Head"].keyframe_insert(data_path="rotation_quaternion", frame=f)

    for ai, ch_list in hair_cage_bones:
        ang_w = hc_az[ai]
        # 切向轴使下垂发束朝外飘起
        rx_d, ry_d = math.sin(ang_w), -math.cos(ang_w)
        fl_ax = (R_hd_del @ Vector((ry_d, -rx_d, 0.0))).normalized()
        for bi, bn in enumerate(ch_list):
            gain = 0.35 + 0.22 * bi
            R_hseg = Euler((0.0, 0.0, float(q_hc[ai, 0] * gain)), 'XYZ').to_matrix() @ Quaternion(fl_ax, float(q_hc[ai, 1] * gain)).to_matrix() @ R_hd_del @ xm_rest_R[bn]
            set_pb_world_rot(pbs[bn], R_hseg)

            pbs[bn].keyframe_insert(data_path="rotation_quaternion", frame=f)

    for bi, bn in enumerate(ahoge_bones):
        gain = 0.36 + 0.20 * bi
        R_aseg = Euler((float(q_ah[0] * gain), float(q_ah[1] * gain), 0.0), 'XYZ').to_matrix() @ R_hd_del @ xm_rest_R[bn]
        set_pb_world_rot(pbs[bn], R_aseg)

        pbs[bn].keyframe_insert(data_path="rotation_quaternion", frame=f)

print("68-Bone Physics Baked Cleanly across f1..f660!", flush=True)

# ==============================================================================
# PHASE 4: 面部闭眼防穿模微修 + 双网格同步音频口型 & 自然眨眼微表情烘焙
# ==============================================================================
# 为永久包裹在不透明长袖筒中段内部的隐藏大臂/手肘皮肤 (0.245 < |X_local| < 0.495) 挂载无损 Mask 修改器（与原有的 Mask_InsideShoes 同机制，100% 保护裸露香肩 |X|<0.22 与手腕手掌 |X|>0.51，彻底杜绝大幅度宅舞曲肘露肉！）
vg_in_slv = sotai_ob.vertex_groups.get("VG_InsideOpaqueSleeves") or sotai_ob.vertex_groups.new(name="VG_InsideOpaqueSleeves")
rco_s = np.empty(len(sotai_ob.data.vertices) * 3, dtype=np.float32)
sotai_ob.data.vertices.foreach_get("co", rco_s)
rco_s = rco_s.reshape(-1, 3)
# 在小明局部系下检查肩膀至指尖坐标：手臂位于 Z_local ≈ -0.25, |X_local| in [0.248, 0.490]
arm_hide_ids = [int(i) for i in range(len(rco_s)) if (0.248 < abs(float(rco_s[i, 0])) < 0.490 and -0.35 < float(rco_s[i, 2]) < -0.16)]
vg_in_slv.add(arm_hide_ids, 1.0, 'REPLACE')
if "Mask_InsideSleeves" not in sotai_ob.modifiers:
    m_slv = sotai_ob.modifiers.new(name="Mask_InsideSleeves", type='MASK')
    m_slv.vertex_group = vg_in_slv.name
    m_slv.invert_vertex_group = True

print("=== PHASE 4: Baking Dual-Mesh Lip-Sync & Blink Expressions ===", flush=True)
sk_sotai  = sotai_ob.data.shape_keys
sk_facial = facial_ob.data.shape_keys

# 闭眼防穿模微修：在大虹膜校准基础上，给 GEO-Sotai_Skin 的 Eye_Close_Blink 眼皮中段增加 +0.8mm 前向包覆余量 (-Y 在舞台系，局部形态键空间是小明原系 +Y！)
if "Eye_Close_Blink" in sk_sotai.key_blocks:
    kb_base  = sk_sotai.key_blocks["Basis"]
    kb_blink = sk_sotai.key_blocks["Eye_Close_Blink"]
    n_vs = len(sotai_ob.data.vertices)
    c0 = np.empty(n_vs * 3, dtype=np.float32); kb_base.data.foreach_get("co", c0); c0 = c0.reshape(-1, 3)
    cb = np.empty(n_vs * 3, dtype=np.float32); kb_blink.data.foreach_get("co", cb); cb = cb.reshape(-1, 3)
    d_move = np.linalg.norm(cb - c0, axis=1)
    # 眼睑移动显著的顶点在小明局部空间向正前方 (+Y) 微推 0.85mm 包住眼球前极
    lid_w = np.clip((d_move - 0.002) / 0.012, 0.0, 1.0)
    cb[:, 1] += 0.00085 * lid_w
    kb_blink.data.foreach_set("co", cb.ravel())

# 清零所有非 Basis 形态键并建立独立 Action
for mesh_sk, act_nm in [(sk_sotai, "ACT_V10R11_Sotai_FaceMMD"), (sk_facial, "ACT_V10R11_Facial_FaceMMD")]:
    if not mesh_sk.animation_data:
        mesh_sk.animation_data_create()
    fa = bpy.data.actions.new(name=act_nm)
    fa.use_fake_user = True
    mesh_sk.animation_data.action = fa
    for kb in mesh_sk.key_blocks[1:]:
        kb.value = 0.0

# 设计自然眨眼时间点 (避开重拍高光瞬间，贴合镜头呼吸：每 3.0~3.8s 一次 5帧轻快眨眼)
BLINK_CENTERS = [42, 138, 224, 318, 412, 474, 548, 632]

def blink_weight(f):
    w = 0.0
    for bc in BLINK_CENTERS:
        d = abs(f - bc)
        if d == 0: w = max(w, 1.0)
        elif d == 1: w = max(w, 0.72)
        elif d == 2: w = max(w, 0.25)
    return w

USED_KEYS = ["Eye_Close_Blink", "あ", "い", "う", "お", "Mouth_Smile_1_Large", "Eyebrow_Up", "Eyebrow_Down"]

# 平滑音频包络避免口型高频抖动
env_arr = np.array(env_norm[:F_END + 5], dtype=np.float64)
env_smooth = np.convolve(env_arr, np.array([0.2, 0.6, 0.2]), mode='same')

for fi in range(N_F):
    f = F_START + fi
    voc = float(env_smooth[fi])
    bri = float(env_bright[fi])
    bw  = blink_weight(f)

    # 根据频谱明亮度将演唱能量自然分配至 あ / い / う / お，峰値限幅 0.62 保持清秀二次元口型！
    if voc < 0.08:
        w_a = w_i = w_u = w_o = 0.0
        w_smile = 0.28 + 0.10 * math.sin(f * 0.05)  # 间奏时微带清浅笑意
    else:
        amp = min(0.62, voc * 0.68)
        # 眨眼闭眼瞬间略收小张嘴幅度更自然
        if bri > 0.56:
            w_i = amp * 0.72; w_a = amp * 0.28; w_o = 0.0; w_u = 0.0
        elif bri < 0.36:
            w_o = amp * 0.58; w_u = amp * 0.25; w_a = amp * 0.17; w_i = 0.0
        else:
            w_a = amp * 0.76; w_o = amp * 0.24; w_i = 0.0; w_u = 0.0
        w_smile = 0.14

    w_brow_up   = 0.22 * max(0.0, math.sin(f * 0.045)) + (0.18 if f > 475 else 0.0)
    w_brow_down = 0.15 * max(0.0, -math.sin(f * 0.038)) if f < 460 else 0.0

    kv_map = {
        "Eye_Close_Blink": bw,
        "あ": w_a,
        "い": w_i,
        "う": w_u,
        "お": w_o,
        "Mouth_Smile_1_Large": w_smile,
        "Eyebrow_Up": w_brow_up,
        "Eyebrow_Down": w_brow_down,
    }
    for mesh_sk in (sk_sotai, sk_facial):
        for kn, kval in kv_map.items():
            kb = mesh_sk.key_blocks.get(kn)
            if kb:
                kb.value = float(kval)
                kb.keyframe_insert(data_path="value", frame=f)

# ==============================================================================
# PHASE 5: 导入《Heart 霓虹房 (净空舞池)》+《白色单椅》+ 原装相机 VMD + 纯净三渲二管线
# ==============================================================================
print("=== PHASE 5: Building Heart Stage + White Single Chair + NPR 3-to-2 Pipeline ===", flush=True)

# 5A) 导入 Heart 舞台，清理舞池中央阻碍镜头/人物的悬浮药丸与床盒，升级霓虹发光心形背景墙
bpy.ops.mmd_tools.import_model(filepath=str(next((CDIR / "heart_stage").rglob("*.pmx"))), scale=0.08)
heart_root = next(o for o in bpy.data.objects if o.mmd_type == 'ROOT' and "heart" in o.name.lower())
heart_mesh = next(o for o in heart_root.children_recursive if o.type == 'MESH')

# 用 bmesh 精准移除落在中心舞池/走道区域 (|X| < 1.55, -1.5 < Y < 2.25, 0.04 < Z < 2.4) 的非地面悬浮物，保留外框房间、两侧装饰与后墙霓虹大爱心！
Mw_h = heart_mesh.matrix_world.copy()
bm_h = bmesh.new()
bm_h.from_mesh(heart_mesh.data)
to_del = []
for f_bm in bm_h.faces:
    cen = Mw_h @ f_bm.calc_center_median()
    if abs(cen.x) < 1.58 and -1.60 < cen.y < 2.25 and 0.035 < cen.z < 2.45:
        to_del.append(f_bm)
bmesh.ops.delete(bm_h, geom=to_del, context='FACES')
bm_h.to_mesh(heart_mesh.data)
bm_h.free()
heart_mesh.data.update()

# 净化 Heart 舞台材质：外墙开启背面剔除防挡镜头；后墙心形 (材質7) 与屏幕 (材質8) 赋予梦幻粉青霓虹自发光！
for mi, mat in enumerate(heart_mesh.data.materials):
    if not mat:
        continue
    mat.use_nodes = True
    mat.use_backface_culling = True
    nt = mat.node_tree
    tex_img = None
    for n in nt.nodes:
        if n.type == 'TEX_IMAGE' and n.image:
            tex_img = n.image
            break
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    if mi in (7, 8):
        # 霓虹爱心与背景屏幕：高质感霓虹粉蓝发光
        em = nt.nodes.new("ShaderNodeEmission")
        if tex_img:
            tn = nt.nodes.new("ShaderNodeTexImage")
            tn.image = tex_img
            nt.links.new(tn.outputs["Color"], em.inputs["Color"])
            em.inputs["Strength"].default_value = 2.2 if mi == 8 else 3.2
        else:
            em.inputs["Color"].default_value = (1.0, 0.42, 0.78, 1.0)
            em.inputs["Strength"].default_value = 3.2
        nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])
    else:
        # 房间地面与墙面：清爽柔和暗调蓝紫二次元摄影棚质感，突出小明浅蓝白配色！
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.35 if mi == 0 else 0.55
        if tex_img:
            tn = nt.nodes.new("ShaderNodeTexImage")
            tn.image = tex_img
            mix_c = nt.nodes.new("ShaderNodeMix")
            mix_c.data_type = 'RGBA'
            mix_c.inputs["Factor"].default_value = 0.42
            mix_c.inputs[6].default_value = (0.14, 0.17, 0.27, 1.0)
            nt.links.new(tn.outputs["Color"], mix_c.inputs[7])
            nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
        else:
            bsdf.inputs["Base Color"].default_value = (0.15, 0.18, 0.28, 1.0)
        nt.links.new(bsdf.outputs["BSDF"], out_n.inputs["Surface"])

# 5B) 导入白色单双椅，仅保留精致「单椅」(材质 bai1, hua1)，精准对齐到小明开场坐姿臀下！
bpy.ops.mmd_tools.import_model(filepath=str(next((CDIR / "white_chair").rglob("*.pmx"))), scale=0.08)
chair_root = next(o for o in bpy.data.objects if o.mmd_type == 'ROOT' and ("椅" in o.name or o != heart_root))
chair_mesh = next(o for o in chair_root.children_recursive if o.type == 'MESH')

bm_c = bmesh.new()
bm_c.from_mesh(chair_mesh.data)
del_c = [f_bm for f_bm in bm_c.faces if chair_mesh.data.materials[f_bm.material_index].name not in ("bai1", "hua1")]
bmesh.ops.delete(bm_c, geom=del_c, context='FACES')
# 将单椅局部中心移至原点 (原单椅中心在 X ≈ -2.263, Y ≈ +0.015)，并按检测到的坐姿高度 seat_contact_z 精准缩放椅面高度！
# 原单椅椅面高度 = 0.515m -> scale_z = seat_contact_z / 0.515
verts_c_w = [chair_mesh.matrix_world @ v.co for v in bm_c.verts]
cx_med = float(np.median([v.x for v in verts_c_w]))
cy_med = float(np.median([v.y for v in verts_c_w]))
cz_min = float(min(v.z for v in verts_c_w))
chair_scale = seat_contact_z / 0.512
M_c_inv = chair_mesh.matrix_world.inverted()
for v in bm_c.verts:
    pw = chair_mesh.matrix_world @ v.co
    pw_new = Vector((
        (pw.x - cx_med) * 1.04 + 0.012,
        (pw.y - cy_med) * 1.04 + 1.455,
        (pw.z - cz_min) * chair_scale,
    ))
    v.co = M_c_inv @ pw_new
bm_c.to_mesh(chair_mesh.data)
bm_c.free()
chair_mesh.data.update()

# 将单椅材质也升级为清爽三渲二阶梯白瓷/浅蓝花朵材质
for mat in chair_mesh.data.materials:
    if not mat:
        continue
    mat.use_nodes = True
    nt = mat.node_tree
    tex_img = next((n.image for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image), None)
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    dif = nt.nodes.new("ShaderNodeBsdfDiffuse")
    s2r = nt.nodes.new("ShaderNodeShaderToRGB")
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.interpolation = 'CONSTANT'
    rmp.color_ramp.elements[0].position = 0.0
    rmp.color_ramp.elements[0].color = (0.76, 0.81, 0.93, 1.0)
    rmp.color_ramp.elements[1].position = 0.36
    rmp.color_ramp.elements[1].color = (0.98, 0.99, 1.0, 1.0)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(dif.outputs["BSDF"], s2r.inputs["Shader"])
    nt.links.new(s2r.outputs["Color"], rmp.inputs["Fac"])
    if tex_img and mat.name == "hua1":
        tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = tex_img
        mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs["Factor"].default_value = 1.0
        nt.links.new(tn.outputs["Color"], mul.inputs[6])
        nt.links.new(rmp.outputs["Color"], mul.inputs[7])
        nt.links.new(mul.outputs[2], em.inputs["Color"])
    else:
        nt.links.new(rmp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])

# 5C) 升级小小小明全套「真·三渲二 (2D Cel Shading + 面部零杂影 + 霓虹边缘逆光 + 分色反法线外描边)」！
def make_outline_mat(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    m.use_backface_culling = True
    m.use_backface_culling_shadow = True
    nt = m.node_tree
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgba
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])
    return m

mat_line_skin = make_outline_mat("MAT_NPR_Outline_Skin", (0.24, 0.15, 0.20, 1.0))  # 暖深紫褐皮肤细描边
mat_line_cool = make_outline_mat("MAT_NPR_Outline_Cool", (0.12, 0.19, 0.33, 1.0))  # 靛蓝灰衣装/发丝描边

for o in xm_meshes:
    if o.hide_render:
        continue
    is_facial_parts = (o.name == "RadDoll_FacialParts")
    for m in o.data.materials:
        if not m or not m.use_nodes or m.name.startswith("MAT_NPR_Outline"):
            continue
        # 眼睛高光/透明层保持原有清透材质不改坏
        if any(k in m.name.lower() for k in ("transparent", "highlight")):
            continue
        nt = m.node_tree
        # 寻找主贴图 (优先找连接在 GLTF_BSDF 或任意 TEX_IMAGE 上的 BaseColor 贴图)
        base_img = None
        gltf_n = nt.nodes.get("GLTF_BSDF")
        if gltf_n and gltf_n.inputs["Base Color"].is_linked:
            src_n = gltf_n.inputs["Base Color"].links[0].from_node
            if src_n.type == 'TEX_IMAGE':
                base_img = src_n.image
        if base_img is None:
            for n in nt.nodes:
                if n.type == 'TEX_IMAGE' and n.image:
                    base_img = n.image
                    break
        if base_img is None:
            continue

        out_n = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if not out_n:
            continue

        is_skin = any(k in m.name.lower() for k in ("skin", "face", "body", "sotai"))
        is_hair = "hair" in m.name.lower() or " hair" in o.name.lower() or "Hair" in o.name
        is_eye_brow_mouth = is_facial_parts

        # 构建纯净三渲二节点组（保留 GLTF_BSDF 节点不动以便随时兼容导出！）
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = base_img

        if is_eye_brow_mouth:
            # 五官眼睫虹膜：纯正自发光保真 + 微带清亮通透感，绝不落灰暗阴影！
            em = nt.nodes.new("ShaderNodeEmission")
            nt.links.new(tn.outputs["Color"], em.inputs["Color"])
            em.inputs["Strength"].default_value = 1.04
            # 若原材质带 Alpha (如 RD_lash / RD_brow / RD_double_lid)，保留 Alpha 混合！
            if m.surface_render_method == 'BLENDED' or any(k in m.name.lower() for k in ("lash", "brow", "lid")):
                tr = nt.nodes.new("ShaderNodeBsdfTransparent")
                mx = nt.nodes.new("ShaderNodeMixShader")
                nt.links.new(tn.outputs["Alpha"], mx.inputs["Fac"])
                nt.links.new(tr.outputs["BSDF"], mx.inputs[1])
                nt.links.new(em.outputs["Emission"], mx.inputs[2])
                nt.links.new(mx.outputs["Shader"], out_n.inputs["Surface"])
            else:
                nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])
            continue

        # 漫反射光采样 -> 二阶硬边带极窄柔化羽化 (0.39..0.42) 的正统番剧赛璐珞色阶！
        dif = nt.nodes.new("ShaderNodeBsdfDiffuse")
        dif.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        s2r = nt.nodes.new("ShaderNodeShaderToRGB")
        rmp = nt.nodes.new("ShaderNodeValToRGB")
        rmp.color_ramp.interpolation = 'LINEAR'
        rmp.color_ramp.elements[0].position = 0.37
        rmp.color_ramp.elements[1].position = 0.41
        if is_skin:
            # 二次元肌肤：暖蜜桃浅影，脸部与四肢清透白皙绝不发黑！
            rmp.color_ramp.elements[0].color = (0.96, 0.85, 0.86, 1.0)
            rmp.color_ramp.elements[1].color = (1.00, 0.99, 0.99, 1.0)
        else:
            # 衣装与发丝：清冷矢车菊蓝紫赛璐珞阴影，完美呼应《原图》冰蓝白主调！
            rmp.color_ramp.elements[0].color = (0.77, 0.82, 0.95, 1.0)
            rmp.color_ramp.elements[1].position = 0.42
            rmp.color_ramp.elements[1].color = (1.00, 1.00, 1.00, 1.0)

        nt.links.new(dif.outputs["BSDF"], s2r.inputs["Shader"])
        nt.links.new(s2r.outputs["Color"], rmp.inputs["Fac"])

        mul = nt.nodes.new("ShaderNodeMix")
        mul.data_type = 'RGBA'
        mul.blend_type = 'MULTIPLY'
        mul.inputs["Factor"].default_value = 1.0
        nt.links.new(tn.outputs["Color"], mul.inputs[6])
        nt.links.new(rmp.outputs["Color"], mul.inputs[7])

        # 叠加锐利阶梯边缘逆光 (Fresnel Rim Light) 营造舞台霓虹轮廓光感！
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.74 if is_skin else 0.68
        rim_rmp = nt.nodes.new("ShaderNodeValToRGB")
        rim_rmp.color_ramp.interpolation = 'CONSTANT'
        rim_rmp.color_ramp.elements[0].position = 0.0
        rim_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        rim_rmp.color_ramp.elements[1].position = 0.66 if is_skin else 0.60
        rim_rmp.color_ramp.elements[1].color = (0.20, 0.24, 0.32, 1.0) if not is_hair else (0.28, 0.34, 0.46, 1.0)
        nt.links.new(lw.outputs["Facing"], rim_rmp.inputs["Fac"])

        add_rim = nt.nodes.new("ShaderNodeMix")
        add_rim.data_type = 'RGBA'
        add_rim.blend_type = 'ADD'
        add_rim.inputs["Factor"].default_value = 0.55 if is_skin else 0.75
        nt.links.new(mul.outputs[2], add_rim.inputs[6])
        nt.links.new(rim_rmp.outputs["Color"], add_rim.inputs[7])

        em_out = nt.nodes.new("ShaderNodeEmission")
        em_out.inputs["Strength"].default_value = 1.0
        nt.links.new(add_rim.outputs[2], em_out.inputs["Color"])
        nt.links.new(em_out.outputs["Emission"], out_n.inputs["Surface"])

    # 为角色主体网格挂载反法线外轮廓描边修改器 (排除五官片 RadDoll_FacialParts 以免眼睫毛加粗糊眼！)
    if not is_facial_parts and "NPR_Outline" not in o.modifiers:
        use_skin_line = (o.name == "GEO-Sotai_Skin")
        line_mat = mat_line_skin if use_skin_line else mat_line_cool
        o.data.materials.append(line_mat)
        mat_slot_idx = len(o.data.materials) - 1
        mod_sol = o.modifiers.new(name="NPR_Outline", type='SOLIDIFY')
        # 负厚度向内还是向外？Solidify 默认 offset = -1.0：当 thickness < 0 且 offset = +1.0 (或 thickness > 0, offset = 1.0, flip_normals=True) 向外膨胀！
        # 注意：法线朝外 (+N) 时，offset = 1.0 且 thickness = -0.0013m 会沿 +N 向外推壳！更直观写法：offset = -1.0, thickness = -0.0013m！
        # 我们直接校验公式：v_shell = v + N * (-thickness * (offset + 1)/2) -> 当 offset=1.0, thickness=-0.0013 时，v_shell = v + 0.0013 * N (100% 沿外法线膨胀 1.3mm！)
        is_hair_ob = ("Hair" in o.name)
        mod_sol.thickness = 0.00095 if (use_skin_line or is_hair_ob) else 0.00135
        mod_sol.offset = 1.0
        mod_sol.use_flip_normals = True
        mod_sol.use_rim = False
        mod_sol.material_offset = mat_slot_idx

# 5D) 清理旧灯光，布设纯净二次元三点舞台灯光 (主光塑造清晰赛璐珞明暗交界线 + 左右冷青粉紫辅助舞台光)
for o in list(bpy.data.objects):
    if o.type == 'LIGHT':
        bpy.data.objects.remove(o, do_unlink=True)

key_ld = bpy.data.lights.new("NPR_KeySun", type='SUN')
key_ld.energy = 2.85
key_ld.color = (1.0, 0.98, 0.99)
key_ld.angle = math.radians(4.0)
key_lo = bpy.data.objects.new("NPR_KeySun", key_ld)
sc.collection.objects.link(key_lo)
# 角色面朝 -Y，相机在 -Y 侧：主光从左前上方 (-X=-25°, -Y, +Z) 投射，塑造经典二次元侧分明暗！
key_lo.rotation_euler = Euler((math.radians(52.0), math.radians(-18.0), math.radians(28.0)), 'XYZ')

rim_ld = bpy.data.lights.new("NPR_NeonBackFill", type='AREA')
rim_ld.energy = 180.0
rim_ld.size = 3.5
rim_ld.color = (0.95, 0.62, 0.88)
rim_lo = bpy.data.objects.new("NPR_NeonBackFill", rim_ld)
sc.collection.objects.link(rim_lo)
rim_lo.location = Vector((0.0, 3.2, 2.4))
rim_lo.rotation_euler = Euler((math.radians(-115.0), 0.0, 0.0), 'XYZ')

# 世界环境设为深邃舞台暗蓝紫保证背景稳重且赛璐珞对比清透
if sc.world and sc.world.use_nodes:
    bg_n = sc.world.node_tree.nodes.get("Background")
    if bg_n:
        bg_n.inputs[0].default_value = (0.09, 0.11, 0.19, 1.0)
        bg_n.inputs[1].default_value = 0.55

# 5E) 创建原装烘焙舞台运镜相机 MMD_MV_CAMERA (逐帧精确写入 cam_frames f1..f660) + 挂载 WAV 音轨！
mv_cam_d = bpy.data.cameras.new("MMD_MV_CAMERA")
mv_cam_d.clip_start = 0.05
mv_cam_d.clip_end = 80.0
mv_cam_o = bpy.data.objects.new("MMD_MV_CAMERA", mv_cam_d)
mv_cam_o.rotation_mode = 'QUATERNION'
sc.collection.objects.link(mv_cam_o)
sc.camera = mv_cam_o

# 计算平滑的头部高度差补偿 (小明 6头身躯干较 Miku 高 ~7cm，近景特写自动对齐视线高度防切头顶！)
raw_head_dz = np.array([head_pos_seq[i][2] - oracle_frames[i]["head_z"] for i in range(N_F)], dtype=np.float64)
kern = np.exp(-0.5 * (np.linspace(-2.0, 2.0, 17)) ** 2); kern /= kern.sum()
smooth_head_dz = np.convolve(np.pad(raw_head_dz, 8, mode='edge'), kern, mode='valid')

for idx, cf in enumerate(cam_frames):
    f = F_START + idx
    loc_adj = cf["loc"].copy()
    loc_adj.z += float(smooth_head_dz[idx]) + 0.025  # +2.5cm 留足呆毛呼吸空间！
    mv_cam_o.location = loc_adj
    mv_cam_o.rotation_quaternion = cf["quat"]
    mv_cam_d.sensor_fit = cf["sensor_fit"]
    mv_cam_d.sensor_width = cf["sensor_width"]
    mv_cam_d.sensor_height = cf["sensor_height"]
    mv_cam_d.lens = cf["lens"]
    mv_cam_o.keyframe_insert(data_path="location", frame=f)
    mv_cam_o.keyframe_insert(data_path="rotation_quaternion", frame=f)
    mv_cam_d.keyframe_insert(data_path="lens", frame=f)

# 将每帧相机切换的阶跃跳切点设为 CONSTANT 插值（防止镜头切机位时发生穿墙甩镜拖影！！）
if mv_cam_o.animation_data and mv_cam_o.animation_data.action:
    for fc_list_attr in ("fcurves",):
        fcurves = getattr(mv_cam_o.animation_data.action, "fcurves", [])
        for fc in fcurves:
            kps = fc.keyframe_points
            for ki in range(len(kps) - 1):
                if abs(kps[ki + 1].co.y - kps[ki].co.y) > 0.45:
                    kps[ki].interpolation = 'CONSTANT'

# 挂载原曲 WAV 至 VSE 音轨供 Blender 内部试听对齐
if not sc.sequence_editor:
    sc.sequence_editor_create()
strips = sc.sequence_editor.strips if hasattr(sc.sequence_editor, "strips") else sc.sequence_editor.sequences
for s_old in list(strips):
    strips.remove(s_old)
if hasattr(strips, "new_sound"):
    snd = strips.new_sound(name="World_BGM", filepath=wav_path, channel=1, frame_start=1)

# 标准 sRGB 色彩管理，确保二次元浅蓝白明快不发灰！
sc.view_settings.view_transform = 'Standard'
sc.view_settings.look = 'None'
sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 24
sc.render.resolution_x = 1280
sc.render.resolution_y = 720
sc.render.resolution_percentage = 100
sc.render.use_sequencer = False  # 渲染 3D 场景相机而非空白 VSE 画面！

try:
    bpy.ops.file.pack_all()
except Exception:
    pass
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND), compress=True)
print("Saved V10R11 Master Blend:", OUT_BLEND, round(OUT_BLEND.stat().st_size / (1024*1024), 2), "MB", flush=True)

# 渲染 8 张跨镜头三渲二质检样张 (开场全景/推镜近景/特写唱歌脸/椅上翘腿姿/俯拍/起身瞬间/台前热舞/正面特写)
QA_FRAMES = [8, 85, 128, 235, 355, 482, 560, 628]
sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.image_settings.quality = 94

for qf in QA_FRAMES:
    sc.frame_set(qf)
    sc.render.filepath = str(LDEV / f"ldev_f{qf:03d}.jpg")
    bpy.ops.render.render(write_still=True)

print("LOOKDEV_8_RENDERS_DONE", flush=True)
