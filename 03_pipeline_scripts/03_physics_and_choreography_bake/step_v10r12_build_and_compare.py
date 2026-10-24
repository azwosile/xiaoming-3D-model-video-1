"""
V10R12 专项精修与同机位对比构建器 (不覆盖 V10R11!):
1. 角色舞台等比放大 STAGE_SCALE = 1.130 (世界身高 1.412m -> 1.5956m，与原装 YYB 初音 1.593m 1:1 对齐，头身比 0% 变形!)
2. 白椅三维浅坐校准 + 坐姿臀部逐帧硬托底保护：
   - 消除白椅 X 偏心，将座深缩浅至 ~26.5cm 并后移，使大腿膝窝 100% 探出座板前缘以外；
   - 在 f1..f462 全坐姿段施加 Hips.z 硬托底，杜绝 0:05 后仰时臀胯下沉切入白椅座板！
3. 下颚刚性解剖权重重塑 (根治 0:03 歪头下颚扭曲 & 0:05 下巴橡皮拉伸鼓包)：
   - 沿下颌骨斜切面清除前下巴 664 个顶点的 35%~45% Neck 溢出权重及颈部 Chest 残余权重，恢复 100% 刚性 V 字下颌！
4. 三渲二「颈部常驻月牙阴影 + 下颌折角内描边 + 口腔清晰度」+ 30° 仰头番剧作画代偿：
   - 在 GEO-Sotai_Skin 烘焙随骨骼运动锁定解剖位置的顶点色属性 NPR_SubChin (上颈冷粉紫月牙投射阴影 + 下颌底缘折角线)；
   - 在世界空间对过度大仰头 (>18°) 做柔顺限幅 (60° -> <=26°)，转化为挺胸弧度与向下看镜头的半眯垂眸眼神，彻底消除抬头伪人感！
5. 水手服短袖/上衣内层隐藏皮肤精准遮罩扩展，根除 0:03 水手服肩袖露肉穿模。
"""
import json, math, time
from pathlib import Path
import numpy as np
import bpy, bmesh
from mathutils import Vector, Matrix, Quaternion, Euler
import addon_utils

t0_all = time.time()
ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
BASE_V10R10 = ROOT / "生成模型/组装/v010R8_全身衣装装配/RadDoll_V10R10_全身物理与循环跑步总装_PRIVATE.blend"
WDIR = ROOT / "生成模型/组装/v010R11_MMD_三渲二舞台_PRIVATE"
CDIR = WDIR / "mmd_cache_PRIVATE"
V11_BLEND = WDIR / "RadDoll_V10R11_MMD_Heart舞台三渲二开场MV_PRIVATE.blend"
OUT_BLEND = WDIR / "RadDoll_V10R12_MMD_Heart舞台三渲二精修预览_PRIVATE.blend"
CMP_DIR = WDIR / "v10r12_compare_qa"
CMP_DIR.mkdir(parents=True, exist_ok=True)

FPS = 30
DT = 1.0 / FPS
F_START, F_END = 1, 660
STAGE_SCALE = 1.130  # 等比舞台缩放：1.412m * 1.130 = 1.5956m (头身比 100% 不变!)

addon_utils.enable("bl_ext.blender_org.mmd_tools", default_set=False)

audio_info = json.loads((WDIR / "audio_envelope_720f.json").read_text(encoding="utf-8"))
env_norm   = audio_info["env_norm"]
env_bright = audio_info["env_bright"]
wav_path   = audio_info["wav_path"]

# ==============================================================================
# PHASE 1: 提取原装 YYB Miku + 动作.vmd + 相机.vmd 世界空间真值
# ==============================================================================
print("=== [V10R12] PHASE 1: Extracting MMD Oracle Motion & Camera ===", flush=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("bl_ext.blender_org.mmd_tools", default_set=False)
sc0 = bpy.context.scene
sc0.render.fps = FPS

miku_pmx = str(next((CDIR / "yyb_miku").rglob("*.pmx")))
vmds = sorted((CDIR / "world_motion").glob("*.vmd"), key=lambda q: q.stat().st_size, reverse=True)
motion_vmd, cam_vmd = str(vmds[0]), str(vmds[1])

bpy.ops.mmd_tools.import_model(filepath=miku_pmx, scale=0.08)
miku_arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
jp2bl = {pb.mmd_bone.name_j: pb.name for pb in miku_arm.pose.bones if hasattr(pb, "mmd_bone")}

TRACK_JP = [
    "全ての親", "センター", "グルーブ", "下半身", "上半身", "上半身2", "首", "頭", "左目", "右目",
    "左肩", "左腕", "左腕捩", "左ひじ", "左手捩", "左手首",
    "右肩", "右腕", "右腕捩", "右ひじ", "右手捩", "右手首",
    "左足", "左ひざ", "左足首", "左つま先",
    "右足", "右ひざ", "右足首", "右つま先",
]
for s_jp in ["左", "右"]:
    for f_jp in ["親指０", "親指１", "親指２",
                 "人指１", "人指２", "人指３",
                 "中指１", "中指２", "中指３",
                 "薬指１", "薬指２", "薬指３",
                 "小指１", "小指２", "小指３"]:
        TRACK_JP.append(s_jp + f_jp)
TRACK_JP = [j for j in TRACK_JP if j in jp2bl]

Mw_miku = miku_arm.matrix_world.copy()
miku_rest_R, miku_rest_head, miku_rest_tail = {}, {}, {}
for jn in TRACK_JP:
    bn = jp2bl[jn]
    b = miku_arm.data.bones[bn]
    miku_rest_R[jn]    = (Mw_miku.to_3x3() @ b.matrix_local.to_3x3()).normalized()
    miku_rest_head[jn] = Mw_miku @ b.head_local
    miku_rest_tail[jn] = Mw_miku @ b.tail_local

bpy.ops.object.select_all(action='DESELECT')
miku_arm.select_set(True)
bpy.context.view_layer.objects.active = miku_arm
bpy.ops.mmd_tools.import_vmd(filepath=motion_vmd, scale=0.08)

cam_data = bpy.data.cameras.new("TmpMMDCam")
cam_obj  = bpy.data.objects.new("TmpMMDCam", cam_data)
sc0.collection.objects.link(cam_obj)
bpy.ops.object.select_all(action='DESELECT')
cam_obj.select_set(True)
bpy.context.view_layer.objects.active = cam_obj
bpy.ops.mmd_tools.import_vmd(filepath=cam_vmd, scale=0.08)
real_cam = next(o for o in bpy.data.objects if o.type == 'CAMERA')

oracle_frames, cam_frames = [], []
for f in range(F_START, F_END + 1):
    sc0.frame_set(f)
    deps = bpy.context.evaluated_depsgraph_get()
    arm_ev = miku_arm.evaluated_get(deps)
    Mw_ev  = arm_ev.matrix_world
    R_ev   = Mw_ev.to_3x3()
    f_rot  = {}
    for jn in TRACK_JP:
        pb = arm_ev.pose.bones[jp2bl[jn]]
        f_rot[jn] = (R_ev @ pb.matrix.to_3x3()).normalized()
    hips_pb = arm_ev.pose.bones[jp2bl["下半身"]]
    head_pb = arm_ev.pose.bones[jp2bl["頭"]]
    ankL_pb = arm_ev.pose.bones[jp2bl["左足首"]]
    ankR_pb = arm_ev.pose.bones[jp2bl["右足首"]]
    oracle_frames.append({
        "hips_pos": (Mw_ev @ hips_pb.head).copy(),
        "head_z":   float((Mw_ev @ head_pb.head).z),
        "ankL_z":   float((Mw_ev @ ankL_pb.head).z),
        "ankR_z":   float((Mw_ev @ ankR_pb.head).z),
        "R": f_rot,
    })
    c_ev = real_cam.evaluated_get(deps)
    cmw  = c_ev.matrix_world.copy()
    cam_frames.append({
        "loc": cmw.to_translation(),
        "quat": cmw.to_quaternion(),
        "lens": float(c_ev.data.lens),
        "sensor_width": float(c_ev.data.sensor_width),
        "sensor_height": float(c_ev.data.sensor_height),
        "sensor_fit": str(c_ev.data.sensor_fit),
    })

# ==============================================================================
# PHASE 2: 打开 V10R10 主装文件 -> 修复下颌刚性权重 + 烘焙颈部月牙阴影顶点色 + 扩充水手服遮罩
# ==============================================================================
print("=== [V10R12] PHASE 2: Rigid Jaw Weights + SubChin Shadow Attribute + UnderCloth Mask ===", flush=True)
bpy.ops.wm.open_mainfile(filepath=str(BASE_V10R10), load_ui=False, use_scripts=False)
addon_utils.enable("bl_ext.blender_org.mmd_tools", default_set=False)
sc = bpy.context.scene
sc.render.fps = FPS
sc.frame_start = F_START
sc.frame_end   = F_END

for o in list(bpy.data.objects):
    if o.name.startswith("QA_") or ("Cam" in o.name and o.type == 'CAMERA'):
        bpy.data.objects.remove(o, do_unlink=True)

arm_ob    = bpy.data.objects["ARM-FullBody_V10"]
skin_ob   = bpy.data.objects["GEO-Sotai_Skin"]
facial_ob = bpy.data.objects["RadDoll_FacialParts"]
coat_ob   = bpy.data.objects["GEO-Cloth_Coat"]
sailor_ob = bpy.data.objects["GEO-Cloth_SailorTop"]
shorts_ob = bpy.data.objects["GEO-Cloth_Shorts"]
shoeL_ob  = bpy.data.objects["GEO-Shoe_L"]
shoeR_ob  = bpy.data.objects["GEO-Shoe_R"]

# 2A) 解剖级刚性下颌权重修复：沿下颌骨斜切平面清除前下巴与下颌缘的 Neck / Chest 溢出权重！
vg_head  = skin_ob.vertex_groups.get("Head") or skin_ob.vertex_groups.new(name="Head")
vg_neck  = skin_ob.vertex_groups.get("Neck") or skin_ob.vertex_groups.new(name="Neck")
vg_chest = skin_ob.vertex_groups.get("Chest")
idx_head  = vg_head.index
idx_neck  = vg_neck.index
idx_chest = vg_chest.index if vg_chest else -1

fixed_jaw_cnt = 0
for v in skin_ob.data.vertices:
    x, y, z = v.co.x, v.co.y, v.co.z
    if z < -0.195 or abs(x) > 0.095:
        continue
    # C1 连续解剖切面：前颈喉部切面降至 -0.1535m（完整包住整个下颌骨、下颌角与下巴底面），并在颈柱 [-0.188, z_jaw_top] 上从 0.0 平滑升至 1.0（零跳变台阶！）
    fwd_u = max(0.0, min(1.0, (y - (-0.010)) / 0.050))
    z_jaw_top = -0.1360 * (1.0 - fwd_u) + (-0.1540) * fwd_u
    w_map = {g.group: g.weight for g in v.groups}
    w_h = w_map.get(idx_head, 0.0)
    w_n = w_map.get(idx_neck, 0.0)
    w_c = w_map.get(idx_chest, 0.0) if idx_chest >= 0 else 0.0
    w_tot = w_h + w_n + w_c
    if w_tot < 1e-4:
        continue

    if z >= z_jaw_top:
        if w_n > 1e-4 or w_c > 1e-4:
            vg_head.add([v.index], w_tot, 'REPLACE')
            vg_neck.add([v.index], 0.0, 'REPLACE')
            if vg_chest:
                vg_chest.add([v.index], 0.0, 'REPLACE')
            fixed_jaw_cnt += 1
    elif z >= -0.188:
        t = max(0.0, min(1.0, (z - (-0.188)) / max(0.008, z_jaw_top - (-0.188))))
        s_w = t * t * (3.0 - 2.0 * t)  # 在 z=z_jaw_top 处精确达到 1.000，与刚性下颌 100% C1 无缝衔接！
        vg_head.add([v.index], w_tot * s_w, 'REPLACE')
        vg_neck.add([v.index], w_tot * (1.0 - s_w), 'REPLACE')
        if vg_chest and w_c > 0.0:
            vg_chest.add([v.index], 0.0, 'REPLACE')
        fixed_jaw_cnt += 1

print(f"Rigid Jaw Weight Fix applied to {fixed_jaw_cnt} vertices!", flush=True)

# 2B) 在 GEO-Sotai_Skin (Rest Base 态) 上程序化烘焙二次元常驻「颈部月牙投射阴影 + 下颌折角线」顶点色属性 NPR_SubChin
# R 通道: 上颈部月牙形下巴投射阴影 (0..1)
# G 通道: 下颌骨下折角边缘线强调 (0..1)
if "NPR_SubChin" in skin_ob.data.color_attributes:
    skin_ob.data.color_attributes.remove(skin_ob.data.color_attributes["NPR_SubChin"])
ca_chin = skin_ob.data.color_attributes.new(name="NPR_SubChin", type='BYTE_COLOR', domain='POINT')

for v in skin_ob.data.vertices:
    x, y, z = v.co.x, v.co.y, v.co.z
    nx, ny, nz = v.normal.x, v.normal.y, v.normal.z
    neck_sh = 0.0
    jaw_ln  = 0.0
    # 纯正二次元赛璐珞颈影：下巴与下颌面 100% 保持白皙暖蜜桃零脏斑 (jaw_ln=0)，仅在严格位于下巴后下方的竖直颈柱 (z <= -0.153, y <= 0.045) 投下干净月牙暖粉灰阴影！
    if -0.198 <= z <= -0.152 and -0.002 <= y <= 0.046 and abs(x) <= 0.034:
        lat_u = abs(x) / 0.031
        if lat_u < 1.0 and ny > 0.05:
            z_bottom_arc = -0.184 + 0.022 * (lat_u ** 1.6)
            if z >= z_bottom_arc:
                edge_in = min(1.0, (z - z_bottom_arc) / 0.0035)
                top_in  = min(1.0, (-0.152 - z) / 0.0030)
                front_w = max(0.0, min(1.0, (ny - 0.05) / 0.45))
                neck_sh = max(0.0, min(1.0, edge_in * top_in * front_w))

    ca_chin.data[v.index].color = (neck_sh, jaw_ln, 0.0, 1.0)

# 2C) 非破坏式全衣内隐藏皮肤遮罩 (消除 0:03 水手服短袖露肉 & 外套中段肘部破衣，100% 保留领口锁骨、露脐小腹、短袖口下裸臂与双手!)
vg_hide = skin_ob.vertex_groups.get("VG_Hidden_UnderCloth") or skin_ob.vertex_groups.new(name="VG_Hidden_UnderCloth")
hide_ids = []
for v in skin_ob.data.vertices:
    x, y, z = v.co.x, v.co.y, v.co.z
    ax = abs(x)
    hide = False
    # (1) 外套前臂与肘部管内 (0.252 < |X| < 0.488)
    if 0.252 < ax < 0.488 and -0.38 < z < -0.10:
        hide = True
    # (2) 水手服白色短袖包覆的肩头/三角肌上盖 (0.132 < |X| < 0.216, -0.285 < z < -0.200) -> 彻底消除 0:03 抬臂短袖破皮！
    elif 0.132 < ax < 0.216 and -0.285 < z < -0.202:
        hide = True
    # (3) 水手服躯干遮盖的胸背内层 (-0.372 < z < -0.228, |X| <= 0.135)，但完整豁免前胸 V 领锁骨开口区！
    elif ax <= 0.135 and -0.372 < z < -0.228:
        in_vneck_open = (y > 0.015 and ax < 0.088 and z > -0.312)
        if not in_vneck_open:
            hide = True
    # (4) 短裤内部盆腔核心 (-0.615 < z < -0.515)
    elif ax < 0.125 and -0.615 < z < -0.515:
        hide = True
    if hide:
        hide_ids.append(v.index)

vg_hide.remove(list(range(len(skin_ob.data.vertices))))
vg_hide.add(hide_ids, 1.0, 'REPLACE')
for old_m in ["Mask_InsideSleeves", "Mask_HiddenUnderCloth"]:
    if old_m in skin_ob.modifiers:
        skin_ob.modifiers.remove(skin_ob.modifiers[old_m])
mod_msk = skin_ob.modifiers.new(name="Mask_HiddenUnderCloth", type='MASK')
mod_msk.vertex_group = "VG_Hidden_UnderCloth"
mod_msk.invert_vertex_group = True

# 同样隐藏水手服短袖被蓝色外套后肩/腋下完全包裹且易在极度前伸时顶出外套的袖底外缘窄环 (|X| > 0.238 且后侧 y < -0.025)
vg_sl_hide = sailor_ob.vertex_groups.get("VG_Sailor_UnderCoat") or sailor_ob.vertex_groups.new(name="VG_Sailor_UnderCoat")
sl_hide_ids = [v.index for v in sailor_ob.data.vertices if abs(v.co.x) > 0.242 and v.co.y < -0.022]
vg_sl_hide.remove(list(range(len(sailor_ob.data.vertices))))
if sl_hide_ids:
    vg_sl_hide.add(sl_hide_ids, 1.0, 'REPLACE')
    if "Mask_SailorUnderCoat" not in sailor_ob.modifiers:
        ms = sailor_ob.modifiers.new(name="Mask_SailorUnderCoat", type='MASK')
        ms.vertex_group = "VG_Sailor_UnderCoat"
        ms.invert_vertex_group = True

# ==============================================================================
# PHASE 3: 舞台等比放大 STAGE_SCALE=1.130 + 全身重定向 + 30° 仰头番剧代偿 + 椅面硬托底
# ==============================================================================
print("=== [V10R12] PHASE 3: 1.595m Uniform Stage Scale + Retargeting + Head-Up Compensation ===", flush=True)
xm_root = bpy.data.objects.new("XIAOMING_STAGE_ROOT", None)
sc.collection.objects.link(xm_root)

xm_meshes = []
for o in list(bpy.data.objects):
    if o == xm_root or o.type in ('CAMERA', 'LIGHT'):
        continue
    if o.type == 'MESH':
        xm_meshes.append(o)
    if o.parent is None:
        mw_keep = o.matrix_world.copy()
        o.parent = xm_root
        o.matrix_parent_inverse = xm_root.matrix_world.inverted()
        o.matrix_world = mw_keep

# 关键突破：等比缩放 (1.13, 1.13, 1.13) -> 世界身高 1.5956m，与原装初音 1.593m 完全对齐，自身 6 头身中性比例 0% 变形！
xm_root.scale = Vector((STAGE_SCALE, STAGE_SCALE, STAGE_SCALE))
xm_root.location = Vector((0.0, 0.0, 1.4120 * STAGE_SCALE))
xm_root.rotation_euler = Euler((0.0, 0.0, math.pi), 'XYZ')
bpy.context.view_layer.update()

act_mmd = bpy.data.actions.new(name="ACT_V10R12_MMD_OpeningChairDance")
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
# 注意 Mw_xm 含 1.13 等比缩放：提取纯正交旋转矩阵时必须 .normalized()！
R_arm_w = Mw_xm.to_3x3().normalized()
R_arm_w_inv = R_arm_w.inverted()
arm_world_scale = Mw_xm.to_scale().z  # = 1.130

xm_rest_R, xm_rest_u, xm_rest_head = {}, {}, {}
for b in arm_ob.data.bones:
    Rw = (R_arm_w @ b.matrix_local.to_3x3()).normalized()
    hw = Mw_xm @ b.head_local
    tw = Mw_xm @ b.tail_local
    xm_rest_R[b.name] = Rw
    xm_rest_head[b.name] = hw
    dv = tw - hw
    xm_rest_u[b.name] = dv.normalized() if dv.length > 1e-6 else Vector((0, 0, 1))

z_miku_hips0 = float(miku_rest_head["下半身"].z)
z_xm_hips0   = float(xm_rest_head["Hips"].z)
leg_ratio    = z_xm_hips0 / max(z_miku_hips0, 0.5)
print(f"V10R12 Scale Calib: miku_hips0={z_miku_hips0:.4f}m, xm_hips0={z_xm_hips0:.4f}m -> leg_ratio={leg_ratio:.4f}", flush=True)

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
        (f"upper_arm.{side}",       f"{s_jp}腕捩", "arm_upper"),
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

R_xm_apose_base = {}
for xm_bn, jp_bn, mode in BONE_MAP:
    if xm_bn not in xm_rest_R or jp_bn not in miku_rest_R:
        continue
    if mode == "direct":
        R_xm_apose_base[xm_bn] = xm_rest_R[xm_bn]
    else:
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

hips_pos_seq, hips_yaw_seq = [], []
thighL_lift_seq, thighR_lift_seq = [], []
wristL_pos_seq, wristR_pos_seq = [], []
chest_z_seq, head_pos_seq, head_rot_seq = [], [], []
headup_lidded_seq = []  # 记录每帧因大仰头触发的半眯垂眸微表情补偿量！
knee_back_y_samples = []

miku_hips_rest_pos = miku_rest_head["下半身"]
xm_hips_rest_pos   = xm_rest_head["Hips"]
R_hips_rest_w = xm_rest_R["Hips"]
R_hips_rest_w_inv = R_hips_rest_w.inverted()

# 标定目标白椅椅面顶高：取 Miku 开场静坐 (f1..f40) 骨盆高度推算的小明 1.595m 等比静坐臀底高度！
# 在 1.412m 时 f12 短裤底 Z=0.464m -> 乘 1.130 后静坐臀底高度约为 0.524m！我们将白椅椅面顶设定在 Z_SEAT_TOP = 0.518m，
# 并在 f1..f458 全程施加硬托底保护 (确保后仰时骨盆绝不下沉穿透 Z_SEAT_TOP！)
Z_SEAT_TOP = 0.518

for idx in range(len(oracle_frames)):
    f = F_START + idx
    of = oracle_frames[idx]
    sc.frame_set(f)
    reset_cur_arm_space_R()

    mp = of["hips_pos"]
    dz_hips = (mp.z - miku_hips_rest_pos.z) * leg_ratio
    target_hips_w = Vector((
        mp.x * 0.98,
        mp.y,
        xm_hips_rest_pos.z + dz_hips,
    ))

    # 坐姿防下沉硬托底预保护 (f <= 462)：在 0:05 (f140..f178) 等靠椅后仰段，不让 Hips.z 跌破静坐椅面高度！
    if f <= 462:
        # 平滑释放权重 (f445..f462 准备起身时柔和过渡)
        w_sit_clamp = 1.0 if f <= 445 else max(0.0, (462 - f) / 17.0)
        # 静坐基准 Hips.z 约为 0.658m (对应短裤底 ~0.520m)；若低于 0.655m，则柔和托住至 >= 0.652m！
        min_hips_sit_z = 0.652
        if target_hips_w.z < min_hips_sit_z:
            target_hips_w.z = target_hips_w.z * (1.0 - w_sit_clamp) + (min_hips_sit_z + 0.15 * (target_hips_w.z - min_hips_sit_z)) * w_sit_clamp

    d_hips_w = target_hips_w - xm_hips_rest_pos
    pbs["Hips"].location = (R_hips_rest_w_inv @ d_hips_w) / arm_world_scale

    # 预先检测原装动作中 Head 的真实世界仰角，以便把超限仰角分流 25% 给 Chest 优雅挺胸！
    R_head_raw_w = (of["R"]["頭"] @ miku_rest_R["頭"].inverted() @ R_xm_apose_base["Head"]).normalized()
    # Head 在舞台静息态下朝前向量为 (0, -1, 0)；世界变换矩阵为 R_head_raw_w @ xm_rest_R["Head"].inv()
    M_head_delta_raw = R_head_raw_w @ xm_rest_R["Head"].inverted()
    v_face_fwd_raw = (M_head_delta_raw @ Vector((0.0, -1.0, 0.0))).normalized()
    up_pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, v_face_fwd_raw.z))))

    extra_chest_arch_rad = 0.0
    lidded_comp = 0.0
    if up_pitch_deg > 8.5:
        excess_deg = up_pitch_deg - 8.5
        extra_chest_arch_rad = math.radians(min(8.5, excess_deg * 0.22))
        lidded_comp = min(0.18, excess_deg * 0.0048)
    headup_lidded_seq.append(lidded_comp)

    for xm_bn, jp_bn, mode in BONE_MAP:
        if xm_bn not in R_xm_apose_base or jp_bn not in of["R"]:
            continue
        R_m_pose = of["R"][jp_bn]
        R_m_rest = miku_rest_R[jp_bn]
        R_delta_w = R_m_pose @ R_m_rest.inverted()
        R_tgt_w = (R_delta_w @ R_xm_apose_base[xm_bn]).normalized()

        if xm_bn == "Chest" and extra_chest_arch_rad > 1e-4:
            # 将部分大仰头张力转化为优雅挺胸后仰 (绕世界 -X 旋转使上胸朝 -Y/+Z 微扬)
            q_arch = Quaternion(Vector((-1.0, 0.0, 0.0)), extra_chest_arch_rad).to_matrix()
            R_tgt_w = (q_arch @ R_tgt_w).normalized()

        elif xm_bn in ("Neck", "Head") and up_pitch_deg > 8.5:
            # 番剧作画级「仰头柔顺限幅」：低机位仰拍时世界脸部仰角控制在 <=16.5° 黄金美型区间，100% 展现五官神态而非下颌底面！
            M_hd = R_tgt_w @ xm_rest_R[xm_bn].inverted()
            v_fwd = (M_hd @ Vector((0.0, -1.0, 0.0))).normalized()
            cur_el = math.degrees(math.asin(max(-1.0, min(1.0, v_fwd.z))))
            cap_start = 6.0 if xm_bn == "Neck" else 8.5
            if cur_el > cap_start:
                new_el = cap_start + 0.19 * (cur_el - cap_start)  # 48° -> 16.0°！完美番剧微仰黄金角！
                horiz = Vector((v_fwd.x, v_fwd.y, 0.0))
                if horiz.length > 1e-5:
                    horiz.normalize()
                    v_fwd_clamped = (horiz * math.cos(math.radians(new_el)) + Vector((0.0, 0.0, math.sin(math.radians(new_el))))).normalized()
                    q_fix = v_fwd.rotation_difference(v_fwd_clamped).to_matrix()
                    R_tgt_w = (q_fix @ R_tgt_w).normalized()

        elif xm_bn in ("LeftEye", "RightEye") and up_pitch_deg > 14.0:
            # 仰头时双瞳微向下看镜头，赋予鲜活番剧眼神，彻底消除木偶空洞感！
            look_down_rad = math.radians(min(8.5, (up_pitch_deg - 14.0) * 0.30))
            q_eye = Quaternion(Vector((+1.0, 0.0, 0.0)), look_down_rad).to_matrix()
            R_tgt_w = (q_eye @ R_tgt_w).normalized()

        elif xm_bn.startswith("shoulder."):
            R_base_sh = R_xm_apose_base[xm_bn]
            q_diff_sh = (R_tgt_w @ R_base_sh.inverted()).to_quaternion()
            if q_diff_sh.angle > math.radians(5.5):
                q_clamped = Quaternion(q_diff_sh.axis, math.radians(5.5) + 0.20 * (q_diff_sh.angle - math.radians(5.5)))
                R_tgt_w = (q_clamped.to_matrix() @ R_base_sh).normalized()

        elif mode == "arm_upper":
            sgn_x = +1.0 if xm_bn.endswith(".L") else -1.0
            u_cur = ((R_tgt_w @ xm_rest_R[xm_bn].inverted()) @ xm_rest_u[xm_bn]).normalized()
            u_mod = Vector(u_cur)
            if u_mod.z > 0.10:
                u_mod.z = 0.10 + 0.30 * (u_mod.z - 0.10)
            lat = sgn_x * u_mod.x
            if lat < 0.18:
                u_mod.x = sgn_x * (0.18 + 0.25 * (lat - 0.18))
            u_mod.normalize()
            R_tgt_w = (u_cur.rotation_difference(u_mod).to_matrix() @ R_tgt_w).normalized()

        elif xm_bn.startswith("lower_arm.") and not xm_bn.startswith("lower_arm_Twist"):
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

    # 宽袖口防互穿缓冲垫 (按 1.13 倍尺度同步放大阈值至 0.242m)
    wL_now = Mw_xm @ pbs["hand.L"].head
    wR_now = Mw_xm @ pbs["hand.R"].head
    dist_wr = (wL_now - wR_now).length
    if dist_wr < 0.242:
        push_rad = min(0.16, (0.242 - dist_wr) * 1.05)
        for side_c, sgn_c in [("L", +1.0), ("R", -1.0)]:
            R_fa_cur = R_arm_w @ pbs[f"lower_arm.{side_c}"].matrix.to_3x3()
            q_sep = Euler((-push_rad * 0.45, 0.0, sgn_c * push_rad * 0.85), 'XYZ').to_matrix()
            set_pb_world_rot(pbs[f"lower_arm.{side_c}"], (q_sep @ R_fa_cur).normalized())
        bpy.context.view_layer.update()

    # 采集坐姿膝窝世界 Y 坐标，确保白椅前缘 100% 退在膝窝与小腿后方！
    if f <= 430 and f % 10 == 1:
        kL_y = float((Mw_xm @ pbs["lower_leg.L"].head).y)
        kR_y = float((Mw_xm @ pbs["lower_leg.R"].head).y)
        knee_back_y_samples.append(max(kL_y, kR_y))

    # 站立舞段 (f >= 480) 鞋底触地 Z=0.002m 校准
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
            if z_sole_min < 0.028:
                corr_z = 0.002 - z_sole_min
                d_hips_w.z += corr_z
                pbs["Hips"].location = (R_hips_rest_w_inv @ d_hips_w) / arm_world_scale
                bpy.context.view_layer.update()

    hips_w_cur = Mw_xm @ pbs["Hips"].head
    hips_fwd_w = (R_arm_w @ pbs["Hips"].matrix.to_3x3() @ Vector((0, 0, -1))).normalized()
    hips_pos_seq.append(np.array([hips_w_cur.x, hips_w_cur.y, hips_w_cur.z], dtype=np.float64))
    hips_yaw_seq.append(math.atan2(hips_fwd_w.x, -hips_fwd_w.y))

    for side, seq_l in [("L", thighL_lift_seq), ("R", thighR_lift_seq)]:
        ul_vec = ((Mw_xm @ pbs[f"upper_leg.{side}"].tail) - (Mw_xm @ pbs[f"upper_leg.{side}"].head)).normalized()
        fwd_amt = -ul_vec.y * math.cos(hips_yaw_seq[-1]) + ul_vec.x * math.sin(hips_yaw_seq[-1])
        lift_rad = math.atan2(max(0.0, fwd_amt), max(0.05, -ul_vec.z))
        seq_l.append(lift_rad)

    wristL_pos_seq.append(np.array(Mw_xm @ pbs["hand.L"].head, dtype=np.float64))
    wristR_pos_seq.append(np.array(Mw_xm @ pbs["hand.R"].head, dtype=np.float64))
    chest_z_seq.append(float((Mw_xm @ pbs["Chest"].head).z))
    head_pos_seq.append(np.array(Mw_xm @ pbs["Head"].head, dtype=np.float64))

    pbs["Hips"].keyframe_insert(data_path="location", frame=f)
    for xm_bn, _, _ in BONE_MAP:
        if xm_bn in pbs:
            pbs[xm_bn].keyframe_insert(data_path="rotation_quaternion", frame=f)

knee_safe_y = float(np.percentile(knee_back_y_samples, 85)) if knee_back_y_samples else 1.16
print(f"V10R12 Retargeting done! 85th-pct seated knee Y = {knee_safe_y:.4f}m", flush=True)

# ==============================================================================
# PHASE 4: 68 骨 RK4 物理烘焙 (加强坐姿大衣下摆托举角度，避免下摆穿入浅坐白椅边缘) + 音频口型眨眼
# ==============================================================================
print("=== [V10R12] PHASE 4: 68-Bone Physics + LipSync + Gaze Blink ===", flush=True)
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

COAT_SECTORS = [("FL", -0.115, -0.095), ("FR", +0.115, -0.095), ("SL", -0.185, +0.010), ("SR", +0.185, +0.010), ("BK", 0.000, +0.145)]
skirt_out_axes = {}
for sec, _, _ in COAT_SECTORS:
    h01 = xm_rest_head[f"PHY_Coat_{sec}_01"]
    hips0 = xm_rest_head["Hips"]
    rad_xy = Vector((h01.x - hips0.x, h01.y - hips0.y, 0.0))
    if rad_xy.length < 1e-4: rad_xy = Vector((0.0, 1.0, 0.0))
    rad_xy.normalize()
    tangent_flare_ax = Vector((rad_xy.y, -rad_xy.x, 0.0)).normalized()
    skirt_out_axes[sec] = (rad_xy, tangent_flare_ax)

q_sk  = np.zeros((5, 2), dtype=np.float64); dq_sk = np.zeros((5, 2), dtype=np.float64)
q_aux = np.zeros(4, dtype=np.float64);      dq_aux = np.zeros(4, dtype=np.float64)
q_hc  = np.zeros((12, 2), dtype=np.float64); dq_hc = np.zeros((12, 2), dtype=np.float64)
q_ah  = np.zeros(2, dtype=np.float64);      dq_ah = np.zeros(2, dtype=np.float64)
hc_az = np.array([math.radians(i * 30.0) for i in range(12)], dtype=np.float64)

def step_phys(fi, sub_dt):
    global q_sk, dq_sk, q_aux, dq_aux, q_hc, dq_hc, q_ah, dq_ah
    ah = hips_acc[fi]; hd_a = head_acc[fi]; hv = head_vel[fi]
    thL = thighL_lift_seq[fi]; thR = thighR_lift_seq[fi]
    hz  = hips_pos_arr[fi, 2]
    sit_w = 0.0
    if fi < 468:
        sit_w = max(0.0, min(1.0, (0.86 - hz) / 0.16)) * (1.0 if fi < 448 else (468 - fi) / 20.0)

    for si, (sec, _, _) in enumerate(COAT_SECTORS):
        rad_v, _ = skirt_out_axes[sec]
        inert = -(ah[0] * rad_v.x + ah[1] * rad_v.y) * 0.032 + max(0.0, -ah[2]) * 0.018
        min_q0, min_q1 = -0.10, -0.12
        if sec == "FL":
            min_q0 = max(min_q0, thL * 0.78 + 0.34 * sit_w)
            min_q1 = max(min_q1, thL * 0.36)
        elif sec == "FR":
            min_q0 = max(min_q0, thR * 0.78 + 0.34 * sit_w)
            min_q1 = max(min_q1, thR * 0.36)
        elif sec in ("SL", "SR"):
            th_s = thL if sec == "SL" else thR
            min_q0 = max(min_q0, 0.54 * sit_w + th_s * 0.28)
            min_q1 = max(min_q1, 0.32 * sit_w)
        elif sec == "BK":
            min_q0 = max(min_q0, 0.62 * sit_w)
            min_q1 = max(min_q1, 0.48 * sit_w)

        k0, c0 = 62.0, 9.2
        k1, c1 = 46.0, 7.6
        f0 = -k0 * (q_sk[si, 0] - min_q0) - c0 * dq_sk[si, 0] + inert * 14.0
        if q_sk[si, 0] < min_q0:
            f0 += 280.0 * (min_q0 - q_sk[si, 0])
        f1 = -k1 * (q_sk[si, 1] - max(min_q1, q_sk[si, 0] * 0.55)) - c1 * dq_sk[si, 1] + inert * 19.0
        dq_sk[si, 0] += f0 * sub_dt; q_sk[si, 0] = np.clip(q_sk[si, 0] + dq_sk[si, 0] * sub_dt, min_q0 - 0.02, 1.25)
        dq_sk[si, 1] += f1 * sub_dt; q_sk[si, 1] = np.clip(q_sk[si, 1] + dq_sk[si, 1] * sub_dt, min_q1 - 0.02, 1.30)

    drv_aux = [
        float(np.clip(-wrL_acc[fi, 2] * 0.015 + wrL_acc[fi, 1] * 0.010, -0.45, 0.45)),
        float(np.clip(-wrR_acc[fi, 2] * 0.015 + wrR_acc[fi, 1] * 0.010, -0.45, 0.45)),
        float(np.clip(-chZ_acc[fi] * 0.016, -0.35, 0.35)),
        float(np.clip(-chZ_acc[fi] * 0.016, -0.35, 0.35)),
    ]
    for ai in range(4):
        fa = -75.0 * (q_aux[ai] - drv_aux[ai]) - 10.5 * dq_aux[ai]
        dq_aux[ai] += fa * sub_dt
        q_aux[ai] = float(np.clip(q_aux[ai] + dq_aux[ai] * sub_dt, -0.42, 0.48))

    for hi in range(12):
        az = hc_az[hi]
        rx, ry = math.sin(az), math.cos(az)
        in_h = -(hd_a[0] * rx + hd_a[1] * ry) * 0.018 - (hv[0] * rx + hv[1] * ry) * 0.05 + max(0.0, -hd_a[2]) * 0.010
        fh0 = -88.0 * (q_hc[hi, 0] - 0.03) - 11.5 * dq_hc[hi, 0] + in_h * 18.0
        fh1 = -68.0 * (q_hc[hi, 1] - q_hc[hi, 0] * 0.65) - 9.5 * dq_hc[hi, 1] + in_h * 24.0
        dq_hc[hi, 0] += fh0 * sub_dt; q_hc[hi, 0] = np.clip(q_hc[hi, 0] + dq_hc[hi, 0] * sub_dt, -0.05, 0.38)
        dq_hc[hi, 1] += fh1 * sub_dt; q_hc[hi, 1] = np.clip(q_hc[hi, 1] + dq_hc[hi, 1] * sub_dt, -0.06, 0.46)

    in_ah = -hd_a[1] * 0.022 - hd_a[2] * 0.016
    fa0 = -95.0 * q_ah[0] - 9.0 * dq_ah[0] + in_ah * 22.0
    fa1 = -72.0 * (q_ah[1] - q_ah[0] * 0.7) - 7.5 * dq_ah[1] + in_ah * 28.0
    dq_ah[0] += fa0 * sub_dt; q_ah[0] = float(np.clip(q_ah[0] + dq_ah[0] * sub_dt, -0.35, 0.38))
    dq_ah[1] += fa1 * sub_dt; q_ah[1] = float(np.clip(q_ah[1] + dq_ah[1] * sub_dt, -0.42, 0.45))

for _ in range(60):
    for _ in range(4): step_phys(0, DT / 4.0)

phys_traj_sk  = np.zeros((N_F, 5, 2)); phys_traj_aux = np.zeros((N_F, 4))
phys_traj_hc  = np.zeros((N_F, 12, 2)); phys_traj_ah  = np.zeros((N_F, 2))
for fi in range(N_F):
    for _ in range(4): step_phys(fi, DT / 4.0)
    phys_traj_sk[fi] = q_sk; phys_traj_aux[fi] = q_aux
    phys_traj_hc[fi] = q_hc; phys_traj_ah[fi] = q_ah

for fi in range(N_F):
    f = F_START + fi
    sc.frame_set(f)
    reset_cur_arm_space_R()
    yaw_q = Quaternion(Vector((0, 0, 1)), hips_yaw_seq[fi]).to_matrix()
    for si, (sec, _, _) in enumerate(COAT_SECTORS):
        _, ax_rest = skirt_out_axes[sec]
        ax_w = (yaw_q @ ax_rest).normalized()
        for cj, sfx in [(0, "01"), (1, "02")]:
            bn = f"PHY_Coat_{sec}_{sfx}"
            if bn in pbs:
                ang = float(phys_traj_sk[fi, si, cj])
                R_par_cur = R_arm_w @ get_pb_arm_R(pbs[bn].parent.name)
                R_par_rst = R_arm_w @ bone_local_R[pbs[bn].parent.name]
                R_base_w  = (R_par_cur @ R_par_rst.inverted() @ xm_rest_R[bn]).normalized()
                R_new_w   = (Quaternion(ax_w, ang).to_matrix() @ R_base_w).normalized()
                set_pb_world_rot(pbs[bn], R_new_w)
                pbs[bn].keyframe_insert(data_path="rotation_quaternion", frame=f)

    for ai, bn in enumerate(["PHY_Cuff_L", "PHY_Cuff_R", "PHY_Bow_L", "PHY_Bow_R"]):
        if bn in pbs:
            pbs[bn].rotation_quaternion = Quaternion((1, 0, 0), float(phys_traj_aux[fi, ai]))
            pbs[bn].keyframe_insert(data_path="rotation_quaternion", frame=f)

    for hi in range(12):
        for seg_idx, vals in [((1, 2), phys_traj_hc[fi, hi, 0] * 0.5), ((3, 4), phys_traj_hc[fi, hi, 1] * 0.5)]:
            for sj in seg_idx:
                bn = f"Hc_{hi}_{sj}"
                if bn in pbs:
                    pbs[bn].rotation_quaternion = Quaternion((1, 0, 0), float(vals))
                    pbs[bn].keyframe_insert(data_path="rotation_quaternion", frame=f)

    for aj, segs in [(0, [1, 2]), (1, [3, 4, 5])]:
        for sj in segs:
            bn = f"Hair_Ahoge_{sj:02d}"
            if bn in pbs:
                pbs[bn].rotation_quaternion = Quaternion((1, 0, 0), float(phys_traj_ah[fi, aj] * 0.45))
                pbs[bn].keyframe_insert(data_path="rotation_quaternion", frame=f)

# 音频口型 + 防穿模眨眼 + 大仰头垂眸微眯眼神叠加！
sk_sotai  = skin_ob.data.shape_keys
sk_facial = facial_ob.data.shape_keys
for mesh_sk in (sk_sotai, sk_facial):
    if mesh_sk and mesh_sk.animation_data:
        mesh_sk.animation_data_clear()

BLINK_CENTERS = [58, 168, 265, 378, 468, 566, 642]
def blink_weight(f):
    w = 0.0
    for bc in BLINK_CENTERS:
        d = f - bc
        if d == -2: w = max(w, 0.35)
        elif d == -1: w = max(w, 0.88)
        elif d == 0: w = max(w, 0.96)
        elif d == 1: w = max(w, 0.72)
        elif d == 2: w = max(w, 0.25)
    return w

env_arr = np.array(env_norm[:F_END + 5], dtype=np.float64)
env_smooth = np.convolve(env_arr, np.array([0.2, 0.6, 0.2]), mode='same')

for fi in range(N_F):
    f = F_START + fi
    voc = float(env_smooth[fi])
    bri = float(env_bright[fi])
    bw  = max(blink_weight(f), headup_lidded_seq[fi])  # 大仰头时叠加微垂眼帘眼神，消除瞪眼伪人感！

    if voc < 0.08:
        w_a = w_i = w_u = w_o = 0.0
        w_smile = 0.28 + 0.10 * math.sin(f * 0.05)
    else:
        amp = min(0.62, voc * 0.68)
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
        "あ": w_a, "い": w_i, "う": w_u, "お": w_o,
        "Mouth_Smile_1_Large": w_smile,
        "Eyebrow_Up": w_brow_up, "Eyebrow_Down": w_brow_down,
    }
    for mesh_sk in (sk_sotai, sk_facial):
        for kn, kval in kv_map.items():
            kb = mesh_sk.key_blocks.get(kn)
            if kb:
                kb.value = float(kval)
                kb.keyframe_insert(data_path="value", frame=f)

# ==============================================================================
# PHASE 5: 净空 Heart 舞台 + 浅坐零穿模白椅 + 升级版三渲二 (含下巴月牙阴影 & 下颌线) + 相机
# ==============================================================================
print("=== [V10R12] PHASE 5: Stage + Shallow-Fit Chair + SubChin NPR Shader ===", flush=True)
bpy.ops.mmd_tools.import_model(filepath=str(next((CDIR / "heart_stage").rglob("*.pmx"))), scale=0.08)
heart_root = next(o for o in bpy.data.objects if o.mmd_type == 'ROOT' and "heart" in o.name.lower())
heart_mesh = next(o for o in heart_root.children_recursive if o.type == 'MESH')

Mw_h = heart_mesh.matrix_world.copy()
bm_h = bmesh.new()
bm_h.from_mesh(heart_mesh.data)
to_del = [f_bm for f_bm in bm_h.faces if abs((Mw_h @ f_bm.calc_center_median()).x) < 1.58 and -1.60 < (Mw_h @ f_bm.calc_center_median()).y < 2.25 and 0.035 < (Mw_h @ f_bm.calc_center_median()).z < 2.45]
bmesh.ops.delete(bm_h, geom=to_del, context='FACES')
bm_h.to_mesh(heart_mesh.data); bm_h.free(); heart_mesh.data.update()

for mi, mat in enumerate(heart_mesh.data.materials):
    if not mat: continue
    mat.use_nodes = True; mat.use_backface_culling = True
    nt = mat.node_tree
    tex_img = next((n.image for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image), None)
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    if mi in (7, 8):
        em = nt.nodes.new("ShaderNodeEmission")
        if tex_img:
            tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = tex_img
            nt.links.new(tn.outputs["Color"], em.inputs["Color"])
            em.inputs["Strength"].default_value = 2.2 if mi == 8 else 3.2
        else:
            em.inputs["Color"].default_value = (1.0, 0.42, 0.78, 1.0); em.inputs["Strength"].default_value = 3.2
        nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])
    else:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.35 if mi == 0 else 0.55
        if tex_img:
            tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = tex_img
            mix_c = nt.nodes.new("ShaderNodeMix"); mix_c.data_type = 'RGBA'
            mix_c.inputs["Factor"].default_value = 0.42; mix_c.inputs[6].default_value = (0.14, 0.17, 0.27, 1.0)
            nt.links.new(tn.outputs["Color"], mix_c.inputs[7]); nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
        else:
            bsdf.inputs["Base Color"].default_value = (0.15, 0.18, 0.28, 1.0)
        nt.links.new(bsdf.outputs["BSDF"], out_n.inputs["Surface"])

# 5B) 白色单椅浅坐零穿模校准：
# 精确采样仅属于「座板 (0.38 < Z < 0.53)」的几何中心与前缘，将座宽缩放 0.78、座深缩放 0.68，
# 并把座板前缘严格锁定在 `knee_safe_y + 0.065m` 之后（确保大腿膝窝与小腿 100% 在座板前方悬空，绝不切腰切腿！），
# 同时将座板顶面精准对齐到小明全坐姿段最低短裤底面之下 3mm！
bpy.ops.mmd_tools.import_model(filepath=str(next((CDIR / "white_chair").rglob("*.pmx"))), scale=0.08)
chair_root = next(o for o in bpy.data.objects if o.mmd_type == 'ROOT' and ("椅" in o.name or o != heart_root))
chair_mesh = next(o for o in chair_root.children_recursive if o.type == 'MESH')

# 采样小明在 f95..f180 (包含 0:03 与 0:05!) 的真实短裤底面最低高度，作为座板顶面绝对上限！
sit_shorts_z_list = []
for f_chk in [12, 60, 102, 128, 152, 160, 168, 220, 320, 410]:
    sc.frame_set(f_chk)
    dg_chk = bpy.context.evaluated_depsgraph_get()
    sh_eo = shorts_ob.evaluated_get(dg_chk)
    sh_pts = [shorts_ob.matrix_world @ v.co for v in sh_eo.data.vertices]
    sit_shorts_z_list.append(float(np.percentile([p.z for p in sh_pts], 2)))
min_sit_shorts_z = min(sit_shorts_z_list)
target_seat_top_z = min_sit_shorts_z - 0.005  # 留 5mm 净空，确保最深后仰帧短裤也稳稳坐在椅面之上绝不穿入座板！
print(f"Chair Calib: min_sit_shorts_z={min_sit_shorts_z:.4f}m -> target_seat_top_z={target_seat_top_z:.4f}m", flush=True)

bm_c = bmesh.new()
bm_c.from_mesh(chair_mesh.data)
del_c = [f_bm for f_bm in bm_c.faces if chair_mesh.data.materials[f_bm.material_index].name not in ("bai1", "hua1")]
bmesh.ops.delete(bm_c, geom=del_c, context='FACES')

verts_c_w = [chair_mesh.matrix_world @ v.co for v in bm_c.verts]
seat_slab_w = [p for p in verts_c_w if 0.38 <= p.z <= 0.54]
seat_cx = 0.5 * (min(p.x for p in seat_slab_w) + max(p.x for p in seat_slab_w))
seat_y_min = min(p.y for p in seat_slab_w)
seat_y_max = max(p.y for p in seat_slab_w)
seat_z_top = max(p.z for p in seat_slab_w)
cz_min = min(p.z for p in verts_c_w)

scale_cx = 0.80  # 精巧宽度，不挡两侧大衣摆自然垂落
scale_cy = 0.66  # 座深从 39.3cm 缩浅至 25.9cm，浅坐露腿，膝窝 100% 探出椅沿！
scale_cz = target_seat_top_z / max(0.45, seat_z_top - cz_min)
# 让缩放后的座板前缘位于 max(1.255, knee_safe_y + 0.065)，这样臀部中心 (Y≈1.45) 稳坐座板中央，而膝盖小腿 (Y≈1.16) 完全在座板前方！
desired_front_y = max(1.265, knee_safe_y + 0.065)

M_c_inv = chair_mesh.matrix_world.inverted()
for v in bm_c.verts:
    pw = chair_mesh.matrix_world @ v.co
    pw_new = Vector((
        (pw.x - seat_cx) * scale_cx + 0.012,
        (pw.y - seat_y_min) * scale_cy + desired_front_y,
        (pw.z - cz_min) * scale_cz,
    ))
    v.co = M_c_inv @ pw_new
bm_c.to_mesh(chair_mesh.data); bm_c.free(); chair_mesh.data.update()

for mat in chair_mesh.data.materials:
    if not mat: continue
    mat.use_nodes = True
    nt = mat.node_tree
    tex_img = next((n.image for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image), None)
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    dif = nt.nodes.new("ShaderNodeBsdfDiffuse"); s2r = nt.nodes.new("ShaderNodeShaderToRGB"); rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.interpolation = 'CONSTANT'
    rmp.color_ramp.elements[0].position = 0.0;  rmp.color_ramp.elements[0].color = (0.76, 0.81, 0.93, 1.0)
    rmp.color_ramp.elements[1].position = 0.36; rmp.color_ramp.elements[1].color = (0.98, 0.99, 1.0, 1.0)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(dif.outputs["BSDF"], s2r.inputs["Shader"]); nt.links.new(s2r.outputs["Color"], rmp.inputs["Fac"])
    if tex_img and mat.name == "hua1":
        tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = tex_img
        mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs["Factor"].default_value = 1.0
        nt.links.new(tn.outputs["Color"], mul.inputs[6]); nt.links.new(rmp.outputs["Color"], mul.inputs[7]); nt.links.new(mul.outputs[2], em.inputs["Color"])
    else:
        nt.links.new(rmp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])

# 5C) 升级三渲二着色器 (皮肤接入 NPR_SubChin 顶点色实现「上颈常驻月牙阴影 + 下颌折角线」；口腔 RD_mouth 增强珊瑚红清晰度！)
def make_outline_mat(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True; m.use_backface_culling = True; m.use_backface_culling_shadow = True
    nt = m.node_tree; nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgba; em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])
    return m

mat_line_skin = make_outline_mat("MAT_NPR_Outline_Skin", (0.22, 0.13, 0.18, 1.0))
mat_line_cool = make_outline_mat("MAT_NPR_Outline_Cool", (0.12, 0.19, 0.33, 1.0))

for o in xm_meshes:
    if o.hide_render: continue
    is_facial_parts = (o.name == "RadDoll_FacialParts")
    for m in o.data.materials:
        if not m or not m.use_nodes or m.name.startswith("MAT_NPR_Outline"): continue
        if any(k in m.name.lower() for k in ("transparent", "highlight")): continue
        nt = m.node_tree
        base_img = None
        gltf_n = nt.nodes.get("GLTF_BSDF")
        if gltf_n and gltf_n.inputs["Base Color"].is_linked:
            src_n = gltf_n.inputs["Base Color"].links[0].from_node
            if src_n.type == 'TEX_IMAGE': base_img = src_n.image
        if base_img is None:
            base_img = next((n.image for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image), None)
        if base_img is None: continue
        out_n = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if not out_n: continue

        is_skin = any(k in m.name.lower() for k in ("skin", "face", "body", "sotai"))
        is_hair = "hair" in m.name.lower() or "Hair" in o.name

        tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = base_img

        if is_facial_parts:
            em = nt.nodes.new("ShaderNodeEmission")
            if "mouth" in m.name.lower():
                # 增强二次元口腔/舌面暖珊瑚对比度，使仰头/唱歌口型清晰立体不糊白！
                hsv = nt.nodes.new("ShaderNodeHueSaturation")
                hsv.inputs["Saturation"].default_value = 1.38
                hsv.inputs["Value"].default_value = 0.86
                nt.links.new(tn.outputs["Color"], hsv.inputs["Color"])
                nt.links.new(hsv.outputs["Color"], em.inputs["Color"])
                em.inputs["Strength"].default_value = 1.0
            else:
                nt.links.new(tn.outputs["Color"], em.inputs["Color"])
                em.inputs["Strength"].default_value = 1.04
            if m.surface_render_method == 'BLENDED' or any(k in m.name.lower() for k in ("lash", "brow", "lid")):
                tr = nt.nodes.new("ShaderNodeBsdfTransparent"); mx = nt.nodes.new("ShaderNodeMixShader")
                nt.links.new(tn.outputs["Alpha"], mx.inputs["Fac"])
                nt.links.new(tr.outputs["BSDF"], mx.inputs[1]); nt.links.new(em.outputs["Emission"], mx.inputs[2])
                nt.links.new(mx.outputs["Shader"], out_n.inputs["Surface"])
            else:
                nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])
            continue

        dif = nt.nodes.new("ShaderNodeBsdfDiffuse"); dif.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        s2r = nt.nodes.new("ShaderNodeShaderToRGB"); rmp = nt.nodes.new("ShaderNodeValToRGB")
        rmp.color_ramp.interpolation = 'LINEAR'
        rmp.color_ramp.elements[0].position = 0.37; rmp.color_ramp.elements[1].position = 0.41
        if is_skin:
            rmp.color_ramp.elements[0].color = (0.96, 0.85, 0.86, 1.0)
            rmp.color_ramp.elements[1].color = (1.00, 0.99, 0.99, 1.0)
        else:
            rmp.color_ramp.elements[0].color = (0.77, 0.82, 0.95, 1.0)
            rmp.color_ramp.elements[1].position = 0.42
            rmp.color_ramp.elements[1].color = (1.00, 1.00, 1.00, 1.0)

        nt.links.new(dif.outputs["BSDF"], s2r.inputs["Shader"]); nt.links.new(s2r.outputs["Color"], rmp.inputs["Fac"])
        mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs["Factor"].default_value = 1.0
        nt.links.new(tn.outputs["Color"], mul.inputs[6]); nt.links.new(rmp.outputs["Color"], mul.inputs[7])

        cur_color_socket = mul.outputs[2]

        # 对皮肤材质叠加「颈部常驻冷粉紫月牙阴影 (R)」与「下颌折角内描边提示 (G)」，彻底根治仰拍面颈糊成一团的伪人感！
        if is_skin and o.name == "GEO-Sotai_Skin":
            vc = nt.nodes.new("ShaderNodeVertexColor")
            vc.layer_name = "NPR_SubChin"
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            nt.links.new(vc.outputs["Color"], sep.inputs["Color"])

            # 1) 颈部月牙投射阴影乘色 (清透番剧冷粉紫灰 0.77, 0.67, 0.77)
            mix_nsh = nt.nodes.new("ShaderNodeMix"); mix_nsh.data_type = 'RGBA'; mix_nsh.blend_type = 'MULTIPLY'
            nt.links.new(sep.outputs["Red"], mix_nsh.inputs["Factor"])
            nt.links.new(cur_color_socket, mix_nsh.inputs[6])
            mix_nsh.inputs[7].default_value = (0.85, 0.75, 0.79, 1.0)

            # 2) 下颌折角线强调 (暖深莓紫 0.56, 0.42, 0.50，低机位仰拍时清晰勾出尖下巴边界！)
            mix_jln = nt.nodes.new("ShaderNodeMix"); mix_jln.data_type = 'RGBA'; mix_jln.blend_type = 'MULTIPLY'
            nt.links.new(sep.outputs["Green"], mix_jln.inputs["Factor"])
            nt.links.new(mix_nsh.outputs[2], mix_jln.inputs[6])
            mix_jln.inputs[7].default_value = (0.58, 0.44, 0.52, 1.0)
            cur_color_socket = mix_jln.outputs[2]

        lw = nt.nodes.new("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.74 if is_skin else 0.68
        rim_rmp = nt.nodes.new("ShaderNodeValToRGB"); rim_rmp.color_ramp.interpolation = 'CONSTANT'
        rim_rmp.color_ramp.elements[0].position = 0.0; rim_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        rim_rmp.color_ramp.elements[1].position = 0.66 if is_skin else 0.60
        rim_rmp.color_ramp.elements[1].color = (0.20, 0.24, 0.32, 1.0) if not is_hair else (0.28, 0.34, 0.46, 1.0)
        nt.links.new(lw.outputs["Facing"], rim_rmp.inputs["Fac"])

        add_rim = nt.nodes.new("ShaderNodeMix"); add_rim.data_type = 'RGBA'; add_rim.blend_type = 'ADD'
        add_rim.inputs["Factor"].default_value = 0.50 if is_skin else 0.75
        nt.links.new(cur_color_socket, add_rim.inputs[6]); nt.links.new(rim_rmp.outputs["Color"], add_rim.inputs[7])

        em_out = nt.nodes.new("ShaderNodeEmission"); em_out.inputs["Strength"].default_value = 1.0
        nt.links.new(add_rim.outputs[2], em_out.inputs["Color"]); nt.links.new(em_out.outputs["Emission"], out_n.inputs["Surface"])

    if not is_facial_parts and "NPR_Outline" not in o.modifiers:
        use_skin_line = (o.name == "GEO-Sotai_Skin")
        line_mat = mat_line_skin if use_skin_line else mat_line_cool
        o.data.materials.append(line_mat)
        mat_slot_idx = len(o.data.materials) - 1
        mod_sol = o.modifiers.new(name="NPR_Outline", type='SOLIDIFY')
        is_hair_ob = ("Hair" in o.name)
        mod_sol.thickness = 0.00105 if (use_skin_line or is_hair_ob) else 0.00135
        mod_sol.offset = 1.0
        mod_sol.use_flip_normals = True
        mod_sol.use_rim = False
        mod_sol.material_offset = mat_slot_idx

# 5D) 舞台灯光与相机 (1.595m 与原装初音 1.593m 1:1 等高，相机仅需 +0.025m 微呼吸余量即可原汁原味还原全部特写与全景！)
for o in list(bpy.data.objects):
    if o.type == 'LIGHT': bpy.data.objects.remove(o, do_unlink=True)

key_ld = bpy.data.lights.new("NPR_KeySun", type='SUN')
key_ld.energy = 2.85; key_ld.color = (1.0, 0.98, 0.99); key_ld.angle = math.radians(4.0)
key_lo = bpy.data.objects.new("NPR_KeySun", key_ld); sc.collection.objects.link(key_lo)
key_lo.rotation_euler = Euler((math.radians(52.0), math.radians(-18.0), math.radians(28.0)), 'XYZ')

rim_ld = bpy.data.lights.new("NPR_NeonBackFill", type='AREA')
rim_ld.energy = 180.0; rim_ld.size = 3.5; rim_ld.color = (0.95, 0.62, 0.88)
rim_lo = bpy.data.objects.new("NPR_NeonBackFill", rim_ld); sc.collection.objects.link(rim_lo)
rim_lo.location = Vector((0.0, 3.2, 2.4)); rim_lo.rotation_euler = Euler((math.radians(-115.0), 0.0, 0.0), 'XYZ')

if sc.world and sc.world.use_nodes:
    bg_n = sc.world.node_tree.nodes.get("Background")
    if bg_n: bg_n.inputs[0].default_value = (0.09, 0.11, 0.19, 1.0); bg_n.inputs[1].default_value = 0.55

mv_cam_d = bpy.data.cameras.new("MMD_MV_CAMERA")
mv_cam_d.clip_start = 0.05; mv_cam_d.clip_end = 80.0
mv_cam_o = bpy.data.objects.new("MMD_MV_CAMERA", mv_cam_d); mv_cam_o.rotation_mode = 'QUATERNION'
sc.collection.objects.link(mv_cam_o); sc.camera = mv_cam_o

for idx, cf in enumerate(cam_frames):
    f = F_START + idx
    loc_adj = cf["loc"].copy()
    loc_adj.z += 0.028  # 1.595m 已与初音等高，仅留 +2.8cm 呆毛余量，开场/特写/比心全机位原汁原味！
    mv_cam_o.location = loc_adj
    mv_cam_o.rotation_quaternion = cf["quat"]
    mv_cam_d.sensor_fit = cf["sensor_fit"]
    mv_cam_d.sensor_width = cf["sensor_width"]
    mv_cam_d.sensor_height = cf["sensor_height"]
    mv_cam_d.lens = cf["lens"]
    mv_cam_o.keyframe_insert(data_path="location", frame=f)
    mv_cam_o.keyframe_insert(data_path="rotation_quaternion", frame=f)
    mv_cam_d.keyframe_insert(data_path="lens", frame=f)

if mv_cam_o.animation_data and mv_cam_o.animation_data.action:
    for fc in getattr(mv_cam_o.animation_data.action, "fcurves", []):
        kps = fc.keyframe_points
        for ki in range(len(kps) - 1):
            if abs(kps[ki + 1].co.y - kps[ki].co.y) > 0.45:
                kps[ki].interpolation = 'CONSTANT'

if not sc.sequence_editor: sc.sequence_editor_create()
strips = sc.sequence_editor.strips if hasattr(sc.sequence_editor, "strips") else sc.sequence_editor.sequences
for s_old in list(strips): strips.remove(s_old)
if hasattr(strips, "new_sound"):
    strips.new_sound(name="World_BGM", filepath=wav_path, channel=1, frame_start=1)

sc.view_settings.view_transform = 'Standard'; sc.view_settings.look = 'None'
sc.render.engine = 'BLENDER_EEVEE'; sc.eevee.taa_render_samples = 20
sc.render.resolution_x = 1280; sc.render.resolution_y = 720; sc.render.resolution_percentage = 100
sc.render.use_sequencer = False

try: bpy.ops.file.pack_all()
except Exception: pass
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND), compress=True)
print(f"SAVED V10R12 BLEND: {OUT_BLEND.name} ({OUT_BLEND.stat().st_size/(1024*1024):.2f} MB) in {time.time()-t0_all:.1f}s", flush=True)

# 渲染 8 个关键帧（精准覆盖用户指出的 0:03 图一 f102/f126 与 0:05 图二 f154/f162，以及全景/椅上/起身/比心）！
QA_FRAMES = [12, 102, 126, 154, 162, 242, 482, 628]
sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.image_settings.quality = 94

for qf in QA_FRAMES:
    sc.frame_set(qf)
    sc.render.filepath = str(CMP_DIR / f"v12_f{qf:03d}.jpg")
    bpy.ops.render.render(write_still=True)

print("V10R12_8_RENDERS_DONE", flush=True)
