import math, time
from pathlib import Path
import bpy
from mathutils import Vector, Matrix

t0 = time.time()
ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
WDIR = ROOT / "生成模型" / "组装" / "v010R11_MMD_三渲二舞台_PRIVATE"
SRC_R18 = WDIR / "RadDoll_V10R18_FullSong6265_KorolevskiyTheater_HSR_MMD_Master_PRIVATE.blend"
DST_R19 = WDIR / "RadDoll_V10R19_FullSong6265_KorolevskiyTheater_HSR_Genshin_Master_PRIVATE.blend"
TEX_DIR = WDIR / "tex_v10r19_PRIVATE"
QA_DIR  = WDIR / "qa_v10r19_ab_compare"
QA_DIR.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=str(SRC_R18))
scn = bpy.context.scene

def en_in(nd, nm):
    return [sk for sk in nd.inputs if sk.enabled and sk.name == nm][0]
def en_out(nd, nm):
    return [sk for sk in nd.outputs if sk.enabled and sk.name == nm][0]

def load_img(filename, colorspace='sRGB'):
    p = str(TEX_DIR / filename)
    im = bpy.data.images.load(p, check_existing=True)
    im.colorspace_settings.name = colorspace
    im.alpha_mode = 'STRAIGHT'
    return im

print("=== [1/6] Loading V10R19 4K Face/Eye Atlases, Head-Anchored FaceSDF & 256x20 Shadow Ramps ===", flush=True)
img_face_4k   = load_img("Xiaoming_Face_4K_V10R19_PRIVATE.png", 'sRGB')
img_eye_4k    = load_img("Xiaoming_Eye_4K_V10R19_PRIVATE.png", 'sRGB')
img_face_sdf  = load_img("Xiaoming_FaceSDF_2048_PRIVATE.png", 'Non-Color')
img_ramp_skin = load_img("Xiaoming_Skin_Shadow_Ramp_256x20_PRIVATE.png", 'sRGB')
img_ramp_hair = load_img("Xiaoming_Hair_Shadow_Ramp_256x20_PRIVATE.png", 'sRGB')
img_ramp_clth = load_img("Xiaoming_Cloth_Shadow_Ramp_256x20_PRIVATE.png", 'sRGB')

print("=== [2/6] Attaching Real-Time Head-Bone SDF Helper Empties (Zero-Handler Pure EEVEE Tracking) ===", flush=True)
arm = bpy.data.objects["ARM-FullBody_V10"]
old_pose_pos = arm.data.pose_position
arm.data.pose_position = 'REST'
bpy.context.view_layer.update()

sdf_col = bpy.data.collections.get("RIG_Genshin_HeadSDF")
if sdf_col is None:
    sdf_col = bpy.data.collections.new("RIG_Genshin_HeadSDF")
    scn.collection.children.link(sdf_col)

def make_head_empty(name, loc_bone_space):
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    em = bpy.data.objects.new(name, None)
    em.empty_display_type = 'PLAIN_AXES'
    em.empty_display_size = 0.02
    em.hide_render = True
    sdf_col.objects.link(em)
    em.parent = arm
    em.parent_type = 'BONE'
    em.parent_bone = 'Head'
    em.matrix_parent_inverse = Matrix.Identity(4)
    em.location = loc_bone_space
    return em

# In ARM-FullBody_V10 rest pose (with ARM_MW = diag(-1.13, -1.13, +1.13)):
# Bone local (0, 0, -1) -> World (0, -1, 0) [Exact Nose Forward!]
# Bone local (-1, 0, 0) -> World (+1, 0, 0) [Exact +X Axis matching sdf_right!]
em_orig  = make_head_empty("Head_SDF_Origin", ( 0.0, 0.0,  0.0))
em_fwd   = make_head_empty("Head_SDF_Fwd",    ( 0.0, 0.0, -1.0))
em_right = make_head_empty("Head_SDF_Right",  (-1.0, 0.0,  0.0))
bpy.context.view_layer.update()

v_fwd_rest   = (em_fwd.matrix_world.translation - em_orig.matrix_world.translation).normalized()
v_right_rest = (em_right.matrix_world.translation - em_orig.matrix_world.translation).normalized()
print("VERIFY_REST_HEAD_AXES: fwd=", [round(x, 4) for x in v_fwd_rest], "right=", [round(x, 4) for x in v_right_rest], flush=True)
assert v_fwd_rest.dot(Vector((0.0, -1.0, 0.0))) > 0.99, f"Unexpected Head_Fwd: {v_fwd_rest}"
assert v_right_rest.dot(Vector((1.0, 0.0, 0.0))) > 0.99, f"Unexpected Head_Right: {v_right_rest}"

# While in REST pose, cleanly split frontmost 3D cornea catchlight & upper-cornea shadow polygons of RadDoll_FacialParts Slot 5!
fp = bpy.data.objects["RadDoll_FacialParts"]
fp_uv = fp.data.uv_layers[0].data

mat_eye_ov = bpy.data.materials.get("RD_eye_overlay")
if mat_eye_ov is None:
    mat_eye_ov = bpy.data.materials.new("RD_eye_overlay")
if "RD_eye_overlay" not in [m.name for m in fp.data.materials if m]:
    fp.data.materials.append(mat_eye_ov)
ov_slot_idx = [i for i, m in enumerate(fp.data.materials) if m and m.name == "RD_eye_overlay"][0]

reassigned_catchlights = 0
for poly in fp.data.polygons:
    if poly.material_index in (5, ov_slot_idx):
        ys = [(fp.matrix_world @ fp.data.vertices[vi].co).y for vi in poly.vertices]
        us = [fp_uv[li].uv[0] for li in poly.loop_indices]
        vs = [fp_uv[li].uv[1] for li in poly.loop_indices]
        # STRICTLY activate only the pure 3D floating cornea catchlight quads (uv_u in [0.495, 0.692], uv_v >= 0.818)
        # so zero outer-corner comic/shadow quads ever appear!
        if min(ys) <= -0.1095 and 0.495 <= min(us) and max(us) <= 0.695 and min(vs) >= 0.818:
            poly.material_index = ov_slot_idx
            reassigned_catchlights += 1
        else:
            poly.material_index = 5

print(f"ACTIVATED_3D_EYE_OVERLAY_POLYS: {reassigned_catchlights} (kept comic props hidden on Slot 5)", flush=True)

arm.data.pose_position = old_pose_pos
bpy.context.view_layer.update()

print("=== [2B/6] Baking Exact 6,265-Frame Head-Anchored Face SDF Light Factors to Object Uniform ===", flush=True)
skin_obj = bpy.data.objects["GEO-Sotai_Skin"]
cam_obj  = bpy.data.objects["MMD_MV_CAMERA"]
if skin_obj.animation_data and skin_obj.animation_data.action:
    bpy.data.actions.remove(skin_obj.animation_data.action)

# Temporarily mute heavy mesh viewport evaluation so bone+camera loop runs in <1.0 second!
keep_eval = {arm, cam_obj, em_orig, em_fwd, em_right}
saved_vp = []
for ob in bpy.data.objects:
    if ob not in keep_eval and not ob.hide_viewport:
        saved_vp.append(ob)
        ob.hide_viewport = True

stage_key = Vector((0.55, -0.65, 0.52)).normalized()
sample_dbg = {}
for f in range(1, 6266, 3):
    scn.frame_set(f)
    h0 = em_orig.matrix_world.translation
    hf = (em_fwd.matrix_world.translation - h0).normalized()
    hr = (em_right.matrix_world.translation - h0).normalized()
    cmw = cam_obj.matrix_world
    to_cam = (cmw.translation - h0).normalized()
    cam_right = cmw.to_3x3().col[0].normalized()
    cam_up    = cmw.to_3x3().col[1].normalized()
    cam_34 = (to_cam + 0.58 * cam_right + 0.36 * cam_up).normalized()
    l_key  = (0.62 * cam_34 + 0.38 * stage_key).normalized()

    lf = hf.dot(l_key)
    lr = hr.dot(l_key)
    h_len = math.sqrt(lf * lf + lr * lr + 1e-6)
    cos_f = lf / h_len
    sin_r = lr / h_len
    u_light = max(0.76, min(0.96, 0.5 * cos_f + 0.5))
    side_01 = max(0.0, min(1.0, 0.5 * sin_r + 0.5))
    if abs(sin_r) < 0.26:
        side_01 = 0.68 if sin_r >= -0.05 else 0.32

    skin_obj.color = (u_light, side_01, 1.0, 1.0)
    skin_obj.keyframe_insert(data_path="color", frame=f)
    if f in (1, 112, 451, 1084, 1270):
        sample_dbg[f] = (round(u_light, 3), round(side_01, 3))

for ob in saved_vp:
    ob.hide_viewport = False
bpy.context.view_layer.update()
print("BAKED_HEAD_SDF_SAMPLES:", sample_dbg, flush=True)


print("=== [3/6] Building Genshin/Morningstar-Grade Shaders (Face SDF + 4K Eyes + 256x20 Ramps + MatCap Metal) ===", flush=True)

def set_opaque_clean(mat):
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'DITHERED'

def set_blended_clean(mat):
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'BLENDED'
    if hasattr(mat, "use_transparency_overlap"):
        mat.use_transparency_overlap = False

def add_key_light_and_rim_nodes(nt, smooth_bias=(0.0, -0.28, 0.46), smooth_fac=0.18, rim_exp=7.2):
    lk = nt.links
    n_geo = nt.nodes.new("ShaderNodeNewGeometry")

    n_nmix = nt.nodes.new("ShaderNodeMix"); n_nmix.data_type = 'VECTOR'
    en_in(n_nmix, "Factor").default_value = smooth_fac
    en_in(n_nmix, "B").default_value = smooth_bias
    n_nnrm = nt.nodes.new("ShaderNodeVectorMath"); n_nnrm.operation = 'NORMALIZE'
    lk.new(n_geo.outputs["Normal"], en_in(n_nmix, "A"))
    lk.new(en_out(n_nmix, "Result"), n_nnrm.inputs[0])

    # Proven bright front-right-top Genshin/HSR key rig (Geometry.Incoming [= To_Cam] + Stage Front-Right-Top (-Y)!):
    n_cadd = nt.nodes.new("ShaderNodeVectorMath"); n_cadd.operation = 'ADD'
    n_cadd.inputs[1].default_value = (0.46, 0.0, 0.36)
    n_cnrm = nt.nodes.new("ShaderNodeVectorMath"); n_cnrm.operation = 'NORMALIZE'
    lk.new(n_geo.outputs["Incoming"], n_cadd.inputs[0])
    lk.new(n_cadd.outputs["Vector"], n_cnrm.inputs[0])

    n_lmix = nt.nodes.new("ShaderNodeMix"); n_lmix.data_type = 'VECTOR'
    en_in(n_lmix, "Factor").default_value = 0.62
    en_in(n_lmix, "A").default_value = (0.55, -0.65, 0.52)  # Stage front-right-top (-Y is character front!)
    n_lnrm = nt.nodes.new("ShaderNodeVectorMath"); n_lnrm.operation = 'NORMALIZE'
    lk.new(n_cnrm.outputs["Vector"], en_in(n_lmix, "B"))
    lk.new(en_out(n_lmix, "Result"), n_lnrm.inputs[0])

    n_dot = nt.nodes.new("ShaderNodeVectorMath"); n_dot.operation = 'DOT_PRODUCT'
    n_mad = nt.nodes.new("ShaderNodeMath"); n_mad.operation = 'MULTIPLY_ADD'
    n_mad.inputs[1].default_value = 0.5
    n_mad.inputs[2].default_value = 0.5
    lk.new(n_nnrm.outputs["Vector"], n_dot.inputs[0])
    lk.new(n_lnrm.outputs["Vector"], n_dot.inputs[1])
    lk.new(n_dot.outputs["Value"], n_mad.inputs[0])

    # Narrow crisp Genshin contour Fresnel rim (higher exponent 7.2 prevents broad cloth washout per R9 recipe!)
    n_vd  = nt.nodes.new("ShaderNodeVectorMath"); n_vd.operation = 'DOT_PRODUCT'
    n_abs = nt.nodes.new("ShaderNodeMath"); n_abs.operation = 'ABSOLUTE'
    n_inv = nt.nodes.new("ShaderNodeMath"); n_inv.operation = 'SUBTRACT'; n_inv.inputs[0].default_value = 1.0; n_inv.use_clamp = True
    n_pow = nt.nodes.new("ShaderNodeMath"); n_pow.operation = 'POWER'; n_pow.inputs[1].default_value = rim_exp
    n_sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    n_upm = nt.nodes.new("ShaderNodeMapRange")
    n_upm.inputs["From Min"].default_value = -0.35; n_upm.inputs["From Max"].default_value = 0.45
    n_upm.inputs["To Min"].default_value = 0.20;   n_upm.inputs["To Max"].default_value = 1.0
    n_rmul = nt.nodes.new("ShaderNodeMath"); n_rmul.operation = 'MULTIPLY'

    lk.new(n_nnrm.outputs["Vector"], n_vd.inputs[0]); lk.new(n_geo.outputs["Incoming"], n_vd.inputs[1])
    lk.new(n_vd.outputs["Value"], n_abs.inputs[0])
    lk.new(n_abs.outputs["Value"], n_inv.inputs[1])
    lk.new(n_inv.outputs["Value"], n_pow.inputs[0])
    lk.new(n_nnrm.outputs["Vector"], n_sep.inputs["Vector"])
    lk.new(n_sep.outputs["Z"], n_upm.inputs["Value"])
    lk.new(n_pow.outputs["Value"], n_rmul.inputs[0]); lk.new(n_upm.outputs["Result"], n_rmul.inputs[1])

    return n_geo, n_nnrm.outputs["Vector"], n_geo.outputs["Incoming"], n_lnrm.outputs["Vector"], n_mad.outputs["Value"], n_rmul.outputs["Value"]

def add_256x20_ramp_sampler(nt, ramp_img, factor_socket, v_row_socket_or_float=0.5):
    lk = nt.links
    n_u = nt.nodes.new("ShaderNodeMapRange")
    n_u.inputs["From Min"].default_value = 0.0; n_u.inputs["From Max"].default_value = 1.0
    n_u.inputs["To Min"].default_value = 0.02;  n_u.inputs["To Max"].default_value = 0.98
    n_u.clamp = True
    lk.new(factor_socket, n_u.inputs["Value"])

    n_xyz = nt.nodes.new("ShaderNodeCombineXYZ")
    lk.new(n_u.outputs["Result"], n_xyz.inputs["X"])
    if isinstance(v_row_socket_or_float, (float, int)):
        n_xyz.inputs["Y"].default_value = float(v_row_socket_or_float)
    else:
        lk.new(v_row_socket_or_float, n_xyz.inputs["Y"])

    n_rt = nt.nodes.new("ShaderNodeTexImage")
    n_rt.name = "Shadow Ramp"
    n_rt.image = ramp_img
    n_rt.extension = 'EXTEND'
    n_rt.interpolation = 'Linear'
    lk.new(n_xyz.outputs["Vector"], n_rt.inputs["Vector"])
    return n_rt.outputs["Color"]

# ------------------------------------------------------------------------------
# 3A) RD_skin: Head-Anchored Face SDF + Bang Drop-Shadow + 4K Face Complexion + 256x20 Skin Ramp!
# ------------------------------------------------------------------------------
mat_face = bpy.data.materials["RD_skin"]
set_opaque_clean(mat_face)
nt = mat_face.node_tree; nt.nodes.clear(); lk = nt.links
n_out = nt.nodes.new("ShaderNodeOutputMaterial")
n_em  = nt.nodes.new("ShaderNodeEmission")
n_uv  = nt.nodes.new("ShaderNodeUVMap"); n_uv.uv_map = "UVMap"

n_ftex = nt.nodes.new("ShaderNodeTexImage"); n_ftex.image = img_face_4k
lk.new(n_uv.outputs["UV"], n_ftex.inputs["Vector"])

n_sdf = nt.nodes.new("ShaderNodeTexImage"); n_sdf.image = img_face_sdf
lk.new(n_uv.outputs["UV"], n_sdf.inputs["Vector"])
n_sdf_sep = nt.nodes.new("ShaderNodeSeparateColor")
lk.new(n_sdf.outputs["Color"], n_sdf_sep.inputs["Color"])

_, _, _, out_Lkey, _, out_rim = add_key_light_and_rim_nodes(nt, smooth_bias=(0.0, -0.45, 0.38), smooth_fac=0.45, rim_exp=8.0)

# Read per-frame Head-Anchored SDF parameters directly from native EEVEE ObjectInfo.Color GPU uniform!
n_obinfo = nt.nodes.new("ShaderNodeObjectInfo")
n_obsep  = nt.nodes.new("ShaderNodeSeparateColor")
lk.new(n_obinfo.outputs["Color"], n_obsep.inputs["Color"])

# Select Left (G) vs Right (R) SDF smoothly across side_01 = 0.5
mr_side = nt.nodes.new("ShaderNodeMapRange"); mr_side.interpolation_type = 'SMOOTHSTEP'
mr_side.inputs["From Min"].default_value = 0.44; mr_side.inputs["From Max"].default_value = 0.56
lk.new(n_obsep.outputs["Green"], mr_side.inputs["Value"])

mix_sdf = nt.nodes.new("ShaderNodeMix"); mix_sdf.data_type = 'FLOAT'
lk.new(mr_side.outputs["Result"], en_in(mix_sdf, "Factor"))
lk.new(n_sdf_sep.outputs["Green"], en_in(mix_sdf, "A"))
lk.new(n_sdf_sep.outputs["Red"],   en_in(mix_sdf, "B"))

# Compare u_light (Red) - SDF_thresh -> crisp Genshin cel transition [-0.032, +0.032] across 256x20 Skin Shadow Ramp!
m_diff = nt.nodes.new("ShaderNodeMath"); m_diff.operation = 'SUBTRACT'
lk.new(n_obsep.outputs["Red"], m_diff.inputs[0])
lk.new(en_out(mix_sdf, "Result"), m_diff.inputs[1])

mr_sdf = nt.nodes.new("ShaderNodeMapRange"); mr_sdf.interpolation_type = 'SMOOTHSTEP'
mr_sdf.inputs["From Min"].default_value = -0.032
mr_sdf.inputs["From Max"].default_value =  0.032
mr_sdf.inputs["To Min"].default_value   =  0.14
mr_sdf.inputs["To Max"].default_value   =  0.97
lk.new(m_diff.outputs["Value"], mr_sdf.inputs["Value"])

skin_ramp_col = add_256x20_ramp_sampler(nt, img_ramp_skin, mr_sdf.outputs["Result"], 0.5)

n_mul = nt.nodes.new("ShaderNodeMix"); n_mul.data_type = 'RGBA'; n_mul.blend_type = 'MULTIPLY'
en_in(n_mul, "Factor").default_value = 1.0
lk.new(n_ftex.outputs["Color"], en_in(n_mul, "A"))
lk.new(skin_ramp_col, en_in(n_mul, "B"))
cur_fcol = en_out(n_mul, "Result")

# Forehead Bang Drop-Shadow (Channel B of FaceSDF) -> warm coral-taupe anime bang shadow!
n_bang_f = nt.nodes.new("ShaderNodeMath"); n_bang_f.operation = 'MULTIPLY'; n_bang_f.inputs[1].default_value = 0.24
lk.new(n_sdf_sep.outputs["Blue"], n_bang_f.inputs[0])
n_bang_m = nt.nodes.new("ShaderNodeMix"); n_bang_m.data_type = 'RGBA'; n_bang_m.blend_type = 'MULTIPLY'
en_in(n_bang_m, "B").default_value = (0.82, 0.66, 0.68, 1.0)
lk.new(n_bang_f.outputs["Value"], en_in(n_bang_m, "Factor"))
lk.new(cur_fcol, en_in(n_bang_m, "A"))
cur_fcol = en_out(n_bang_m, "Result")

# Preserve NPR_SubChin crescent neck shadow + warm cheek vertex blush
n_vc  = nt.nodes.new("ShaderNodeVertexColor"); n_vc.layer_name = "NPR_SubChin"
n_spc = nt.nodes.new("ShaderNodeSeparateColor")
lk.new(n_vc.outputs["Color"], n_spc.inputs["Color"])

n_sc = nt.nodes.new("ShaderNodeMix"); n_sc.data_type = 'RGBA'; n_sc.blend_type = 'MULTIPLY'
en_in(n_sc, "B").default_value = (0.76, 0.60, 0.64, 1.0)
n_scf = nt.nodes.new("ShaderNodeMath"); n_scf.operation = 'MULTIPLY'; n_scf.inputs[1].default_value = 0.68
lk.new(n_spc.outputs["Red"], n_scf.inputs[0])
lk.new(n_scf.outputs["Value"], en_in(n_sc, "Factor"))
lk.new(cur_fcol, en_in(n_sc, "A"))
cur_fcol = en_out(n_sc, "Result")

n_bl = nt.nodes.new("ShaderNodeMix"); n_bl.data_type = 'RGBA'; n_bl.blend_type = 'MIX'
en_in(n_bl, "B").default_value = (0.82, 0.52, 0.53, 1.0)
n_blf = nt.nodes.new("ShaderNodeMath"); n_blf.operation = 'MULTIPLY'; n_blf.inputs[1].default_value = 0.12
lk.new(n_spc.outputs["Green"], n_blf.inputs[0])
lk.new(n_blf.outputs["Value"], en_in(n_bl, "Factor"))
lk.new(cur_fcol, en_in(n_bl, "A"))
cur_fcol = en_out(n_bl, "Result")

# Ultra-subtle warm champagne grazing sheen ONLY (eliminates cold blue plastic sheen on facial skin!)
n_radd = nt.nodes.new("ShaderNodeMix"); n_radd.data_type = 'RGBA'; n_radd.blend_type = 'ADD'
en_in(n_radd, "B").default_value = (0.55, 0.46, 0.48, 1.0)
n_rfac = nt.nodes.new("ShaderNodeMath"); n_rfac.operation = 'MULTIPLY'; n_rfac.inputs[1].default_value = 0.03
lk.new(out_rim, n_rfac.inputs[0])
lk.new(n_rfac.outputs["Value"], en_in(n_radd, "Factor"))
lk.new(cur_fcol, en_in(n_radd, "A"))

# Calibrate linear output intensity (0.86) for Standard + Medium High Contrast so lit skin = ~0.94 sRGB luminous porcelain!
n_gain = nt.nodes.new("ShaderNodeHueSaturation")
n_gain.inputs["Saturation"].default_value = 1.03
n_gain.inputs["Value"].default_value = 0.86
lk.new(en_out(n_radd, "Result"), n_gain.inputs["Color"])
lk.new(n_gain.outputs["Color"], n_em.inputs["Color"])
lk.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

# ------------------------------------------------------------------------------
# 3B) RD_body_skin & RD_ear: Matched 256x20 Skin Shadow Ramp (0.000 dE neck seam!)
# ------------------------------------------------------------------------------
for mname, img_nm, has_sub in [("RD_body_skin", "RadDoll_body_tex_v2", True), ("RD_ear", "Tex_Ears_512", False)]:
    mat = bpy.data.materials.get(mname)
    img = bpy.data.images.get(img_nm)
    if not mat or not img:
        continue
    set_opaque_clean(mat)
    nt = mat.node_tree; nt.nodes.clear(); lk = nt.links
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_em  = nt.nodes.new("ShaderNodeEmission")
    n_tex = nt.nodes.new("ShaderNodeTexImage"); n_tex.image = img
    _, _, _, _, out_NL, out_rim = add_key_light_and_rim_nodes(nt, smooth_bias=(0.0, -0.28, 0.45), smooth_fac=0.20, rim_exp=7.4)

    mr_b = nt.nodes.new("ShaderNodeMapRange"); mr_b.interpolation_type = 'SMOOTHSTEP'
    mr_b.inputs["From Min"].default_value = 0.45; mr_b.inputs["From Max"].default_value = 0.62
    mr_b.inputs["To Min"].default_value = 0.14;   mr_b.inputs["To Max"].default_value = 0.96
    lk.new(out_NL, mr_b.inputs["Value"])
    b_ramp = add_256x20_ramp_sampler(nt, img_ramp_skin, mr_b.outputs["Result"], 0.5)

    n_mul = nt.nodes.new("ShaderNodeMix"); n_mul.data_type = 'RGBA'; n_mul.blend_type = 'MULTIPLY'
    en_in(n_mul, "Factor").default_value = 1.0
    lk.new(n_tex.outputs["Color"], en_in(n_mul, "A")); lk.new(b_ramp, en_in(n_mul, "B"))
    cc = en_out(n_mul, "Result")

    if has_sub:
        n_vc  = nt.nodes.new("ShaderNodeVertexColor"); n_vc.layer_name = "NPR_SubChin"
        n_spc = nt.nodes.new("ShaderNodeSeparateColor")
        lk.new(n_vc.outputs["Color"], n_spc.inputs["Color"])
        n_sc = nt.nodes.new("ShaderNodeMix"); n_sc.data_type = 'RGBA'; n_sc.blend_type = 'MULTIPLY'
        en_in(n_sc, "B").default_value = (0.72, 0.56, 0.60, 1.0)
        n_scf = nt.nodes.new("ShaderNodeMath"); n_scf.operation = 'MULTIPLY'; n_scf.inputs[1].default_value = 0.72
        lk.new(n_spc.outputs["Red"], n_scf.inputs[0]); lk.new(n_scf.outputs["Value"], en_in(n_sc, "Factor"))
        lk.new(cc, en_in(n_sc, "A")); cc = en_out(n_sc, "Result")

    n_radd = nt.nodes.new("ShaderNodeMix"); n_radd.data_type = 'RGBA'; n_radd.blend_type = 'ADD'
    en_in(n_radd, "B").default_value = (0.34, 0.58, 0.88, 1.0)
    n_rf = nt.nodes.new("ShaderNodeMath"); n_rf.operation = 'MULTIPLY'; n_rf.inputs[1].default_value = 0.12
    lk.new(out_rim, n_rf.inputs[0]); lk.new(n_rf.outputs["Value"], en_in(n_radd, "Factor"))
    lk.new(cc, en_in(n_radd, "A"))

    n_gain = nt.nodes.new("ShaderNodeHueSaturation")
    n_gain.inputs["Saturation"].default_value = 1.03
    n_gain.inputs["Value"].default_value = 0.86
    lk.new(en_out(n_radd, "Result"), n_gain.inputs["Color"])
    lk.new(n_gain.outputs["Color"], n_em.inputs["Color"])
    lk.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

# ------------------------------------------------------------------------------
# 3C) Multi-Layer 4K Crystalline Eyes + 3D Parallax Catchlight Layer + Gradient Lashes/Brows/Mouth
# ------------------------------------------------------------------------------
# RD_eye (Eyeball + 4K Crystalline Ice-Ocean Sapphire Iris)
mat_eye = bpy.data.materials["RD_eye"]
set_opaque_clean(mat_eye)
nt = mat_eye.node_tree; nt.nodes.clear(); lk = nt.links
no = nt.nodes.new("ShaderNodeOutputMaterial"); ne = nt.nodes.new("ShaderNodeEmission")
ntx = nt.nodes.new("ShaderNodeTexImage"); ntx.image = img_eye_4k; ntx.interpolation = 'Cubic'
nhsv = nt.nodes.new("ShaderNodeHueSaturation")
nhsv.inputs["Saturation"].default_value = 1.12
nhsv.inputs["Value"].default_value = 0.82
lk.new(ntx.outputs["Color"], nhsv.inputs["Color"])
lk.new(nhsv.outputs["Color"], ne.inputs["Color"])
lk.new(ne.outputs["Emission"], no.inputs["Surface"])

# RD_eye_overlay (3D Floating Cornea Catchlights + Soft Upper-Cornea Shadow!)
set_blended_clean(mat_eye_ov)
nt = mat_eye_ov.node_tree; nt.nodes.clear(); lk = nt.links
no = nt.nodes.new("ShaderNodeOutputMaterial")
nmix = nt.nodes.new("ShaderNodeMixShader")
ntr = nt.nodes.new("ShaderNodeBsdfTransparent")
ne = nt.nodes.new("ShaderNodeEmission")
ntx = nt.nodes.new("ShaderNodeTexImage"); ntx.image = img_eye_4k; ntx.interpolation = 'Cubic'
# Boost bright white catchlights slightly (1.12) so they kiss the 0.88 Bloom threshold cleanly!
ne.inputs["Strength"].default_value = 1.12
lk.new(ntx.outputs["Alpha"], nmix.inputs["Fac"])
lk.new(ntr.outputs["BSDF"], nmix.inputs[1])
lk.new(ntx.outputs["Color"], ne.inputs["Color"])
lk.new(ne.outputs["Emission"], nmix.inputs[2])
lk.new(nmix.outputs["Shader"], no.inputs["Surface"])

# RD_lash, RD_brow, RD_double_lid -> Sample 4K Warm Plum-Sienna Gradient Atlas!
for fm_nm, val_g in [("RD_lash", 0.72), ("RD_brow", 0.78), ("RD_double_lid", 0.74)]:
    fm = bpy.data.materials.get(fm_nm)
    if not fm: continue
    set_opaque_clean(fm)
    nt = fm.node_tree; nt.nodes.clear(); lk = nt.links
    no = nt.nodes.new("ShaderNodeOutputMaterial"); ne = nt.nodes.new("ShaderNodeEmission")
    ntx = nt.nodes.new("ShaderNodeTexImage"); ntx.image = img_eye_4k
    nhsv = nt.nodes.new("ShaderNodeHueSaturation"); nhsv.inputs["Value"].default_value = val_g
    lk.new(ntx.outputs["Color"], nhsv.inputs["Color"])
    lk.new(nhsv.outputs["Color"], ne.inputs["Color"])
    lk.new(ne.outputs["Emission"], no.inputs["Surface"])

# RD_mouth -> Sample 4K Sculpted Teeth + Coral Tongue + Warm Oral Cavity on Xiaoming_Face_4K!
fm_mouth = bpy.data.materials.get("RD_mouth")
if fm_mouth:
    set_opaque_clean(fm_mouth)
    nt = fm_mouth.node_tree; nt.nodes.clear(); lk = nt.links
    no = nt.nodes.new("ShaderNodeOutputMaterial"); ne = nt.nodes.new("ShaderNodeEmission")
    ntx = nt.nodes.new("ShaderNodeTexImage"); ntx.image = img_face_4k
    nhsv = nt.nodes.new("ShaderNodeHueSaturation"); nhsv.inputs["Value"].default_value = 0.72
    lk.new(ntx.outputs["Color"], nhsv.inputs["Color"])
    lk.new(nhsv.outputs["Color"], ne.inputs["Color"])
    lk.new(ne.outputs["Emission"], no.inputs["Surface"])

# Refine See-Through Brow over Bangs (delicate warm plum-slate alpha stroke, zero thick outline!)
mat_bt = bpy.data.materials.get("RD_brow_thru")
if mat_bt and mat_bt.node_tree:
    set_blended_clean(mat_bt)
    for nd in mat_bt.node_tree.nodes:
        if nd.type == 'EMISSION':
            nd.inputs["Color"].default_value = (0.20, 0.15, 0.21, 1.0)
        elif nd.type == 'MATH' and nd.operation == 'MULTIPLY':
            nd.inputs[1].default_value = 0.42

# ------------------------------------------------------------------------------
# 3D) Hair Shader: 256x20 Sapphire-Pearl Ramp + Dual-Band Crisp Angel-Ring Ribbon!
# ------------------------------------------------------------------------------
mat_hair = bpy.data.materials.get("RD_hair")
img_hair = bpy.data.images.get("Tex_Hair_Atlas_2048")
if mat_hair and img_hair:
    set_opaque_clean(mat_hair)
    nt = mat_hair.node_tree; nt.nodes.clear(); lk = nt.links
    n_out = nt.nodes.new("ShaderNodeOutputMaterial"); n_em = nt.nodes.new("ShaderNodeEmission")
    n_uv  = nt.nodes.new("ShaderNodeUVMap"); n_uv.uv_map = "UVMap"
    n_tex = nt.nodes.new("ShaderNodeTexImage"); n_tex.image = img_hair
    lk.new(n_uv.outputs["UV"], n_tex.inputs["Vector"])

    n_hsv_in = nt.nodes.new("ShaderNodeHueSaturation")
    n_hsv_in.inputs["Saturation"].default_value = 1.14  # Enrich sea-blue hair tips!
    lk.new(n_tex.outputs["Color"], n_hsv_in.inputs["Color"])

    _, out_Ns, out_V, out_Lc, out_NL, out_rim = add_key_light_and_rim_nodes(nt, smooth_bias=(0.0, -0.20, 0.40), smooth_fac=0.20, rim_exp=7.0)

    mr_h = nt.nodes.new("ShaderNodeMapRange"); mr_h.interpolation_type = 'SMOOTHSTEP'
    mr_h.inputs["From Min"].default_value = 0.43; mr_h.inputs["From Max"].default_value = 0.61
    mr_h.inputs["To Min"].default_value = 0.10;   mr_h.inputs["To Max"].default_value = 0.96
    lk.new(out_NL, mr_h.inputs["Value"])
    h_ramp = add_256x20_ramp_sampler(nt, img_ramp_hair, mr_h.outputs["Result"], 0.5)

    n_mul = nt.nodes.new("ShaderNodeMix"); n_mul.data_type = 'RGBA'; n_mul.blend_type = 'MULTIPLY'
    en_in(n_mul, "Factor").default_value = 1.0
    lk.new(n_hsv_in.outputs["Color"], en_in(n_mul, "A")); lk.new(h_ramp, en_in(n_mul, "B"))

    # Dual-Band Anime Angel-Ring Highlight (Primary Razor Ribbon + Secondary Soft Ice Halo!)
    n_hadd = nt.nodes.new("ShaderNodeVectorMath"); n_hadd.operation = 'ADD'
    n_hnrm = nt.nodes.new("ShaderNodeVectorMath"); n_hnrm.operation = 'NORMALIZE'
    lk.new(out_Lc, n_hadd.inputs[0]); lk.new(out_V, n_hadd.inputs[1])
    lk.new(n_hadd.outputs["Vector"], n_hnrm.inputs[0])

    n_nsft = nt.nodes.new("ShaderNodeVectorMath"); n_nsft.operation = 'ADD'; n_nsft.inputs[1].default_value = (0.0, 0.0, -0.21)
    n_nsn  = nt.nodes.new("ShaderNodeVectorMath"); n_nsn.operation = 'NORMALIZE'
    lk.new(out_Ns, n_nsft.inputs[0]); lk.new(n_nsft.outputs["Vector"], n_nsn.inputs[0])

    n_nh = nt.nodes.new("ShaderNodeVectorMath"); n_nh.operation = 'DOT_PRODUCT'
    lk.new(n_nsn.outputs["Vector"], n_nh.inputs[0]); lk.new(n_hnrm.outputs["Vector"], n_nh.inputs[1])

    n_lgate = nt.nodes.new("ShaderNodeMapRange"); n_lgate.interpolation_type = 'SMOOTHSTEP'
    n_lgate.inputs["From Min"].default_value = 0.44; n_lgate.inputs["From Max"].default_value = 0.56
    lk.new(out_NL, n_lgate.inputs["Value"])

    # Secondary wide ice-blue sheen + Primary crisp silver-white angel ring!
    mr_sp1 = nt.nodes.new("ShaderNodeMapRange"); mr_sp1.interpolation_type = 'SMOOTHSTEP'
    mr_sp1.inputs["From Min"].default_value = 0.855; mr_sp1.inputs["From Max"].default_value = 0.915
    mr_sp2 = nt.nodes.new("ShaderNodeMapRange"); mr_sp2.interpolation_type = 'SMOOTHSTEP'
    mr_sp2.inputs["From Min"].default_value = 0.922; mr_sp2.inputs["From Max"].default_value = 0.958
    lk.new(n_nh.outputs["Value"], mr_sp1.inputs["Value"])
    lk.new(n_nh.outputs["Value"], mr_sp2.inputs["Value"])

    m_s1 = nt.nodes.new("ShaderNodeMath"); m_s1.operation = 'MULTIPLY'
    m_s2 = nt.nodes.new("ShaderNodeMath"); m_s2.operation = 'MULTIPLY'
    lk.new(mr_sp1.outputs["Result"], m_s1.inputs[0]); lk.new(n_lgate.outputs["Result"], m_s1.inputs[1])
    lk.new(mr_sp2.outputs["Result"], m_s2.inputs[0]); lk.new(n_lgate.outputs["Result"], m_s2.inputs[1])

    n_sa1 = nt.nodes.new("ShaderNodeMix"); n_sa1.data_type = 'RGBA'; n_sa1.blend_type = 'ADD'
    en_in(n_sa1, "B").default_value = (0.14, 0.24, 0.38, 1.0)
    lk.new(m_s1.outputs["Value"], en_in(n_sa1, "Factor")); lk.new(en_out(n_mul, "Result"), en_in(n_sa1, "A"))

    n_sa2 = nt.nodes.new("ShaderNodeMix"); n_sa2.data_type = 'RGBA'; n_sa2.blend_type = 'ADD'
    en_in(n_sa2, "B").default_value = (0.28, 0.36, 0.46, 1.0)
    lk.new(m_s2.outputs["Value"], en_in(n_sa2, "Factor")); lk.new(en_out(n_sa1, "Result"), en_in(n_sa2, "A"))

    n_radd = nt.nodes.new("ShaderNodeMix"); n_radd.data_type = 'RGBA'; n_radd.blend_type = 'ADD'
    en_in(n_radd, "B").default_value = (0.38, 0.68, 0.96, 1.0)
    n_rf = nt.nodes.new("ShaderNodeMath"); n_rf.operation = 'MULTIPLY'; n_rf.inputs[1].default_value = 0.24
    lk.new(out_rim, n_rf.inputs[0]); lk.new(n_rf.outputs["Value"], en_in(n_radd, "Factor"))
    lk.new(en_out(n_sa2, "Result"), en_in(n_radd, "A"))

    n_gain = nt.nodes.new("ShaderNodeHueSaturation"); n_gain.inputs["Value"].default_value = 0.87
    lk.new(en_out(n_radd, "Result"), n_gain.inputs["Color"])
    lk.new(n_gain.outputs["Color"], n_em.inputs["Color"])
    lk.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

# ------------------------------------------------------------------------------
# 3E) 5 Outfit Shaders: Multi-Row 256x20 Cloth/Metal Ramp + R9 Anti-Washout + MatCap Silver Glints!
# ------------------------------------------------------------------------------
OUTFIT_LIST = [
    ("MAT_V10_Toon_Coat",      "Tex_Coat_4096_V10R15_Clean_PRIVATE.png",      0.13, 0.85),
    ("MAT_V10_Toon_SailorTop", "Tex_SailorTop_2048_V10R15_Clean_PRIVATE.png", 0.09, 0.84),
    ("MAT_V10_Toon_Shorts",    "Tex_Shorts_2048_V10R15_Clean_PRIVATE.png",    0.12, 0.85),
    ("MAT_V10_Toon_SocksAcc",  "Tex_Socks_Acc_2048_V10R15_Clean_PRIVATE.png", 0.10, 0.83),
    ("MAT_V10_Toon_Shoes",     "Tex_Shoes_2048_V10R15_Clean_PRIVATE.png",     0.11, 0.83),
]
for mname, png_name, rim_k, gain_v in OUTFIT_LIST:
    mat = bpy.data.materials.get(mname)
    img = bpy.data.images.get(png_name)
    if not mat or not img: continue
    set_opaque_clean(mat)
    img.colorspace_settings.name = 'sRGB'
    nt = mat.node_tree; nt.nodes.clear(); lk = nt.links
    n_out = nt.nodes.new("ShaderNodeOutputMaterial"); n_em = nt.nodes.new("ShaderNodeEmission")
    n_uv  = nt.nodes.new("ShaderNodeUVMap")
    n_tex = nt.nodes.new("ShaderNodeTexImage"); n_tex.image = img
    lk.new(n_uv.outputs["UV"], n_tex.inputs["Vector"])

    # R9 Cloth Saturation Recovery (1.06)!
    n_hsv = nt.nodes.new("ShaderNodeHueSaturation"); n_hsv.inputs["Saturation"].default_value = 1.06
    lk.new(n_tex.outputs["Color"], n_hsv.inputs["Color"])

    _, out_Ns, out_V, out_Lc, out_NL, out_rim = add_key_light_and_rim_nodes(nt, smooth_bias=(0.0, -0.26, 0.46), smooth_fac=0.15, rim_exp=7.5)

    # Detect material sub-family per pixel from HSV:
    # - Silver Metallic Hardware (buttons, O-ring, whale clip, choker clasp, eyelets): Sat < 0.14 & Val in [0.42..0.87]
    # - Colored Blue/Navy Cloth: Sat >= 0.14 -> V_row = 0.50
    # - White Sailor/Sock Cloth: Sat < 0.14 & Val > 0.87 -> V_row = 0.82
    n_sep = nt.nodes.new("ShaderNodeSeparateColor"); n_sep.mode = 'HSV'
    lk.new(n_tex.outputs["Color"], n_sep.inputs["Color"])

    mr_low_s = nt.nodes.new("ShaderNodeMapRange"); mr_low_s.interpolation_type = 'SMOOTHSTEP'
    mr_low_s.inputs["From Min"].default_value = 0.16; mr_low_s.inputs["From Max"].default_value = 0.08
    lk.new(n_sep.outputs["Green"], mr_low_s.inputs["Value"])  # 1 when neutral grey/white

    mr_mid_v1 = nt.nodes.new("ShaderNodeMapRange"); mr_mid_v1.interpolation_type = 'SMOOTHSTEP'
    mr_mid_v1.inputs["From Min"].default_value = 0.36; mr_mid_v1.inputs["From Max"].default_value = 0.48
    mr_mid_v2 = nt.nodes.new("ShaderNodeMapRange"); mr_mid_v2.interpolation_type = 'SMOOTHSTEP'
    mr_mid_v2.inputs["From Min"].default_value = 0.89; mr_mid_v2.inputs["From Max"].default_value = 0.81
    lk.new(n_sep.outputs["Blue"], mr_mid_v1.inputs["Value"])
    lk.new(n_sep.outputs["Blue"], mr_mid_v2.inputs["Value"])

    m_mv = nt.nodes.new("ShaderNodeMath"); m_mv.operation = 'MULTIPLY'
    m_metal = nt.nodes.new("ShaderNodeMath"); m_metal.operation = 'MULTIPLY'
    lk.new(mr_mid_v1.outputs["Result"], m_mv.inputs[0]); lk.new(mr_mid_v2.outputs["Result"], m_mv.inputs[1])
    lk.new(m_mv.outputs["Value"], m_metal.inputs[0]); lk.new(mr_low_s.outputs["Result"], m_metal.inputs[1])

    # Row selector V in 256x20 Cloth Ramp: White=0.82, Blue=0.50, Metal=0.14
    mix_wb = nt.nodes.new("ShaderNodeMix"); mix_wb.data_type = 'FLOAT'
    en_in(mix_wb, "A").default_value = 0.50; en_in(mix_wb, "B").default_value = 0.82
    lk.new(mr_low_s.outputs["Result"], en_in(mix_wb, "Factor"))

    mix_row = nt.nodes.new("ShaderNodeMix"); mix_row.data_type = 'FLOAT'
    en_in(mix_row, "B").default_value = 0.14
    lk.new(m_metal.outputs["Value"], en_in(mix_row, "Factor"))
    lk.new(en_out(mix_wb, "Result"), en_in(mix_row, "A"))

    mr_c = nt.nodes.new("ShaderNodeMapRange"); mr_c.interpolation_type = 'SMOOTHSTEP'
    mr_c.inputs["From Min"].default_value = 0.43; mr_c.inputs["From Max"].default_value = 0.59
    mr_c.inputs["To Min"].default_value = 0.10;   mr_c.inputs["To Max"].default_value = 0.96
    lk.new(out_NL, mr_c.inputs["Value"])

    c_ramp = add_256x20_ramp_sampler(nt, img_ramp_clth, mr_c.outputs["Result"], en_out(mix_row, "Result"))

    n_mul = nt.nodes.new("ShaderNodeMix"); n_mul.data_type = 'RGBA'; n_mul.blend_type = 'MULTIPLY'
    en_in(n_mul, "Factor").default_value = 1.0
    lk.new(n_hsv.outputs["Color"], en_in(n_mul, "A")); lk.new(c_ramp, en_in(n_mul, "B"))

    # MatCap Metallic Specular Glint strictly on Silver Hardware Islands (buttons, O-ring, choker, whale pin clip)!
    n_hadd = nt.nodes.new("ShaderNodeVectorMath"); n_hadd.operation = 'ADD'
    n_hnrm = nt.nodes.new("ShaderNodeVectorMath"); n_hnrm.operation = 'NORMALIZE'
    lk.new(out_Lc, n_hadd.inputs[0]); lk.new(out_V, n_hadd.inputs[1])
    lk.new(n_hadd.outputs["Vector"], n_hnrm.inputs[0])
    n_nh = nt.nodes.new("ShaderNodeVectorMath"); n_nh.operation = 'DOT_PRODUCT'
    lk.new(out_Ns, n_nh.inputs[0]); lk.new(n_hnrm.outputs["Vector"], n_nh.inputs[1])

    mr_mspec = nt.nodes.new("ShaderNodeMapRange"); mr_mspec.interpolation_type = 'SMOOTHSTEP'
    mr_mspec.inputs["From Min"].default_value = 0.86; mr_mspec.inputs["From Max"].default_value = 0.97
    lk.new(n_nh.outputs["Value"], mr_mspec.inputs["Value"])
    m_mg = nt.nodes.new("ShaderNodeMath"); m_mg.operation = 'MULTIPLY'
    lk.new(mr_mspec.outputs["Result"], m_mg.inputs[0]); lk.new(m_metal.outputs["Value"], m_mg.inputs[1])

    n_madd = nt.nodes.new("ShaderNodeMix"); n_madd.data_type = 'RGBA'; n_madd.blend_type = 'ADD'
    en_in(n_madd, "B").default_value = (0.52, 0.58, 0.70, 1.0)
    lk.new(m_mg.outputs["Value"], en_in(n_madd, "Factor"))
    lk.new(en_out(n_mul, "Result"), en_in(n_madd, "A"))

    # Subtle narrow contour rim (R9 anti-washout rule: rim_k = 0.08..0.11!)
    n_radd = nt.nodes.new("ShaderNodeMix"); n_radd.data_type = 'RGBA'; n_radd.blend_type = 'ADD'
    en_in(n_radd, "B").default_value = (0.36, 0.66, 0.96, 1.0)
    n_rf = nt.nodes.new("ShaderNodeMath"); n_rf.operation = 'MULTIPLY'; n_rf.inputs[1].default_value = rim_k
    lk.new(out_rim, n_rf.inputs[0]); lk.new(n_rf.outputs["Value"], en_in(n_radd, "Factor"))
    lk.new(en_out(n_madd, "Result"), en_in(n_radd, "A"))

    n_gain = nt.nodes.new("ShaderNodeHueSaturation"); n_gain.inputs["Value"].default_value = gain_v
    lk.new(en_out(n_radd, "Result"), n_gain.inputs["Color"])
    lk.new(n_gain.outputs["Color"], n_em.inputs["Color"])
    lk.new(n_em.outputs["Emission"], n_out.inputs["Surface"])


print("=== [4/6] Refining Single Thin Plum-Brown Outlines (odette.json Spec + Zero Brow Outline) ===", flush=True)
mat_out_hair = bpy.data.materials.get("MAT_NPR_Outline_Hair")
if mat_out_hair is None:
    mat_out_hair = bpy.data.materials["MAT_NPR_Outline_Cool"].copy()
    mat_out_hair.name = "MAT_NPR_Outline_Hair"

for om, col in [
    (bpy.data.materials["MAT_NPR_Outline_Skin"], (0.145, 0.075, 0.105, 1.0)),  # Warm plum-rose thin face line
    (mat_out_hair,                               (0.115, 0.082, 0.145, 1.0)),  # Deep plum-indigo hair line (odette.json!)
    (bpy.data.materials["MAT_NPR_Outline_Cool"], (0.082, 0.095, 0.175, 1.0)),  # Crisp navy-slate garment line
]:
    set_opaque_clean(om)
    om.use_backface_culling = True
    for nd in om.node_tree.nodes:
        if nd.type == 'EMISSION':
            nd.inputs["Color"].default_value = col

for o in bpy.data.objects:
    if o.type != 'MESH':
        continue
    if o.name == "GEO-Hair-Brow_SeeThrough":
        for md in o.modifiers:
            if md.name == "NPR_Outline":
                md.show_render = False
                md.show_viewport = False
    elif o.name == "GEO-Sotai_Skin":
        for md in o.modifiers:
            if md.name == "NPR_Outline":
                md.thickness = 0.00058
    elif o.name == "GEO-Head-Ears":
        for md in o.modifiers:
            if md.name == "NPR_Outline":
                md.thickness = 0.00065
        for si, m in enumerate(o.data.materials):
            if m and m.name == "MAT_NPR_Outline_Cool":
                o.data.materials[si] = bpy.data.materials["MAT_NPR_Outline_Skin"]
    elif o.name.startswith("GEO-Hair-"):
        for md in o.modifiers:
            if md.name == "NPR_Outline":
                md.thickness = 0.00078
        for si, m in enumerate(o.data.materials):
            if m and m.name == "MAT_NPR_Outline_Cool":
                o.data.materials[si] = mat_out_hair
    elif o.name.startswith(("GEO-Cloth_", "GEO-Shoe_")):
        for md in o.modifiers:
            if md.name == "NPR_Outline":
                md.thickness = 0.00110


print("=== [5/6] Calibrating EEVEE 48-Sample Lossless PNG + Festivity-Style Bloom & Saving V10R19 ===", flush=True)
scn.camera = bpy.data.objects["MMD_MV_CAMERA"]
scn.render.engine = 'BLENDER_EEVEE'
scn.view_settings.view_transform = 'Standard'
scn.view_settings.look = 'Medium High Contrast'
scn.view_settings.exposure = -0.02
scn.view_settings.gamma = 1.0

ev = scn.eevee
ev.taa_render_samples = 48
ev.use_raytracing = True
ev.ray_tracing_method = 'SCREEN'
ev.use_fast_gi = False
ev.volumetric_samples = 32
ev.volumetric_tile_size = '8'
ev.use_volumetric_shadows = False
ev.use_shadows = True
ev.shadow_pool_size = '512'
ev.shadow_resolution_scale = 0.68

cg = scn.compositing_node_group
if cg:
    gl_nodes = [n for n in cg.nodes if n.type == 'GLARE']
    if gl_nodes:
        gl = gl_nodes[0]
        gl.inputs["Type"].default_value = 'Bloom'
        gl.inputs["Quality"].default_value = 'High'
        gl.inputs["Threshold"].default_value = 0.88
        gl.inputs["Smoothness"].default_value = 0.22
        gl.inputs["Strength"].default_value = 0.36
        gl.inputs["Size"].default_value = 0.45

scn.render.resolution_x = 1920
scn.render.resolution_y = 1080
scn.render.resolution_percentage = 100
scn.render.fps = 30
scn.render.image_settings.file_format = 'PNG'
scn.render.image_settings.color_mode = 'RGB'
scn.render.image_settings.compression = 15

bpy.ops.wm.save_as_mainfile(filepath=str(DST_R19), compress=True)
print(f"SAVED_V10R19_MASTER: {DST_R19.name} ({DST_R19.stat().st_size / 1048576:.2f} MB)", flush=True)

print("=== [6/6] Rendering 6x 1080p Lossless PNG Close-Up & Full-Body Audit Shots ===", flush=True)
audit_shots = [
    (111,  "f0111_closeup_front"),
    (451,  "f0451_closeup_low"),
    (1271, "f1271_closeup_34side"),
    (241,  "f0241_bust_whalepin"),
    (586,  "f0586_mid_crystal"),
    (1086, "f1086_fullbody_kick"),
]
times = []
for idx, (fno, tag) in enumerate(audit_shots):
    ts0 = time.time()
    scn.frame_set(fno)
    bpy.context.view_layer.update()
    scn.render.filepath = str(QA_DIR / f"{tag}.png")
    bpy.ops.render.render(write_still=True)
    dt = time.time() - ts0
    times.append(dt)
    print(f"  AUDIT_SHOT {idx+1}/6 {tag}: {dt:.2f}s", flush=True)

print(f"V10R19_BUILD_DONE: avg_render={sum(times[1:])/max(1,len(times)-1):.3f}s/f total_elapsed={time.time()-t0:.1f}s", flush=True)
