import math, random, time
from pathlib import Path
import numpy as np
import bpy, bmesh, addon_utils
from mathutils import Vector, Matrix, Euler

t_start = time.time()
ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
WDIR = ROOT / "生成模型" / "组装" / "v010R11_MMD_三渲二舞台_PRIVATE"
SRC_R17 = WDIR / "RadDoll_V10R17_FullSong6265_MMD_Heart舞台三渲二零交拧精修母带_PRIVATE.blend"
THEATER_BLEND = WDIR / "mmd_cache_PRIVATE" / "korolevskiy_theater" / "The Korolevskiy Theater.blend"
DST_R18 = WDIR / "RadDoll_V10R18_FullSong6265_KorolevskiyTheater_HSR_MMD_Master_PRIVATE.blend"
PREVIEW_DIR = WDIR / "qa_v10r18_theater_hsr"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

ora = np.load(str(WDIR / "oracle_full_6265.npz"), allow_pickle=True)
hips_pos = ora["hips_pos"]  # (6265, 3)

addon_utils.enable("bl_ext.blender_org.mmd_tools", default_set=False)
bpy.ops.wm.open_mainfile(filepath=str(SRC_R17))
scn = bpy.context.scene

print("=== [1/6] Removing cramped Heart box stage & legacy flat lights ===", flush=True)
for old_name in ["heart_mesh", "heart_arm", "heart", "NPR_KeySun", "NPR_NeonBackFill"]:
    ob = bpy.data.objects.get(old_name)
    if ob:
        bpy.data.objects.remove(ob, do_unlink=True)

for old_mat in [f"材質{i}" for i in range(9)]:
    m = bpy.data.materials.get(old_mat)
    if m and m.users == 0:
        bpy.data.materials.remove(m)

print("=== [2/6] Appending The Korolevskiy Theater (Scene + Lights + fog + World + Compositor) ===", flush=True)
with bpy.data.libraries.load(str(THEATER_BLEND), link=False) as (data_from, data_to):
    data_to.collections = [c for c in ("Scene", "Lights", "fog") if c in data_from.collections]
    data_to.worlds = list(data_from.worlds)
    data_to.node_groups = [ng for ng in data_from.node_groups if "Compositing" in ng]

th_master_col = bpy.data.collections.new("STAGE_Korolevskiy_Theater")
scn.collection.children.link(th_master_col)

th_root = bpy.data.objects.new("KOROLEVSKIY_THEATER_ROOT", None)
th_root.empty_display_type = 'ARROWS'
th_root.empty_display_size = 2.0
th_master_col.objects.link(th_root)

all_th_objs = set()
for col in data_to.collections:
    if col is None:
        continue
    th_master_col.children.link(col)
    for o in col.all_objects:
        all_th_objs.add(o)

# CRITICAL: Must update view_layer BEFORE reading matrix_world on freshly appended objects!
bpy.context.view_layer.update()

for o in all_th_objs:
    if o.parent is None:
        mw = o.matrix_world.copy()
        o.parent = th_root
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.matrix_basis = mw

# Rotate 180 deg around Z + DY=-3.0m, DZ=+0.002m -> 0/6265 blocked frames, crystal star-window behind dancer!
th_root.location = (0.0, -3.0, 0.002)
th_root.rotation_euler = Euler((0.0, 0.0, math.pi), 'XYZ')
bpy.context.view_layer.update()

if data_to.worlds and data_to.worlds[0]:
    scn.world = data_to.worlds[0]

if data_to.node_groups and data_to.node_groups[0]:
    scn.use_nodes = True
    scn.compositing_node_group = data_to.node_groups[0]

print("=== [3/6] Optimizing Theater Lights, Grounding Contact AO & HSR/Ray-MMD Volumetric VFX ===", flush=True)
primary_shadow_names = {"Point", "Point.011", "Spot"}
kept_sh = 0
for o in all_th_objs:
    if o.type == 'LIGHT':
        if o.data.energy >= 1000.0 and hasattr(o.data, "specular_factor"):
            # Keep specular reflections only from physical 300W golden lanterns & SSR architecture, zero fake orb blobs!
            o.data.specular_factor = 0.0
        if o.name.split(".")[0] == "Spot" or o.name in primary_shadow_names:
            o.data.use_shadow = True
            kept_sh += 1
        else:
            o.data.use_shadow = False

# Calibrate close-up stage volume haze so inside-stage cameras stay crystal-clear HD while keeping atmospheric depth!
for fn_v in ("Cube", "Cube.001"):
    fo = bpy.data.objects.get(fn_v)
    if fo and fo.type == 'MESH':
        for vm in fo.data.materials:
            if vm and vm.node_tree:
                for vn in vm.node_tree.nodes:
                    if vn.type == 'PRINCIPLED_VOLUME':
                        vn.inputs["Density"].default_value *= 0.55
                        vn.inputs["Emission Strength"].default_value *= 0.55
                    elif vn.type == 'EMISSION':
                        vn.inputs["Strength"].default_value *= 0.35

# Keep stage floor rich deep sapphire with smooth water-mirror sheen (no harsh spotlight blowout!)
for o in all_th_objs:
    if o.type == 'MESH' and o.name.startswith("mesh_80"):
        for m in o.data.materials:
            if m and m.node_tree:
                for n in m.node_tree.nodes:
                    if n.type == 'BSDF_PRINCIPLED':
                        n.inputs["Roughness"].default_value = 0.19

vfx_col = bpy.data.collections.new("VFX_HSR_Genshin_Stage")
scn.collection.children.link(vfx_col)

# Build HoYoverse-style soft Grounding Contact Shadow Discs (under dancer & under white chair — zero white puddle!)
def build_contact_ao_mat(name, max_dark=0.52):
    mat = bpy.data.materials.new(name)
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'BLENDED'
    nt = mat.node_tree; nt.nodes.clear(); lk = nt.links
    no = nt.nodes.new("ShaderNodeOutputMaterial")
    nm = nt.nodes.new("ShaderNodeMixShader")
    ntr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nem = nt.nodes.new("ShaderNodeEmission")
    nem.inputs["Color"].default_value = (0.015, 0.025, 0.085, 1.0)  # Deep royal navy shadow
    nem.inputs["Strength"].default_value = 1.0

    nuv = nt.nodes.new("ShaderNodeUVMap")
    nvsub = nt.nodes.new("ShaderNodeVectorMath"); nvsub.operation = 'SUBTRACT'
    nvsub.inputs[1].default_value = (0.5, 0.5, 0.0)
    nvlen = nt.nodes.new("ShaderNodeVectorMath"); nvlen.operation = 'LENGTH'
    nmr = nt.nodes.new("ShaderNodeMapRange"); nmr.interpolation_type = 'SMOOTHSTEP'
    nmr.inputs["From Min"].default_value = 0.50
    nmr.inputs["From Max"].default_value = 0.04
    nmr.inputs["To Min"].default_value = 0.0
    nmr.inputs["To Max"].default_value = max_dark

    lk.new(nuv.outputs["UV"], nvsub.inputs[0])
    lk.new(nvsub.outputs["Vector"], nvlen.inputs[0])
    lk.new(nvlen.outputs["Value"], nmr.inputs["Value"])
    lk.new(nmr.outputs["Result"], nm.inputs["Fac"])
    lk.new(ntr.outputs["BSDF"], nm.inputs[1])
    lk.new(nem.outputs["Emission"], nm.inputs[2])
    lk.new(nm.outputs["Shader"], no.inputs["Surface"])
    return mat

def make_uv_disc(name, radius, mat):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=24, y_segments=24, size=radius)
    uv_lay = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for lp in f.loops:
            lp[uv_lay].uv = (lp.vert.co.x / (2.0 * radius) + 0.5, lp.vert.co.y / (2.0 * radius) + 0.5)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    if hasattr(ob, "visible_shadow"):
        ob.visible_shadow = False
    vfx_col.objects.link(ob)
    return ob

mat_ao_dancer = build_contact_ao_mat("MAT_HSR_ContactAO_Dancer", max_dark=0.55)
mat_ao_chair  = build_contact_ao_mat("MAT_HSR_ContactAO_Chair",  max_dark=0.48)

ob_ao_chair = make_uv_disc("HSR_ContactAO_Chair", 0.58, mat_ao_chair)
ob_ao_chair.location = (-0.02, 0.55, 0.004)

ob_ao_dancer = make_uv_disc("HSR_ContactAO_Dancer", 0.62, mat_ao_dancer)
ob_ao_dancer.animation_data_create()
act_ao = bpy.data.actions.new(name="ACT_HSR_ContactAO_Dancer")
ob_ao_dancer.animation_data.action = act_ao
for f in range(1, 6266, 2):
    idx = f - 1
    ob_ao_dancer.location = (float(hips_pos[idx, 0]), float(hips_pos[idx, 1]), 0.005)
    ob_ao_dancer.keyframe_insert(data_path="location", frame=f)

# Build 4 translucent volumetric spotlight shafts (Tyndall stage beams)
def build_beam_mat(name, col_rgb, peak_alpha=0.036):
    mat = bpy.data.materials.new(name)
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'BLENDED'
    nt = mat.node_tree; nt.nodes.clear(); lk = nt.links
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_mix = nt.nodes.new("ShaderNodeMixShader")
    n_tr  = nt.nodes.new("ShaderNodeBsdfTransparent")
    n_em  = nt.nodes.new("ShaderNodeEmission")
    n_em.inputs["Color"].default_value = (*col_rgb, 1.0)
    n_em.inputs["Strength"].default_value = 1.25

    n_geo = nt.nodes.new("ShaderNodeNewGeometry")
    n_lw  = nt.nodes.new("ShaderNodeLayerWeight"); n_lw.inputs["Blend"].default_value = 0.5
    n_sub = nt.nodes.new("ShaderNodeMath"); n_sub.operation = 'SUBTRACT'; n_sub.inputs[0].default_value = 1.0
    n_pow = nt.nodes.new("ShaderNodeMath"); n_pow.operation = 'POWER'; n_pow.inputs[1].default_value = 2.8

    n_sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    n_mr1 = nt.nodes.new("ShaderNodeMapRange"); n_mr1.inputs["From Min"].default_value = 0.4; n_mr1.inputs["From Max"].default_value = 4.0
    n_mr2 = nt.nodes.new("ShaderNodeMapRange"); n_mr2.inputs["From Min"].default_value = 26.0; n_mr2.inputs["From Max"].default_value = 12.0
    n_mul1 = nt.nodes.new("ShaderNodeMath"); n_mul1.operation = 'MULTIPLY'
    n_mul2 = nt.nodes.new("ShaderNodeMath"); n_mul2.operation = 'MULTIPLY'
    n_mul3 = nt.nodes.new("ShaderNodeMath"); n_mul3.operation = 'MULTIPLY'; n_mul3.inputs[1].default_value = peak_alpha

    lk.new(n_lw.outputs["Fresnel"], n_sub.inputs[1])
    lk.new(n_sub.outputs["Value"], n_pow.inputs[0])
    lk.new(n_geo.outputs["Position"], n_sep.inputs["Vector"])
    lk.new(n_sep.outputs["Z"], n_mr1.inputs["Value"])
    lk.new(n_sep.outputs["Z"], n_mr2.inputs["Value"])
    lk.new(n_mr1.outputs["Result"], n_mul1.inputs[0])
    lk.new(n_mr2.outputs["Result"], n_mul1.inputs[1])
    lk.new(n_pow.outputs["Value"], n_mul2.inputs[0])
    lk.new(n_mul1.outputs["Value"], n_mul2.inputs[1])
    lk.new(n_mul2.outputs["Value"], n_mul3.inputs[0])
    lk.new(n_mul3.outputs["Value"], n_mix.inputs["Fac"])
    lk.new(n_tr.outputs["BSDF"], n_mix.inputs[1])
    lk.new(n_em.outputs["Emission"], n_mix.inputs[2])
    lk.new(n_mix.outputs["Shader"], n_out.inputs["Surface"])
    return mat

mat_beam_blue = build_beam_mat("MAT_VFX_Beam_Sapphire", (0.42, 0.72, 1.00), peak_alpha=0.036)
mat_beam_gold = build_beam_mat("MAT_VFX_Beam_Champagne", (1.00, 0.84, 0.58), peak_alpha=0.025)

beam_specs = [
    ("VFX_Beam_L1", (-10.5, 10.5, 25.5), (-1.8, 1.5, 0.0), 0.55, 4.2, mat_beam_blue),
    ("VFX_Beam_R1", ( 10.5, 10.5, 25.5), ( 1.8, 1.5, 0.0), 0.55, 4.2, mat_beam_blue),
    ("VFX_Beam_L2", (-15.0,  6.5, 24.5), (-4.5, 0.5, 0.0), 0.50, 4.8, mat_beam_gold),
    ("VFX_Beam_R2", ( 15.0,  6.5, 24.5), ( 4.5, 0.5, 0.0), 0.50, 4.8, mat_beam_gold),
]
for bname, p_top, p_bot, r_top, r_bot, bmat in beam_specs:
    v_top = Vector(p_top); v_bot = Vector(p_bot)
    mid = 0.5 * (v_top + v_bot)
    depth = (v_top - v_bot).length
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False, segments=32, radius1=r_bot, radius2=r_top, depth=depth)
    me = bpy.data.meshes.new(bname)
    bm.to_mesh(me); bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    me.materials.append(bmat)
    ob = bpy.data.objects.new(bname, me)
    ob.location = mid
    ob.rotation_euler = (v_top - v_bot).to_track_quat('Z', 'Y').to_euler()
    if hasattr(ob, "visible_shadow"):
        ob.visible_shadow = False
    vfx_col.objects.link(ob)

# Pinpoint distant stage starlight motes strictly in upper/back stage arch (never near front/rear camera paths!)
mat_mote = bpy.data.materials.new("MAT_VFX_StarlightMotes")
nt_m = mat_mote.node_tree; nt_m.nodes.clear()
mo_out = nt_m.nodes.new("ShaderNodeOutputMaterial")
mo_em  = nt_m.nodes.new("ShaderNodeEmission")
mo_vc  = nt_m.nodes.new("ShaderNodeVertexColor"); mo_vc.layer_name = "MoteCol"
mo_em.inputs["Strength"].default_value = 1.85
nt_m.links.new(mo_vc.outputs["Color"], mo_em.inputs["Color"])
nt_m.links.new(mo_em.outputs["Emission"], mo_out.inputs["Surface"])

rng = random.Random(20261023)
bm_p = bmesh.new()
cl_layer = bm_p.verts.layers.color.new("MoteCol")
for i in range(140):
    px = rng.uniform(-16.0, 16.0)
    py = rng.uniform(4.5, 15.5)
    # Keep clear of rear camera corridor (|x|<4, y<11.5, z<4.5)
    pz = rng.uniform(4.6, 12.5) if (abs(px) < 5.0 and py < 12.0) else rng.uniform(1.2, 11.5)
    sz = rng.uniform(0.007, 0.014)
    ret = bmesh.ops.create_icosphere(bm_p, subdivisions=1, radius=sz, matrix=Matrix.Translation((px, py, pz)))
    col = (1.0, 0.85, 0.58, 1.0) if (i % 3 == 0) else (0.52, 0.82, 1.0, 1.0)
    for v in ret["verts"]:
        v[cl_layer] = col

me_motes = bpy.data.meshes.new("VFX_StarlightMotes_Mesh")
bm_p.to_mesh(me_motes); bm_p.free()
me_motes.materials.append(mat_mote)
ob_motes = bpy.data.objects.new("VFX_StarlightMotes", me_motes)
if hasattr(ob_motes, "visible_shadow"):
    ob_motes.visible_shadow = False
vfx_col.objects.link(ob_motes)

ob_motes.animation_data_create()
ob_motes.animation_data.action = bpy.data.actions.new(name="ACT_VFX_StarlightMotes")
ob_motes.location = (0.0, 0.0, 0.0)
ob_motes.keyframe_insert(data_path="location", frame=1)
ob_motes.location = (0.35, 0.0, 0.85)
ob_motes.keyframe_insert(data_path="location", frame=6265)


print("=== [4/6] Upgrading All Character & Chair Shaders (True Genshin/HSR Contrast-Calibrated 3D Cel) ===", flush=True)
def en_in(nd, nm):
    return [sk for sk in nd.inputs if sk.enabled and sk.name == nm][0]
def en_out(nd, nm):
    return [sk for sk in nd.outputs if sk.enabled and sk.name == nm][0]

def add_hsr_light_rig_nodes(nt, smooth_bias=(0.0, -0.32, 0.52), smooth_fac=0.18):
    lk = nt.links
    n_geo = nt.nodes.new("ShaderNodeNewGeometry")

    n_nmix = nt.nodes.new("ShaderNodeMix"); n_nmix.data_type = 'VECTOR'
    en_in(n_nmix, "Factor").default_value = smooth_fac
    en_in(n_nmix, "B").default_value = smooth_bias
    n_nnrm = nt.nodes.new("ShaderNodeVectorMath"); n_nnrm.operation = 'NORMALIZE'
    lk.new(n_geo.outputs["Normal"], en_in(n_nmix, "A"))
    lk.new(en_out(n_nmix, "Result"), n_nnrm.inputs[0])

    # Camera-offset key (40% camera-side-top + 60% fixed theatrical 3/4 key so every pose has rich 55/45 light/shadow sculpting!)
    n_cadd = nt.nodes.new("ShaderNodeVectorMath"); n_cadd.operation = 'ADD'
    n_cadd.inputs[1].default_value = (0.42, 0.0, 0.36)
    n_cnrm = nt.nodes.new("ShaderNodeVectorMath"); n_cnrm.operation = 'NORMALIZE'
    lk.new(n_geo.outputs["Incoming"], n_cadd.inputs[0])
    lk.new(n_cadd.outputs["Vector"], n_cnrm.inputs[0])

    n_lmix = nt.nodes.new("ShaderNodeMix"); n_lmix.data_type = 'VECTOR'
    en_in(n_lmix, "Factor").default_value = 0.38
    en_in(n_lmix, "A").default_value = (0.58, -0.62, 0.52)
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

    # Tight outer-contour Fresnel rim: pow(clamp(1 - abs(dot(N_s, Incoming))), 5.8)
    n_vd  = nt.nodes.new("ShaderNodeVectorMath"); n_vd.operation = 'DOT_PRODUCT'
    n_abs = nt.nodes.new("ShaderNodeMath"); n_abs.operation = 'ABSOLUTE'
    n_inv = nt.nodes.new("ShaderNodeMath"); n_inv.operation = 'SUBTRACT'; n_inv.inputs[0].default_value = 1.0; n_inv.use_clamp = True
    n_pow = nt.nodes.new("ShaderNodeMath"); n_pow.operation = 'POWER'; n_pow.inputs[1].default_value = 5.8
    n_sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    n_upm = nt.nodes.new("ShaderNodeMapRange")
    n_upm.inputs["From Min"].default_value = -0.35
    n_upm.inputs["From Max"].default_value = 0.45
    n_upm.inputs["To Min"].default_value = 0.20
    n_upm.inputs["To Max"].default_value = 1.0
    n_rmul = nt.nodes.new("ShaderNodeMath"); n_rmul.operation = 'MULTIPLY'

    lk.new(n_nnrm.outputs["Vector"], n_vd.inputs[0])
    lk.new(n_geo.outputs["Incoming"], n_vd.inputs[1])
    lk.new(n_vd.outputs["Value"], n_abs.inputs[0])
    lk.new(n_abs.outputs["Value"], n_inv.inputs[1])
    lk.new(n_inv.outputs["Value"], n_pow.inputs[0])
    lk.new(n_nnrm.outputs["Vector"], n_sep.inputs["Vector"])
    lk.new(n_sep.outputs["Z"], n_upm.inputs["Value"])
    lk.new(n_pow.outputs["Value"], n_rmul.inputs[0])
    lk.new(n_upm.outputs["Result"], n_rmul.inputs[1])

    return n_geo, n_nnrm.outputs["Vector"], n_geo.outputs["Incoming"], n_lnrm.outputs["Vector"], n_mad.outputs["Value"], n_rmul.outputs["Value"]

# 4A) Skin Shaders: Calibrated linear [0.34 -> 0.54 SSS -> 0.62 Lit] so Medium High Contrast renders creamy peach-ivory + sculpted shadow!
SKIN_SPECS = [
    ("RD_skin",      "RadDoll_face_tex.png", True,  0.42),
    ("RD_body_skin", "RadDoll_body_tex_v2",  True,  0.18),
    ("RD_ear",       "Tex_Ears_512",         False, 0.22),
]
for mname, img_name, has_subchin, sfac in SKIN_SPECS:
    mat = bpy.data.materials.get(mname)
    img = bpy.data.images.get(img_name)
    if not mat or not img:
        continue
    nt = mat.node_tree; nt.nodes.clear(); lk = nt.links
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_em  = nt.nodes.new("ShaderNodeEmission")
    n_tex = nt.nodes.new("ShaderNodeTexImage"); n_tex.image = img

    _, _, _, _, out_NL, out_rim = add_hsr_light_rig_nodes(
        nt, smooth_bias=(0.0, -0.45, 0.40) if mname == "RD_skin" else (0.0, -0.28, 0.45),
        smooth_fac=sfac
    )

    n_ramp = nt.nodes.new("ShaderNodeValToRGB")
    n_ramp.color_ramp.interpolation = 'EASE'
    els = n_ramp.color_ramp.elements
    els[0].position = 0.48
    els[0].color = (0.37, 0.30, 0.40, 1.0)      # Rich theatrical rose-violet shadow (sculpts legs, jaw, fingers!)
    e_sss = els.new(0.545)
    e_sss.color = (0.58, 0.40, 0.39, 1.0)       # Genshin/HSR coral-peach subsurface terminator band!
    els[1].position = 0.61
    els[1].color = (0.62, 0.56, 0.56, 1.0)      # Creamy warm anime skin (~0.87 sRGB post-contrast, zero whiteout!)
    lk.new(out_NL, n_ramp.inputs["Factor"])

    n_mul = nt.nodes.new("ShaderNodeMix"); n_mul.data_type = 'RGBA'; n_mul.blend_type = 'MULTIPLY'
    en_in(n_mul, "Factor").default_value = 1.0
    lk.new(n_tex.outputs["Color"], en_in(n_mul, "A"))
    lk.new(n_ramp.outputs["Color"], en_in(n_mul, "B"))
    cur_col = en_out(n_mul, "Result")

    if has_subchin:
        n_vc  = nt.nodes.new("ShaderNodeVertexColor"); n_vc.layer_name = "NPR_SubChin"
        n_spc = nt.nodes.new("ShaderNodeSeparateColor")
        lk.new(n_vc.outputs["Color"], n_spc.inputs["Color"])
        n_sc = nt.nodes.new("ShaderNodeMix"); n_sc.data_type = 'RGBA'; n_sc.blend_type = 'MULTIPLY'
        en_in(n_sc, "B").default_value = (0.68, 0.55, 0.62, 1.0)
        n_scf = nt.nodes.new("ShaderNodeMath"); n_scf.operation = 'MULTIPLY'; n_scf.inputs[1].default_value = 0.85
        lk.new(n_spc.outputs["Red"], n_scf.inputs[0])
        lk.new(n_scf.outputs["Value"], en_in(n_sc, "Factor"))
        lk.new(cur_col, en_in(n_sc, "A"))
        cur_col = en_out(n_sc, "Result")

        n_bl = nt.nodes.new("ShaderNodeMix"); n_bl.data_type = 'RGBA'; n_bl.blend_type = 'MIX'
        en_in(n_bl, "B").default_value = (0.64, 0.39, 0.40, 1.0)
        n_blf = nt.nodes.new("ShaderNodeMath"); n_blf.operation = 'MULTIPLY'; n_blf.inputs[1].default_value = 0.24
        lk.new(n_spc.outputs["Green"], n_blf.inputs[0])
        lk.new(n_blf.outputs["Value"], en_in(n_bl, "Factor"))
        lk.new(cur_col, en_in(n_bl, "A"))
        cur_col = en_out(n_bl, "Result")

    n_radd = nt.nodes.new("ShaderNodeMix"); n_radd.data_type = 'RGBA'; n_radd.blend_type = 'ADD'
    en_in(n_radd, "B").default_value = (0.34, 0.58, 0.88, 1.0)
    n_rfac = nt.nodes.new("ShaderNodeMath"); n_rfac.operation = 'MULTIPLY'; n_rfac.inputs[1].default_value = 0.26
    lk.new(out_rim, n_rfac.inputs[0])
    lk.new(n_rfac.outputs["Value"], en_in(n_radd, "Factor"))
    lk.new(cur_col, en_in(n_radd, "A"))

    lk.new(en_out(n_radd, "Result"), n_em.inputs["Color"])
    lk.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

# 4B) Hair Shader: Rich sapphire-periwinkle shadow + crisp silver-pearl lit + distinct Angel-Ring + glowing cyan rim
mat_hair = bpy.data.materials.get("RD_hair")
img_hair = bpy.data.images.get("Tex_Hair_Atlas_2048")
if mat_hair and img_hair:
    nt = mat_hair.node_tree; nt.nodes.clear(); lk = nt.links
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_em  = nt.nodes.new("ShaderNodeEmission")
    n_uv  = nt.nodes.new("ShaderNodeUVMap"); n_uv.uv_map = "UVMap"
    n_tex = nt.nodes.new("ShaderNodeTexImage"); n_tex.image = img_hair
    lk.new(n_uv.outputs["UV"], n_tex.inputs["Vector"])

    n_geo, out_Ns, out_V, out_Lc, out_NL, out_rim = add_hsr_light_rig_nodes(
        nt, smooth_bias=(0.0, -0.22, 0.42), smooth_fac=0.22
    )

    n_ramp = nt.nodes.new("ShaderNodeValToRGB")
    n_ramp.color_ramp.interpolation = 'EASE'
    els = n_ramp.color_ramp.elements
    els[0].position = 0.45
    els[0].color = (0.32, 0.39, 0.58, 1.0)      # Deep sapphire-periwinkle strand shadow
    e_mid = els.new(0.53)
    e_mid.color = (0.48, 0.58, 0.78, 1.0)       # Sky-ice transition band
    els[1].position = 0.61
    els[1].color = (0.66, 0.70, 0.80, 1.0)      # Silky pearl-silver lit
    lk.new(out_NL, n_ramp.inputs["Factor"])

    n_mul = nt.nodes.new("ShaderNodeMix"); n_mul.data_type = 'RGBA'; n_mul.blend_type = 'MULTIPLY'
    en_in(n_mul, "Factor").default_value = 1.0
    lk.new(n_tex.outputs["Color"], en_in(n_mul, "A"))
    lk.new(n_ramp.outputs["Color"], en_in(n_mul, "B"))

    # Angel-Ring Specular Band
    n_hadd = nt.nodes.new("ShaderNodeVectorMath"); n_hadd.operation = 'ADD'
    n_hnrm = nt.nodes.new("ShaderNodeVectorMath"); n_hnrm.operation = 'NORMALIZE'
    lk.new(out_Lc, n_hadd.inputs[0]); lk.new(out_V, n_hadd.inputs[1])
    lk.new(n_hadd.outputs["Vector"], n_hnrm.inputs[0])

    n_nsft = nt.nodes.new("ShaderNodeVectorMath"); n_nsft.operation = 'ADD'
    n_nsft.inputs[1].default_value = (0.0, 0.0, -0.22)
    n_nsn  = nt.nodes.new("ShaderNodeVectorMath"); n_nsn.operation = 'NORMALIZE'
    lk.new(out_Ns, n_nsft.inputs[0]); lk.new(n_nsft.outputs["Vector"], n_nsn.inputs[0])

    n_nh   = nt.nodes.new("ShaderNodeVectorMath"); n_nh.operation = 'DOT_PRODUCT'
    n_smr  = nt.nodes.new("ShaderNodeMapRange"); n_smr.interpolation_type = 'SMOOTHSTEP'
    n_smr.inputs["From Min"].default_value = 0.875
    n_smr.inputs["From Max"].default_value = 0.945
    n_lgate = nt.nodes.new("ShaderNodeMapRange"); n_lgate.interpolation_type = 'SMOOTHSTEP'
    n_lgate.inputs["From Min"].default_value = 0.44
    n_lgate.inputs["From Max"].default_value = 0.58
    n_smul = nt.nodes.new("ShaderNodeMath"); n_smul.operation = 'MULTIPLY'

    lk.new(n_nsn.outputs["Vector"], n_nh.inputs[0]); lk.new(n_hnrm.outputs["Vector"], n_nh.inputs[1])
    lk.new(n_nh.outputs["Value"], n_smr.inputs["Value"])
    lk.new(out_NL, n_lgate.inputs["Value"])
    lk.new(n_smr.outputs["Result"], n_smul.inputs[0]); lk.new(n_lgate.outputs["Result"], n_smul.inputs[1])

    n_sadd = nt.nodes.new("ShaderNodeMix"); n_sadd.data_type = 'RGBA'; n_sadd.blend_type = 'ADD'
    en_in(n_sadd, "B").default_value = (0.24, 0.34, 0.48, 1.0)
    lk.new(n_smul.outputs["Value"], en_in(n_sadd, "Factor"))
    lk.new(en_out(n_mul, "Result"), en_in(n_sadd, "A"))

    n_radd = nt.nodes.new("ShaderNodeMix"); n_radd.data_type = 'RGBA'; n_radd.blend_type = 'ADD'
    en_in(n_radd, "B").default_value = (0.45, 0.76, 1.05, 1.0)
    n_rfac = nt.nodes.new("ShaderNodeMath"); n_rfac.operation = 'MULTIPLY'; n_rfac.inputs[1].default_value = 0.40
    lk.new(out_rim, n_rfac.inputs[0])
    lk.new(n_rfac.outputs["Value"], en_in(n_radd, "Factor"))
    lk.new(en_out(n_sadd, "Result"), en_in(n_radd, "A"))

    lk.new(en_out(n_radd, "Result"), n_em.inputs["Color"])
    lk.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

# 4C) Facial Parts: Jewel eyes + crisp dark lashes
mat_eye = bpy.data.materials.get("RD_eye")
img_eye = bpy.data.images.get("RadDoll_eye_tex.png")
if mat_eye and img_eye:
    nt = mat_eye.node_tree; nt.nodes.clear(); lk = nt.links
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_em  = nt.nodes.new("ShaderNodeEmission")
    n_tex = nt.nodes.new("ShaderNodeTexImage"); n_tex.image = img_eye
    n_hsv = nt.nodes.new("ShaderNodeHueSaturation")
    n_hsv.inputs["Saturation"].default_value = 1.24
    n_hsv.inputs["Value"].default_value = 0.68
    n_sep = nt.nodes.new("ShaderNodeSeparateColor"); n_sep.mode = 'HSV'
    n_mr  = nt.nodes.new("ShaderNodeMapRange"); n_mr.interpolation_type = 'SMOOTHSTEP'
    n_mr.inputs["From Min"].default_value = 0.94
    n_mr.inputs["From Max"].default_value = 0.995
    n_spk = nt.nodes.new("ShaderNodeMix"); n_spk.data_type = 'RGBA'; n_spk.blend_type = 'ADD'
    en_in(n_spk, "B").default_value = (0.35, 0.42, 0.52, 1.0)
    lk.new(n_tex.outputs["Color"], n_hsv.inputs["Color"])
    lk.new(n_tex.outputs["Color"], n_sep.inputs["Color"])
    lk.new(n_sep.outputs["Blue"], n_mr.inputs["Value"])
    lk.new(n_mr.outputs["Result"], en_in(n_spk, "Factor"))
    lk.new(n_hsv.outputs["Color"], en_in(n_spk, "A"))
    lk.new(en_out(n_spk, "Result"), n_em.inputs["Color"])
    lk.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

for fmat_name, c_col in [
    ("RD_lash",       (0.045, 0.018, 0.026, 1.0)),
    ("RD_mouth",      (0.320, 0.072, 0.095, 1.0)),
    ("RD_brow",       (0.155, 0.105, 0.135, 1.0)),
    ("RD_double_lid", (0.125, 0.070, 0.085, 1.0)),
]:
    fm = bpy.data.materials.get(fmat_name)
    if fm and fm.node_tree:
        nt = fm.node_tree; nt.nodes.clear()
        no = nt.nodes.new("ShaderNodeOutputMaterial")
        ne = nt.nodes.new("ShaderNodeEmission")
        ne.inputs["Color"].default_value = c_col
        nt.links.new(ne.outputs["Emission"], no.inputs["Surface"])

# 4D) 5 Outfit & Accessory Shaders: Crisp 3D fold sculpting (smooth_fac=0.18) + rich royal periwinkle shadows + zero whiteout!
OUTFIT_SPECS = {
    "MAT_V10_Toon_Coat":      ("Tex_Coat_4096_V10R15_Clean_PRIVATE.png",      (0.30, 0.38, 0.58, 1.0), (0.46, 0.54, 0.72, 1.0), (0.64, 0.68, 0.76, 1.0), 0.32),
    "MAT_V10_Toon_SailorTop": ("Tex_SailorTop_2048_V10R15_Clean_PRIVATE.png", (0.33, 0.40, 0.58, 1.0), (0.46, 0.53, 0.68, 1.0), (0.58, 0.61, 0.68, 1.0), 0.22),
    "MAT_V10_Toon_Shorts":    ("Tex_Shorts_2048_V10R15_Clean_PRIVATE.png",    (0.30, 0.38, 0.58, 1.0), (0.46, 0.54, 0.72, 1.0), (0.64, 0.68, 0.76, 1.0), 0.28),
    "MAT_V10_Toon_SocksAcc":  ("Tex_Socks_Acc_2048_V10R15_Clean_PRIVATE.png", (0.32, 0.39, 0.57, 1.0), (0.45, 0.52, 0.67, 1.0), (0.57, 0.60, 0.67, 1.0), 0.24),
    "MAT_V10_Toon_Shoes":     ("Tex_Shoes_2048_V10R15_Clean_PRIVATE.png",     (0.31, 0.38, 0.55, 1.0), (0.44, 0.50, 0.65, 1.0), (0.58, 0.61, 0.68, 1.0), 0.24),
}
for mname, (png_name, c_shd, c_mid, c_lit, rim_k) in OUTFIT_SPECS.items():
    mat = bpy.data.materials.get(mname)
    img = bpy.data.images.get(png_name)
    if not mat or not img:
        continue
    img.colorspace_settings.name = 'sRGB'

    nt = mat.node_tree; nt.nodes.clear(); lk = nt.links
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_em  = nt.nodes.new("ShaderNodeEmission")
    n_uv  = nt.nodes.new("ShaderNodeUVMap")
    n_tex = nt.nodes.new("ShaderNodeTexImage"); n_tex.image = img
    lk.new(n_uv.outputs["UV"], n_tex.inputs["Vector"])

    _, _, _, _, out_NL, out_rim = add_hsr_light_rig_nodes(
        nt, smooth_bias=(0.0, -0.30, 0.50), smooth_fac=0.18
    )

    n_ramp = nt.nodes.new("ShaderNodeValToRGB")
    n_ramp.color_ramp.interpolation = 'EASE'
    els = n_ramp.color_ramp.elements
    els[0].position = 0.45
    els[0].color = c_shd
    em = els.new(0.53)
    em.color = c_mid
    els[1].position = 0.60
    els[1].color = c_lit
    lk.new(out_NL, n_ramp.inputs["Factor"])

    n_mul = nt.nodes.new("ShaderNodeMix"); n_mul.data_type = 'RGBA'; n_mul.blend_type = 'MULTIPLY'
    en_in(n_mul, "Factor").default_value = 1.0
    lk.new(n_tex.outputs["Color"], en_in(n_mul, "A"))
    lk.new(n_ramp.outputs["Color"], en_in(n_mul, "B"))

    n_radd = nt.nodes.new("ShaderNodeMix"); n_radd.data_type = 'RGBA'; n_radd.blend_type = 'ADD'
    en_in(n_radd, "B").default_value = (0.40, 0.70, 0.98, 1.0)
    n_rfac = nt.nodes.new("ShaderNodeMath"); n_rfac.operation = 'MULTIPLY'; n_rfac.inputs[1].default_value = rim_k
    lk.new(out_rim, n_rfac.inputs[0])
    lk.new(n_rfac.outputs["Value"], en_in(n_radd, "Factor"))
    lk.new(en_out(n_mul, "Result"), en_in(n_radd, "A"))

    lk.new(en_out(n_radd, "Result"), n_em.inputs["Color"])
    lk.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

for out_mname, out_col in [
    ("MAT_NPR_Outline_Cool", (0.075, 0.100, 0.195, 1.0)),
    ("MAT_NPR_Outline_Skin", (0.185, 0.095, 0.110, 1.0)),
]:
    om = bpy.data.materials.get(out_mname)
    if om and om.node_tree:
        for n in om.node_tree.nodes:
            if n.type == 'EMISSION':
                n.inputs["Color"].default_value = out_col

# 4E) White Chair Shaders: Sculpted porcelain & ice-sapphire roses integrated cleanly into theater lighting!
CHAIR_TINTS = {
    "bai1": ((0.24, 0.30, 0.48, 1.0), (0.50, 0.54, 0.64, 1.0)),
    "bai2": ((0.24, 0.30, 0.48, 1.0), (0.50, 0.54, 0.64, 1.0)),
    "bai3": ((0.26, 0.31, 0.47, 1.0), (0.53, 0.56, 0.65, 1.0)),
    "hua1": ((0.26, 0.32, 0.50, 1.0), (0.52, 0.56, 0.66, 1.0)),
    "hua2": ((0.18, 0.32, 0.58, 1.0), (0.42, 0.58, 0.80, 1.0)),
    "hua3": ((0.22, 0.36, 0.62, 1.0), (0.46, 0.62, 0.84, 1.0)),
    "shu2": ((0.14, 0.22, 0.36, 1.0), (0.28, 0.38, 0.52, 1.0)),
    "shu3": ((0.14, 0.22, 0.36, 1.0), (0.28, 0.38, 0.52, 1.0)),
    "ye2":  ((0.16, 0.26, 0.40, 1.0), (0.30, 0.44, 0.58, 1.0)),
    "ye3":  ((0.16, 0.26, 0.40, 1.0), (0.30, 0.44, 0.58, 1.0)),
}
img_yizi = bpy.data.images.get("yizi.png")
for cm_name, (c_shd, c_lit) in CHAIR_TINTS.items():
    cm = bpy.data.materials.get(cm_name)
    if not cm:
        continue
    nt = cm.node_tree; nt.nodes.clear(); lk = nt.links
    no = nt.nodes.new("ShaderNodeOutputMaterial")
    ne = nt.nodes.new("ShaderNodeEmission")
    _, _, _, _, out_NL, out_rim = add_hsr_light_rig_nodes(nt, smooth_bias=(0.0, -0.25, 0.50), smooth_fac=0.15)
    nr = nt.nodes.new("ShaderNodeValToRGB"); nr.color_ramp.interpolation = 'EASE'
    nr.color_ramp.elements[0].position = 0.42; nr.color_ramp.elements[0].color = c_shd
    nr.color_ramp.elements[1].position = 0.58; nr.color_ramp.elements[1].color = c_lit
    lk.new(out_NL, nr.inputs["Factor"])
    base_out = nr.outputs["Color"]
    if cm_name == "hua1" and img_yizi:
        ntx = nt.nodes.new("ShaderNodeTexImage"); ntx.image = img_yizi
        nm  = nt.nodes.new("ShaderNodeMix"); nm.data_type = 'RGBA'; nm.blend_type = 'MULTIPLY'
        en_in(nm, "Factor").default_value = 0.65
        lk.new(base_out, en_in(nm, "A"))
        lk.new(ntx.outputs["Color"], en_in(nm, "B"))
        base_out = en_out(nm, "Result")
    n_radd = nt.nodes.new("ShaderNodeMix"); n_radd.data_type = 'RGBA'; n_radd.blend_type = 'ADD'
    en_in(n_radd, "B").default_value = (0.30, 0.54, 0.85, 1.0)
    n_rf = nt.nodes.new("ShaderNodeMath"); n_rf.operation = 'MULTIPLY'; n_rf.inputs[1].default_value = 0.20
    lk.new(out_rim, n_rf.inputs[0])
    lk.new(n_rf.outputs["Value"], en_in(n_radd, "Factor"))
    lk.new(base_out, en_in(n_radd, "A"))
    lk.new(en_out(n_radd, "Result"), ne.inputs["Color"])
    lk.new(ne.outputs["Emission"], no.inputs["Surface"])


print("=== [5/6] Configuring EEVEE Next (512MB Shadow Pool!) + Bloom Compositor & Saving V10R18 ===", flush=True)
scn.camera = bpy.data.objects["MMD_MV_CAMERA"]
scn.render.engine = 'BLENDER_EEVEE'
scn.view_settings.view_transform = 'Standard'
scn.view_settings.look = 'High Contrast'
scn.view_settings.exposure = -0.06
scn.view_settings.gamma = 1.0

ev = scn.eevee
ev.taa_render_samples = 16
ev.use_raytracing = True
ev.ray_tracing_method = 'SCREEN'
ev.use_fast_gi = False
ev.volumetric_samples = 32
ev.volumetric_tile_size = '8'
ev.use_volumetric_shadows = False
ev.use_shadows = True
ev.shadow_pool_size = '512'
ev.shadow_resolution_scale = 0.65

cg = scn.compositing_node_group
if cg:
    gl_nodes = [n for n in cg.nodes if n.type == 'GLARE']
    if gl_nodes:
        gl = gl_nodes[0]
        gl.inputs["Type"].default_value = 'Bloom'
        gl.inputs["Quality"].default_value = 'High'
        gl.inputs["Threshold"].default_value = 0.80
        gl.inputs["Smoothness"].default_value = 0.25
        gl.inputs["Strength"].default_value = 0.48
        gl.inputs["Size"].default_value = 0.50

scn.render.resolution_x = 1920
scn.render.resolution_y = 1080
scn.render.resolution_percentage = 100
scn.render.fps = 30
scn.render.image_settings.file_format = 'JPEG'
scn.render.image_settings.quality = 94

bpy.ops.wm.save_as_mainfile(filepath=str(DST_R18), compress=True)
print(f"SAVED_V10R18_V2: {DST_R18.name} ({DST_R18.stat().st_size / 1048576:.2f} MB, kept_shadow_lights={kept_sh})", flush=True)

print("=== [6/6] Rendering 6x 1080p Genshin/HSR/MMD Beauty Shots (v2 Calibrated) ===", flush=True)
test_frames = [
    (120,  "f0120_白椅唤醒近景"),
    (586,  "f0586_水晶巨窗中景"),
    (1086, "f1086_高踢腿水镜全景"),
    (1350, "f1350_歌剧院远景"),
    (1708, "f1708_0分56背身反打观众席"),
    (3600, "f3600_后半曲华彩全景"),
]
times = []
for idx, (fno, tag) in enumerate(test_frames):
    tf0 = time.time()
    scn.frame_set(fno)
    bpy.context.view_layer.update()
    scn.render.filepath = str(PREVIEW_DIR / f"{tag}.jpg")
    bpy.ops.render.render(write_still=True)
    dt = time.time() - tf0
    times.append(dt)
    print(f"  SHOT {idx+1}/6 {tag}: {dt:.2f}s", flush=True)

steady_avg = sum(times[1:]) / max(1, len(times) - 1)
print(f"V10R18_V2_BENCHMARK: first_frame={times[0]:.2f}s steady_avg={steady_avg:.3f}s/frame total_elapsed={time.time()-t_start:.1f}s", flush=True)
