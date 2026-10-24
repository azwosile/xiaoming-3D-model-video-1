"""V10R7 外观：二次元柔和赛璐璐/无脏阴影材质（眼睛自发光保真贴图、瓷白面部贴图+暖调柔影、发束分缕描边银蓝材质、视角衰减刘海透眉）与 EEVEE 设置。"""
import math

import bpy
from mathutils import Vector

SLOTS = ['eye', 'lash', 'brow', 'double_lid', 'mouth', 'hidden']
CLASS_TO_SLOT = {
    'eye': 0,
    'lash': 1,
    'brow': 2,
    'double_lid': 3,
    'mouth': 4,
    'ear': 5,
    'hidden': 5,
    'pupil': 5,
    'hl': 5,
    'line': 5,
    'other': 5,
}


def srgb_to_lin(c):
    return tuple(((x / 255.0) ** 2.2 if x > 10 else x / 255.0 / 12.92) for x in c)


def _clear_nodes(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (620, 0)
    return nt, out


def _mk_flat_or_toon(name, srgb_lit, srgb_sh=None):
    mat = bpy.data.materials.new(name)
    nt, out = _clear_nodes(mat)
    c_lit = srgb_to_lin(srgb_lit)
    mat.diffuse_color = (*c_lit, 1.0)
    em = nt.nodes.new('ShaderNodeEmission')
    em.location = (320, 0)
    if srgb_sh is None:
        em.inputs['Color'].default_value = (*c_lit, 1.0)
    else:
        c_sh = srgb_to_lin(srgb_sh)
        dif = nt.nodes.new('ShaderNodeBsdfDiffuse')
        dif.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        s2r = nt.nodes.new('ShaderNodeShaderToRGB')
        ramp = nt.nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.interpolation = 'EASE'
        ramp.color_ramp.elements[0].position = 0.22
        ramp.color_ramp.elements[0].color = (*c_sh, 1.0)
        ramp.color_ramp.elements[1].position = 0.56
        ramp.color_ramp.elements[1].color = (*c_lit, 1.0)
        nt.links.new(dif.outputs['BSDF'], s2r.inputs['Shader'])
        nt.links.new(s2r.outputs['Color'], ramp.inputs['Fac'])
        nt.links.new(ramp.outputs['Color'], em.inputs['Color'])
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    return mat


def _mk_transparent(name='RD_transparent'):
    mat = bpy.data.materials.new(name)
    for attr, val in (('surface_render_method', 'DITHERED'), ('blend_method', 'CLIP'), ('use_backface_culling', False)):
        try:
            setattr(mat, attr, val)
        except Exception:
            pass
    nt, out = _clear_nodes(mat)
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    tr.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(tr.outputs['BSDF'], out.inputs['Surface'])
    mat.diffuse_color = (0.0, 0.0, 0.0, 0.0)
    return mat


def _mk_brow_thru(name='RD_brow_thru'):
    """二次元刘海透眉材质：贴合前刘海表面，正视/斜视柔和显现暖灰褐眉弧，侧视/顶视/背视按视线角自动归零透明。"""
    mat = bpy.data.materials.new(name)
    for attr, val in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND'), ('use_backface_culling', True)):
        try:
            setattr(mat, attr, val)
        except Exception:
            pass
    nt, out = _clear_nodes(mat)
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    em = nt.nodes.new('ShaderNodeEmission')
    c_brow = srgb_to_lin((132, 108, 116))
    em.inputs['Color'].default_value = (*c_brow, 1.0)

    at = nt.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'Col'

    geo = nt.nodes.new('ShaderNodeNewGeometry')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(geo.outputs['Incoming'], sep.inputs['Vector'])

    # Incoming 指向相机：正视时 Y≈+1.0，45°斜视 Y≈+0.71，侧视/顶视 Y=0，背视 Y=-1
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.interpolation_type = 'SMOOTHERSTEP'
    mr.inputs['From Min'].default_value = 0.84
    mr.inputs['From Max'].default_value = 0.96
    mr.inputs['To Min'].default_value = 0.0
    mr.inputs['To Max'].default_value = 0.64
    nt.links.new(sep.outputs['Y'], mr.inputs['Value'])

    mul_a = nt.nodes.new('ShaderNodeMath')
    mul_a.operation = 'MULTIPLY'
    nt.links.new(at.outputs['Fac'], mul_a.inputs[0])
    nt.links.new(mr.outputs['Result'], mul_a.inputs[1])

    mx = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(mul_a.outputs['Value'], mx.inputs['Fac'])
    nt.links.new(tr.outputs['BSDF'], mx.inputs[1])
    nt.links.new(em.outputs['Emission'], mx.inputs[2])
    nt.links.new(mx.outputs['Shader'], out.inputs['Surface'])
    mat.diffuse_color = (*c_brow, 0.45)
    return mat


def _mk_eye(tex_path):
    mat = bpy.data.materials.new('RD_eye')
    nt, out = _clear_nodes(mat)
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(str(tex_path))
    tex.image.colorspace_settings.name = 'sRGB'
    tex.extension = 'EXTEND'
    tex.image.pack()
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1.0
    nt.links.new(tex.outputs['Color'], em.inputs['Color'])
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    mat.diffuse_color = (0.95, 0.95, 0.98, 1.0)
    return mat


def _mk_skin(face_tex_path):
    mat = bpy.data.materials.new('RD_skin')
    nt, out = _clear_nodes(mat)
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(str(face_tex_path))
    tex.image.colorspace_settings.name = 'sRGB'
    tex.extension = 'EXTEND'
    tex.image.pack()

    dif = nt.nodes.new('ShaderNodeBsdfDiffuse')
    dif.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    s2r = nt.nodes.new('ShaderNodeShaderToRGB')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.interpolation = 'EASE'
    ramp.color_ramp.elements[0].position = 0.14
    ramp.color_ramp.elements[0].color = (0.93, 0.79, 0.79, 1.0)
    ramp.color_ramp.elements[1].position = 0.44
    ramp.color_ramp.elements[1].color = (1.00, 1.00, 1.00, 1.0)

    mul = nt.nodes.new('ShaderNodeMix')
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.inputs['Factor'].default_value = 1.0

    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1.0

    nt.links.new(dif.outputs['BSDF'], s2r.inputs['Shader'])
    nt.links.new(s2r.outputs['Color'], ramp.inputs['Fac'])
    nt.links.new(tex.outputs['Color'], mul.inputs[6])
    nt.links.new(ramp.outputs['Color'], mul.inputs[7])
    nt.links.new(mul.outputs[2], em.inputs['Color'])
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    mat.diffuse_color = (0.95, 0.85, 0.85, 1.0)
    return mat


def _mk_hair():
    """二次元分缕层次发材质：顶点色（前刘海暖银白+发梢肤透微粉、两侧与后发银白→钴天蓝）
    × 赛璐璐双阶漫反射 × 掠射角发束边缘淡紫灰分缝线（让重叠刘海根根分明）+ 额顶柔白天使环高光。"""
    mat = bpy.data.materials.new('RD_hair')
    nt, out = _clear_nodes(mat)
    at = nt.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'Col'

    dif = nt.nodes.new('ShaderNodeBsdfDiffuse')
    dif.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    s2r = nt.nodes.new('ShaderNodeShaderToRGB')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.interpolation = 'EASE'
    ramp.color_ramp.elements[0].position = 0.24
    ramp.color_ramp.elements[0].color = (0.76, 0.74, 0.84, 1.0)
    ramp.color_ramp.elements[1].position = 0.64
    ramp.color_ramp.elements[1].color = (1.01, 1.01, 1.02, 1.0)

    # 双面对称掠射角柔边（使用 |dot(Normal, Incoming)| 替代单面 LayerWeight，彻底消除薄发梢背面与发帽反法线面在转动时的深紫闪烁！）
    geo_e = nt.nodes.new('ShaderNodeNewGeometry')
    dot_ni = nt.nodes.new('ShaderNodeVectorMath')
    dot_ni.operation = 'DOT_PRODUCT'
    nt.links.new(geo_e.outputs['Normal'], dot_ni.inputs[0])
    nt.links.new(geo_e.outputs['Incoming'], dot_ni.inputs[1])
    abs_ni = nt.nodes.new('ShaderNodeMath')
    abs_ni.operation = 'ABSOLUTE'
    nt.links.new(dot_ni.outputs['Value'], abs_ni.inputs[0])
    edge_ramp = nt.nodes.new('ShaderNodeValToRGB')
    edge_ramp.color_ramp.interpolation = 'EASE'
    # |dot|=0 (掠射边缘) -> 柔和淡紫灰 0.74；|dot|>=0.42 (正面或背面主体) -> 纯净 1.00
    edge_ramp.color_ramp.elements[0].position = 0.06
    edge_ramp.color_ramp.elements[0].color = (0.74, 0.71, 0.82, 1.0)
    edge_ramp.color_ramp.elements[1].position = 0.44
    edge_ramp.color_ramp.elements[1].color = (1.00, 1.00, 1.00, 1.0)
    nt.links.new(abs_ni.outputs['Value'], edge_ramp.inputs['Fac'])

    mul1 = nt.nodes.new('ShaderNodeMix')
    mul1.data_type = 'RGBA'
    mul1.blend_type = 'MULTIPLY'
    mul1.inputs['Factor'].default_value = 1.0

    mul2 = nt.nodes.new('ShaderNodeMix')
    mul2.data_type = 'RGBA'
    mul2.blend_type = 'MULTIPLY'
    mul2.inputs['Factor'].default_value = 1.0

    # 基于表面法线朝向的柔和颅顶光泽（消除原世界 Z 坐标在侧面/背面切出的生硬水平白横带，且转头无滑动频闪）
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    dot_n = nt.nodes.new('ShaderNodeVectorMath')
    dot_n.operation = 'DOT_PRODUCT'
    dot_n.inputs[1].default_value = (0.0, 0.42, 0.907)
    nt.links.new(geo.outputs['Normal'], dot_n.inputs[0])
    halo_mr = nt.nodes.new('ShaderNodeMapRange')
    halo_mr.interpolation_type = 'SMOOTHERSTEP'
    halo_mr.inputs['From Min'].default_value = 0.72
    halo_mr.inputs['From Max'].default_value = 0.97
    nt.links.new(dot_n.outputs['Value'], halo_mr.inputs['Value'])

    add_halo = nt.nodes.new('ShaderNodeMix')
    add_halo.data_type = 'RGBA'
    add_halo.blend_type = 'SCREEN'
    halo_fac = nt.nodes.new('ShaderNodeMath')
    halo_fac.operation = 'MULTIPLY'
    halo_fac.inputs[1].default_value = 0.065
    nt.links.new(halo_mr.outputs['Result'], halo_fac.inputs[0])
    nt.links.new(halo_fac.outputs['Value'], add_halo.inputs['Factor'])
    add_halo.inputs[7].default_value = (1.0, 0.99, 1.0, 1.0)

    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1.0

    nt.links.new(dif.outputs['BSDF'], s2r.inputs['Shader'])
    nt.links.new(s2r.outputs['Color'], ramp.inputs['Fac'])
    nt.links.new(at.outputs['Color'], mul1.inputs[6])
    nt.links.new(ramp.outputs['Color'], mul1.inputs[7])
    nt.links.new(mul1.outputs[2], mul2.inputs[6])
    nt.links.new(edge_ramp.outputs['Color'], mul2.inputs[7])
    nt.links.new(mul2.outputs[2], add_halo.inputs[6])
    nt.links.new(add_halo.outputs[2], em.inputs['Color'])
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    mat.diffuse_color = (0.90, 0.90, 0.94, 1.0)
    return mat


def _mk_ear(name='RD_ear'):
    mat = bpy.data.materials.new(name)
    nt, out = _clear_nodes(mat)
    at = nt.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'Col'
    dif = nt.nodes.new('ShaderNodeBsdfDiffuse')
    dif.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    s2r = nt.nodes.new('ShaderNodeShaderToRGB')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.interpolation = 'EASE'
    ramp.color_ramp.elements[0].position = 0.16
    ramp.color_ramp.elements[0].color = (0.91, 0.76, 0.77, 1.0)
    ramp.color_ramp.elements[1].position = 0.48
    ramp.color_ramp.elements[1].color = (1.00, 1.00, 1.00, 1.0)
    mul = nt.nodes.new('ShaderNodeMix')
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.inputs['Factor'].default_value = 1.0
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(dif.outputs['BSDF'], s2r.inputs['Shader'])
    nt.links.new(s2r.outputs['Color'], ramp.inputs['Fac'])
    nt.links.new(at.outputs['Color'], mul.inputs[6])
    nt.links.new(ramp.outputs['Color'], mul.inputs[7])
    nt.links.new(mul.outputs[2], em.inputs['Color'])
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    mat.diffuse_color = (0.95, 0.84, 0.84, 1.0)
    return mat


def make_materials(eye_tex_path, face_tex_path=None):
    if face_tex_path is None:
        face_tex_path = eye_tex_path.parent / 'RadDoll_face_tex.png'
    return {
        'skin': _mk_skin(face_tex_path),
        'eye': _mk_eye(eye_tex_path),
        'lash': _mk_flat_or_toon('RD_lash', (92, 60, 64), (66, 42, 46)),
        'brow': _mk_flat_or_toon('RD_brow', (136, 112, 118)),
        'double_lid': _mk_flat_or_toon('RD_double_lid', (120, 92, 96)),
        'mouth': _mk_flat_or_toon('RD_mouth', (196, 98, 108), (156, 70, 82)),
        'hidden': _mk_transparent('RD_transparent'),
        'ear': _mk_ear('RD_ear'),
        'hair': _mk_hair(),
        'brow_thru': _mk_brow_thru('RD_brow_thru'),
    }


def setup_scene(sc, res=900, transparent=False, exposure=0.0, samples=48):
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA' if transparent else 'RGB'
    sc.render.film_transparent = transparent
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.view_settings.exposure = exposure
    for attr, val in (('taa_render_samples', samples), ('use_gtao', False), ('use_raytracing', False), ('use_soft_shadows', True)):
        try:
            setattr(sc.eevee, attr, val)
        except Exception:
            pass
    w = bpy.data.worlds.new('W')
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputWorld')
    lp = nt.nodes.new('ShaderNodeLightPath')
    bg_env = nt.nodes.new('ShaderNodeBackground')
    bg_env.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_env.inputs[1].default_value = 0.08
    bg_cam = nt.nodes.new('ShaderNodeBackground')
    bg_cam.inputs[0].default_value = (0.86, 0.88, 0.91, 1.0)
    bg_cam.inputs[1].default_value = 1.0
    mx = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(lp.outputs['Is Camera Ray'], mx.inputs['Fac'])
    nt.links.new(bg_env.outputs['Background'], mx.inputs[1])
    nt.links.new(bg_cam.outputs['Background'], mx.inputs[2])
    nt.links.new(mx.outputs['Shader'], out.inputs['Surface'])
    sc.world = w


def add_lights(sc):
    def sun(name, frm, energy, angle_deg=18, shadow=False):
        ld = bpy.data.lights.new(name, 'SUN')
        ld.energy = energy
        ld.angle = math.radians(angle_deg)
        try:
            ld.use_shadow = shadow
        except Exception:
            pass
        ob = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(ob)
        ob.rotation_euler = Vector([-c for c in frm]).to_track_quat('-Z', 'Y').to_euler()
    sun('KEY', (-0.24, 1.0, 0.45), 1.80, angle_deg=20, shadow=False)
    sun('FILL', (0.58, 0.82, 0.15), 0.58, angle_deg=24, shadow=False)
    sun('BACK', (0.0, -1.0, 0.45), 0.85, angle_deg=20, shadow=False)
