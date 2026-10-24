import io, zipfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
WDIR = ROOT / "生成模型" / "组装" / "v010R11_MMD_三渲二舞台_PRIVATE"
OUT_TEX_DIR = WDIR / "tex_v10r19_PRIVATE"
OUT_TEX_DIR.mkdir(parents=True, exist_ok=True)

geom = np.load(r"C:\Temp\xm_face_uv_geom.npz")
tri_uv  = geom["tri_uv"]    # (N, 3, 2)
tri_xyz = geom["tri_xyz"]   # (N, 3, 3)
tri_nrm = geom["tri_nrm"]   # (N, 3, 3)
bang_pts = geom["bang_pts"] # (M, 3)
eye_xyz  = geom["eye_xyz"]
lash_xyz = geom["lash_xyz"]
mouth_xyz = geom["mouth_xyz"]

# Compute exact 3D facial landmarks in rest pose
eye_l_pts = eye_xyz[eye_xyz[:, 0] > 0.005]   # +X
eye_r_pts = eye_xyz[eye_xyz[:, 0] < -0.005]  # -X
eye_l_c = eye_l_pts.mean(axis=0)
eye_r_c = eye_r_pts.mean(axis=0)
eye_z = 0.5 * (eye_l_c[2] + eye_r_c[2])
eye_abs_x = 0.5 * (abs(eye_l_c[0]) + abs(eye_r_c[0]))
mid_x = 0.5 * (eye_l_c[0] + eye_r_c[0])

# Nose tip: frontmost (-Y) skin vertex near midline
all_skin_v = tri_xyz.reshape(-1, 3)
mid_mask = (np.abs(all_skin_v[:, 0] - mid_x) < 0.008) & (all_skin_v[:, 2] > eye_z - 0.055) & (all_skin_v[:, 2] < eye_z + 0.01)
nose_tip = all_skin_v[mid_mask][np.argmin(all_skin_v[mid_mask][:, 1])]

# Mouth slit frontmost center: frontmost vertices of mouth_xyz
mouth_front = mouth_xyz[mouth_xyz[:, 1] < np.percentile(mouth_xyz[:, 1], 18)]
lip_c = np.array([mid_x, mouth_front[:, 1].min(), np.median(mouth_front[:, 2])], dtype=np.float32)

print(f"LANDMARKS: mid_x={mid_x:.4f}, eye_L={eye_l_c.round(4)}, eye_R={eye_r_c.round(4)}, nose_tip={nose_tip.round(4)}, lip_c={lip_c.round(4)}")

# ==============================================================================
# HELPER: Rasterize 3D attributes onto UV map (S x S) with barycentric coords
# ==============================================================================
def rasterize_uv_coords(res=2048):
    pos_map = np.zeros((res, res, 3), dtype=np.float32)
    nrm_map = np.zeros((res, res, 3), dtype=np.float32)
    valid   = np.zeros((res, res), dtype=np.uint8)

    for idx in range(len(tri_uv)):
        uv = tri_uv[idx].copy()
        # Image row 0 = top (V=1), row res-1 = bottom (V=0)
        px = uv[:, 0] * (res - 1)
        py = (1.0 - uv[:, 1]) * (res - 1)

        x0 = max(0, int(np.floor(px.min())) - 1)
        x1 = min(res - 1, int(np.ceil(px.max())) + 1)
        y0 = max(0, int(np.floor(py.min())) - 1)
        y1 = min(res - 1, int(np.ceil(py.max())) + 1)
        if x1 < x0 or y1 < y0:
            continue

        denom = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2])
        if abs(denom) < 1e-8:
            continue

        gx, gy = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float32),
                             np.arange(y0, y1 + 1, dtype=np.float32))
        w0 = ((py[1] - py[2]) * (gx - px[2]) + (px[2] - px[1]) * (gy - py[2])) / denom
        w1 = ((py[2] - py[0]) * (gx - px[2]) + (px[0] - px[2]) * (gy - py[2])) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            continue

        xyz = tri_xyz[idx]
        nrm = tri_nrm[idx]
        p_interp = (w0[..., None] * xyz[0] + w1[..., None] * xyz[1] + w2[..., None] * xyz[2])
        n_interp = (w0[..., None] * nrm[0] + w1[..., None] * nrm[1] + w2[..., None] * nrm[2])
        n_len = np.linalg.norm(n_interp, axis=-1, keepdims=True) + 1e-8
        n_interp /= n_len

        sub_pos = pos_map[y0:y1+1, x0:x1+1]
        sub_nrm = nrm_map[y0:y1+1, x0:x1+1]
        sub_val = valid[y0:y1+1, x0:x1+1]
        sub_pos[inside] = p_interp[inside]
        sub_nrm[inside] = n_interp[inside]
        sub_val[inside] = 1

    # Dilate valid pixels across UV seams by 24px using nearest valid pixel (distanceTransformWithLabels)
    inv_mask = (1 - valid).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(inv_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # Build lookup from label index (1-based over zero pixels of inv_mask == valid pixels!)
    vy, vx = np.where(valid > 0)
    # cv2 assigns labels 1..K in row-major order of zero pixels in inv_mask!
    seam_band = (valid == 0) & (dist <= 28.0)
    lbl_idx = np.clip(labels[seam_band] - 1, 0, len(vy) - 1)
    pos_map[seam_band] = pos_map[vy[lbl_idx], vx[lbl_idx]]
    nrm_map[seam_band] = nrm_map[vy[lbl_idx], vx[lbl_idx]]
    dilated_valid = ((valid > 0) | seam_band).astype(np.uint8)
    return pos_map, nrm_map, valid, dilated_valid

print("Rasterizing 2048x2048 3D UV position/normal field...", flush=True)
pos2k, nrm2k, valid2k, dvalid2k = rasterize_uv_coords(2048)

# ==============================================================================
# MODULE A1: Head-Anchored Face SDF (2048x2048 RGB Non-Color) + Bang Drop Shadow
# ==============================================================================
# Convention in shader:
#   u_light = 0.5 * dot(Head_Fwd, L_horiz) + 0.5  in [0, 1]  (1.0 = pure front light, 0.5 = 90 deg side light, 0.0 = back light)
#   side_r  = dot(Head_Right, L_horiz)            (>0 means light on character's +X side, <0 means light on -X side)
#   Pixel is LIT when u_light > SDF_Threshold!
#   So SMALL threshold (e.g. 0.20) = stays LIT almost always; LARGE threshold (e.g. 0.85) = enters shadow early!
def compute_anime_face_sdf_side(pos, nrm, dval, light_sign=+1.0):
    # sx > 0 means lit side of face; sx < 0 means away-from-light (shadow) side of face!
    sx = (pos[..., 0] - mid_x) * light_sign
    dy_fwd = -(pos[..., 1] - nose_tip[1])  # 0 at nose tip, negative going back toward ears
    dz = pos[..., 2] - nose_tip[2]         # 0 at nose tip, +0.032 at eyes, -0.035 at mouth

    # Idealized smooth anime face horizontal azimuth angle phi in [-pi, pi]:
    # Instead of bumpy 3D cheek/mouth normals, use smooth elliptic cylinder azimuth + curated anime features!
    # Normalize lateral x across face half-width (~0.088 m)
    lat = np.clip(sx / 0.088, -1.25, 1.25)

    # Calibrated around 3/4 anime key light (theta ~ 36-42 deg -> u_light = 0.5*cos(theta)+0.5 ~ 0.87..0.90):
    # - Lit side of face (lat >= 0) AND inner 55% of shadow cheek stay CLEAN & LUMINOUS (thresh = 0.20..0.82 < 0.89)!
    # - Outer 30% of shadow cheek/temple curves cleanly into warm peach shadow (thresh = 0.88..0.95 > 0.89)!
    # - Crisp anime nose-side triangle shadow sits at thresh = 0.925 so it appears cleanly at 3/4 key light!
    thresh = np.where(
        lat >= 0.0,
        0.52 - 0.34 * np.power(np.clip(lat, 0.0, 1.0), 0.75),
        0.52 + 0.44 * np.power(np.clip(-lat, 0.0, 1.0), 1.15)
    )

    # Feature 1: Classic Genshin/Anime Nose-Side Crisp Triangle Shadow on away-from-light side of nose bridge!
    nose_tri_h = np.clip((dz + 0.005) / 0.029, 0.0, 1.0)
    nose_tri_w = 0.015 * (1.0 - 0.68 * nose_tri_h)
    nt_dist_x = np.clip((-(0.0015) - sx) / np.maximum(nose_tri_w, 1e-4), 0.0, 1.5)
    nt_mask = np.exp(-2.8 * np.maximum(0.0, nt_dist_x - 0.72)**2) * \
              np.clip((-0.001 - sx) / 0.0028, 0.0, 1.0) * \
              np.clip((dz + 0.005) / 0.0045, 0.0, 1.0) * \
              np.clip((0.024 - dz) / 0.0065, 0.0, 1.0)
    thresh = np.maximum(thresh, 0.928 * nt_mask)

    # Feature 2: Iconic Anime Rembrandt Lit Cheek Triangle on the shadow cheek (under shadow-side eye)!
    rem_cx, rem_cz = -0.042, 0.012
    rem_r = ((sx - rem_cx) / 0.024)**2 + ((dz - rem_cz) / 0.014)**2
    rem_mask = np.exp(-1.4 * rem_r)
    thresh = thresh - 0.095 * rem_mask

    # Feature 3: Tiny delicate under-lower-lip shadow accent
    lip_dx = (pos[..., 0] - mid_x) / 0.011
    lip_dz = (pos[..., 2] - (lip_c[2] - 0.0070)) / 0.0038
    under_lip = np.exp(-1.6 * (lip_dx**2 + lip_dz**2)) * (pos[..., 1] < nose_tip[1] + 0.032)
    thresh = np.maximum(thresh, 0.905 * under_lip)

    # Back-of-head / behind-ear wrap (dy_fwd < -0.078 -> push threshold to 0.94)
    back_wrap = np.clip((-0.075 - dy_fwd) / 0.050, 0.0, 1.0)
    thresh = thresh * (1.0 - back_wrap) + 0.94 * back_wrap

    thresh = np.where(dval > 0, np.clip(thresh, 0.05, 0.97), 0.50).astype(np.float32)
    thresh = cv2.GaussianBlur(thresh, (7, 7), 1.8)
    return thresh

sdf_right = compute_anime_face_sdf_side(pos2k, nrm2k, dvalid2k, light_sign=+1.0)
sdf_left  = compute_anime_face_sdf_side(pos2k, nrm2k, dvalid2k, light_sign=-1.0)

# Bang Drop-Shadow Mask (Channel B):
# Measure proximity of forehead / upper-brow skin pixels (z > eye_z + 0.006, front face y < -0.07) to front hair bangs!
from scipy.spatial import cKDTree
bang_tree = cKDTree(bang_pts)
# Query shifted slightly upward (+Z = +0.010, -Y = -0.008) to simulate top-front stage spotlight casting bang shadow slightly downward onto forehead!
q_pts = pos2k.copy()
q_pts[..., 1] -= 0.009
q_pts[..., 2] += 0.013
dists, _ = bang_tree.query(q_pts.reshape(-1, 3), k=1)
dists = dists.reshape(2048, 2048)

forehead_gate = np.clip((pos2k[..., 2] - (eye_z + 0.004)) / 0.022, 0.0, 1.0) * \
                np.clip((-0.065 - pos2k[..., 1]) / 0.030, 0.0, 1.0)
bang_shd = np.clip(1.0 - (dists - 0.005) / 0.020, 0.0, 1.0) * forehead_gate * (dvalid2k > 0)
bang_shd = cv2.GaussianBlur(bang_shd.astype(np.float32), (17, 17), 4.5)

sdf_rgb = np.stack([
    np.clip(sdf_right * 255.0, 0, 255).astype(np.uint8),
    np.clip(sdf_left  * 255.0, 0, 255).astype(np.uint8),
    np.clip(bang_shd  * 255.0, 0, 255).astype(np.uint8),
], axis=-1)
sdf_path = OUT_TEX_DIR / "Xiaoming_FaceSDF_2048_PRIVATE.png"
Image.fromarray(sdf_rgb, mode="RGB").save(sdf_path)
print("SAVED_FACE_SDF:", sdf_path.name, sdf_path.stat().st_size)

# ==============================================================================
# MODULE A2 + D: Three 256x20 Genshin/HSR Shadow_Ramp PNG LUTs
# ==============================================================================
def make_ramp_256x20(row_stops):
    """row_stops: list of 20 lists of (pos_0_1, (R,G,B) in 0..255 sRGB)."""
    arr = np.zeros((20, 256, 4), dtype=np.uint8)
    arr[..., 3] = 255
    xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    for r_i, stops in enumerate(row_stops):
        ps = np.array([s[0] for s in stops], dtype=np.float32)
        cs = np.array([s[1] for s in stops], dtype=np.float32)
        for ch in range(3):
            arr[r_i, :, ch] = np.clip(np.interp(xs, ps, cs[:, ch]), 0, 255).astype(np.uint8)
    return arr

# 1) Skin Shadow Ramp (256x20): Warm rose-taupe shadow -> Genshin coral-peach SSS band -> creamy ivory-peach lit!
# Calibrated for Standard + Medium High Contrast so lit skin = ~0.89 sRGB (zero blowout, preserves 100% blush & nose details!)
skin_row = [
    (0.00, (192, 152, 160)),
    (0.38, (204, 164, 170)),
    (0.47, (216, 174, 176)),
    (0.515,(238, 186, 182)),  # Warm Genshin coral-peach subsurface scattering fringe!
    (0.560,(248, 236, 235)),
    (1.00, (252, 244, 243)),
]
Image.fromarray(make_ramp_256x20([skin_row] * 20), "RGBA").save(OUT_TEX_DIR / "Xiaoming_Skin_Shadow_Ramp_256x20_PRIVATE.png")

# 2) Hair Shadow Ramp (256x20): Deep periwinkle-sapphire shadow -> icy sky mid-band -> luminous pearl-white lit!
hair_row = [
    (0.00, (128, 148, 196)),
    (0.42, (142, 164, 210)),
    (0.49, (172, 194, 232)),
    (0.56, (224, 232, 245)),
    (1.00, (236, 240, 248)),
]
Image.fromarray(make_ramp_256x20([hair_row] * 20), "RGBA").save(OUT_TEX_DIR / "Xiaoming_Hair_Shadow_Ramp_256x20_PRIVATE.png")

# 3) Cloth & Hardware Multi-Row Shadow Ramp (256x20):
# Rows 0..6: Crisp lavender-periwinkle shadow for white sailor cloth (prevents grey muddiness & prevents white washout!)
# Rows 7..13: Rich cerulean-royal shadow for light-blue coat & shorts (R9 saturation recovery!)
# Rows 14..19: High-contrast cool chrome-silver ramp for metallic buttons, O-ring, whale clip & choker hardware!
cloth_white = [
    (0.00, (138, 154, 198)),
    (0.44, (152, 168, 208)),
    (0.51, (186, 198, 226)),
    (0.57, (224, 228, 238)),
    (1.00, (232, 235, 242)),
]
cloth_blue = [
    (0.00, (124, 150, 204)),
    (0.44, (142, 168, 218)),
    (0.51, (182, 202, 236)),
    (0.57, (226, 234, 246)),
    (1.00, (236, 240, 248)),
]
cloth_metal = [
    (0.00, (92,  106, 142)),
    (0.42, (118, 134, 170)),
    (0.50, (175, 192, 222)),
    (0.58, (238, 244, 252)),
    (1.00, (252, 254, 255)),
]
cloth_rows = [cloth_white]*7 + [cloth_blue]*7 + [cloth_metal]*6
Image.fromarray(make_ramp_256x20(cloth_rows), "RGBA").save(OUT_TEX_DIR / "Xiaoming_Cloth_Shadow_Ramp_256x20_PRIVATE.png")
print("SAVED_3_SHADOW_RAMPS_256x20", flush=True)

# ==============================================================================
# MODULE B: 4096x4096 Crystalline Ice-Ocean Sapphire Multi-Layer Eye Atlas
# ==============================================================================
with zipfile.ZipFile(ROOT / "别人的建模项目参考" / "外部VRChat模型参考" / "RadDollV3_v3.02.zip", "r") as zf:
    eye_w = np.array(Image.open(io.BytesIO(zf.read("RadDollV3_v3.02/Tex/WhiteColor/RadDollV3_Eye_White.png"))).convert("RGBA"), dtype=np.float32)
    face_up = np.array(Image.open(io.BytesIO(zf.read("RadDollV3_v3.02/Tex/RadDollV3_Face_BaseColor.png"))).convert("RGBA"), dtype=np.float32)

H, W = 4096, 4096
yy, xx = np.meshgrid(np.linspace(0.0, 1.0, H, dtype=np.float32),
                     np.linspace(0.0, 1.0, W, dtype=np.float32), indexing='ij')
# Note: image row 0 is UV.y = 1.0 (top), row 4095 is UV.y = 0.0 (bottom)!
uv_u = xx
uv_v = 1.0 - yy

eye_out = np.zeros((H, W, 4), dtype=np.float32)
lum = (0.299 * eye_w[..., 0] + 0.587 * eye_w[..., 1] + 0.114 * eye_w[..., 2]) / 255.0
alpha_in = eye_w[..., 3] / 255.0

# --- 1) LOWER HALF (uv_v < 0.528): Left & Right Eyeballs (Sclera + Crystalline Ice-Sapphire Iris) ---
# Locate the two circular iris regions in RadDollV3_Eye_White (centers at u ~ 0.273 & 0.727, uv_v ~ 0.245, rx ~ 0.168, ry ~ 0.188)
r_left  = np.sqrt(((uv_u - 0.2735) / 0.166)**2 + ((uv_v - 0.244) / 0.188)**2)
r_right = np.sqrt(((uv_u - 0.7265) / 0.166)**2 + ((uv_v - 0.244) / 0.188)**2)
r_iris  = np.minimum(r_left, r_right)
in_lower = uv_v < 0.528

# Iris vertical gradient parameter (0 = bottom crescent caustic, 1 = top deep royal indigo under eyelid)
iris_v = np.clip((uv_v - 0.085) / 0.320, 0.0, 1.0)

# Map RadDollV3_Eye_White crystalline luminance (lum in 0..1) into rich Genshin/原图 Ice-Ocean Sapphire palette!
# Enhance micro-contrast of radial crystalline fibers in midtones!
lum_blur = cv2.GaussianBlur(lum, (0, 0), 9.0)
fiber_detail = np.clip(lum - lum_blur, -0.18, 0.18) * 1.65
lum_c = np.clip(lum + fiber_detail, 0.0, 1.0)

# Top palette (royal midnight indigo -> rich cobalt -> bright sky sapphire)
top_dark = np.array([12.0,  26.0,  76.0], dtype=np.float32)
top_mid  = np.array([34.0,  86.0, 182.0], dtype=np.float32)
top_hi   = np.array([115.0, 182.0, 248.0], dtype=np.float32)

# Bottom palette (deep sapphire rim -> luminous ocean cyan -> glowing glacier-white crescent caustic!)
bot_dark = np.array([20.0,  52.0, 124.0], dtype=np.float32)
bot_mid  = np.array([76.0, 168.0, 242.0], dtype=np.float32)
bot_hi   = np.array([218.0, 248.0, 255.0], dtype=np.float32)

def tri_lerp(l_val, c0, c1, c2):
    t1 = np.clip(l_val / 0.52, 0.0, 1.0)[..., None]
    t2 = np.clip((l_val - 0.52) / 0.48, 0.0, 1.0)[..., None]
    return np.where(l_val[..., None] < 0.52, c0 + (c1 - c0) * t1, c1 + (c2 - c1) * t2)

col_top = tri_lerp(lum_c, top_dark, top_mid, top_hi)
col_bot = tri_lerp(lum_c, bot_dark, bot_mid, bot_hi)
vv = np.power(iris_v, 0.88)[..., None]
iris_rgb = col_bot * (1.0 - vv) + col_top * vv

# Add luminous lower crescent caustic boost + crisp dark outer limbal ring + jewel pupil depth!
crescent = np.exp(-((r_iris - 0.64) / 0.18)**2) * np.clip((0.52 - iris_v) / 0.38, 0.0, 1.0)
iris_rgb += crescent[..., None] * np.array([42.0, 68.0, 52.0], dtype=np.float32)

limbal = np.exp(-((r_iris - 0.93) / 0.085)**2)
iris_rgb = iris_rgb * (1.0 - 0.48 * limbal[..., None]) + np.array([14.0, 28.0, 72.0]) * (0.48 * limbal[..., None])

# Pure diamond-white baked micro-sparkles where lum > 0.945 inside iris
spark = np.clip((lum - 0.935) / 0.055, 0.0, 1.0)[..., None]
iris_rgb = iris_rgb * (1.0 - spark) + 255.0 * spark

# Sclera: clean porcelain ivory with soft upper-eyelid blue-lavender cast shadow across top of eyeball!
sclera_base = np.array([250.0, 246.0, 249.0], dtype=np.float32)
upper_sclera_shd = np.clip((uv_v - 0.345) / 0.105, 0.0, 1.0)[..., None]
sclera_rgb = sclera_base * (1.0 - 0.24 * upper_sclera_shd) + np.array([168.0, 178.0, 212.0]) * (0.24 * upper_sclera_shd)
# Subtle warm pink caruncle at far left/right corners
caruncle = (np.clip((0.085 - uv_u) / 0.065, 0.0, 1.0) + np.clip((uv_u - 0.915) / 0.065, 0.0, 1.0))[..., None]
sclera_rgb = sclera_rgb * (1.0 - 0.22 * caruncle) + np.array([236.0, 182.0, 188.0]) * (0.22 * caruncle)

# Smooth blend between iris (r_iris <= 0.98) and sclera (r_iris >= 1.04) using RadDoll's exact iris boundary!
iris_mask = np.clip((1.03 - r_iris) / 0.065, 0.0, 1.0) * np.clip((1.0 - lum * 0.15), 0.0, 1.0)
iris_mask = np.clip((1.02 - r_iris) / 0.055, 0.0, 1.0)[..., None]

# Also apply soft upper-eyelid shadow band across top of iris too (Genshin/BV1aenu6YE59 signature!)
top_eye_shd = np.clip((uv_v - 0.355) / 0.085, 0.0, 1.0)[..., None]
iris_rgb = iris_rgb * (1.0 - 0.30 * top_eye_shd) + np.array([18.0, 32.0, 82.0]) * (0.30 * top_eye_shd)

lower_rgb = sclera_rgb * (1.0 - iris_mask) + iris_rgb * iris_mask
eye_out[in_lower, :3] = np.clip(lower_rgb[in_lower], 0.0, 255.0)
eye_out[in_lower, 3]  = 255.0

# --- 2) UPPER HALF (uv_v >= 0.528): 3D Floating Catchlights + Lashes + Brows + Double-Lid + Cornea Shadow ---
in_upper = ~in_lower
# Identify sub-zones in upper half:
# (a) 3D Floating Catchlight islands: uv_u in [0.495, 0.692] & uv_v in [0.818, 0.998] -> Pure glowing diamond white with crisp alpha!
is_catchlight = in_upper & (uv_u >= 0.495) & (uv_u <= 0.692) & (uv_v >= 0.818)
# (b) Upper-cornea soft shadow band: uv_u in [0.700, 1.000] & uv_v in [0.880, 0.996]
is_cornea_shd = in_upper & (uv_u >= 0.700) & (uv_v >= 0.880)
# (c) Double-eyelid crease: uv_u in [0.225, 0.360] & uv_v in [0.805, 0.845]
is_dbl_lid    = in_upper & (uv_u >= 0.225) & (uv_u <= 0.360) & (uv_v >= 0.805) & (uv_v <= 0.845)
# (d) Eyelashes & Eyebrows: uv_u in [0.525, 1.000] & uv_v in [0.528, 0.880] (plus lash tips)
is_lash_brow  = in_upper & (uv_u >= 0.525) & (~is_catchlight) & (~is_cornea_shd)

# Default upper: transparent
eye_out[in_upper, 3] = 0.0

# (a) Catchlights: crisp diamond white with slight ice-cyan halo at soft edges!
cl_a = np.clip(alpha_in * 1.15, 0.0, 1.0)
eye_out[is_catchlight, 0] = 248.0
eye_out[is_catchlight, 1] = 252.0
eye_out[is_catchlight, 2] = 255.0
eye_out[is_catchlight, 3] = cl_a[is_catchlight] * 255.0

# (b) Upper-cornea shadow band: translucent navy-plum shadow over top of eyeball!
cs_a = np.clip(alpha_in * 0.38, 0.0, 0.45)
eye_out[is_cornea_shd, 0] = 36.0
eye_out[is_cornea_shd, 1] = 44.0
eye_out[is_cornea_shd, 2] = 88.0
eye_out[is_cornea_shd, 3] = cs_a[is_cornea_shd] * 255.0

# (c) Double-lid crease: delicate warm rose-taupe line (not harsh black!)
eye_out[is_dbl_lid, 0] = 108.0
eye_out[is_dbl_lid, 1] = 64.0
eye_out[is_dbl_lid, 2] = 74.0
eye_out[is_dbl_lid, 3] = 255.0

# (d) Eyelashes & Brows: Rich plum-espresso core -> warm sienna-rose outer tips & inner/outer corner taper!
# Horizontal taper along lash UV (uv_u from 0.53 inner -> 0.99 outer corner!)
lash_t = np.clip((uv_u - 0.55) / 0.42, 0.0, 1.0)
tip_warm = np.power(lash_t, 1.6) * 0.65 + (1.0 - alpha_in) * 0.45
tip_warm = np.clip(tip_warm, 0.0, 1.0)
# Brow sub-band (uv_v in 0.528..0.795) is softer silver-taupe-slate to harmonize with Xiaoming's silver-blue hair!
is_brow_zone = is_lash_brow & (uv_v < 0.785) & (uv_u > 0.545)

lash_core = np.array([34.0, 18.0, 28.0], dtype=np.float32)
lash_tip  = np.array([124.0, 68.0, 78.0], dtype=np.float32)
lash_col  = lash_core * (1.0 - tip_warm[..., None]) + lash_tip * tip_warm[..., None]

brow_core = np.array([86.0, 74.0, 92.0], dtype=np.float32)
brow_tip  = np.array([138.0, 118.0, 132.0], dtype=np.float32)
brow_col  = brow_core * (1.0 - tip_warm[..., None]) + brow_tip * tip_warm[..., None]

eye_out[is_lash_brow, :3] = lash_col[is_lash_brow]
eye_out[is_brow_zone, :3] = brow_col[is_brow_zone]
eye_out[is_lash_brow, 3]  = 255.0

eye_path = OUT_TEX_DIR / "Xiaoming_Eye_4K_V10R19_PRIVATE.png"
Image.fromarray(np.clip(eye_out, 0, 255).astype(np.uint8), "RGBA").save(eye_path)
print("SAVED_EYE_4K:", eye_path.name, eye_path.stat().st_size, flush=True)

# ==============================================================================
# MODULE C: 4096x4096 Neutral-Anime Radiant "No-Makeup" Natural Complexion Face Map
# ==============================================================================
# Upsample 2K 3D position map cleanly to 4K for sub-pixel landmark makeup placement!
pos4k = cv2.resize(pos2k, (H, W), interpolation=cv2.INTER_LINEAR)
dval4k = cv2.resize(dvalid2k, (H, W), interpolation=cv2.INTER_NEAREST)

# Start from upstream 4K RadDollV3_Face_BaseColor.png:
# Keep 100% of left half (UV.u < 0.45: 4K inner-mouth teeth, tongue, oral cavity, ears!)
# On the face skin island (dval4k > 0), calibrate base skin tone to match RadDoll_body_tex_v2 ([252, 240, 238]) seamlessly
# while retaining 45% of upstream subtle skin contour + adding calibrated 原图 neutral-boyish healthy complexion!
face_out = face_up.copy()
target_skin = np.array([252.0, 240.0, 238.0], dtype=np.float32)
up_med = np.array([255.0, 232.6, 228.1], dtype=np.float32)

# Gentle shift on skin region so base matches body neck [252, 240, 238] while preserving upstream soft eye/cheek gradients!
skin_shifted = target_skin + (face_up[..., :3] - up_med) * 0.58
skin_shifted = np.clip(skin_shifted, 0.0, 255.0)

fx = pos4k[..., 0] - mid_x
fy = pos4k[..., 1]
fz = pos4k[..., 2]
front_mask = np.clip((-0.075 - fy) / 0.030, 0.0, 1.0) * (dval4k > 0)

# 1) Under-Eye to Cheekbone Healthy Peach-Coral Warmth (原图 signature — subtle, fresh, neutral!)
blush_z = eye_z - 0.022
blush_x = eye_abs_x * 1.04
b_l = np.exp(-(((fx - blush_x) / 0.025)**2 + ((fz - blush_z) / 0.015)**2))
b_r = np.exp(-(((fx + blush_x) / 0.025)**2 + ((fz - blush_z) / 0.015)**2))
blush = np.clip(b_l + b_r, 0.0, 1.0) * front_mask
peach_blush_col = np.array([248.0, 190.0, 188.0], dtype=np.float32)
skin_shifted = skin_shifted * (1.0 - 0.24 * blush[..., None]) + peach_blush_col * (0.24 * blush[..., None])

# 2) Upper/Lower Eyelid & Outer Canthus Warm Taupe-Rose Contour (eliminates harsh black-on-white lash contrast!)
el_l = np.exp(-(((fx - eye_abs_x*1.06) / 0.023)**2 + ((fz - (eye_z + 0.002)) / 0.014)**2))
el_r = np.exp(-(((fx + eye_abs_x*1.06) / 0.023)**2 + ((fz - (eye_z + 0.002)) / 0.014)**2))
eyeshadow = np.clip(el_l + el_r, 0.0, 1.0) * front_mask
taupe_rose = np.array([228.0, 182.0, 184.0], dtype=np.float32)
skin_shifted = skin_shifted * (1.0 - 0.24 * eyeshadow[..., None]) + taupe_rose * (0.24 * eyeshadow[..., None])

# 3) Nose Bridge Subtle Luminous Strip + Nose-Tip Warm Blushed Touch + Crisp Tiny Anime Nose Tick!
nt_glow = np.exp(-((fx / 0.009)**2 + ((fz - nose_tip[2]) / 0.008)**2)) * np.clip((-0.125 - fy) / 0.020, 0.0, 1.0)
skin_shifted = skin_shifted * (1.0 - 0.28 * nt_glow[..., None]) + np.array([244.0, 184.0, 182.0]) * (0.28 * nt_glow[..., None])

# Tiny delicate warm-brown anime nose dot/tick right on nose tip (原图 & BV1aenu6YE59 style!)
nose_dot = np.exp(-(((fx - 0.0012) / 0.0014)**2 + ((fz - (nose_tip[2] + 0.0015)) / 0.0024)**2)) * np.clip((-0.138 - fy) / 0.010, 0.0, 1.0)
skin_shifted = skin_shifted * (1.0 - 0.52 * nose_dot[..., None]) + np.array([148.0, 96.0, 98.0]) * (0.52 * nose_dot[..., None])

# Tiny crisp porcelaneous nose-bridge highlight just above nose tip!
nose_hi = np.exp(-(((fx + 0.0010) / 0.0018)**2 + ((fz - (nose_tip[2] + 0.0085)) / 0.0045)**2)) * np.clip((-0.132 - fy) / 0.014, 0.0, 1.0)
skin_shifted = skin_shifted * (1.0 - 0.45 * nose_hi[..., None]) + np.array([255.0, 252.0, 251.0]) * (0.45 * nose_hi[..., None])

# 4) Natural Clean Boyish/Neutral Coral-Peach Lip Center Warmth (NOT glossy feminine lipstick!)
lip_w = np.exp(-((fx / 0.0135)**2 + ((fz - lip_c[2]) / 0.0052)**2)) * np.clip((-0.112 - fy) / 0.020, 0.0, 1.0)
lip_tint = np.array([238.0, 168.0, 170.0], dtype=np.float32)
skin_shifted = skin_shifted * (1.0 - 0.36 * lip_w[..., None]) + lip_tint * (0.36 * lip_w[..., None])
# Tiny subtle lower-lip center moist highlight speck
lip_spec = np.exp(-(((fx - 0.0015) / 0.0028)**2 + ((fz - (lip_c[2] - 0.0020)) / 0.0014)**2)) * np.clip((-0.118 - fy) / 0.015, 0.0, 1.0)
skin_shifted = skin_shifted * (1.0 - 0.35 * lip_spec[..., None]) + np.array([255.0, 240.0, 240.0]) * (0.35 * lip_spec[..., None])

# Blend smoothly only onto the main right-side face skin UV region (uv_u > 0.44 & dval4k > 0), keeping mouth cavity & ear atlas intact!
apply_face_mask = ((uv_u > 0.44) & (dval4k > 0))
face_out[apply_face_mask, :3] = np.clip(skin_shifted[apply_face_mask], 0.0, 255.0)
face_out[..., 3] = 255.0

face_path = OUT_TEX_DIR / "Xiaoming_Face_4K_V10R19_PRIVATE.png"
Image.fromarray(np.clip(face_out, 0, 255).astype(np.uint8), "RGBA").save(face_path)
print("SAVED_FACE_4K:", face_path.name, face_path.stat().st_size, flush=True)
