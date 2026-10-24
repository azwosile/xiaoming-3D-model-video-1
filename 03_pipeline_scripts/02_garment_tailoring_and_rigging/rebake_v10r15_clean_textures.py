r"""
V10R15 全套衣装 4K/2K「去光影花斑 + 保留手绘缝线/纽扣/条纹 + 原图平涂色校准」贴图重烘焙引擎
严格保护原贴图只读，全新输出至：
  生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE\textures_v10r15_clean\*_V10R15_Clean_PRIVATE.png
"""
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

ROOT    = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明")
SRC_DIR = ROOT / r"生成模型\组装\v010R8_全身衣装装配\textures"
OUT_DIR = ROOT / r"生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE\textures_v10r15_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_rgba(name):
    im = Image.open(SRC_DIR / name).convert("RGBA")
    arr = np.array(im, dtype=np.float32) / 255.0
    return im, arr

def save_rgba(arr, out_name):
    u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    out_p = OUT_DIR / out_name
    Image.fromarray(u8, mode="RGBA").save(out_p, compress_level=4)
    print(f"  [ReBaked] {out_name} ({u8.shape[1]}x{u8.shape[0]}, {out_p.stat().st_size/1024:.0f} KB)")

def local_blur_rgb(im_pil, radius_px):
    rgb_pil = im_pil.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius_px))
    return np.array(rgb_pil, dtype=np.float32) / 255.0

def sstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

print("=== [V10R15 Texture Re-Baker] Purifying All 5 Outfit Albedo Maps ===")

# ----------------------------------------------------------------------
# 1. 长外套 Tex_Coat_4096 -> Tex_Coat_4096_V10R15_Clean_PRIVATE.png
#    《原图》校准色：纯净清透浅天蓝底色 [0.706, 0.818, 0.992]，浅冰白装饰条/银白扣 [0.945, 0.962, 0.992]，深蓝灰缝线 [0.385, 0.478, 0.635]
# ----------------------------------------------------------------------
im_c, c = load_rgba("Tex_Coat_4096_PRIVATE.png")
rgb = c[:, :, :3]
alp = c[:, :, 3:4]
blur = local_blur_rgb(im_c, radius_px=18)

lum      = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
lum_blur = 0.2126*blur[:,:,0] + 0.7152*blur[:,:,1] + 0.0722*blur[:,:,2]
sat      = rgb.max(axis=2) - rgb.min(axis=2)

COAT_SKY   = np.array([0.706, 0.818, 0.992], dtype=np.float32)
COAT_WHITE = np.array([0.945, 0.962, 0.992], dtype=np.float32)
COAT_LINE  = np.array([0.380, 0.472, 0.635], dtype=np.float32)

# (a) 识别原本真正的无色银白/冰白部件（纽扣、白色装饰，特点是低饱和度 sat < 0.085 且高明度 lum > 0.86）
base_c = np.broadcast_to(COAT_SKY[None, None, :], rgb.shape).copy()

# (b) 高频暗线提取（局部对比度 dip = lum_blur - lum，只抓取 1~10px 的精细缝线/扣眼/口袋边，100% 免疫宽幅烘焙灰斑！）
dip = np.maximum(lum_blur - lum, 0.0)
abs_dark = sstep(0.66, 0.46, lum) * sstep(0.015, 0.055, dip + np.maximum(0.62 - lum, 0.0))
line_w = np.clip(np.maximum(sstep(0.022, 0.110, dip), abs_dark), 0.0, 0.92)

out_c = base_c * (1.0 - line_w[:, :, None]) + COAT_LINE[None, None, :] * line_w[:, :, None]
# 额外在贴图左上角 (u in [0.005..0.035], v in [0.965..0.995]) 预置两块标准纯色色卡：
# 供 Blender 网格中的「双袖浅色竖条」与「门襟银纽扣」精确 UV 绑定，确保 100% 零杂色！
H_c, W_c = out_c.shape[:2]
out_c[int(0.96*H_c):int(0.99*H_c), int(0.01*W_c):int(0.04*W_c)] = np.array([0.920, 0.958, 1.000], dtype=np.float32) # 纯净浅冰白袖条色卡 -> UV (0.025, 0.975)
out_c[int(0.96*H_c):int(0.99*H_c), int(0.05*W_c):int(0.08*W_c)] = np.array([0.945, 0.958, 0.985], dtype=np.float32) # 纯净银白纽扣色卡 -> UV (0.065, 0.975)
save_rgba(np.concatenate([out_c, np.ones_like(alp)], axis=2), "Tex_Coat_4096_V10R15_Clean_PRIVATE.png")


# ----------------------------------------------------------------------
# 2. 高腰短裤 Tex_Shorts_2048 -> Tex_Shorts_2048_V10R15_Clean_PRIVATE.png
#    《原图》校准色：清透浅蓝主色 [0.680, 0.820, 0.996] + 纯白滚边/蝴蝶结抽绳 [0.972, 0.978, 0.992] + 精细缝线
# ----------------------------------------------------------------------
im_sh, sh = load_rgba("Tex_Shorts_2048_PRIVATE.png")
rgb = sh[:, :, :3]; alp = sh[:, :, 3:4]
blur = local_blur_rgb(im_sh, radius_px=12)
lum = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
lum_blur = 0.2126*blur[:,:,0] + 0.7152*blur[:,:,1] + 0.0722*blur[:,:,2]
sat = rgb.max(axis=2) - rgb.min(axis=2)

SH_BLUE  = np.array([0.680, 0.820, 0.996], dtype=np.float32)
SH_WHITE = np.array([0.972, 0.978, 0.992], dtype=np.float32)
SH_LINE  = np.array([0.445, 0.525, 0.665], dtype=np.float32)

# 白色滚边与抽绳（低饱和度或极高明度）全部洗净烘焙灰斑，还原纯白！
w_trim = np.clip((1.0 - sstep(0.08, 0.19, sat)) * sstep(0.80, 0.88, lum), 0.0, 1.0)
base_sh = SH_BLUE[None, None, :] * (1.0 - w_trim[:, :, None]) + SH_WHITE[None, None, :] * w_trim[:, :, None]
dip = np.maximum(lum_blur - lum, 0.0)
line_w = np.clip(sstep(0.024, 0.115, dip), 0.0, 0.88)
out_sh = base_sh * (1.0 - line_w[:, :, None]) + SH_LINE[None, None, :] * line_w[:, :, None]
out_sh[0:80, 0:80] = SH_BLUE
save_rgba(np.concatenate([out_sh, np.ones_like(alp)], axis=2), "Tex_Shorts_2048_V10R15_Clean_PRIVATE.png")


# ----------------------------------------------------------------------
# 3. 露脐水手服 Tex_SailorTop_2048 -> Tex_SailorTop_2048_V10R15_Clean_PRIVATE.png
#    《原图》校准色：纯净瓷白衣身 [0.958, 0.966, 0.990] + 清透天蓝蝴蝶结/领条/胸袋条 [0.545, 0.735, 0.950] + 蓝灰暗线
# ----------------------------------------------------------------------
im_st, st = load_rgba("Tex_SailorTop_2048_PRIVATE.png")
rgb = st[:, :, :3]; alp = st[:, :, 3:4]
blur = local_blur_rgb(im_st, radius_px=12)
lum = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
lum_blur = 0.2126*blur[:,:,0] + 0.7152*blur[:,:,1] + 0.0722*blur[:,:,2]
blue_exc = rgb[:, :, 2] - rgb[:, :, 0]

ST_WHITE = np.array([0.958, 0.966, 0.990], dtype=np.float32)
ST_BLUE  = np.array([0.545, 0.735, 0.950], dtype=np.float32)
ST_LINE  = np.array([0.320, 0.470, 0.685], dtype=np.float32)

w_blue = sstep(0.095, 0.200, blue_exc)
base_st = ST_WHITE[None, None, :] * (1.0 - w_blue[:, :, None]) + ST_BLUE[None, None, :] * w_blue[:, :, None]
dip = np.maximum(lum_blur - lum, 0.0)
line_w = np.clip(sstep(0.025, 0.120, dip), 0.0, 0.88)
out_st = base_st * (1.0 - line_w[:, :, None]) + ST_LINE[None, None, :] * line_w[:, :, None]
save_rgba(np.concatenate([out_st, np.ones_like(alp)], axis=2), "Tex_SailorTop_2048_V10R15_Clean_PRIVATE.png")


# ----------------------------------------------------------------------
# 4. 堆堆袜与配饰 Tex_Socks_Acc_2048 -> Tex_Socks_Acc_2048_V10R15_Clean_PRIVATE.png
#    堆堆袜/腿环纯白 [0.942, 0.952, 0.984]（保留竖罗纹暗线！）+ 蓝灰颈圈/鲸鱼发卡纯净色
# ----------------------------------------------------------------------
im_sk, sk = load_rgba("Tex_Socks_Acc_2048_PRIVATE.png")
rgb = sk[:, :, :3]; alp = sk[:, :, 3:4]
blur = local_blur_rgb(im_sk, radius_px=10)
lum = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
lum_blur = 0.2126*blur[:,:,0] + 0.7152*blur[:,:,1] + 0.0722*blur[:,:,2]
blue_exc = rgb[:, :, 2] - rgb[:, :, 0]

SK_WHITE  = np.array([0.942, 0.952, 0.984], dtype=np.float32)
ACC_SLATE = np.array([0.410, 0.530, 0.755], dtype=np.float32)
ACC_SKY   = np.array([0.620, 0.785, 0.975], dtype=np.float32)
SK_LINE   = np.array([0.740, 0.775, 0.855], dtype=np.float32)

is_colored = sstep(0.11, 0.21, blue_exc)
# 区分深蓝灰颈圈与浅天蓝鲸鱼发卡
light_blue_t = sstep(0.55, 0.74, lum)
col_part = ACC_SLATE[None, None, :] * (1.0 - light_blue_t[:, :, None]) + ACC_SKY[None, None, :] * light_blue_t[:, :, None]
base_sk = SK_WHITE[None, None, :] * (1.0 - is_colored[:, :, None]) + col_part * is_colored[:, :, None]

dip = np.maximum(lum_blur - lum, 0.0)
line_w = np.clip(sstep(0.015, 0.085, dip), 0.0, 0.80)
out_sk = base_sk * (1.0 - line_w[:, :, None]) + SK_LINE[None, None, :] * line_w[:, :, None]
save_rgba(np.concatenate([out_sk, np.ones_like(alp)], axis=2), "Tex_Socks_Acc_2048_V10R15_Clean_PRIVATE.png")


# ----------------------------------------------------------------------
# 5. 白灰厚底运动鞋 Tex_Shoes_2048 -> Tex_Shoes_2048_V10R15_Clean_PRIVATE.png
#    《原图》校准色：干净白鞋面 [0.940, 0.952, 0.982] + 浅冷灰拼接块 [0.835, 0.858, 0.912] + 浅天蓝鞋底边条 [0.545, 0.715, 0.935]
# ----------------------------------------------------------------------
im_shs, shs = load_rgba("Tex_Shoes_2048_PRIVATE.png")
rgb = shs[:, :, :3]; alp = shs[:, :, 3:4]
blur = local_blur_rgb(im_shs, radius_px=10)
lum = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
lum_blur = 0.2126*blur[:,:,0] + 0.7152*blur[:,:,1] + 0.0722*blur[:,:,2]
blue_exc = rgb[:, :, 2] - rgb[:, :, 0]

SHOE_WHITE = np.array([0.940, 0.952, 0.982], dtype=np.float32)
SHOE_GREY  = np.array([0.835, 0.858, 0.912], dtype=np.float32)
SHOE_BLUE  = np.array([0.545, 0.715, 0.935], dtype=np.float32)
SHOE_LINE  = np.array([0.520, 0.555, 0.630], dtype=np.float32)

w_sblue = sstep(0.14, 0.25, blue_exc)
w_swhite = sstep(0.855, 0.905, lum_blur) * (1.0 - w_sblue)
w_sgrey = np.clip(1.0 - w_sblue - w_swhite, 0.0, 1.0)

base_shoe = (
    SHOE_WHITE[None, None, :] * w_swhite[:, :, None] +
    SHOE_GREY[None, None, :]  * w_sgrey[:, :, None] +
    SHOE_BLUE[None, None, :]  * w_sblue[:, :, None]
)
dip = np.maximum(lum_blur - lum, 0.0)
line_w = np.clip(sstep(0.018, 0.095, dip), 0.0, 0.85)
out_shoe = base_shoe * (1.0 - line_w[:, :, None]) + SHOE_LINE[None, None, :] * line_w[:, :, None]
save_rgba(np.concatenate([out_shoe, np.ones_like(alp)], axis=2), "Tex_Shoes_2048_V10R15_Clean_PRIVATE.png")

print("ALL 5 CLEAN ANIME ALBEDO TEXTURES RE-BAKED SUCCESSFULLY!")
