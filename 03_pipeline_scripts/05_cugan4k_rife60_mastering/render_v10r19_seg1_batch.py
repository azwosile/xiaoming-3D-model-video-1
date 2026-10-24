import sys, time
from pathlib import Path
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["1", "300"]
f_start, f_end = int(argv[0]), int(argv[1])

WDIR  = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
BLEND = WDIR / "RadDoll_V10R19_FullSong6265_KorolevskiyTheater_HSR_Genshin_Master_PRIVATE.blend"
ODIR  = WDIR / "mv_v10r19_seg1_raw720p"
ODIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
bpy.ops.wm.open_mainfile(filepath=str(BLEND))

sc = bpy.context.scene
sc.camera = bpy.data.objects["MMD_MV_CAMERA"]
sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 48
sc.render.use_persistent_data = True
sc.render.resolution_x = 1280
sc.render.resolution_y = 720
sc.render.resolution_percentage = 100
sc.render.fps = 30
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.image_settings.compression = 15
sc.render.use_overwrite = False
sc.render.use_placeholder = False
sc.render.filepath = str(ODIR / "frame_")
sc.frame_start = f_start
sc.frame_end   = f_end

bpy.ops.render.render(animation=True)
done_n = sum(1 for f in range(f_start, f_end + 1) if (ODIR / f"frame_{f:04d}.png").exists() and (ODIR / f"frame_{f:04d}.png").stat().st_size > 20000)
print(f"V10R19_SEG1_BATCH_DONE {f_start}..{f_end}: {done_n}/{f_end - f_start + 1} frames in {time.time() - t0:.1f}s", flush=True)
