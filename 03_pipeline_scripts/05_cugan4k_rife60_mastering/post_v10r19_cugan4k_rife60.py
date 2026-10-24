import json, shutil, subprocess, sys, time
from pathlib import Path
from PIL import Image

WDIR = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
RAW720_DIR = WDIR / "showcase_v10r19_raw720p_f0425_0604"
SR4K_DIR   = WDIR / "showcase_v10r19_cugan4k_f0425_0604"
RIFE_DIR   = WDIR / "showcase_v10r19_rife4k60_f0425_0604"
QA19       = WDIR / "qa_v10r19_ab_compare"
MV18       = WDIR / "mv_frames_v10r18_theater_hsr_6265"

RC_DIR   = r"C:\Tools\ncnn_anime_post\realcugan-ncnn-vulkan-20220728-windows"
RC_EXE   = rf"{RC_DIR}\realcugan-ncnn-vulkan.exe"
RIFE_CWD = r"C:\Tools\ncnn_anime_post\rife-ncnn-vulkan-20221029-windows"
RIFE_EXE = rf"{RIFE_CWD}\rife-ncnn-vulkan.exe"
FFMPEG_BIN  = r"C:\Users\Kurami_Utaho\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
FFPROBE_BIN = r"C:\Users\Kurami_Utaho\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe"

stage = sys.argv[1] if len(sys.argv) > 1 else "all"

if stage in ("cugan", "all"):
    t0 = time.time()
    assert len(list(RAW720_DIR.glob("frame_*.png"))) == 180, "Expected 180 raw 720p PNG frames"
    SR4K_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [RC_EXE, "-i", str(RAW720_DIR), "-o", str(SR4K_DIR), "-s", "3", "-n", "-1", "-t", "256",
         "-m", "models-pro", "-g", "1", "-j", "2:2:2", "-f", "jpg"],
        cwd=RC_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    sr_files = sorted(SR4K_DIR.glob("frame_*.jpg"))
    assert len(sr_files) == 180, f"Expected 180 4K SR frames, got {len(sr_files)}"
    assert Image.open(sr_files[0]).size == (3840, 2160), "Expected 3840x2160 4K output"
    print(f"STAGE_CUGAN_4K_OK: 180x (3840x2160) in {time.time()-t0:.1f}s", flush=True)

if stage in ("rife", "all"):
    t0 = time.time()
    if RIFE_DIR.exists():
        shutil.rmtree(RIFE_DIR)
    RIFE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_seg = WDIR / "_tmp_rife_seg"

    # Exact camera hard-cut boundaries inside f0425..f0604 (0-based indices [0..179]):
    # Hard cuts at (446,447)->idx 21|22, (457,458)->idx 32|33, (473,474)->idx 48|49, (540,541)->idx 115|116
    segments = [
        (0,   21),   # 22 frames -> 44 @ 60fps
        (22,  32),   # 11 frames -> 22 @ 60fps (0.90m ultra-close-up face!)
        (33,  48),   # 16 frames -> 32 @ 60fps (bust + whale hairpin!)
        (49,  115),  # 67 frames -> 134 @ 60fps (full-body dance!)
        (116, 179),  # 64 frames -> 128 @ 60fps (crystal star-window!)
    ]
    out_cursor = 1
    for s_i, (a, b) in enumerate(segments):
        if tmp_seg.exists():
            shutil.rmtree(tmp_seg)
        seg_in = tmp_seg / "in"
        seg_out = tmp_seg / "out"
        seg_in.mkdir(parents=True, exist_ok=True)
        seg_out.mkdir(parents=True, exist_ok=True)
        k_len = b - a + 1
        for k_idx, src_idx in enumerate(range(a, b + 1)):
            shutil.copy2(SR4K_DIR / f"frame_{src_idx:04d}.jpg", seg_in / f"{k_idx:08d}.jpg")

        subprocess.run(
            [RIFE_EXE, "-i", str(seg_in), "-o", str(seg_out), "-n", str(k_len * 2),
             "-m", "rife-v4.6", "-g", "1", "-u", "-j", "2:2:2", "-f", "%08d.jpg"],
            cwd=RIFE_CWD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        gen = sorted(seg_out.glob("*.jpg"))
        assert len(gen) == k_len * 2, f"Seg {s_i}: expected {k_len*2}, got {len(gen)}"
        for gf in gen:
            shutil.move(str(gf), str(RIFE_DIR / f"frame_{out_cursor:04d}.jpg"))
            out_cursor += 1
        print(f"  RIFE_SEG {s_i+1}/5 [{a}..{b}] -> {len(gen)} frames OK", flush=True)

    if tmp_seg.exists():
        shutil.rmtree(tmp_seg)
    assert out_cursor - 1 == 360, f"Expected 360 total 60fps frames, got {out_cursor-1}"
    print(f"STAGE_RIFE_4K60_OK: 360 cut-preserved 4K frames in {time.time()-t0:.1f}s", flush=True)

if stage in ("encode", "all"):
    t0 = time.time()
    audio_info = json.loads((WDIR / "audio_envelope_6265f.json").read_text(encoding="utf-8"))
    wav_path = audio_info["wav_path"]
    ss_sec = f"{(425 - 1) / 30.0:.6f}"

    out_4k60   = WDIR / "Xiaoming_V10R19_Genshin_CUGAN_RIFE_4K60_Showcase_PRIVATE.mp4"
    out_1080p60 = WDIR / "Xiaoming_V10R19_Genshin_FaceUpgrade_1080p60_Showcase_PRIVATE.mp4"

    # 1) Encode True 4K60 Master (BT.709, CRF 19, medium, yuv420p, AAC 192k Stereo)
    subprocess.run([
        FFMPEG_BIN, "-y",
        "-framerate", "60", "-i", str(RIFE_DIR / "frame_%04d.jpg"),
        "-ss", ss_sec, "-t", "6.000", "-i", wav_path,
        "-vf", "scale=3840:2160:out_color_matrix=bt709,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-movflags", "+faststart", str(out_4k60)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # 2) Encode 1080p60 Viewer Master (downsampled from 4K60 via Lanczos, BT.709, CRF 19)
    subprocess.run([
        FFMPEG_BIN, "-y",
        "-framerate", "60", "-i", str(RIFE_DIR / "frame_%04d.jpg"),
        "-ss", ss_sec, "-t", "6.000", "-i", wav_path,
        "-vf", "scale=1920:1080:flags=lanczos:out_color_matrix=bt709,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-movflags", "+faststart", str(out_1080p60)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Full CPU decode + ffprobe verification per CXP-2024/genshin-blender-render/references/upscale-interpolate.md!
    for mp4 in (out_4k60, out_1080p60):
        subprocess.run([FFMPEG_BIN, "-v", "error", "-i", str(mp4), "-f", "null", "-"], check=True)
        pr = subprocess.run([
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,color_space",
            "-of", "json", str(mp4)
        ], capture_output=True, text=True, check=True)
        st = json.loads(pr.stdout)["streams"][0]
        print(f"VERIFIED_MP4: {mp4.name} ({mp4.stat().st_size/1048576:.2f} MB) -> {st}", flush=True)

    # Also update the v4 (+5% luminosity) Real-CUGAN 2x 1080p->4K Face & FullBody A/B Comparison Sheets!
    for tag in ["f0111_closeup_front", "f0451_closeup_low", "f1271_closeup_34side"]:
        subprocess.run([RC_EXE, "-i", str(QA19 / f"{tag}.png"), "-o", str(QA19 / f"{tag}_cugan2x.png"),
                        "-s", "2", "-n", "-1", "-m", "models-pro", "-g", "1"],
                       cwd=RC_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    crops_1080 = [
        (111,  "f0111_closeup_front",  (735, 165, 1255, 685)),
        (451,  "f0451_closeup_low",    (700, 265, 1220, 785)),
        (1271, "f1271_closeup_34side", (600, 265, 1120, 785)),
    ]
    S = 700
    sh_face = Image.new("RGB", (S * 3, S * 2), (16, 16, 22))
    for i, (fno, tag, box) in enumerate(crops_1080):
        im18 = Image.open(MV18 / f"frame_{fno:04d}.jpg").convert("RGB").crop(box).resize((S, S), Image.Resampling.LANCZOS)
        box2x = tuple(c * 2 for c in box)
        im19 = Image.open(QA19 / f"{tag}_cugan2x.png").convert("RGB").crop(box2x).resize((S, S), Image.Resampling.LANCZOS)
        sh_face.paste(im18, (i * S, 0))
        sh_face.paste(im19, (i * S, S))
    sh_face.save(QA19 / "V10R19_原神崩铁清透少年感面部升级_前后特写对比.jpg", quality=94)

    BW, BH = 800, 450
    sh_body = Image.new("RGB", (BW * 3, BH * 2), (16, 16, 22))
    for i, (fno, tag) in enumerate([(241, "f0241_bust_whalepin"), (586, "f0586_mid_crystal"), (1086, "f1086_fullbody_kick")]):
        im18 = Image.open(MV18 / f"frame_{fno:04d}.jpg").convert("RGB").resize((BW, BH), Image.Resampling.LANCZOS)
        im19 = Image.open(QA19 / f"{tag}.png").convert("RGB").resize((BW, BH), Image.Resampling.LANCZOS)
        sh_body.paste(im18, (i * BW, 0))
        sh_body.paste(im19, (i * BW, BH))
    sh_body.save(QA19 / "V10R19_全身衣料发丝金属细描边升级_前后对比.jpg", quality=94)
    print(f"STAGE_ENCODE_AND_QA_OK in {time.time()-t0:.1f}s", flush=True)
