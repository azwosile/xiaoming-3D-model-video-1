import json, shutil, subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WDIR = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
RAW720_DIR = WDIR / "mv_v10r19_seg2_raw720p"
SR4K_DIR   = WDIR / "mv_v10r19_seg2_cugan4k"
RIFE_DIR   = WDIR / "mv_v10r19_seg2_rife4k60"
CUTS_JSON  = WDIR / "camera_hard_cuts_6265.json"

SEG_F0, SEG_F1 = 2401, 4320
SEG_LEN = SEG_F1 - SEG_F0 + 1  # 1920

RC_DIR   = r"C:\Tools\ncnn_anime_post\realcugan-ncnn-vulkan-20220728-windows"
RC_EXE   = rf"{RC_DIR}\realcugan-ncnn-vulkan.exe"
RIFE_CWD = r"C:\Tools\ncnn_anime_post\rife-ncnn-vulkan-20221029-windows"
RIFE_EXE = rf"{RIFE_CWD}\rife-ncnn-vulkan.exe"
FFMPEG_BIN  = r"C:\Users\Kurami_Utaho\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
FFPROBE_BIN = r"C:\Users\Kurami_Utaho\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe"

stage = sys.argv[1]

def build_shots_seg2():
    cuts = json.loads(CUTS_JSON.read_text(encoding="utf-8"))["cuts_seg2_2401_4320"]
    shots = []
    cur = SEG_F0
    for a, b in cuts:
        assert b == a + 1 and cur <= a < SEG_F1
        shots.append((cur, a))
        cur = b
    shots.append((cur, SEG_F1))
    assert sum(b - a + 1 for a, b in shots) == SEG_LEN
    return shots

if stage == "verify_png":
    t0 = time.time()
    bad = []
    for f in range(SEG_F0, SEG_F1 + 1):
        p = RAW720_DIR / f"frame_{f:04d}.png"
        if not p.exists() or p.stat().st_size < 20000:
            bad.append(f)
            continue
        with open(p, "rb") as fp:
            fp.seek(-8, 2)
            if fp.read(8) != b"IEND\xaeB`\x82":
                bad.append(f)
    if bad:
        print(f"BAD_PNG_FRAMES: {bad}", flush=True)
        sys.exit(2)
    SR4K_DIR.mkdir(parents=True, exist_ok=True)
    shots = build_shots_seg2()
    one_f = [s for s in shots if s[0] == s[1]]
    print(f"VERIFY_SEG2_PNG_1920_OK in {time.time()-t0:.2f}s | shots={len(shots)} | 1f_flash={one_f}", flush=True)

elif stage == "cugan":
    f_start, f_end = int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    SR4K_DIR.mkdir(parents=True, exist_ok=True)
    todo = [f for f in range(f_start, f_end + 1)
            if not ((SR4K_DIR / f"frame_{f:04d}.jpg").exists() and (SR4K_DIR / f"frame_{f:04d}.jpg").stat().st_size > 50000)]
    if todo:
        tmp_in  = WDIR / "_tmp_seg2_cugan_in"
        tmp_out = WDIR / "_tmp_seg2_cugan_out"
        for d in (tmp_in, tmp_out):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        for f in todo:
            shutil.copy2(RAW720_DIR / f"frame_{f:04d}.png", tmp_in / f"frame_{f:04d}.png")
        subprocess.run(
            [RC_EXE, "-i", str(tmp_in), "-o", str(tmp_out), "-s", "3", "-n", "-1", "-t", "256",
             "-m", "models-pro", "-g", "1", "-j", "2:2:2", "-f", "jpg"],
            cwd=RC_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        for f in todo:
            gp = tmp_out / f"frame_{f:04d}.jpg"
            assert gp.exists() and Image.open(gp).size == (3840, 2160), f"Missing/bad 4K frame {f}"
            shutil.move(str(gp), str(SR4K_DIR / f"frame_{f:04d}.jpg"))
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)
    total_4k = len(list(SR4K_DIR.glob("frame_*.jpg")))
    if total_4k == SEG_LEN and RAW720_DIR.exists():
        shutil.rmtree(RAW720_DIR, ignore_errors=True)
    print(f"CUGAN_SEG2_BATCH_OK {f_start}..{f_end} (ran={len(todo)}) in {time.time()-t0:.1f}s | total_4k={total_4k}/{SEG_LEN}", flush=True)

elif stage == "rife":
    s_start, s_end = int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    RIFE_DIR.mkdir(parents=True, exist_ok=True)
    shots = build_shots_seg2()
    tmp_seg = WDIR / "_tmp_seg2_rife"
    ran_shots = 0
    ran_out = 0

    for s_i in range(s_start, min(s_end + 1, len(shots))):
        a, b = shots[s_i]
        k_len = b - a + 1
        rel_a = a - SEG_F0 + 1
        rel_b = b - SEG_F0 + 1
        out_a = 2 * (rel_a - 1) + 1
        out_b = 2 * rel_b
        if all((RIFE_DIR / f"frame_{k:05d}.jpg").exists() and (RIFE_DIR / f"frame_{k:05d}.jpg").stat().st_size > 50000
               for k in range(out_a, out_b + 1)):
            continue
        if k_len == 1:
            shutil.copy2(SR4K_DIR / f"frame_{a:04d}.jpg", RIFE_DIR / f"frame_{out_a:05d}.jpg")
            shutil.copy2(SR4K_DIR / f"frame_{a:04d}.jpg", RIFE_DIR / f"frame_{out_b:05d}.jpg")
            ran_shots += 1
            ran_out += 2
            continue

        if tmp_seg.exists():
            shutil.rmtree(tmp_seg)
        seg_in  = tmp_seg / "in"
        seg_out = tmp_seg / "out"
        seg_in.mkdir(parents=True, exist_ok=True)
        seg_out.mkdir(parents=True, exist_ok=True)
        for k_idx, f in enumerate(range(a, b + 1)):
            shutil.copy2(SR4K_DIR / f"frame_{f:04d}.jpg", seg_in / f"{k_idx:08d}.jpg")

        subprocess.run(
            [RIFE_EXE, "-i", str(seg_in), "-o", str(seg_out), "-n", str(k_len * 2),
             "-m", "rife-v4.6", "-g", "1", "-u", "-j", "2:2:2", "-f", "%08d.jpg"],
            cwd=RIFE_CWD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        gen = sorted(seg_out.glob("*.jpg"))
        assert len(gen) == k_len * 2, f"Shot {s_i} [{a}..{b}]: expected {k_len*2}, got {len(gen)}"
        for idx_g, gf in enumerate(gen):
            shutil.move(str(gf), str(RIFE_DIR / f"frame_{out_a + idx_g:05d}.jpg"))
        ran_shots += 1
        ran_out += len(gen)

    if tmp_seg.exists():
        shutil.rmtree(tmp_seg, ignore_errors=True)
    total_60 = len(list(RIFE_DIR.glob("frame_*.jpg")))
    print(f"RIFE_SEG2_BATCH_OK shots[{s_start}..{s_end}] (ran_shots={ran_shots}, ran_out={ran_out}) in {time.time()-t0:.1f}s | total_60={total_60}/{SEG_LEN*2}", flush=True)

elif stage == "encode_4k":
    t0 = time.time()
    cnt_60 = len(list(RIFE_DIR.glob("frame_*.jpg")))
    assert cnt_60 == SEG_LEN * 2, f"Expected {SEG_LEN*2} RIFE 4K60 frames, got {cnt_60}"
    wav_path = json.loads((WDIR / "audio_envelope_6265f.json").read_text(encoding="utf-8"))["wav_path"]

    out_4k60 = WDIR / "Xiaoming_V10R19_KorolevskiyTheater_Genshin_CUGAN_RIFE_Seg2_Chorus2_4K60_Master_PRIVATE.mp4"
    subprocess.run([
        FFMPEG_BIN, "-y",
        "-framerate", "60", "-i", str(RIFE_DIR / "frame_%05d.jpg"),
        "-ss", "80.000000", "-t", "64.000", "-i", wav_path,
        "-vf", "scale=3840:2160:out_color_matrix=bt709,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-movflags", "+faststart", str(out_4k60)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    subprocess.run([FFMPEG_BIN, "-v", "error", "-i", str(out_4k60), "-f", "null", "-"], check=True)
    pr = subprocess.run([
        FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,color_space",
        "-of", "json", str(out_4k60)
    ], capture_output=True, text=True, check=True)
    st = json.loads(pr.stdout)["streams"][0]
    print(f"ENCODE_SEG2_4K60_OK in {time.time()-t0:.1f}s | {out_4k60.name} ({out_4k60.stat().st_size/1048576:.2f} MB) -> {st}", flush=True)

elif stage == "encode_1080_and_qa":
    t0 = time.time()
    wav_path = json.loads((WDIR / "audio_envelope_6265f.json").read_text(encoding="utf-8"))["wav_path"]

    out_1080_host   = WDIR / "Xiaoming_V10R19_KorolevskiyTheater_Genshin_CUGAN_RIFE_Seg2_Chorus2_1080p60_Master_PRIVATE.mp4"
    out_1080_viewer = WDIR / "Xiaoming_V10R19_KorolevskiyTheater_Genshin_CUGAN_RIFE_Seg2_Chorus2_1080p60_MV_PRIVATE.mp4"

    subprocess.run([
        FFMPEG_BIN, "-y",
        "-framerate", "60", "-i", str(RIFE_DIR / "frame_%05d.jpg"),
        "-ss", "80.000000", "-t", "64.000", "-i", wav_path,
        "-vf", "scale=1920:1080:flags=lanczos:out_color_matrix=bt709,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-movflags", "+faststart", str(out_1080_host)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    subprocess.run([
        FFMPEG_BIN, "-y",
        "-framerate", "60", "-i", str(RIFE_DIR / "frame_%05d.jpg"),
        "-ss", "80.000000", "-t", "64.000", "-i", wav_path,
        "-vf", "scale=1920:1080:flags=lanczos:out_color_matrix=bt709,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "22.5", "-maxrate", "2350k", "-bufsize", "4700k",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "160k", "-ac", "2",
        "-movflags", "+faststart", str(out_1080_viewer)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    for mp4 in (out_1080_host, out_1080_viewer):
        subprocess.run([FFMPEG_BIN, "-v", "error", "-i", str(mp4), "-f", "null", "-"], check=True)
        pr = subprocess.run([
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,color_space",
            "-of", "json", str(mp4)
        ], capture_output=True, text=True, check=True)
        st = json.loads(pr.stdout)["streams"][0]
        print(f"VERIFIED_SEG2_1080P60: {mp4.name} ({mp4.stat().st_size/1048576:.2f} MB) -> {st}", flush=True)

    picks = [2445, 2530, 2660, 2835, 2960, 3140, 3350, 3560, 3760, 3960, 4135, 4295]
    CW, CH = 640, 360
    grid = Image.new("RGB", (CW * 4, CH * 3), (14, 15, 22))
    try:
        fnt = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 20)
    except Exception:
        fnt = ImageFont.load_default()
    for idx_p, fno in enumerate(picks):
        im = Image.open(SR4K_DIR / f"frame_{fno:04d}.jpg").convert("RGB").resize((CW, CH), Image.Resampling.LANCZOS)
        dr = ImageDraw.Draw(im)
        sec = (fno - 1) / 30.0
        lbl = f"f{fno:04d} ({int(sec//60)}:{sec%60:04.1f}) · V10R19 4K60"
        dr.rectangle((10, 10, 292, 42), fill=(12, 16, 28))
        dr.text((18, 13), lbl, fill=(235, 244, 255), font=fnt)
        gx, gy = (idx_p % 4) * CW, (idx_p // 4) * CH
        grid.paste(im, (gx, gy))
    qa_out = WDIR / "V10R19_皇家大剧院_原神崩铁满血4K60管线_第2副歌段1920帧12宫格总览.jpg"
    grid.save(qa_out, quality=93)

    shutil.rmtree(RIFE_DIR, ignore_errors=True)
    print(f"ENCODE_SEG2_1080_AND_QA_OK in {time.time()-t0:.1f}s | qa={qa_out.stat().st_size/1024:.1f} KB", flush=True)
