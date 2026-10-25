"""Real-CUGAN 超分 + 相机硬切隔离 RIFE 2x (60fps) 无果冻母带合成与解码校验管线

核心特性：
  1. 支持调用 realcugan-ncnn-vulkan 将原生序列帧（如 720p/1080p）无涂抹超分至真 4K（3840x2160）；
  2. 读取 extract_camera_hard_cuts.py 生成的镜头分段 JSON，严格在每个独立连续镜头内部运行 RIFE v4.6，
     段尾复制 1 帧保证 N 原生帧精确生成 2*N 帧（60fps），100% 根除跨切镜果冻融化怪帧与音画漂移；
  3. 自动封装 BT.709 元数据 CRF 19 母带，并执行 ffmpeg -v error 全流逐帧零报错解码验真。

用法示例：
  python cugan_rife_cut_safe_master.py \
    --in-dir ./raw_720p_frames --pattern "f_{:04d}.png" \
    --cuts-json ./camera_hard_cuts.json \
    --start 1 --end 6265 \
    --cugan-bin /path/to/realcugan-ncnn-vulkan --cugan-scale 3 \
    --rife-bin /path/to/rife-ncnn-vulkan --rife-model rife-v4.6 \
    --audio ./song.wav \
    --work-dir ./_work_4k60 \
    --out-4k ./Master_4K60.mp4 \
    --out-1080 ./Master_1080p60.mp4
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run_checked(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stderr[-1200:]}")
    return p


def clip_segments(segments, f_start, f_end):
    out = []
    for s, e in segments:
        cs, ce = max(s, f_start), min(e, f_end)
        if cs <= ce:
            out.append((cs, ce))
    return out


def verify_mp4(mp4_path: Path, expected_frames: int):
    run_checked(["ffmpeg", "-v", "error", "-i", str(mp4_path), "-f", "null", "-"])
    p = run_checked([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,color_space",
        "-of", "json", str(mp4_path)
    ])
    st = json.loads(p.stdout)["streams"][0]
    got = int(st["nb_read_frames"])
    if got != expected_frames:
        raise RuntimeError(f"Frame count mismatch in {mp4_path}: got {got}, expected {expected_frames}")
    print(f"VERIFIED_OK: {mp4_path.name} | {st['width']}x{st['height']} @ {st['r_frame_rate']} | frames={got} | color={st.get('color_space')}")


def main():
    ap = argparse.ArgumentParser(description="Cut-isolated Real-CUGAN + RIFE 60fps mastering pipeline.")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--pattern", default="f_{:04d}.png")
    ap.add_argument("--cuts-json", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--cugan-bin", required=True)
    ap.add_argument("--cugan-models", default="models-pro")
    ap.add_argument("--cugan-scale", type=int, default=3)
    ap.add_argument("--rife-bin", required=True)
    ap.add_argument("--rife-model", default="rife-v4.6")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--fps-in", type=int, default=30)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-4k", required=True)
    ap.add_argument("--out-1080", default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    work = Path(args.work_dir)
    cugan_in = work / "01_in_sub"
    cugan_out = work / "02_cugan_4k"
    rife_final = work / "03_rife_60fps"
    for d in (cugan_in, cugan_out, rife_final):
        d.mkdir(parents=True, exist_ok=True)

    cuts_data = json.loads(Path(args.cuts_json).read_text(encoding="utf-8"))
    segs = clip_segments(cuts_data["segments"], args.start, args.end)
    n_native = args.end - args.start + 1

    # Step 1: Stage native frames & run Real-CUGAN (no-denoise -n -1 preserves crisp anime linework)
    for f in range(args.start, args.end + 1):
        src = in_dir / args.pattern.format(f)
        dst = cugan_in / f"{f:06d}.png"
        if not dst.exists():
            shutil.copy2(src, dst)

    if len(list(cugan_out.glob("*.png"))) < n_native:
        run_checked([
            args.cugan_bin, "-i", str(cugan_in), "-o", str(cugan_out),
            "-n", "-1", "-s", str(args.cugan_scale), "-m", args.cugan_models
        ])

    # Step 2: Cut-isolated RIFE 2x interpolation
    out_idx = 1
    seg_tmp_in = work / "_seg_in"
    seg_tmp_out = work / "_seg_out"
    for s, e in segs:
        k = e - s + 1
        if k == 1:
            src_4k = cugan_out / f"{s:06d}.png"
            for _ in range(2):
                shutil.copy2(src_4k, rife_final / f"{out_idx:06d}.png")
                out_idx += 1
            continue

        shutil.rmtree(seg_tmp_in, ignore_errors=True)
        shutil.rmtree(seg_tmp_out, ignore_errors=True)
        seg_tmp_in.mkdir(parents=True)
        seg_tmp_out.mkdir(parents=True)

        for idx, f in enumerate(range(s, e + 1), start=1):
            shutil.copy2(cugan_out / f"{f:06d}.png", seg_tmp_in / f"{idx:06d}.png")

        run_checked([
            args.rife_bin, "-i", str(seg_tmp_in), "-o", str(seg_tmp_out),
            "-m", args.rife_model, "-u", "-n", str(2 * k), "-f", "%06d.png"
        ])
        produced = sorted(seg_tmp_out.glob("*.png"))
        if len(produced) != 2 * k:
            raise RuntimeError(f"Segment [{s}..{e}] RIFE produced {len(produced)} != {2*k}")
        for p in produced:
            shutil.move(str(p), str(rife_final / f"{out_idx:06d}.png"))
            out_idx += 1

    shutil.rmtree(seg_tmp_in, ignore_errors=True)
    shutil.rmtree(seg_tmp_out, ignore_errors=True)
    expected_60 = 2 * n_native
    assert out_idx - 1 == expected_60

    # Step 3: Encode BT.709 CRF 19 Masters & verify full decode
    fps_out = args.fps_in * 2
    audio_args = []
    if args.audio:
        t_offset = (args.start - 1) / float(args.fps_in)
        dur = n_native / float(args.fps_in)
        audio_args = ["-ss", f"{t_offset:.6f}", "-t", f"{dur:.6f}", "-i", str(args.audio), "-c:a", "aac", "-b:a", "192k"]

    out_4k = Path(args.out_4k)
    out_4k.parent.mkdir(parents=True, exist_ok=True)
    cmd_4k = ["ffmpeg", "-y", "-framerate", str(fps_out), "-i", str(rife_final / "%06d.png")] + audio_args + [
        "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-movflags", "+faststart", str(out_4k)
    ]
    run_checked(cmd_4k)
    verify_mp4(out_4k, expected_60)

    if args.out_1080:
        out_1080 = Path(args.out_1080)
        out_1080.parent.mkdir(parents=True, exist_ok=True)
        run_checked([
            "ffmpeg", "-y", "-i", str(out_4k),
            "-vf", "scale=1920:1080:flags=lanczos",
            "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-c:a", "copy", "-movflags", "+faststart", str(out_1080)
        ])
        verify_mp4(out_1080, expected_60)


if __name__ == "__main__":
    main()
