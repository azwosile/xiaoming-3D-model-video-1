"""全时间轴相机硬切检测与连续镜头分段导出工具（Blender 5.x 后台 CLI）

用途：
  为 RIFE / 光流补帧提供精确到帧的连续镜头分段表，彻底消灭 MMD 硬切处的「果冻融化怪帧」。

用法示例：
  blender -b stage_mv.blend --factory-startup --python-exit-code 1 \
    -P extract_camera_hard_cuts.py -- \
    --start 1 --end 6265 \
    --loc-thresh 0.35 --rot-deg-thresh 18.0 --lens-thresh 3.0 \
    --out camera_hard_cuts.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="Extract exact camera hard cuts across timeline.")
    ap.add_argument("--camera", default=None, help="Camera object name (default: scene.camera)")
    ap.add_argument("--start", type=int, default=None, help="Start frame")
    ap.add_argument("--end", type=int, default=None, help="End frame")
    ap.add_argument("--loc-thresh", type=float, default=0.35, help="World position jump threshold in meters")
    ap.add_argument("--rot-deg-thresh", type=float, default=18.0, help="World rotation jump threshold in degrees")
    ap.add_argument("--lens-thresh", type=float, default=3.0, help="Focal length jump threshold in mm")
    ap.add_argument("--out", required=True, help="Output JSON file path")
    return ap.parse_args(argv)


def quat_angle_deg(q1, q2) -> float:
    dot = min(1.0, abs(q1.normalized().dot(q2.normalized())))
    return math.degrees(2.0 * math.acos(dot))


def main():
    args = parse_args()
    scene = bpy.context.scene
    cam = bpy.data.objects.get(args.camera) if args.camera else scene.camera
    if cam is None or cam.type != "CAMERA":
        raise RuntimeError("Valid active camera not found.")

    f_start = args.start if args.start is not None else scene.frame_start
    f_end = args.end if args.end is not None else scene.frame_end

    cuts = []
    prev_loc = None
    prev_quat = None
    prev_lens = None

    for f in range(f_start, f_end + 1):
        scene.frame_set(f)
        deps = bpy.context.evaluated_depsgraph_get()
        ecam = cam.evaluated_get(deps)
        mw = ecam.matrix_world
        loc = mw.to_translation()
        quat = mw.to_quaternion()
        lens = float(ecam.data.lens)

        if prev_loc is not None:
            d_loc = (loc - prev_loc).length
            d_rot = quat_angle_deg(prev_quat, quat)
            d_lens = abs(lens - prev_lens)
            if d_loc > args.loc_thresh or d_rot > args.rot_deg_thresh or d_lens > args.lens_thresh:
                cuts.append({
                    "cut_before_frame": f,
                    "prev_frame": f - 1,
                    "delta_loc_m": round(d_loc, 4),
                    "delta_rot_deg": round(d_rot, 2),
                    "delta_lens_mm": round(d_lens, 2),
                })
        prev_loc, prev_quat, prev_lens = loc.copy(), quat.copy(), lens

    segments = []
    seg_s = f_start
    for c in cuts:
        cf = c["cut_before_frame"]
        segments.append([seg_s, cf - 1])
        seg_s = cf
    segments.append([seg_s, f_end])

    total_frames = sum(e - s + 1 for s, e in segments)
    assert total_frames == (f_end - f_start + 1), "Segment partition mismatch!"

    payload = {
        "camera": cam.name,
        "frame_start": f_start,
        "frame_end": f_end,
        "native_frame_count": total_frames,
        "expected_2x_frame_count": total_frames * 2,
        "thresholds": {
            "loc_m": args.loc_thresh,
            "rot_deg": args.rot_deg_thresh,
            "lens_mm": args.lens_thresh,
        },
        "hard_cut_count": len(cuts),
        "segment_count": len(segments),
        "segments": segments,
        "cuts": cuts,
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EXTRACTED_CAMERA_CUTS: {len(cuts)} cuts -> {len(segments)} segments across {total_frames} frames")


if __name__ == "__main__":
    main()
