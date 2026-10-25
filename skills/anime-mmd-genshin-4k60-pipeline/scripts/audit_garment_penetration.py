"""通用素体×多层服装逐帧 BVHTree 穿模量化审计工具（Blender 5.x 后台 CLI）

用法示例：
  blender -b character.blend --factory-startup --python-exit-code 1 \
    -P audit_garment_penetration.py -- \
    --body GEO-Body \
    --garments GEO-SailorTop GEO-Shorts GEO-Coat \
    --start 1 --end 6265 --step 2 \
    --out penetration_audit.json
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils.bvhtree import BVHTree


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="Audit BVHTree overlaps between body mesh and garments.")
    ap.add_argument("--body", required=True, help="Evaluated body mesh object name")
    ap.add_argument("--garments", nargs="+", required=True, help="Garment mesh object names")
    ap.add_argument("--start", type=int, default=None, help="Start frame (default: scene.frame_start)")
    ap.add_argument("--end", type=int, default=None, help="End frame (default: scene.frame_end)")
    ap.add_argument("--step", type=int, default=1, help="Frame step")
    ap.add_argument("--ignore-modifiers", nargs="*", default=["Outline", "SOLIDIFY"],
                    help="Substrings of modifier names (e.g. Solidify outline shells) to mute during audit")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    return ap.parse_args(argv)


def build_world_bvh(obj, depsgraph):
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mesh.calc_loop_triangles()
    mw = eval_obj.matrix_world
    verts = [mw @ v.co for v in mesh.vertices]
    tris = [tuple(t.vertices) for t in mesh.loop_triangles]
    tree = BVHTree.FromPolygons(verts, tris, all_triangles=True) if tris else None
    eval_obj.to_mesh_clear()
    return tree, len(tris)


def main():
    args = parse_args()
    scene = bpy.context.scene
    f_start = args.start if args.start is not None else scene.frame_start
    f_end = args.end if args.end is not None else scene.frame_end

    body_obj = bpy.data.objects.get(args.body)
    if body_obj is None:
        raise RuntimeError(f"Body object not found: {args.body}")

    garment_objs = []
    for gname in args.garments:
        g = bpy.data.objects.get(gname)
        if g is None:
            raise RuntimeError(f"Garment object not found: {gname}")
        garment_objs.append(g)

    # Temporarily mute outline shell modifiers so inverted hulls do not create false overlaps
    muted_restore = []
    for obj in [body_obj] + garment_objs:
        for mod in obj.modifiers:
            if any(k.lower() in mod.name.lower() or k.lower() in mod.type.lower() for k in args.ignore_modifiers):
                if mod.show_viewport:
                    muted_restore.append((mod, True))
                    mod.show_viewport = False

    summary = {g.name: {"frames_with_overlap": 0, "max_overlap_pairs": 0, "worst_frames": []} for g in garment_objs}
    per_frame_hits = {g.name: [] for g in garment_objs}

    try:
        for f in range(f_start, f_end + 1, args.step):
            scene.frame_set(f)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            body_bvh, _ = build_world_bvh(body_obj, depsgraph)
            if body_bvh is None:
                continue
            for g in garment_objs:
                g_bvh, _ = build_world_bvh(g, depsgraph)
                if g_bvh is None:
                    continue
                pairs = body_bvh.overlap(g_bvh)
                cnt = len(pairs)
                if cnt > 0:
                    summary[g.name]["frames_with_overlap"] += 1
                    if cnt > summary[g.name]["max_overlap_pairs"]:
                        summary[g.name]["max_overlap_pairs"] = cnt
                    per_frame_hits[g.name].append((cnt, f))
    finally:
        for mod, prev in muted_restore:
            mod.show_viewport = prev

    for gname, hits in per_frame_hits.items():
        hits.sort(reverse=True)
        summary[gname]["worst_frames"] = [{"frame": f, "overlap_pairs": c} for c, f in hits[:20]]

    report = {
        "blend_file": bpy.data.filepath,
        "body": args.body,
        "frame_range": [f_start, f_end, args.step],
        "audited_frames": len(range(f_start, f_end + 1, args.step)),
        "garments": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PENETRATION_AUDIT_DONE:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
