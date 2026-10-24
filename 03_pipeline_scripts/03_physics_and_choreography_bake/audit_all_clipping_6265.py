import math, time
from pathlib import Path
import numpy as np
import bpy, addon_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

t0 = time.time()
WDIR  = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
BLEND = WDIR / "RadDoll_V10R15_FullSong6265_MMD_Heart舞台三渲二全曲母带_PRIVATE.blend"

addon_utils.enable("bl_ext.blender_org.mmd_tools", default_set=False)
bpy.ops.wm.open_mainfile(filepath=str(BLEND))
sc     = bpy.context.scene
arm_ob = bpy.data.objects["ARM-FullBody_V10"]
pbs    = arm_ob.pose.bones
Mw_xm  = arm_ob.matrix_world.copy()

# Disable mesh modifiers temporarily for ultra-fast 6,265f bone/capsule sweep (0.8s!)
mod_states = []
for o in bpy.data.objects:
    if o.type == 'MESH':
        for m in o.modifiers:
            mod_states.append((m, m.show_viewport))
            m.show_viewport = False

def seg_seg_dist(p1, q1, p2, q2):
    # Exact 3D closest distance between segment [p1,q1] and [p2,q2]
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = u.dot(u); b = u.dot(v); c = v.dot(v); d = u.dot(w); e = v.dot(w)
    D = a * c - b * b
    sc_n, sN, sD = D, D, D
    tc_n, tN, tD = D, D, D
    if D < 1e-8:
        sN = 0.0; sD = 1.0; tN = e; tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0; tN = e; tD = c
        elif sN > sD:
            sN = sD; tN = e + b; tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else: sN = -d; sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0: sN = 0.0
        elif (-d + b) > a: sN = sD
        else: sN = (-d + b); sD = a
    sc_v = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc_v = 0.0 if abs(tN) < 1e-8 else tN / tD
    c1 = p1 + u * sc_v
    c2 = p2 + v * tc_v
    return (c1 - c2).length, sc_v, tc_v

arm_cross_hits = []      # L/R forearm-hand sleeves colliding (< 0.145m!)
arm_torso_hits = []      # forearm/hand colliding into Chest/Spine (< 0.135m!) or Head (< 0.135m!)
elbow_fold_hits = []     # acute elbow bend (> 108 deg) or high upper_arm z (> 0.08) like 0:58!
leg_cross_hits = []      # L/R knees/shins colliding (< 0.085m)

for f in range(1, 6266):
    sc.frame_set(f)
    bpy.context.view_layer.update()

    uaL0 = Mw_xm @ pbs["upper_arm.L"].head; uaL1 = Mw_xm @ pbs["upper_arm.L"].tail
    faL0 = Mw_xm @ pbs["lower_arm.L"].head; faL1 = Mw_xm @ pbs["hand.L"].tail
    uaR0 = Mw_xm @ pbs["upper_arm.R"].head; uaR1 = Mw_xm @ pbs["upper_arm.R"].tail
    faR0 = Mw_xm @ pbs["lower_arm.R"].head; faR1 = Mw_xm @ pbs["hand.R"].tail

    sp0  = Mw_xm @ pbs["Hips"].head;        sp1  = Mw_xm @ pbs["Neck"].head
    hd0  = Mw_xm @ pbs["Head"].head;        hd1  = hd0 + Vector((0, 0, 0.14))

    # 1. L vs R forearm+hand segment distance (coat sleeve diameter is ~0.15m!)
    d_ff, s_l, s_r = seg_seg_dist(faL0, faL1, faR0, faR1)
    d_uf1, _, _    = seg_seg_dist(uaL0, uaL1, faR0, faR1)
    d_uf2, _, _    = seg_seg_dist(faL0, faL1, uaR0, uaR1)
    d_min_lr = min(d_ff, d_uf1, d_uf2)
    if d_min_lr < 0.145:
        arm_cross_hits.append((f, round(d_min_lr * 1000, 1), round((faL1 - faR1).length * 1000, 1)))

    # 2. Forearm/Hand penetrating Torso [sp0, sp1] or Head [hd0, hd1]
    d_tL, _, _ = seg_seg_dist(faL0, faL1, sp0, sp1)
    d_tR, _, _ = seg_seg_dist(faR0, faR1, sp0, sp1)
    d_hL, _, _ = seg_seg_dist(faL0, faL1, hd0, hd1)
    d_hR, _, _ = seg_seg_dist(faR0, faR1, hd0, hd1)
    d_body = min(d_tL, d_tR, d_hL, d_hR)
    if d_body < 0.135:
        arm_torso_hits.append((f, round(d_body * 1000, 1), round(min(d_tL, d_tR)*1000,1), round(min(d_hL, d_hR)*1000,1)))

    # 3. Acute elbow fold / high upper arm (like 0:58 = f1740!)
    uL = (uaL1 - uaL0).normalized(); fL = (faL1 - faL0).normalized()
    uR = (uaR1 - uaR0).normalized(); fR = (faR1 - faR0).normalized()
    bL = math.degrees(math.acos(max(-1.0, min(1.0, uL.dot(fL)))))
    bR = math.degrees(math.acos(max(-1.0, min(1.0, uR.dot(fR)))))
    if max(bL, bR) > 108.0 or max(uL.z, uR.z) > 0.06:
        elbow_fold_hits.append((f, round(bL, 1), round(bR, 1), round(uL.z, 2), round(uR.z, 2)))

def summarize_bouts(hits, gap=4):
    if not hits: return []
    bouts = []
    s = e = hits[0][0]
    best = hits[0]
    for h in hits[1:]:
        if h[0] <= e + gap:
            e = h[0]
            if h[1] < best[1]: best = h
        else:
            bouts.append((s, e, best))
            s = e = h[0]; best = h
    bouts.append((s, e, best))
    return bouts

print(f"=== SCAN DONE in {time.time()-t0:.2f}s ===", flush=True)
b_cross = summarize_bouts(arm_cross_hits)
print(f"\n[A] L/R ARM-SLEEVE CROSSING (<145mm): {len(arm_cross_hits)} frames in {len(b_cross)} bouts:", flush=True)
for s, e, b in b_cross:
    tag = " [IN FIRST 80s!]" if s <= 2400 else ""
    print(f"  f{s:04d}..f{e:04d} ({s/30:.1f}s..{e/30:.1f}s): min_seg={b[1]}mm at f{b[0]} (wrist_d={b[2]}mm){tag}", flush=True)

b_torso = summarize_bouts(arm_torso_hits)
print(f"\n[B] ARM-INTO-TORSO/HEAD (<135mm): {len(arm_torso_hits)} frames in {len(b_torso)} bouts:", flush=True)
for s, e, b in b_torso[:30]:
    tag = " [IN FIRST 80s!]" if s <= 2400 else ""
    print(f"  f{s:04d}..f{e:04d} ({s/30:.1f}s..{e/30:.1f}s): min_d={b[1]}mm at f{b[0]} (torso={b[2]}mm, head={b[3]}mm){tag}", flush=True)

# Specifically inspect f1730..f1765 (0:57.7 .. 0:58.8 — user's screenshot!)
print("\n[C] DETAIL AROUND 0:58 (f1730..f1765):", flush=True)
for f in range(1730, 1766, 3):
    sc.frame_set(f); bpy.context.view_layer.update()
    uaL0 = Mw_xm @ pbs["upper_arm.L"].head; uaL1 = Mw_xm @ pbs["upper_arm.L"].tail
    faL0 = Mw_xm @ pbs["lower_arm.L"].head; faL1 = Mw_xm @ pbs["hand.L"].tail
    uaR0 = Mw_xm @ pbs["upper_arm.R"].head; uaR1 = Mw_xm @ pbs["upper_arm.R"].tail
    faR0 = Mw_xm @ pbs["lower_arm.R"].head; faR1 = Mw_xm @ pbs["hand.R"].tail
    d_ff, _, _ = seg_seg_dist(faL0, faL1, faR0, faR1)
    uL = (uaL1 - uaL0).normalized(); fL = (faL1 - faL0).normalized()
    uR = (uaR1 - uaR0).normalized(); fR = (faR1 - faR0).normalized()
    bL = math.degrees(math.acos(max(-1.0, min(1.0, uL.dot(fL)))))
    bR = math.degrees(math.acos(max(-1.0, min(1.0, uR.dot(fR)))))
    print(f"  f{f} ({f/30:.2f}s): seg_LR={d_ff*1000:.0f}mm, bendL={bL:.0f}deg, bendR={bR:.0f}deg, uaL.z={uL.z:.2f}, uaR.z={uR.z:.2f}, handL=({faL1.x:.2f},{faL1.y:.2f},{faL1.z:.2f}), handR=({faR1.x:.2f},{faR1.y:.2f},{faR1.z:.2f})", flush=True)
