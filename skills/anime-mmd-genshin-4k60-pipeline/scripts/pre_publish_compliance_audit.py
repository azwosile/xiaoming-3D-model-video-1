"""二次元展示仓库开源推送前合规与体积自动审计门禁（Python 3 CLI）

拦截规则：
  1. 严禁出现任何受再配布限制的三维网格、骨骼动作或原始配布压缩包（.blend/.fbx/.glb/.pmx/.vmd/.zip/.rar/.7z/.npz 等）；
  2. 严禁文件名或路径残留内部隔离标记（如 PRIVATE）；
  3. 严禁单文件超过阈值（默认 48 MB，确保零 Git LFS 配额消耗且低于 GitHub 50 MB 警告线）。

用法示例：
  python pre_publish_compliance_audit.py --repo ./xiaoming-3D-model-video-1 --max-mb 48
"""
import argparse
import os
import sys
from pathlib import Path

BANNED_EXTENSIONS = {
    ".blend", ".blend1", ".blend2",
    ".fbx", ".glb", ".gltf", ".obj", ".dae", ".stl", ".usd", ".usda", ".usdc",
    ".pmx", ".pmd", ".x", ".vmd", ".vpd",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".npz", ".npy", ".b64", ".pyc",
}


def main():
    ap = argparse.ArgumentParser(description="Audit showcase repository before git commit/push.")
    ap.add_argument("--repo", required=True, help="Repository root directory")
    ap.add_argument("--max-mb", type=float, default=48.0, help="Maximum allowed single file size in MB")
    ap.add_argument("--forbid-keyword", default="PRIVATE", help="Case-insensitive forbidden filename marker")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"ERROR: repo directory not found: {repo}", file=sys.stderr)
        sys.exit(2)

    violations = []
    total_bytes = 0
    file_count = 0
    largest = ("", 0)

    for dp, dn, fn in os.walk(repo):
        dn[:] = [d for d in dn if d != ".git"]
        for f in fn:
            fp = Path(dp) / f
            rel = fp.relative_to(repo)
            sz = fp.stat().st_size
            file_count += 1
            total_bytes += sz
            if sz > largest[1]:
                largest = (str(rel), sz)

            ext = fp.suffix.lower()
            if ext in BANNED_EXTENSIONS:
                violations.append(f"[BANNED_EXT] {rel} ({ext})")
            if args.forbid_keyword and args.forbid_keyword.lower() in str(rel).lower():
                violations.append(f"[FORBIDDEN_MARKER] {rel}")
            if sz > args.max_mb * 1048576:
                violations.append(f"[OVERSIZE] {rel} ({sz / 1048576:.2f} MB > {args.max_mb:.2f} MB)")

    if violations:
        print(f"COMPLIANCE_AUDIT_FAILED ({len(violations)} violations):", file=sys.stderr)
        for v in violations:
            print(" ", v, file=sys.stderr)
        sys.exit(1)

    print(
        f"COMPLIANCE_AUDIT_PASSED: {file_count} files, "
        f"total={total_bytes / 1048576:.2f} MB, "
        f"largest={largest[0]} ({largest[1] / 1048576:.2f} MB)"
    )


if __name__ == "__main__":
    main()
