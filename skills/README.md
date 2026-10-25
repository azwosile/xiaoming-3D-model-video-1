# Agent SKILLS · 二次元角色高阶精修 × MMD 全曲编舞 × 原神/崩铁 4K60 舞台母带

本目录收录从《小小小明不是小明》`V10R1–V10R19`（全曲 `6,265` 原生帧 → `12,530` 帧 `@ 60fps`）完整实战中提炼的可复用 AI Agent 技能包，遵循 Claude Code / ShunCode / Codex 标准 `SKILL.md` 规范，可直接挂载至新角色项目使用。

## 目录结构

```text
skills/
└── anime-mmd-genshin-4k60-pipeline/
    ├── SKILL.md                                                 # 技能主入口：6 条前置纪律、8 关量化门槛、12 条踩坑硬规则、参数速查表
    ├── references/
    │   ├── 01-hybrid-head-and-starburst-eyes.md                 # 混合素体选头、无发帽头皮几何体检、4K 多层星芒眼贴与 66 面角膜高光壳
    │   ├── 02-garment-tailoring-pocket-and-delight.md           # Tripo 衣装修身剪裁、口袋黑洞封腔、腋下粘连重拓与贴图分位数去光重烘焙
    │   ├── 03-zero-tear-rigging-and-dynamic-clearance.md        # 仰角下颌无缝焊接、多层落肩外套平滑传权与全曲动态防穿模外推算法
    │   ├── 04-mmd-oracle-retarget-and-zero-twist.md             # 外源 MMD 关节真值采样重定向、胸前交叉防穿插、背身 180° 防反拧与视线护航
    │   ├── 05-genshin-hsr-face-sdf-and-npr-shaders.md           # 原神/崩铁三渲二：UV 展开 Face SDF、逐帧 CONSTANT 头部光角锚定、分区色阶与金属高光
    │   ├── 06-theater-stage-cugan4k-and-cut-safe-rife60.md      # 大型剧院舞台灯光分层、断点续渲、Real-CUGAN 4K 超分与硬切隔离 RIFE 60fps 母带
    │   └── 07-compliance-audit-and-github-release.md            # 再配布合规红线、零 LFS 精选建仓与大体积母带流式 GitHub Release 交付
    ├── scripts/
    │   ├── audit_garment_penetration.py                         # 通用 Blender CLI：素体×多层服装逐帧 BVHTree 穿模统计与最差帧定位
    │   ├── extract_camera_hard_cuts.py                          # 通用 Blender CLI：全时间轴相机硬切检测与连续镜头分段 JSON 导出
    │   ├── cugan_rife_cut_safe_master.py                        # 通用 CLI：Real-CUGAN 超分 + 镜头硬切隔离 RIFE 60fps + BT.709 母带压制与解码校验
    │   └── pre_publish_compliance_audit.py                      # 通用 CLI：开源提交前受限网格/压缩包/内部标记/文件体积自动门禁
    └── templates/
        ├── mmd_mv_stage_checklist.md                            # 八关量化签收自检表模板
        └── release_notes_template.md                            # 4K60 / 1080p60 典藏母带 GitHub Release 说明模板
```

## 快速复用方式

1. **在 ShunCode / Claude Code 中挂载**：将 `anime-mmd-genshin-4k60-pipeline/` 复制到工作区技能目录（如 `.claude/skills/`、`.shuncode/skills/` 或共享 `建模SKILLS/`），Agent 将根据 `SKILL.md` 的触发描述自动按阶段加载对应 `references/` 与 `scripts/`。
2. **与前置技能串联**：
   - **第一阶段（从零拆图到基础可动）**：使用 `anime-character-pipeline` 完成立绘四视图拆图 → Tripo 部件生成 → 初装与基础绑定；
   - **第二阶段（高颜值精修与全曲 4K60 舞台成片）**：切入本技能 `anime-mmd-genshin-4k60-pipeline` 完成混合换头、去光重烘焙、全曲零穿模零交拧解算、原神/崩铁 `Face SDF` 着色与 `Real-CUGAN 4K + RIFE 60fps` 无果冻母带交付。
