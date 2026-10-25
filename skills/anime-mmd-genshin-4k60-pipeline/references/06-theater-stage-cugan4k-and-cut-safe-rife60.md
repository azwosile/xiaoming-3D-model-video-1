# 06 大型剧院舞台灯光、断点续渲与 Real-CUGAN 4K + 硬切隔离 RIFE 60fps 母带管线

## 6.1 大型 MMD 舞台（如科罗廖夫大剧院）与三渲二角色的光照解耦

将 `180+ MB` 的精细欧式歌剧院 / 大舞台导入 Blender EEVEE 后，最容易犯的错误是：**用同一套强聚光灯同时照亮舞台和二次元角色**——结果不是剧院背景漆黑一片，就是角色白衬衫与脸颊惨白过曝！

### 舞台×角色双层照明标准架构（`build_v10r18_theater_hsr_mmd.py`）
1. **角色材质自控曝光上限**：
   - 角色三渲二着色器内部对 `Shader to RGB` 提取的光照强度做 `clamp(0.0, 1.0)` 归一化后再查 `Shadow_Ramp` 色表，从数学根源上保证**无论舞台追光多亮，角色皮肤和白水手服永远不会爆白死光**！
2. **剧院建筑氛围光组**：
   - 顶部枝形水晶吊灯与壁灯挂暖金点光源（色温约 `3200K–3800K`），地面大石材开启适度粗糙度反射（`Roughness = 0.22–0.35`）映出角色倒影微光；
   - 添加极低浓度体积雾盒（Volume Scatter `Density = 0.004–0.007`，`Anisotropy = 0.55`），赋予剧院纵深空气感与追光光束轮廓，且不糊掉前景人物对比度。
3. **音频包络联动呼吸灯效（Audio Envelope Reactive Rim/Spot）**：
   - 预先从原曲 `.wav` 提取每帧 RMS 响度包络（`audio_envelope_6265f.json`），温和驱动舞台后方逆光灯阵与吊灯微幅脉动（幅度控制在 `±15%`），让副歌鼓点具备现场演出张力。

---

## 6.2 数千帧长片的分段式断点续渲策略

全曲 `6,265` 帧切忌在一个进程里盲跑到底！推荐按音乐结构切分为 3 段（如：`Seg1 f1..2400` 前奏+第一副歌、`Seg2 f2401..4320` 二段主副歌、`Seg3 f4321..6265` 桥段+终章）：

1. **进程隔离与自动断点跳过**：
   - 每渲染 `200–400` 帧或每段独立启动干净后台 Blender 进程，释放 EEVEE GPU 显存碎片；
   - 逐帧渲染循环前检查目标 PNG 是否已存在且字节数 `> 50,000`（防止半截损坏空文件假阳性），中断重跑零浪费：
     ```python
     out_png = out_dir / f"f_{frame:04d}.png"
     if out_png.exists() and out_png.stat().st_size > 50000:
         continue
     ```
2. **严禁同工程多进程并发开 EEVEE**：
   - 实测同一 `.blend` 同时跑两个后台 EEVEE 进程偶发着色器缓存/贴图句柄竞争，导致整段画面变成品红色丢失贴图！单 GPU 顺序满载跑 `48` TAA 采样反而最快最稳。

---

## 6.3 为什么 `Real-CUGAN 4K` + `RIFE 60fps` 是三渲二母带的黄金放大器

原生直接渲染 `3840×2160 @ 60fps`（`12,530` 帧）不仅耗时翻 `6–8` 倍，而且 EEVEE 原生多边形边缘在 `4K` 下仍可能暴露细微几何折角。
采用 **`1280×720`（高采样干净源）→ `Real-CUGAN models-pro up3x-no-denoise`（`3840×2160` 真 4K）→ 镜头硬切隔离 `RIFE v4.6`（`60fps`）**：
- `Real-CUGAN (no-denoise)` 专为二次元线条与色块训练，能把眼睫毛、星芒虹膜与 `0.7 mm` 紫褐细描边重构得如矢量原画般锋利，同时完整保留衣料织物纹理（绝不涂抹成塑料感，故严禁开强降噪档 `denoise3x`）！

---

## 6.4 致命大坑与根治方案：MMD 相机硬切隔离补帧（零果冻融化帧）

### 灾难现场复盘
MMD 相机 `.vmd` 大量采用「第 `k` 帧人脸大特写 → 第 `k+1` 帧瞬间跳切到剧院全景」的硬切剪辑（《小小小明》全曲共检出 **109 处硬切、112 个独立连续镜头**）。
如果把 `f_0001.png .. f_6265.png` 整包丢给 `rife-ncnn-vulkan` 做 `2×` 补帧：
- RIFE 会在第 `k` 帧与第 `k+1` 帧之间强行光流插值，生成 **109 张五官撕裂飞溅、人物与吊灯融化粘连的「果冻恐怖帧」**！

### 工业级零果冻硬切隔离算法（详见 `scripts/extract_camera_hard_cuts.py` & `scripts/cugan_rife_cut_safe_master.py`）
1. **第一步：从 Blender 相机精确提取硬切真值表（不用视觉阈值瞎猜）**
   - 直接读取 `1..6265` 每帧活动相机的世界坐标 `loc(t)`、旋转四元数 `quat(t)` 与焦距 `lens(t)`；
   - 凡满足以下任一条件，即判定 `[t-1 -> t]` 为**相机硬切边界**：
     - `||loc(t) - loc(t-1)|| > 0.35 m`
     - `angle(quat(t), quat(t-1)) > 18.0°`
     - `|lens(t) - lens(t-1)| > 3.0 mm`
   - 据此将全曲切分为互不跨越硬切的连续镜头区间列表：`[(s_0, e_0), (s_1, e_1), ..., (s_M, e_M)]`。
2. **第二步：镜头段物理隔离补帧与精确帧数守恒**
   - 对每个长度为 `K = e_i - s_i + 1` 的连续镜头段：
     - 若 `K == 1`（单帧闪切镜头）：直接输出 **2 张**完全相同的该帧（`2×1 = 2` 帧），不调 RIFE；
     - 若 `K >= 2`：仅将该镜头段的 `K` 张 4K 图拷入临时子目录调 `rife-ncnn-vulkan -m rife-v4.6 -u -n (2*K)`（注意核验 RIFE 输出的最后一张与第 `2*K-1` 张是否为原末帧复用，确保段内 `0..2K-2` 为真实运动内插、段尾第 `2K-1` 帧干净保持末帧，**绝不读取下一镜头第 1 帧**）！
   - **帧数与音画同步数学铁律**：设原生总帧数为 `N`（`@ 30fps`），所有镜头段合并后总帧数严格等于 **`2 × N`（`@ 60fps`）**，总时长分毫不差（`6265 / 30 = 12530 / 60 = 208.833333s`），音乐鼓点与口型卡点零累积漂移！

---

## 6.5 母带压制、分段无损合流与全量解码验真

1. **色彩元数据与编码参数（防止播放器偏色发灰）**：
   ```bash
   ffmpeg -y -framerate 60 -i out_60fps/%06d.png -i audio.wav \
     -c:v libx264 -preset slow -crf 19 -pix_fmt yuv420p \
     -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
     -c:a aac -b:a 192k -movflags +faststart -shortest Master_4K60.mp4
   ```
2. **多段视频无损合流（Lossless Stream Concat）**：
   - 若分段产出了 `Seg1`、`Seg2`、`Seg3` 的同规格无音频或纯视频流 H.264 母带（完全相同的分辨率、帧率、`SPS/PPS`、`CRF` 与 `pix_fmt`），可用 `ffmpeg -f concat -safe 0 -i list.txt -i full_audio.wav -c:v copy -c:a aac -b:a 192k` 毫秒级无损合流全曲，避免二次有损重编码画质折损。
3. **交付前强制全流解码体检**：
   - 任何母带交付前必跑：
     ```bash
     ffmpeg -v error -i Master_4K60.mp4 -f null -
     ffprobe -v error -count_frames -select_streams v:0 \
       -show_entries stream=width,height,r_frame_rate,nb_read_frames,color_space Master_4K60.mp4
     ```
   - 必须确认 `stderr` 为空（`0` 解码错）、`nb_read_frames == 2 * N`、`r_frame_rate == 60/1`、`color_space == bt709`。
