## 🎬 <版本里程碑> · <舞台名称> × 原神/崩铁风格三渲二全曲 4K60 / 1080p60 典藏母带

本 Release 归档《<角色/项目名称>》3D 角色工程全曲（`f1..f<N>` 原生 `30fps` → `<C>` 处切镜隔离补帧 `<2N>` 帧 `@ 60fps`，时长 `<mm:ss.xx>`）最高画质典藏母带：

| 附件文件名 | 分辨率 / 帧率 / 帧数 | 编码规格 | 体积 | SHA-256 |
| :--- | :--- | :--- | ---: | :--- |
| `<Character>_FullSong_4K60_Master.mp4` | `3840×2160` · `60fps` · `<2N>` 帧 | `H.264 High@L5.2` · `CRF 19` · `BT.709` · `AAC 192k` | `<XXX.XX> MB` | `<sha256_prefix>...` |
| `<Character>_FullSong_1080p60_Master.mp4` | `1920×1080` · `60fps` · `<2N>` 帧 | `4K` Lanczos 下采样 · `CRF 19` · `BT.709` · `AAC 192k` | `<XXX.XX> MB` | `<sha256_prefix>...` |

### 管线技术规格
- **渲染器**：`Blender 5.2 LTS` (`BLENDER_EEVEE`, `48` TAA Samples, `Standard + Medium High Contrast`)
- **三渲二核心管线**：全曲逐帧 `CONSTANT` 头部锚定 `Face SDF` 光影 + `4K` 多层星芒虹膜与三维角膜高光壳 + 分区 `Shadow_Ramp` + 五金独立锐利高光 + 暖暗紫褐细描边
- **超分与防果冻补帧**：`Real-CUGAN models-pro (up3x-no-denoise)` 真 `4K` 超分 + 相机硬切真值隔离 `RIFE v4.6 (-u)` `60fps` 补帧（零跨镜融化伪影）
- **全量解码校验**：全部母带均通过 `ffmpeg -v error` 全流逐帧解码零报错验证

> **致谢署名**：编舞 `<作品号 / UP主 / 制作社团>` ｜ 舞台 `<舞台名称与作者>` ｜ 道具 `<道具名称与作者>` ｜ 素体规范 `<素体名称与作者>`。受「再配布禁止」保护的三维网格与原始动作包均未包含在本仓库与附件中。
