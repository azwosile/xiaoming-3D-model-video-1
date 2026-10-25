# 07 二次元开源合规边界、零 LFS 精选建仓与大体积母带 GitHub Release 交付

## 7.1 二次元 3D 工程的法律与社区红线（严禁踩线）

二次元建模与 MMD 生态高度依赖创作者互信，以下资产在绝大多数使用规约（ReadMe / 利用規約）中明确规定 **「允许个人非商用改造与渲染视频发布，但严禁二次配布原始或衍生三维网格/动作文件（再配布禁止）」**：
1. **社区二次元素体与头模**（如 `RadDoll V3`、部分 Booth 非商用素体、米哈游官方配布改造模型）；
2. **MMD 场景舞台与道具**（`.pmx` / `.pmd` / `.x` 及随附贴图包，如 *The Korolevskiy Theater* by Lummi、单双椅 by 果天、*Heart Stage* by BLue）；
3. **MMD 编舞动作、表情与镜头数据**（`.vmd` / `.vpd` 及提取出的原始动作包）。

### 工程化防泄漏三道闸门
1. **本机工作区命名隔离**：凡合入了上述网格或缓存的文件/目录，创建时一律带 `_PRIVATE` 后缀（如 `RadDoll_V10R19_..._PRIVATE.blend`、`mmd_cache_PRIVATE/`）。
2. **独立精选展示仓库（Clean Showcase Repo）**：
   - 不要在塞满几十 GB 历史实验缓存的老目录里硬做 `git filter-repo`；
   - 在同级新建干净的公开展示仓库目录，仅拷入**全自研 Python 管线源码、原画与拆件规范文档、交互式 HTML 报告、各轮渲染对比大图、`<50 MB` 成片视频以及可复用的 `skills/` 技能包**，并自动剥离成片/图片文件名中的内部 `_PRIVATE` 后缀。
3. **提交前自动化合规拦截（`scripts/pre_publish_compliance_audit.py`）**：
   - 在 `git add` 前强制扫描工作树：一旦发现任何 `.blend`、`.blend1`、`.fbx`、`.glb`、`.gltf`、`.obj`、`.pmx`、`.pmd`、`.vmd`、`.vpd`、`.zip`、`.rar`、`.7z`、`.npz`（若含源模型采样数据），或文件名残留 `PRIVATE`、或单文件 `>= 48 MB`，直接抛错中止提交！
4. **完整规范署名**：逐字核对所有外源资产文件夹内的 `ReadMe.txt` / `说明.txt`（注意日文 Shift-JIS / CP932 与中文 GBK 编码转换，切勿乱码瞎猜作者名），在公开仓库 `README.md` 与视频简介中完整列出素体、舞台、道具、编舞 UP 主与开源工具致谢。

---

## 7.2 为什么坚持「零 Git LFS」+「GitHub Release 承载 4K60 母带」

- **Git LFS 的隐形陷阱**：GitHub 免费账户 Git LFS 存储与月度带宽配额仅 **1 GB**。一旦把 `875 MB` 的 `4K60` 母带或几个 `.blend` 放进 LFS，不仅自己推送容易超限失败，别人克隆两次就会耗尽带宽把整个仓库锁死！
- **最佳实践双轨制**：
  1. **Git 仓库主干（`main`，`< 120 MB` 总量，单文件 `< 48 MB`）**：
     - 放 1 支 `CRF 25–27` 压制的 **`40 MB` 级 `1080p60` 全曲直看版 MP4** + 几支 `5–15 MB` 局部特写/循环动作短片，无需 LFS 即可秒克隆、网页直接预览。
  2. **GitHub Releases（单附件上限高达 `2.0 GB`，且完全不占 LFS 配额，无限免费高速下载）**：
     - 将 `875.54 MB` 的 `CRF 19` 真 `4K60` 典藏母带与 `314.79 MB` 的高码率 `1080p60` 母带作为 `v1.0.0` Release 附件上传，并在 Release 正文中附上精确字节数与 `SHA-256` 校验码。

---

## 7.3 Windows 环境下大体积 Release 附件流式上传防断连技巧

在 Windows + Python / Git Credential Manager 环境下上传 `300 MB–1.5 GB` 视频至 `uploads.github.com` 时，有三个常见系统坑：

1. **Git Credential Manager 非交互取令牌防卡死**：
   - 后台脚本调用 `git-credential-manager.exe get` 时，必须设置环境变量 `GCM_INTERACTIVE=never` 与 `GIT_TERMINAL_PROMPT=0`，且**令牌仅保存在 Python 进程内存中写入 HTTPS Header，严禁打印到终端或拼接到命令行参数**。
2. **避免 `WinError 10054`（远程主机强迫关闭连接）**：
   - 不要一次性 `fp.read()` 几百 MB 进内存，也不要用 `2 MB+` 超大块猛灌套接字（易在跨境 TLS 链路触发缓冲拥塞重置）；
   - 使用 `http.client.HTTPSConnection("uploads.github.com", timeout=360)` 显式发送准确 `Content-Length`，以 **`256 KB`（`262,144` 字节）** 步长平稳流式 `conn.send(chunk)`，实测近 `900 MB` 母带可稳定跑满 `6 MB/s` 一次成功！
3. **重传前必须清理同名 `starter` 残余附件**：
   - 若中途网络抖动断连，GitHub Release 下会残留一个 `state == "starter"` 的同名半截附件，直接重传会报 `HTTP 422 already_exists`；
   - 重试循环开头必须先 `GET /repos/{owner}/{repo}/releases/{id}`，若发现同名附件且 `state != "uploaded"` 或 `size != local_size`，先调 `DELETE /repos/{owner}/{repo}/releases/assets/{asset_id}` 清理后再重传。
