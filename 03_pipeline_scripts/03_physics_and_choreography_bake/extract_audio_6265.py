import wave, json
import numpy as np
from pathlib import Path

WDIR = Path(r"D:\DeepSeek Harness WorkSpace\Blender DSH\小小小明不是小明\生成模型\组装\v010R11_MMD_三渲二舞台_PRIVATE")
wav_fp = next((WDIR / "mmd_cache_PRIVATE/world_motion").glob("*.wav"))

with wave.open(str(wav_fp), "rb") as wf:
    sr = wf.getframerate()
    nch = wf.getnchannels()
    sw = wf.getsampwidth()
    nframes = wf.getnframes()
    raw_pcm = wf.readframes(nframes)

if sw == 2:
    audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
elif sw == 4:
    audio = np.frombuffer(raw_pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
else:
    audio = np.zeros(nframes, dtype=np.float32)
if nch > 1:
    audio = audio.reshape(-1, nch).mean(axis=1)

fps = 30
total_f = int(round((len(audio) / sr) * fps))  # 6265
hop = sr / fps
env_vocal, env_bright = [], []
for fi in range(total_f + 15):
    s0 = int(round(fi * hop))
    seg = audio[s0:s0 + int(round(hop))]
    if len(seg) < 64:
        env_vocal.append(0.0); env_bright.append(0.5); continue
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    mv = (freqs >= 280.0) & (freqs <= 3200.0)
    mh = (freqs >= 1400.0) & (freqs <= 3800.0)
    ve = float(np.sqrt(np.mean(spec[mv] ** 2))) if mv.any() else 0.0
    he = float(np.sqrt(np.mean(spec[mh] ** 2))) if mh.any() else 0.0
    env_vocal.append(ve)
    env_bright.append(round(he / (ve + 1e-6), 4))

p95 = float(np.percentile([v for v in env_vocal if v > 1e-4], 95)) if any(v > 1e-4 for v in env_vocal) else 1.0
env_norm = [round(min(1.25, v / max(p95, 1e-6)), 4) for v in env_vocal]

out = {
    "wav_path": str(wav_fp),
    "sr": sr,
    "fps": fps,
    "total_frames": total_f,
    "env_norm": env_norm,
    "env_bright": env_bright,
}
(WDIR / "audio_envelope_6265f.json").write_text(json.dumps(out), encoding="utf-8")
print(f"AUDIO_6265_OK: total_frames={total_f} ({len(audio)/sr:.2f}s)")
