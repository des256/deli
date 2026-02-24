#!/usr/bin/env python3
"""Decode at various scales to find the right encoder output magnitude."""

import numpy as np
import onnxruntime as ort
import soundfile as sf

audio, sr_val = sf.read("test.wav", dtype="int16")
if sr_val != 16000:
    ratio = sr_val / 16000
    out_len = int(len(audio) / ratio)
    indices = np.arange(out_len) * ratio
    idx_arr = indices.astype(int)
    frac = indices - idx_arr
    idx_next = np.minimum(idx_arr + 1, len(audio) - 1)
    audio = (
        audio[idx_arr].astype(float) * (1 - frac) + audio[idx_next].astype(float) * frac
    ).astype(np.int16)
print(f"Audio: {len(audio)} samples at 16000 Hz ({len(audio) / 16000:.1f}s)")

window_size, hop_size, n_fft, n_mels, preemph_coef = 400, 160, 512, 128, 0.97

sig = audio.astype(np.float32) / 32768.0
pre = np.zeros_like(sig)
pre[0] = sig[0]
pre[1:] = sig[1:] - preemph_coef * sig[:-1]
hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / window_size)


def hz2mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


mel_pts = np.linspace(hz2mel(0), hz2mel(8000), n_mels + 2)
hz_pts = mel2hz(mel_pts)
bin_pts = np.floor(hz_pts * n_fft / 16000 + 0.5).astype(int)
fbank = np.zeros((n_mels, n_fft // 2 + 1))
for i in range(n_mels):
    lo, c, r = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
    for b in range(lo, c):
        if c > lo:
            fbank[i, b] = (b - lo) / (c - lo)
    for b in range(c, r):
        if r > c:
            fbank[i, b] = (r - b) / (r - c)

nf = (len(pre) - window_size) // hop_size + 1
feats = np.zeros((n_mels, nf))
for t in range(nf):
    s = t * hop_size
    frame = pre[s : s + window_size] * hann
    padded = np.zeros(n_fft)
    padded[:window_size] = frame
    spec = np.fft.rfft(padded)
    pw = np.abs(spec) ** 2
    mel_e = fbank @ pw
    feats[:, t] = np.log(mel_e + 1e-10)
for b in range(n_mels):
    m = feats[b].mean()
    s = max(feats[b].std(), 1e-5)
    feats[b] = (feats[b] - m) / s

print(f"Features: {n_mels}x{nf}")

sess_enc = ort.InferenceSession(
    "data/parakeet/encoder.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
sess_dec = ort.InferenceSession(
    "data/parakeet/decoder_joint.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

try:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file="data/parakeet/tokenizer.model")
except ImportError:
    sp = None

enc_out = sess_enc.run(
    None,
    {
        "processed_signal": feats.astype(np.float32).reshape(1, 128, nf),
        "processed_signal_length": np.array([nf], dtype=np.int64),
        "cache_last_channel": np.zeros((1, 24, 70, 1024), dtype=np.float32),
        "cache_last_time": np.zeros((1, 24, 1024, 8), dtype=np.float32),
        "cache_last_channel_len": np.array([0], dtype=np.int64),
        "spk_targets": np.zeros((1, nf), dtype=np.float32),
        "bg_spk_targets": np.zeros((1, nf), dtype=np.float32),
    },
)
encoded = enc_out[0]
print(f"Encoder: shape={encoded.shape}, std={encoded.std():.6f}")


def greedy_decode_full(enc, max_sym=10):
    """Full greedy decode returning tokens."""
    enc_t = enc.shape[2]
    state1 = np.zeros((2, 1, 640), dtype=np.float32)
    state2 = np.zeros((2, 1, 640), dtype=np.float32)
    last_tok = 1024
    tokens = []
    for t in range(enc_t):
        frame = enc[:, :, t : t + 1].transpose(0, 2, 1)
        for _ in range(max_sym):
            s1b, s2b = state1.copy(), state2.copy()
            out = sess_dec.run(
                None,
                {
                    "encoder_outputs": frame,
                    "targets": np.array([[last_tok]], dtype=np.int64),
                    "input_states_1": state1,
                    "input_states_2": state2,
                },
            )
            logits = out[0].flatten()[:1025]
            pred = int(np.argmax(logits))
            if pred == 1024:
                state1, state2 = s1b, s2b
                break
            tokens.append(pred)
            last_tok = pred
            state1, state2 = out[2], out[3]
    return tokens


for scale in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
    scaled = encoded * scale
    toks = greedy_decode_full(scaled)
    if sp and toks:
        text = sp.decode(toks)
    elif toks:
        text = f"[{len(toks)} tokens]"
    else:
        text = "(blank)"
    print(f"\nscale={scale:.1f}: {len(toks)} tokens")
    print(f"  text: {text}")

print("\n--- Checking encoder model weights ---")
try:
    import onnx

    model = onnx.load("data/parakeet/encoder.onnx")
    ln_gammas = []
    for init in model.graph.initializer:
        if "norm" in init.name.lower() and "weight" in init.name.lower():
            data = (
                np.frombuffer(init.raw_data, dtype=np.float32)
                if init.raw_data
                else None
            )
            if data is not None and len(data) == 1024:
                ln_gammas.append(
                    (init.name, data.mean(), data.std(), data.min(), data.max())
                )
    print(f"Found {len(ln_gammas)} LayerNorm gammas with dim=1024")
    if ln_gammas:
        for name, mean, std, mn, mx in ln_gammas[-3:]:
            print(
                f"  {name}: mean={mean:.6f}, std={std:.6f}, min={mn:.6f}, max={mx:.6f}"
            )
except ImportError:
    print("onnx package not available, skipping weight inspection")
