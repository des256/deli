#!/usr/bin/env python3
"""Check model structure and try alternative inputs."""

import numpy as np
import onnxruntime as ort
import soundfile as sf

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
    vocab_size = sp.get_piece_size()
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Token 0: '{sp.id_to_piece(0)}'")
    print(f"Token {vocab_size - 1}: '{sp.id_to_piece(vocab_size - 1)}' (last token)")
    print(f"Blank ID should be {vocab_size} (one past last token)")
    for i in [vocab_size - 2, vocab_size - 1]:
        print(f"  Token {i}: '{sp.id_to_piece(i)}'")
except ImportError:
    print("sentencepiece not available")

print("\n--- Test with random noise features ---")
np.random.seed(42)
rand_features = np.random.randn(1, 128, 98).astype(np.float32)

cache_ch = np.zeros((1, 24, 70, 1024), dtype=np.float32)
cache_t = np.zeros((1, 24, 1024, 8), dtype=np.float32)
cache_ch_len = np.array([0], dtype=np.int64)
spk = np.zeros((1, 98), dtype=np.float32)
bg = np.zeros((1, 98), dtype=np.float32)

enc_out = sess_enc.run(
    None,
    {
        "processed_signal": rand_features,
        "processed_signal_length": np.array([98], dtype=np.int64),
        "cache_last_channel": cache_ch,
        "cache_last_time": cache_t,
        "cache_last_channel_len": cache_ch_len,
        "spk_targets": spk,
        "bg_spk_targets": bg,
    },
)
print(f"Random features -> encoder std: {enc_out[0].std():.6f}")
print(f"  range: [{enc_out[0].min():.4f}, {enc_out[0].max():.4f}]")

print("\n--- Test with spk_targets=1.0 ---")
spk_high = np.ones((1, 98), dtype=np.float32)
enc_out2 = sess_enc.run(
    None,
    {
        "processed_signal": rand_features,
        "processed_signal_length": np.array([98], dtype=np.int64),
        "cache_last_channel": np.zeros((1, 24, 70, 1024), dtype=np.float32),
        "cache_last_time": np.zeros((1, 24, 1024, 8), dtype=np.float32),
        "cache_last_channel_len": np.array([0], dtype=np.int64),
        "spk_targets": spk_high,
        "bg_spk_targets": bg,
    },
)
print(f"spk=1.0 -> encoder std: {enc_out2[0].std():.6f}")
print(f"  range: [{enc_out2[0].min():.4f}, {enc_out2[0].max():.4f}]")

print("\n--- Test with 10x scaled features ---")
enc_out3 = sess_enc.run(
    None,
    {
        "processed_signal": rand_features * 10.0,
        "processed_signal_length": np.array([98], dtype=np.int64),
        "cache_last_channel": np.zeros((1, 24, 70, 1024), dtype=np.float32),
        "cache_last_time": np.zeros((1, 24, 1024, 8), dtype=np.float32),
        "cache_last_channel_len": np.array([0], dtype=np.int64),
        "spk_targets": np.zeros((1, 98), dtype=np.float32),
        "bg_spk_targets": np.zeros((1, 98), dtype=np.float32),
    },
)
print(f"10x features -> encoder std: {enc_out3[0].std():.6f}")
print(f"  range: [{enc_out3[0].min():.4f}, {enc_out3[0].max():.4f}]")

print("\n--- Test decoder-joint with synthetic strong encoder output ---")
syn_enc = np.random.randn(1, 1, 1024).astype(np.float32) * 0.1
state1 = np.zeros((2, 1, 640), dtype=np.float32)
state2 = np.zeros((2, 1, 640), dtype=np.float32)

dec_out = sess_dec.run(
    None,
    {
        "encoder_outputs": syn_enc,
        "targets": np.array([[1024]], dtype=np.int64),
        "input_states_1": state1,
        "input_states_2": state2,
    },
)
logits = dec_out[0].flatten()[:1025]
top5 = np.argsort(logits)[-5:][::-1]
print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
print("Top 5:")
for i in top5:
    print(f"  token {i}: {logits[i]:.4f}")

syn_enc2 = np.random.randn(1, 1, 1024).astype(np.float32) * 0.3
dec_out2 = sess_dec.run(
    None,
    {
        "encoder_outputs": syn_enc2,
        "targets": np.array([[1024]], dtype=np.int64),
        "input_states_1": state1,
        "input_states_2": state2,
    },
)
logits2 = dec_out2[0].flatten()[:1025]
top5_2 = np.argsort(logits2)[-5:][::-1]
print(f"\nSynthetic std=0.3: Logits range: [{logits2.min():.4f}, {logits2.max():.4f}]")
print("Top 5:")
for i in top5_2:
    print(f"  token {i}: {logits2[i]:.4f}")

print("\n--- Test: scale actual encoder output by various factors ---")

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

sig = audio.astype(np.float32) / 32768.0
pre = np.zeros_like(sig)
pre[0] = sig[0]
pre[1:] = sig[1:] - 0.97 * sig[:-1]
hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)


def hz2mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


mel_pts = np.linspace(hz2mel(0), hz2mel(8000), 130)
hz_pts = mel2hz(mel_pts)
bin_pts = np.floor(hz_pts * 512 / 16000 + 0.5).astype(int)
fbank = np.zeros((128, 257))
for i in range(128):
    lo, c, r = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
    for b in range(lo, c):
        if c > lo:
            fbank[i, b] = (b - lo) / (c - lo)
    for b in range(c, r):
        if r > c:
            fbank[i, b] = (r - b) / (r - c)

nf = (len(pre) - 400) // 160 + 1
feats = np.zeros((128, nf))
for t in range(nf):
    s = t * 160
    frame = pre[s : s + 400] * hann
    padded = np.zeros(512)
    padded[:400] = frame
    spec = np.fft.rfft(padded)
    pw = np.abs(spec) ** 2
    mel_e = fbank @ pw
    feats[:, t] = np.log(mel_e + 1e-10)
for b in range(128):
    m = feats[b].mean()
    s = max(feats[b].std(), 1e-5)
    feats[b] = (feats[b] - m) / s

enc_full = sess_enc.run(
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

encoded = enc_full[0]
print(f"Real encoder output: std={encoded.std():.6f}")

for scale in [1.0, 3.0, 5.0, 10.0, 20.0]:
    scaled = encoded * scale
    T = scaled.shape[2]
    state1 = np.zeros((2, 1, 640), dtype=np.float32)
    state2 = np.zeros((2, 1, 640), dtype=np.float32)
    last_tok = 1024
    tok_count = 0
    first_logits = None
    for t in range(T):
        frame = scaled[:, :, t : t + 1].transpose(0, 2, 1)
        for _ in range(10):
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
            if first_logits is None:
                first_logits = logits.copy()
            if pred == 1024:
                state1, state2 = s1b, s2b
                break
            tok_count += 1
            last_tok = pred
            state1, state2 = out[2], out[3]
    blank_logit = first_logits[1024]
    best_nonblank = np.max(first_logits[:1024])
    print(
        f"  scale={scale:5.1f}: {tok_count:3d} tokens, "
        f"blank_logit={blank_logit:.2f}, best_nonblank={best_nonblank:.2f}, "
        f"gap={blank_logit - best_nonblank:.2f}"
    )
