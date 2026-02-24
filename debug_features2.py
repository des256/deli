#!/usr/bin/env python3
"""Debug: try feeding full audio as single chunk, and multi-chunk streaming."""

import numpy as np
import onnxruntime as ort
import soundfile as sf

audio, sr = sf.read("test.wav", dtype="int16")
if sr != 16000:
    ratio = sr / 16000
    out_len = int(len(audio) / ratio)
    indices = np.arange(out_len) * ratio
    idx = indices.astype(int)
    frac = indices - idx
    idx_next = np.minimum(idx + 1, len(audio) - 1)
    audio = (
        audio[idx].astype(float) * (1 - frac) + audio[idx_next].astype(float) * frac
    ).astype(np.int16)
    sr = 16000
print(f"Audio: {len(audio)} samples at {sr} Hz ({len(audio) / sr:.1f}s)")

window_size = 400
hop_size = 160
n_fft = 512
n_mels = 128
preemph = 0.97


def compute_features(pcm_i16, do_normalize=True):
    """Compute mel features from int16 PCM."""
    sig = pcm_i16.astype(np.float32) / 32768.0
    pre = np.zeros_like(sig)
    pre[0] = sig[0]
    pre[1:] = sig[1:] - preemph * sig[:-1]

    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / window_size)

    def hz2mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz2mel(0), hz2mel(8000), n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bin_pts = np.floor(hz_pts * n_fft / sr + 0.5).astype(int)

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

    if do_normalize:
        for b in range(n_mels):
            m = feats[b].mean()
            s = max(feats[b].std(), 1e-5)
            feats[b] = (feats[b] - m) / s

    return feats, nf


sess_enc = ort.InferenceSession(
    "data/parakeet/encoder.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
sess_dec = ort.InferenceSession(
    "data/parakeet/decoder_joint.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)


def run_encoder(
    features, num_frames, cache_ch, cache_t, cache_ch_len, spk=None, bg=None
):
    """Run encoder, return (encoded, enc_len, new_caches)."""
    if spk is None:
        spk = np.zeros((1, num_frames), dtype=np.float32)
    if bg is None:
        bg = np.zeros((1, num_frames), dtype=np.float32)

    inp = features.astype(np.float32).reshape(1, 128, num_frames)
    outs = sess_enc.run(
        None,
        {
            "processed_signal": inp,
            "processed_signal_length": np.array([num_frames], dtype=np.int64),
            "cache_last_channel": cache_ch,
            "cache_last_time": cache_t,
            "cache_last_channel_len": cache_ch_len,
            "spk_targets": spk,
            "bg_spk_targets": bg,
        },
    )
    return outs[0], outs[1], outs[2], outs[3], outs[4]


def greedy_decode(encoded, enc_len):
    """Greedy RNNT decode."""
    T = encoded.shape[2]
    state1 = np.zeros((2, 1, 640), dtype=np.float32)
    state2 = np.zeros((2, 1, 640), dtype=np.float32)
    last_tok = 1024
    tokens = []
    for t in range(T):
        frame = encoded[:, :, t : t + 1].transpose(0, 2, 1)
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
            if pred == 1024:
                state1, state2 = s1b, s2b
                break
            tokens.append(pred)
            last_tok = pred
            state1, state2 = out[2], out[3]
    return tokens


def decode_text(token_ids):
    """Decode token IDs to text."""
    try:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file="data/parakeet/tokenizer.model")
        return sp.decode(token_ids)
    except ImportError:
        return f"[{len(token_ids)} tokens: {token_ids[:20]}...]"


print("=" * 60)
print("TEST 1: Full audio as single chunk")
print("=" * 60)
feats_full, nf_full = compute_features(audio)
print(
    f"Features: {n_mels}x{nf_full}, range=[{feats_full.min():.2f}, {feats_full.max():.2f}]"
)

cache_ch = np.zeros((1, 24, 70, 1024), dtype=np.float32)
cache_t = np.zeros((1, 24, 1024, 8), dtype=np.float32)
cache_ch_len = np.array([0], dtype=np.int64)

enc, enc_len, _, _, _ = run_encoder(
    feats_full, nf_full, cache_ch, cache_t, cache_ch_len
)
print(
    f"Encoder output: shape={enc.shape}, range=[{enc.min():.4f}, {enc.max():.4f}], mean={enc.mean():.6f}"
)
print(f"Encoder output frames: {enc_len}")

tokens = greedy_decode(enc, enc_len)
print(f"Decoded {len(tokens)} tokens")
if tokens:
    print(f"Text: {decode_text(tokens)}")
else:
    print("No tokens decoded (all blank)")

print("\n" + "=" * 60)
print("TEST 2: Streaming 1s chunks with cache propagation")
print("=" * 60)
cache_ch = np.zeros((1, 24, 70, 1024), dtype=np.float32)
cache_t = np.zeros((1, 24, 1024, 8), dtype=np.float32)
cache_ch_len = np.array([0], dtype=np.int64)

chunk_size = 16000
all_tokens = []
for i in range(0, len(audio), chunk_size):
    chunk = audio[i : i + chunk_size]
    if len(chunk) < window_size:
        break
    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

    feats, nf = compute_features(chunk)
    enc, enc_len, cache_ch, cache_t, cache_ch_len = run_encoder(
        feats, nf, cache_ch, cache_t, cache_ch_len
    )

    toks = greedy_decode(enc, enc_len)
    all_tokens.extend(toks)
    print(
        f"  Chunk {i // chunk_size}: {nf} frames -> {enc.shape[2]} enc frames, "
        f"enc range=[{enc.min():.4f}, {enc.max():.4f}], {len(toks)} tokens"
    )

print(f"\nTotal: {len(all_tokens)} tokens")
if all_tokens:
    print(f"Text: {decode_text(all_tokens)}")
else:
    print("No tokens decoded")

print("\n" + "=" * 60)
print("TEST 3: Full audio, NO normalization")
print("=" * 60)
feats_raw, nf_raw = compute_features(audio, do_normalize=False)
print(
    f"Raw features: range=[{feats_raw.min():.2f}, {feats_raw.max():.2f}], mean={feats_raw.mean():.2f}"
)

cache_ch = np.zeros((1, 24, 70, 1024), dtype=np.float32)
cache_t = np.zeros((1, 24, 1024, 8), dtype=np.float32)
cache_ch_len = np.array([0], dtype=np.int64)

enc_raw, enc_len_raw, _, _, _ = run_encoder(
    feats_raw, nf_raw, cache_ch, cache_t, cache_ch_len
)
print(
    f"Encoder output: shape={enc_raw.shape}, range=[{enc_raw.min():.4f}, {enc_raw.max():.4f}]"
)

tokens_raw = greedy_decode(enc_raw, enc_len_raw)
print(f"Decoded {len(tokens_raw)} tokens")
if tokens_raw:
    print(f"Text: {decode_text(tokens_raw)}")

print("\n" + "=" * 60)
print("TEST 4: Full audio with dithering (NeMo default 1e-5)")
print("=" * 60)
sig = audio.astype(np.float32) / 32768.0
np.random.seed(42)
sig_dithered = sig + np.random.randn(len(sig)).astype(np.float32) * 1e-5
audio_dithered = (sig_dithered * 32768.0).astype(np.int16)
feats_d, nf_d = compute_features(audio_dithered)
print(f"Features (dithered): range=[{feats_d.min():.2f}, {feats_d.max():.2f}]")

cache_ch = np.zeros((1, 24, 70, 1024), dtype=np.float32)
cache_t = np.zeros((1, 24, 1024, 8), dtype=np.float32)
cache_ch_len = np.array([0], dtype=np.int64)

enc_d, enc_len_d, _, _, _ = run_encoder(feats_d, nf_d, cache_ch, cache_t, cache_ch_len)
print(
    f"Encoder output: shape={enc_d.shape}, range=[{enc_d.min():.4f}, {enc_d.max():.4f}]"
)

tokens_d = greedy_decode(enc_d, enc_len_d)
print(f"Decoded {len(tokens_d)} tokens")
if tokens_d:
    print(f"Text: {decode_text(tokens_d)}")

print("\n" + "=" * 60)
print("TEST 5: Per-frame encoder output statistics (full audio)")
print("=" * 60)
for t in range(min(5, enc.shape[2])):
    frame_data = enc[0, :, t]
    print(
        f"  Frame {t}: mean={frame_data.mean():.6f}, std={frame_data.std():.6f}, "
        f"min={frame_data.min():.4f}, max={frame_data.max():.4f}"
    )

overall_std = enc.std()
print(f"\nOverall encoder output std: {overall_std:.6f}")
print("Expected for trained model with LayerNorm: ~0.05-0.3")
