#!/usr/bin/env python3
"""Debug script to compare feature computation with NeMo reference."""

import numpy as np
import onnxruntime as ort
import soundfile as sf

audio, sr = sf.read("test.wav", dtype="int16")
print(f"Audio: {len(audio)} samples, {sr} Hz, {len(audio) / sr:.1f}s")

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
    print(f"Resampled to {len(audio)} samples at {sr} Hz")

window_size = 400
hop_size = 160
n_fft = 512
n_mels = 128
sample_rate = 16000
preemph = 0.97

signal = audio.astype(np.float32) / 32768.0

preemph_signal = np.zeros_like(signal)
preemph_signal[0] = signal[0]
preemph_signal[1:] = signal[1:] - preemph * signal[:-1]

hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / window_size)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


nyquist = sample_rate / 2.0
mel_low = hz_to_mel(0.0)
mel_high = hz_to_mel(nyquist)
mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
hz_points = mel_to_hz(mel_points)

bin_points = np.floor(hz_points * n_fft / sample_rate + 0.5).astype(int)

mel_filters = np.zeros((n_mels, n_fft // 2 + 1))
for i in range(n_mels):
    left = bin_points[i]
    center = bin_points[i + 1]
    right = bin_points[i + 2]
    for b in range(left, center):
        if center > left:
            mel_filters[i, b] = (b - left) / (center - left)
    for b in range(center, right):
        if right > center:
            mel_filters[i, b] = (right - b) / (right - center)

chunk = preemph_signal[:16000]
num_frames = (len(chunk) - window_size) // hop_size + 1
print(f"First chunk: {len(chunk)} samples, {num_frames} frames")

features = np.zeros((n_mels, num_frames))
for t in range(num_frames):
    start = t * hop_size
    frame = chunk[start : start + window_size] * hann
    frame_padded = np.zeros(n_fft)
    frame_padded[:window_size] = frame
    spectrum = np.fft.rfft(frame_padded)
    power = np.abs(spectrum) ** 2
    mel_energy = mel_filters @ power
    features[:, t] = np.log(mel_energy + 1e-10)

print(
    f"Raw features: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}"
)

for b in range(n_mels):
    mean = features[b].mean()
    std = max(features[b].std(), 1e-5)
    features[b] = (features[b] - mean) / std

print(
    f"Normalized features: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}"
)

pad_len = n_fft // 2
chunk_padded = np.pad(chunk, (pad_len, pad_len), mode="reflect")
num_frames_centered = (len(chunk_padded) - window_size) // hop_size + 1
print(
    f"\nWith center padding: {len(chunk_padded)} samples, {num_frames_centered} frames"
)

features_centered = np.zeros((n_mels, num_frames_centered))
for t in range(num_frames_centered):
    start = t * hop_size
    frame = chunk_padded[start : start + window_size] * hann
    frame_padded = np.zeros(n_fft)
    frame_padded[:window_size] = frame
    spectrum = np.fft.rfft(frame_padded)
    power = np.abs(spectrum) ** 2
    mel_energy = mel_filters @ power
    features_centered[:, t] = np.log(mel_energy + 1e-10)

print(
    f"Raw features (centered): min={features_centered.min():.4f}, max={features_centered.max():.4f}, mean={features_centered.mean():.4f}"
)

for b in range(n_mels):
    mean = features_centered[b].mean()
    std = max(features_centered[b].std(), 1e-5)
    features_centered[b] = (features_centered[b] - mean) / std

print(
    f"Normalized features (centered): min={features_centered.min():.4f}, max={features_centered.max():.4f}, mean={features_centered.mean():.4f}"
)

print("\n--- Encoder Model Inputs ---")
sess = ort.InferenceSession(
    "data/parakeet/encoder.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
for inp in sess.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")
print("\n--- Encoder Model Outputs ---")
for out in sess.get_outputs():
    print(f"  {out.name}: shape={out.shape}, type={out.type}")

print("\n--- Decoder-Joint Model Inputs ---")
sess2 = ort.InferenceSession(
    "data/parakeet/decoder_joint.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
for inp in sess2.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")
print("\n--- Decoder-Joint Model Outputs ---")
for out in sess2.get_outputs():
    print(f"  {out.name}: shape={out.shape}, type={out.type}")

print("\n--- Running encoder (no center padding) ---")
cache_last_channel = np.zeros((1, 24, 70, 1024), dtype=np.float32)
cache_last_time = np.zeros((1, 24, 1024, 8), dtype=np.float32)
cache_last_channel_len = np.array([0], dtype=np.int64)
spk_targets = np.zeros((1, num_frames), dtype=np.float32)
bg_spk_targets = np.zeros((1, num_frames), dtype=np.float32)

enc_input = features.astype(np.float32).reshape(1, 128, num_frames)
print(f"Encoder input shape: {enc_input.shape}, dtype: {enc_input.dtype}")
enc_out = sess.run(
    None,
    {
        "processed_signal": enc_input,
        "processed_signal_length": np.array([num_frames], dtype=np.int64),
        "cache_last_channel": cache_last_channel,
        "cache_last_time": cache_last_time,
        "cache_last_channel_len": cache_last_channel_len,
        "spk_targets": spk_targets,
        "bg_spk_targets": bg_spk_targets,
    },
)
print(f"Encoder output: shape={enc_out[0].shape}, dtype={enc_out[0].dtype}")
print(
    f"  min={enc_out[0].min():.4f}, max={enc_out[0].max():.4f}, mean={enc_out[0].mean():.6f}"
)
print(f"Encoder output length: {enc_out[1]}")

encoded = enc_out[0]
T_out = encoded.shape[2]
first_frame = encoded[:, :, 0:1]
first_frame_btd = first_frame.transpose(0, 2, 1)
print(f"\nDecoder input (first frame): shape={first_frame_btd.shape}")

state1 = np.zeros((2, 1, 640), dtype=np.float32)
state2 = np.zeros((2, 1, 640), dtype=np.float32)
targets = np.array([[1024]], dtype=np.int64)

dec_out = sess2.run(
    None,
    {
        "encoder_outputs": first_frame_btd,
        "targets": targets,
        "input_states_1": state1,
        "input_states_2": state2,
    },
)
logits = dec_out[0]
print(f"Logits shape: {logits.shape}")
logits_flat = logits.flatten()
print(f"Logits: min={logits_flat.min():.4f}, max={logits_flat.max():.4f}")

top5_idx = np.argsort(logits_flat)[-5:][::-1]
print("Top 5 logits:")
for idx in top5_idx:
    print(f"  token {idx}: {logits_flat[idx]:.4f}")

print("\n--- Running encoder (with center padding) ---")
spk_targets_c = np.zeros((1, num_frames_centered), dtype=np.float32)
bg_spk_targets_c = np.zeros((1, num_frames_centered), dtype=np.float32)

enc_input_c = features_centered.astype(np.float32).reshape(1, 128, num_frames_centered)
cache_last_channel = np.zeros((1, 24, 70, 1024), dtype=np.float32)
cache_last_time = np.zeros((1, 24, 1024, 8), dtype=np.float32)
cache_last_channel_len = np.array([0], dtype=np.int64)

enc_out_c = sess.run(
    None,
    {
        "processed_signal": enc_input_c,
        "processed_signal_length": np.array([num_frames_centered], dtype=np.int64),
        "cache_last_channel": cache_last_channel,
        "cache_last_time": cache_last_time,
        "cache_last_channel_len": cache_last_channel_len,
        "spk_targets": spk_targets_c,
        "bg_spk_targets": bg_spk_targets_c,
    },
)
print(f"Encoder output: shape={enc_out_c[0].shape}, dtype={enc_out_c[0].dtype}")
print(
    f"  min={enc_out_c[0].min():.4f}, max={enc_out_c[0].max():.4f}, mean={enc_out_c[0].mean():.6f}"
)

encoded_c = enc_out_c[0]
T_out_c = encoded_c.shape[2]
first_frame_c = encoded_c[:, :, 0:1].transpose(0, 2, 1)
state1 = np.zeros((2, 1, 640), dtype=np.float32)
state2 = np.zeros((2, 1, 640), dtype=np.float32)

dec_out_c = sess2.run(
    None,
    {
        "encoder_outputs": first_frame_c,
        "targets": targets,
        "input_states_1": state1,
        "input_states_2": state2,
    },
)
logits_c = dec_out_c[0].flatten()
print(f"\nLogits (centered): min={logits_c.min():.4f}, max={logits_c.max():.4f}")
top5_idx_c = np.argsort(logits_c)[-5:][::-1]
print("Top 5 logits (centered):")
for idx in top5_idx_c:
    print(f"  token {idx}: {logits_c[idx]:.4f}")

print("\n--- Full greedy decode (centered features) ---")
state1 = np.zeros((2, 1, 640), dtype=np.float32)
state2 = np.zeros((2, 1, 640), dtype=np.float32)
last_token = 1024
decoded_tokens = []

for t in range(T_out_c):
    frame = encoded_c[:, :, t : t + 1].transpose(0, 2, 1)
    for _ in range(10):
        s1_backup = state1.copy()
        s2_backup = state2.copy()
        dec_out = sess2.run(
            None,
            {
                "encoder_outputs": frame,
                "targets": np.array([[last_token]], dtype=np.int64),
                "input_states_1": state1,
                "input_states_2": state2,
            },
        )
        logits = dec_out[0].flatten()[:1025]
        predicted = np.argmax(logits)
        if predicted == 1024:
            state1 = s1_backup
            state2 = s2_backup
            break
        decoded_tokens.append(predicted)
        last_token = predicted
        state1 = dec_out[2]
        state2 = dec_out[3]

print(f"Decoded tokens: {decoded_tokens}")
print(f"Number of tokens: {len(decoded_tokens)}")

tokens = {}
try:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file="data/parakeet/tokenizer.model")
    if decoded_tokens:
        text = sp.decode(decoded_tokens)
        print(f"Decoded text: {text}")
except ImportError:
    print("(sentencepiece not available for text decoding)")
