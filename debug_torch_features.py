#!/usr/bin/env python3
"""Use torch.stft to compute features, with manual mel filterbank."""

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch


def make_mel_filterbank(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=None, htk=True):
    """Build mel filterbank matrix [n_mels, n_fft//2+1]."""
    if fmax is None:
        fmax = sr / 2.0

    def hz2mel(hz):
        if htk:
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        return 1127.0 * np.log(1.0 + hz / 700.0)

    def mel2hz(m):
        if htk:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        return 700.0 * (np.exp(m / 1127.0) - 1.0)

    mel_lo = hz2mel(fmin)
    mel_hi = hz2mel(fmax)
    mels = np.linspace(mel_lo, mel_hi, n_mels + 2)
    freqs = mel2hz(mels)

    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, center, hi = freqs[i], freqs[i + 1], freqs[i + 2]
        if center > lo:
            mask = (fft_freqs >= lo) & (fft_freqs <= center)
            fb[i, mask] = (fft_freqs[mask] - lo) / (center - lo)
        if hi > center:
            mask = (fft_freqs >= center) & (fft_freqs <= hi)
            fb[i, mask] = (hi - fft_freqs[mask]) / (hi - center)
    return fb


def compute_features_torch(pcm_i16, sr=16000, center=True, normalize=True, htk=True):
    """Compute mel features using torch.stft."""
    sig = pcm_i16.astype(np.float32) / 32768.0
    pre = np.zeros_like(sig)
    pre[0] = sig[0]
    pre[1:] = sig[1:] - 0.97 * sig[:-1]

    x = torch.from_numpy(pre).unsqueeze(0)
    window = torch.hann_window(400, periodic=True)
    stft_out = torch.stft(
        x,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=window,
        center=center,
        return_complex=True,
    )
    power = stft_out.abs() ** 2

    fb = make_mel_filterbank(sr=sr, n_fft=512, n_mels=128, htk=htk)
    fb_t = torch.from_numpy(fb).float()
    mel = torch.matmul(fb_t, power.squeeze(0))
    log_mel = torch.log(mel + 1e-10)

    if normalize:
        mean = log_mel.mean(dim=1, keepdim=True)
        std = log_mel.std(dim=1, keepdim=True).clamp(min=1e-5)
        log_mel = (log_mel - mean) / std

    return log_mel.numpy(), log_mel.shape[1]


audio, sr_val = sf.read("test.wav", dtype="int16")
if sr_val != 16000:
    ratio = sr_val / 16000
    n = int(len(audio) / ratio)
    idx = (np.arange(n) * ratio).astype(int)
    frac = np.arange(n) * ratio - idx
    nxt = np.minimum(idx + 1, len(audio) - 1)
    audio = (audio[idx] * (1 - frac) + audio[nxt] * frac).astype(np.int16)
print(f"Audio: {len(audio)} samples at 16000 Hz ({len(audio) / 16000:.1f}s)")

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


def run_pipeline(feats, nf, label):
    """Run encoder + greedy decode."""
    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")
    print(
        f"Features: ({feats.shape[0]}, {nf}), "
        f"range=[{feats.min():.2f}, {feats.max():.2f}], mean={feats.mean():.4f}"
    )

    out = sess_enc.run(
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
    enc = out[0]
    print(
        f"Encoder: shape={enc.shape}, std={enc.std():.6f}, "
        f"range=[{enc.min():.4f}, {enc.max():.4f}]"
    )

    state1 = np.zeros((2, 1, 640), dtype=np.float32)
    state2 = np.zeros((2, 1, 640), dtype=np.float32)
    last_tok = 1024
    tokens = []
    for t in range(enc.shape[2]):
        frame = enc[:, :, t : t + 1].transpose(0, 2, 1)
        for _ in range(10):
            s1b, s2b = state1.copy(), state2.copy()
            d = sess_dec.run(
                None,
                {
                    "encoder_outputs": frame,
                    "targets": np.array([[last_tok]], dtype=np.int64),
                    "input_states_1": state1,
                    "input_states_2": state2,
                },
            )
            logits = d[0].flatten()[:1025]
            pred = int(np.argmax(logits))
            if t == 0 and not tokens:
                top3 = np.argsort(logits)[-3:][::-1]
                gap = logits[1024] - np.max(logits[:1024])
                print(
                    f"Frame 0: blank={logits[1024]:.2f}, gap={gap:.2f}, "
                    f"top3={[(int(i), round(float(logits[i]), 2)) for i in top3]}"
                )
            if pred == 1024:
                state1, state2 = s1b, s2b
                break
            tokens.append(pred)
            last_tok = pred
            state1, state2 = d[2], d[3]

    print(f"Tokens: {len(tokens)}")
    if sp and tokens:
        print(f"Text: {sp.decode(tokens)}")
    elif not tokens:
        print("(all blank)")


run_pipeline(
    *compute_features_torch(audio, center=True, normalize=True, htk=True),
    "torch center=True, normalize, htk=True",
)

run_pipeline(
    *compute_features_torch(audio, center=False, normalize=True, htk=True),
    "torch center=False, normalize, htk=True",
)

run_pipeline(
    *compute_features_torch(audio, center=True, normalize=False, htk=True),
    "torch center=True, NO normalize, htk=True",
)

run_pipeline(
    *compute_features_torch(audio, center=True, normalize=True, htk=False),
    "torch center=True, normalize, htk=False (Slaney)",
)

run_pipeline(
    *compute_features_torch(audio, center=False, normalize=False, htk=True),
    "torch center=False, NO normalize, htk=True",
)
