"""
Verify streaming ASR pipeline matching the exact Rust implementation.

This script replicates the Rust streaming pipeline step-by-step:
1. Load WAV file (same as Rust wav-asr)
2. Compute mel features per audio chunk (same as Rust compute_features)
3. Run encoder with cache propagation
4. Run decoder-joint per frame with greedy decode
5. Print tokens and compare with Rust output

Run with: /usr/bin/python3 verify_streaming.py test.wav
"""

import math
import sys
import wave

import numpy as np
import onnxruntime as ort

REQUIRED_SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 25
HOP_SIZE_MS = 10
NUM_MEL_BINS = 128
PRE_EMPHASIS = 0.97
FFT_SIZE = 512
BLANK_ID = 1024
MAX_SYMBOLS_PER_STEP = 10
VOCAB_SIZE = 1025
ENCODER_DIM = 1024
NUM_LAYERS = 24
CACHE_CHANNEL_CONTEXT = 70
CACHE_TIME_CONTEXT = 8
DECODER_STATE_DIM = 640


def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(sample_rate, fft_size, num_bins):
    """Exact match of Rust create_mel_filterbank."""
    nyquist = sample_rate / 2.0
    mel_low = hz_to_mel(0.0)
    mel_high = hz_to_mel(nyquist)

    mel_points = []
    for i in range(num_bins + 2):
        mel_val = mel_low + (mel_high - mel_low) * i / (num_bins + 1)
        mel_points.append(mel_to_hz(mel_val))

    bin_points = []
    for freq in mel_points:
        bin_points.append(int(math.floor(freq * fft_size / sample_rate + 0.5)))

    filters = []
    for i in range(num_bins):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        filt = []
        if center > left:
            for b in range(left, center):
                weight = (b - left) / (center - left)
                if weight > 0.0:
                    filt.append((b, weight))
        if right > center:
            for b in range(center, right):
                weight = (right - b) / (right - center)
                if weight > 0.0:
                    filt.append((b, weight))
        filters.append(filt)

    return filters


def compute_power_spectrum(windowed):
    """DFT-based power spectrum matching Rust."""
    n = len(windowed)
    power = np.zeros(n // 2 + 1, dtype=np.float32)
    for k in range(n // 2 + 1):
        real = 0.0
        imag = 0.0
        for t in range(n):
            angle = -2.0 * math.pi * k * t / n
            real += windowed[t] * math.cos(angle)
            imag += windowed[t] * math.sin(angle)
        power[k] = real * real + imag * imag
    return power


def compute_features_rust_style(pcm_i16, sample_rate):
    """Exact match of Rust compute_raw_features (no normalization)."""
    window_size = (WINDOW_SIZE_MS * sample_rate) // 1000
    hop_size = (HOP_SIZE_MS * sample_rate) // 1000

    if len(pcm_i16) < window_size:
        raise ValueError(
            f"Audio too short: {len(pcm_i16)} samples, need at least {window_size}"
        )

    signal = np.zeros(len(pcm_i16), dtype=np.float32)
    signal[0] = pcm_i16[0] / 32768.0
    for i in range(1, len(pcm_i16)):
        signal[i] = pcm_i16[i] / 32768.0 - PRE_EMPHASIS * pcm_i16[i - 1] / 32768.0

    hann = np.array(
        [
            0.5 - 0.5 * math.cos(2.0 * math.pi * i / window_size)
            for i in range(window_size)
        ],
        dtype=np.float32,
    )

    mel_filters = create_mel_filterbank(sample_rate, FFT_SIZE, NUM_MEL_BINS)

    num_frames = (len(signal) - window_size) // hop_size + 1

    features = np.zeros(NUM_MEL_BINS * num_frames, dtype=np.float32)

    for frame_idx in range(num_frames):
        start = frame_idx * hop_size
        end = start + window_size

        windowed = signal[start:end] * hann

        padded = np.zeros(FFT_SIZE, dtype=np.float32)
        padded[:window_size] = windowed

        fft_result = np.fft.rfft(padded)
        power_spectrum = np.abs(fft_result) ** 2

        for bin_idx, filt in enumerate(mel_filters):
            mel_energy = 0.0
            for freq_bin, weight in filt:
                mel_energy += power_spectrum[freq_bin] * weight
            features[bin_idx * num_frames + frame_idx] = math.log(mel_energy + 1e-10)

    return features, num_frames


def load_tokens(path):
    """Load SentencePiece tokenizer.model and extract tokens."""
    try:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=path)
        tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        return tokens
    except ImportError:
        pass

    tokens = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                tokens.append(parts[0])
    return tokens


def load_wav(path):
    """Load WAV file as int16 PCM, matching Rust wav-asr behavior."""
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

        if sample_width == 2:
            samples = np.frombuffer(raw, dtype=np.int16)
        elif sample_width == 4:
            samples_32 = np.frombuffer(raw, dtype=np.int32)
            max_val = float(1 << (sample_width * 8 - 1))
            samples = (samples_32.astype(np.float32) / max_val * 32767.0).astype(
                np.int16
            )
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

        print(
            f"WAV: {sample_rate} Hz, {channels} ch, {sample_width * 8} bit, {n_frames} frames"
        )
        return samples, sample_rate


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "data/parakeet"

    pcm, sr = load_wav(wav_path)
    print(f"Audio: {len(pcm) / sr:.1f}s, {len(pcm)} samples")

    if sr != REQUIRED_SAMPLE_RATE:
        print(f"WARNING: sample rate {sr} != {REQUIRED_SAMPLE_RATE}, need resampling")
        ratio = sr / REQUIRED_SAMPLE_RATE
        out_len = int(len(pcm) / ratio)
        new_pcm = np.zeros(out_len, dtype=np.int16)
        for i in range(out_len):
            src = i * ratio
            idx = int(src)
            frac = src - idx
            if idx + 1 < len(pcm):
                new_pcm[i] = int(pcm[idx] * (1 - frac) + pcm[idx + 1] * frac)
            else:
                new_pcm[i] = pcm[idx]
        pcm = new_pcm
        sr = REQUIRED_SAMPLE_RATE
        print(f"Resampled to: {len(pcm) / sr:.1f}s, {len(pcm)} samples")

    tokens = load_tokens(f"{model_dir}/tokenizer.model")
    print(f"Loaded {len(tokens)} tokens")

    print("Loading encoder...")
    enc_sess = ort.InferenceSession(
        f"{model_dir}/encoder.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print("Loading decoder_joint...")
    dec_sess = ort.InferenceSession(
        f"{model_dir}/decoder_joint.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    print("\nEncoder inputs:")
    for inp in enc_sess.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("Encoder outputs:")
    for out in enc_sess.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")
    print("\nDecoder inputs:")
    for inp in dec_sess.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("Decoder outputs:")
    for out in dec_sess.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    cache_channel = np.zeros(
        (1, NUM_LAYERS, CACHE_CHANNEL_CONTEXT, ENCODER_DIM), dtype=np.float32
    )
    cache_time = np.zeros(
        (1, NUM_LAYERS, ENCODER_DIM, CACHE_TIME_CONTEXT), dtype=np.float32
    )
    cache_channel_len = np.array([0], dtype=np.int64)

    state1 = np.zeros((2, 1, DECODER_STATE_DIM), dtype=np.float32)
    state2 = np.zeros((2, 1, DECODER_STATE_DIM), dtype=np.float32)
    last_token = BLANK_ID

    chunk_samples = REQUIRED_SAMPLE_RATE
    decoded_text = ""
    total_tokens = 0

    chunk_idx = 0
    offset = 0
    while offset < len(pcm):
        end = min(offset + chunk_samples, len(pcm))
        chunk = pcm[offset:end]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")

        features, num_frames = compute_features_rust_style(
            chunk.astype(np.float32), REQUIRED_SAMPLE_RATE
        )

        feat_min = features.min()
        feat_max = features.max()
        feat_mean = features.mean()
        print(
            f"\nChunk {chunk_idx}: {num_frames} mel frames, features: min={feat_min:.4f}, max={feat_max:.4f}, mean={feat_mean:.4f}"
        )

        audio_signal = features.reshape(NUM_MEL_BINS, num_frames)[np.newaxis, :, :]
        length = np.array([num_frames], dtype=np.int64)

        enc_outputs = enc_sess.run(
            [
                "outputs",
                "encoded_lengths",
                "cache_last_channel_next",
                "cache_last_time_next",
                "cache_last_channel_next_len",
            ],
            {
                "audio_signal": audio_signal,
                "length": length,
                "cache_last_channel": cache_channel,
                "cache_last_time": cache_time,
                "cache_last_channel_len": cache_channel_len,
            },
        )

        encoder_out = enc_outputs[0]
        encoded_len = enc_outputs[1]
        cache_channel = enc_outputs[2]
        cache_time = enc_outputs[3]
        cache_channel_len = enc_outputs[4]

        enc_out_frames = encoder_out.shape[2]
        enc_stats = f"min={encoder_out.min():.4f}, max={encoder_out.max():.4f}, mean={encoder_out.mean():.6f}, std={encoder_out.std():.4f}"
        print(
            f"  Encoder output: shape={encoder_out.shape}, dtype={encoder_out.dtype}, {enc_stats}"
        )
        print(f"  Encoded length: {encoded_len}")

        chunk_tokens = []
        for frame_idx in range(enc_out_frames):
            enc_frame = encoder_out[:, :, frame_idx : frame_idx + 1]

            for sym_step in range(MAX_SYMBOLS_PER_STEP):
                state1_save = state1.copy()
                state2_save = state2.copy()

                targets = np.array([[last_token]], dtype=np.int32)
                target_length = np.array([1], dtype=np.int32)

                dec_outputs = dec_sess.run(
                    [
                        "outputs",
                        "prednet_lengths",
                        "output_states_1",
                        "output_states_2",
                    ],
                    {
                        "encoder_outputs": enc_frame.astype(np.float32),
                        "targets": targets,
                        "target_length": target_length,
                        "input_states_1": state1,
                        "input_states_2": state2,
                    },
                )

                logits = dec_outputs[0]
                new_state1 = dec_outputs[2]
                new_state2 = dec_outputs[3]

                if frame_idx == 0 and len(chunk_tokens) == 0 and sym_step == 0:
                    logits_flat = logits.flatten()
                    print(
                        f"  Decoder output shape: {logits.shape}, dtype: {logits.dtype}"
                    )
                    top5 = sorted(
                        enumerate(logits_flat[:VOCAB_SIZE]), key=lambda x: -x[1]
                    )[:5]
                    blank_logit = (
                        logits_flat[BLANK_ID]
                        if BLANK_ID < len(logits_flat)
                        else float("nan")
                    )
                    print(
                        f"  Frame0 logits: predicted={top5[0][0]}, blank_logit={blank_logit:.4f}"
                    )
                    print(f"  Top5: {[(idx, f'{val:.4f}') for idx, val in top5]}")

                logits_flat = logits.flatten()[:VOCAB_SIZE]
                predicted = int(np.argmax(logits_flat))

                if predicted == BLANK_ID:
                    state1 = state1_save
                    state2 = state2_save
                    break
                else:
                    state1 = new_state1
                    state2 = new_state2
                    last_token = predicted
                    chunk_tokens.append(predicted)

        chunk_text = ""
        for tid in chunk_tokens:
            if tid < len(tokens):
                chunk_text += tokens[tid].replace("▁", " ")

        decoded_text += chunk_text
        total_tokens += len(chunk_tokens)
        print(
            f"  Chunk {chunk_idx}: {len(chunk_tokens)} tokens, text='{chunk_text.strip()}'"
        )

        chunk_idx += 1
        offset += chunk_samples

    print(f"\n{'=' * 60}")
    print(f"Final: {total_tokens} total tokens")
    print(f"Text: '{decoded_text.strip()}'")

    print(f"\n{'=' * 60}")
    print("RETRY with 2.5s chunks (40000 samples):")
    chunk_samples = 40000

    cache_channel = np.zeros(
        (1, NUM_LAYERS, CACHE_CHANNEL_CONTEXT, ENCODER_DIM), dtype=np.float32
    )
    cache_time = np.zeros(
        (1, NUM_LAYERS, ENCODER_DIM, CACHE_TIME_CONTEXT), dtype=np.float32
    )
    cache_channel_len = np.array([0], dtype=np.int64)
    state1 = np.zeros((2, 1, DECODER_STATE_DIM), dtype=np.float32)
    state2 = np.zeros((2, 1, DECODER_STATE_DIM), dtype=np.float32)
    last_token = BLANK_ID
    decoded_text = ""
    total_tokens = 0

    chunk_idx = 0
    offset = 0
    while offset < len(pcm):
        end = min(offset + chunk_samples, len(pcm))
        chunk = pcm[offset:end]
        if len(chunk) < 400:
            break
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")

        features, num_frames = compute_features_rust_style(
            chunk.astype(np.float32), REQUIRED_SAMPLE_RATE
        )
        print(f"\nChunk {chunk_idx}: {num_frames} mel frames")

        audio_signal = features.reshape(NUM_MEL_BINS, num_frames)[np.newaxis, :, :]
        length = np.array([num_frames], dtype=np.int64)

        enc_outputs = enc_sess.run(
            [
                "outputs",
                "encoded_lengths",
                "cache_last_channel_next",
                "cache_last_time_next",
                "cache_last_channel_next_len",
            ],
            {
                "audio_signal": audio_signal,
                "length": length,
                "cache_last_channel": cache_channel,
                "cache_last_time": cache_time,
                "cache_last_channel_len": cache_channel_len,
            },
        )

        encoder_out = enc_outputs[0]
        cache_channel = enc_outputs[2]
        cache_time = enc_outputs[3]
        cache_channel_len = enc_outputs[4]

        enc_out_frames = encoder_out.shape[2]
        print(f"  Encoder: {encoder_out.shape}, {enc_out_frames} frames")

        chunk_tokens = []
        for frame_idx in range(enc_out_frames):
            enc_frame = encoder_out[:, :, frame_idx : frame_idx + 1]

            for sym_step in range(MAX_SYMBOLS_PER_STEP):
                state1_save = state1.copy()
                state2_save = state2.copy()

                targets = np.array([[last_token]], dtype=np.int32)
                target_length = np.array([1], dtype=np.int32)

                dec_outputs = dec_sess.run(
                    [
                        "outputs",
                        "prednet_lengths",
                        "output_states_1",
                        "output_states_2",
                    ],
                    {
                        "encoder_outputs": enc_frame.astype(np.float32),
                        "targets": targets,
                        "target_length": target_length,
                        "input_states_1": state1,
                        "input_states_2": state2,
                    },
                )

                logits = dec_outputs[0]
                new_state1 = dec_outputs[2]
                new_state2 = dec_outputs[3]

                if frame_idx == 0 and len(chunk_tokens) == 0 and sym_step == 0:
                    logits_flat = logits.flatten()
                    top5 = sorted(
                        enumerate(logits_flat[:VOCAB_SIZE]), key=lambda x: -x[1]
                    )[:5]
                    print(
                        f"  Frame0 top5: {[(idx, f'{val:.4f}') for idx, val in top5]}"
                    )

                logits_flat = logits.flatten()[:VOCAB_SIZE]
                predicted = int(np.argmax(logits_flat))

                if predicted == BLANK_ID:
                    state1 = state1_save
                    state2 = state2_save
                    break
                else:
                    state1 = new_state1
                    state2 = new_state2
                    last_token = predicted
                    chunk_tokens.append(predicted)

        chunk_text = ""
        for tid in chunk_tokens:
            if tid < len(tokens):
                chunk_text += tokens[tid].replace("▁", " ")
        decoded_text += chunk_text
        total_tokens += len(chunk_tokens)
        print(
            f"  Chunk {chunk_idx}: {len(chunk_tokens)} tokens, text='{chunk_text.strip()}'"
        )

        chunk_idx += 1
        offset += chunk_samples

    print(f"\n{'=' * 60}")
    print(f"Final (2.5s chunks): {total_tokens} total tokens")
    print(f"Text: '{decoded_text.strip()}'")

    print(f"\n{'=' * 60}")
    print("BATCH mode: all audio as single chunk:")

    cache_channel = np.zeros(
        (1, NUM_LAYERS, CACHE_CHANNEL_CONTEXT, ENCODER_DIM), dtype=np.float32
    )
    cache_time = np.zeros(
        (1, NUM_LAYERS, ENCODER_DIM, CACHE_TIME_CONTEXT), dtype=np.float32
    )
    cache_channel_len = np.array([0], dtype=np.int64)
    state1 = np.zeros((2, 1, DECODER_STATE_DIM), dtype=np.float32)
    state2 = np.zeros((2, 1, DECODER_STATE_DIM), dtype=np.float32)
    last_token = BLANK_ID

    features, num_frames = compute_features_rust_style(
        pcm.astype(np.float32), REQUIRED_SAMPLE_RATE
    )
    print(f"Total: {num_frames} mel frames")

    audio_signal = features.reshape(NUM_MEL_BINS, num_frames)[np.newaxis, :, :]
    length = np.array([num_frames], dtype=np.int64)

    enc_outputs = enc_sess.run(
        [
            "outputs",
            "encoded_lengths",
            "cache_last_channel_next",
            "cache_last_time_next",
            "cache_last_channel_next_len",
        ],
        {
            "audio_signal": audio_signal,
            "length": length,
            "cache_last_channel": cache_channel,
            "cache_last_time": cache_time,
            "cache_last_channel_len": cache_channel_len,
        },
    )

    encoder_out = enc_outputs[0]
    enc_out_frames = encoder_out.shape[2]
    print(
        f"Encoder: {encoder_out.shape}, {enc_out_frames} frames, dtype={encoder_out.dtype}"
    )
    print(
        f"Stats: min={encoder_out.min():.4f}, max={encoder_out.max():.4f}, mean={encoder_out.mean():.6f}"
    )

    all_tokens = []
    for frame_idx in range(enc_out_frames):
        enc_frame = encoder_out[:, :, frame_idx : frame_idx + 1]

        for sym_step in range(MAX_SYMBOLS_PER_STEP):
            state1_save = state1.copy()
            state2_save = state2.copy()

            targets = np.array([[last_token]], dtype=np.int32)
            target_length = np.array([1], dtype=np.int32)

            dec_outputs = dec_sess.run(
                ["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
                {
                    "encoder_outputs": enc_frame.astype(np.float32),
                    "targets": targets,
                    "target_length": target_length,
                    "input_states_1": state1,
                    "input_states_2": state2,
                },
            )

            logits = dec_outputs[0]
            new_state1 = dec_outputs[2]
            new_state2 = dec_outputs[3]

            if frame_idx == 0 and len(all_tokens) == 0 and sym_step == 0:
                logits_flat = logits.flatten()
                top5 = sorted(enumerate(logits_flat[:VOCAB_SIZE]), key=lambda x: -x[1])[
                    :5
                ]
                print(f"Frame0 top5: {[(idx, f'{val:.4f}') for idx, val in top5]}")

            logits_flat = logits.flatten()[:VOCAB_SIZE]
            predicted = int(np.argmax(logits_flat))

            if predicted == BLANK_ID:
                state1 = state1_save
                state2 = state2_save
                break
            else:
                state1 = new_state1
                state2 = new_state2
                last_token = predicted
                all_tokens.append(predicted)

    batch_text = ""
    for tid in all_tokens:
        if tid < len(tokens):
            batch_text += tokens[tid].replace("▁", " ")

    print(f"\nBatch: {len(all_tokens)} total tokens")
    print(f"Text: '{batch_text.strip()}'")


if __name__ == "__main__":
    main()
