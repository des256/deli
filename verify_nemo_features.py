"""Compare Rust-style mel features vs NeMo preprocessor features.
Tests the ONNX encoder+decoder with NeMo's features to check if
mel feature mismatch is the root cause of 0-token decode.
"""

# ruff: noqa: E402
import sys

import numpy as np

for key in list(sys.modules.keys()):
    if "numba" in key:
        del sys.modules[key]

import onnxruntime as ort
import soundfile as sf
import torch

BLANK_ID = 1024
VOCAB_SIZE = 1025
ENCODER_DIM = 1024
NUM_LAYERS = 24
CACHE_CHANNEL_CONTEXT = 70
CACHE_TIME_CONTEXT = 8
DECODER_STATE_DIM = 640
MAX_SYMBOLS_PER_STEP = 10


def load_nemo_preprocessor():
    """Load just the preprocessor from the NeMo checkpoint."""
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    model = EncDecHybridRNNTCTCBPEModel.restore_from(
        "data/nemo/multitalker-parakeet-streaming-0.6b-v1.nemo", map_location="cpu"
    )
    model.eval()
    return model.preprocessor


def compute_nemo_features(preprocessor, wav_path):
    """Compute features using NeMo's preprocessor."""
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import torchaudio

        waveform = torch.from_numpy(data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000
    else:
        waveform = torch.from_numpy(data).unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    print(f"Waveform: shape={waveform.shape}, sr={sr}")

    with torch.no_grad():
        length = torch.tensor([waveform.shape[1]], dtype=torch.long)
        features, feat_length = preprocessor(input_signal=waveform, length=length)

    print(f"NeMo features: shape={features.shape}, length={feat_length}")
    print(
        f"  min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}"
    )

    return features.numpy(), feat_length.item()


def greedy_decode_onnx(enc_sess, dec_sess, features_np, num_frames):
    """Run encoder + greedy decode on features."""
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
            "audio_signal": features_np.astype(np.float32),
            "length": length,
            "cache_last_channel": cache_channel,
            "cache_last_time": cache_time,
            "cache_last_channel_len": cache_channel_len,
        },
    )

    encoder_out = enc_outputs[0]
    enc_out_frames = encoder_out.shape[2]
    print(f"Encoder: shape={encoder_out.shape}, frames={enc_out_frames}")
    print(
        f"  min={encoder_out.min():.4f}, max={encoder_out.max():.4f}, mean={encoder_out.mean():.6f}, std={encoder_out.std():.4f}"
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

            logits = dec_outputs[0].flatten()[:VOCAB_SIZE]

            if frame_idx < 3 and sym_step == 0:
                top5 = sorted(enumerate(logits), key=lambda x: -x[1])[:5]
                print(
                    f"  Frame{frame_idx} top5: {[(idx, f'{val:.4f}') for idx, val in top5]}"
                )

            predicted = int(np.argmax(logits))

            if predicted == BLANK_ID:
                state1 = state1_save
                state2 = state2_save
                break
            else:
                state1 = dec_outputs[2]
                state2 = dec_outputs[3]
                last_token = predicted
                all_tokens.append(predicted)

    return all_tokens


def main():
    model_dir = "data/parakeet"

    print("=== Loading NeMo preprocessor ===")
    preprocessor = load_nemo_preprocessor()
    print(f"Preprocessor config: {preprocessor.featurizer}")

    print("\n=== Computing NeMo features ===")
    nemo_features, nemo_num_frames = compute_nemo_features(preprocessor, "test.wav")

    print("\n=== Loading ONNX models ===")
    enc_sess = ort.InferenceSession(
        f"{model_dir}/encoder.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    dec_sess = ort.InferenceSession(
        f"{model_dir}/decoder_joint.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    print("\n=== Test 1: NeMo features, full audio as single chunk ===")
    tokens = greedy_decode_onnx(enc_sess, dec_sess, nemo_features, nemo_num_frames)
    print(f"Tokens: {len(tokens)}")

    print("\n=== Test 2: NeMo features, 1-second chunks ===")
    chunk_frames = 98
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
    all_tokens = []

    frame_offset = 0
    chunk_idx = 0
    while frame_offset < nemo_num_frames:
        frames_in_chunk = min(chunk_frames, nemo_num_frames - frame_offset)
        chunk_features = nemo_features[
            :, :, frame_offset : frame_offset + frames_in_chunk
        ]

        length = np.array([frames_in_chunk], dtype=np.int64)
        enc_outputs = enc_sess.run(
            [
                "outputs",
                "encoded_lengths",
                "cache_last_channel_next",
                "cache_last_time_next",
                "cache_last_channel_next_len",
            ],
            {
                "audio_signal": chunk_features.astype(np.float32),
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

                logits_flat = dec_outputs[0].flatten()[:VOCAB_SIZE]
                predicted = int(np.argmax(logits_flat))

                if predicted == BLANK_ID:
                    state1 = state1_save
                    state2 = state2_save
                    break
                else:
                    state1 = dec_outputs[2]
                    state2 = dec_outputs[3]
                    last_token = predicted
                    chunk_tokens.append(predicted)

        if chunk_idx < 3 or chunk_tokens:
            logits_flat = None
            print(
                f"  Chunk {chunk_idx}: {frames_in_chunk} frames -> {enc_out_frames} enc frames, {len(chunk_tokens)} tokens"
            )

        all_tokens.extend(chunk_tokens)
        frame_offset += frames_in_chunk
        chunk_idx += 1

    print(f"Total tokens (chunked NeMo features): {len(all_tokens)}")

    print("\n=== Feature comparison ===")
    print(f"NeMo features: shape={nemo_features.shape}")
    print(f"  min={nemo_features.min():.4f}, max={nemo_features.max():.4f}")
    print(f"  mean={nemo_features.mean():.4f}, std={nemo_features.std():.4f}")

    for bin_idx in [0, 32, 64, 96, 127]:
        bin_data = nemo_features[0, bin_idx, :]
        print(f"  Bin {bin_idx}: mean={bin_data.mean():.4f}, std={bin_data.std():.4f}")


if __name__ == "__main__":
    main()
