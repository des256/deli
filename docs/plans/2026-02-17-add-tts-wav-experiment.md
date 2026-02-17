# TTS WAV Experiment Implementation Plan

Created: 2026-02-17
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No

> **Status Lifecycle:** PENDING → COMPLETE → VERIFIED
> **Iterations:** Tracks implement→verify cycles (incremented by verify phase)
>
> - PENDING: Initial state, awaiting implementation
> - COMPLETE: All tasks implemented
> - VERIFIED: All checks passed
>
> **Approval Gate:** Implementation CANNOT proceed until `Approved: Yes`
> **Worktree:** No — working directly on current branch

## Summary

**Goal:** Create a standalone experiment binary `tts-wav` that synthesizes the sentence "To be, or not to be, equals, minus one." using Kokoro TTS and writes the result as a WAV file whose path is given as a CLI argument.

**Architecture:** A single `main.rs` binary in `experiments/tts-wav/`. It initializes `Inference::cpu()`, loads the Kokoro model via `use_kokoro()`, calls `speak()` to get a `Tensor<i16>` of 24kHz mono PCM audio, then writes it to a WAV file using the `hound` crate. The output filename is the sole required CLI argument.

**Tech Stack:** `deli-infer` (Kokoro TTS), `deli-base` (Tensor, logging), `hound` (WAV writing), `tokio` (async runtime for `speak()`).

## Scope

### In Scope

- New experiment binary at `experiments/tts-wav/`
- CLI argument for output WAV filename
- WAV file output: 24kHz, mono, 16-bit signed integer PCM
- Hardcoded sentence: "To be, or not to be, equals, minus one."
- Hardcoded model paths: `models/kokoro/kokoro-v1.0.onnx` and `models/kokoro/bf_emma.npy`
- Hardcoded espeak-ng data path: `/usr/lib/x86_64-linux-gnu/espeak-ng-data`

### Out of Scope

- Configurable sentence text (hardcoded per spec)
- Configurable model paths or voice selection
- GPU/CUDA support (CPU only for this experiment)
- Audio playback
- Any changes to `deli-infer` or other crates

## Prerequisites

- Kokoro model files present at `models/kokoro/kokoro-v1.0.onnx` and `models/kokoro/bf_emma.npy`
- espeak-ng installed with data at `/usr/lib/x86_64-linux-gnu/espeak-ng-data`
- Working `deli-infer` crate with Kokoro TTS (just implemented and verified)

## Context for Implementer

- **Patterns to follow:** The `wav-asr` experiment at `experiments/wav-asr/src/main.rs` is the closest analog — it's a binary that reads/writes WAV files using `hound` and calls `deli-infer`. Follow its Cargo.toml structure (bin name, path, dependencies) and main.rs patterns (arg parsing, error handling, logging).
- **Conventions:** Experiment binaries use `[[bin]]` with `name = "tts_wav"` and `path = "src/main.rs"`. They use `#[tokio::main]` async main and `deli_base::init_stdout_logger()` for log output. CLI args are parsed via `std::env::args()`.
- **Key files:**
  - `experiments/wav-asr/Cargo.toml` — reference Cargo.toml structure
  - `experiments/wav-asr/src/main.rs` — reference main.rs structure and `hound` usage
  - `crates/deli-infer/src/inference.rs:99-107` — `use_kokoro()` API
  - `crates/deli-infer/src/tts/kokoro.rs:90` — `speak(&self, text) -> Result<Tensor<i16>>` returns 24kHz mono PCM
- **Gotchas:**
  - `speak()` takes `&self` (not `&mut self`) — the Kokoro instance can be shared
  - The returned `Tensor<i16>` has `data: Vec<i16>` with samples at 24kHz sample rate
  - `hound::WavWriter` needs `WavSpec` with `channels: 1`, `sample_rate: 24000`, `bits_per_sample: 16`, `sample_format: SampleFormat::Int`
  - Don't use the `cuda` feature for deli-infer — this experiment is CPU-only

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Create tts-wav experiment binary

**Total Tasks:** 1 | **Completed:** 1 | **Remaining:** 0

## Implementation Tasks

### Task 1: Create tts-wav experiment binary

**Objective:** Create a complete experiment binary that synthesizes speech and writes WAV output.

**Dependencies:** None

**Files:**

- Create: `experiments/tts-wav/Cargo.toml`
- Create: `experiments/tts-wav/src/main.rs`

**Key Decisions / Notes:**

- Follow `experiments/wav-asr/Cargo.toml` structure for Cargo.toml layout
- Use `Inference::cpu()` (no CUDA feature needed)
- Use `hound::WavWriter::create()` with spec for 24kHz/16-bit/mono
- Write each `i16` sample via `writer.write_sample(sample)`, then call `writer.finalize()?` to complete the WAV header
- Exit with code 1 and usage message if no filename argument provided
- Print sample count and output path on success
- CPU-only by design — this experiment demonstrates CPU-path functionality. For CUDA speedup, change to `Inference::cuda(0)` and enable the `cuda` feature

**Definition of Done:**

- [ ] `cargo build -p tts-wav` compiles without errors
- [ ] Running `cargo run -p tts-wav -- /tmp/test_tts.wav` produces a valid WAV file
- [ ] The WAV file is playable with `ffplay /tmp/test_tts.wav` and contains audible speech
- [ ] Running without arguments prints usage and exits with code 1
- [ ] WAV file header: 24000 Hz, 1 channel, 16-bit signed integer
- [ ] Synthesized speech is intelligible and pronounces all words including "equals" and "minus"

**Verify:**

- `cargo build -p tts-wav` — compiles
- `cargo run -p tts-wav -- /tmp/test_tts.wav` — runs successfully, prints sample count
- `ffprobe /tmp/test_tts.wav 2>&1 | grep -E 'Hz|mono|pcm_s16le'` — confirms 24kHz mono 16-bit

## Testing Strategy

- Unit tests: Not applicable — this is a thin integration binary with no business logic
- Integration tests: Not applicable — the TTS pipeline is already tested in `deli-infer`
- Manual verification: Run the binary, verify WAV file is playable and contains speech matching the input sentence

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| Model files missing at runtime | Low | High | Check model paths exist before loading; print clear error message with expected paths |
| espeak-ng not installed | Low | High | Error from `use_kokoro()` will propagate; the error message from espeak_init is already descriptive |

## Open Questions

None — requirements are fully specified.
