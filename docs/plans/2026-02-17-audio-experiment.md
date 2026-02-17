# Audio Experiment Implementation Plan

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
> **Worktree:** Set at plan creation (from dispatcher). `Yes` uses git worktree isolation; `No` works directly on current branch (default)

## Summary

**Goal:** Create an `audio` experiment with two binaries — `play` (WAV file playback via `AudioOut`) and `record` (5-second capture via `AudioIn` saved as WAV).

**Architecture:** A single experiment crate at `experiments/audio/` with two `[[bin]]` targets following the existing `camera-viewer` multi-binary pattern. `play` reads a WAV file using the `hound` crate, converts samples to `i16` via `hound`'s built-in `.samples::<i16>()` iterator, and sends chunks to `AudioOut`. `record` captures 5 seconds of audio from `AudioIn` by collecting chunks until the total sample count reaches `sample_rate * 5`, then writes them as a mono 16-bit WAV file using `hound::WavWriter`.

**Tech Stack:** `deli-audio` (AudioIn, AudioOut), `hound` (WAV I/O), `deli-base` (logging), `tokio`

## Scope

### In Scope

- `experiments/audio/` crate with Cargo.toml
- `play` binary: load WAV file, convert to i16, play through AudioOut default device
- `record` binary: record 5 seconds from AudioIn default device, save as WAV file
- CLI argument parsing via `std::env::args()` (no clap — matches existing experiment patterns)

### Out of Scope

- Device selection (both use default device via `None`)
- Stereo/multi-channel support (AudioIn/AudioOut are mono)
- Streaming to/from non-WAV formats
- Tests for the experiment binaries (experiments are manual-run tools, not library code — matching existing experiments which have zero tests)
- Changes to `deli-audio`, `deli-base`, or any other crate

## Prerequisites

- `deli-audio` crate with `AudioIn` and `AudioOut` (already implemented)
- PulseAudio server running for actual playback/recording

## Context for Implementer

- **Patterns to follow:** `experiments/camera-viewer/Cargo.toml` for multi-binary experiment layout. `experiments/camera-viewer/src/camera.rs:8` for `#[tokio::main]` entry point with `deli_base::init_stdout_logger()` and `std::env::args()`.
- **Conventions:** Edition 2024, `deli_base::log` for logging, `Result<(), Box<dyn std::error::Error>>` return type for main, path dependencies to `../../crates/`.
- **Key files:**
  - `crates/deli-audio/src/audio_out.rs` — `AudioOut::new(None, sample_rate)`, `send(&[i16]).await`, `cancel().await`
  - `crates/deli-audio/src/audio_in.rs` — `AudioIn::new(None, sample_rate, chunk_frames)`, `recv().await -> Result<Vec<i16>, AudioError>`
  - `experiments/camera-viewer/Cargo.toml` — reference for multi-binary experiment layout
- **Gotchas:**
  - `AudioOut::new()` and `AudioIn::new()` panic outside a tokio runtime — must use `#[tokio::main]`.
  - `hound::WavReader::samples::<i16>()` handles format conversion automatically (float→int, 8→16 bit, etc.). It returns an iterator of `Result<i16, hound::Error>`.
  - `AudioOut::send()` accepts arbitrary-sized `&[i16]` buffers. The entire file can be sent in one call — the background task writes it to PA in a single `simple.write()` call.
  - `AudioIn::recv()` returns `chunk_frames` samples per call. For 5 seconds at 16kHz with 1600 chunk_frames (100ms chunks), that's 50 recv() calls.
  - WAV files can be mono or stereo. If the input WAV is stereo, the play binary must downmix to mono (average L+R channels) since AudioOut is mono-only.
  - `hound::WavSpec.channels` gives the channel count; `hound::WavSpec.sample_rate` gives the sample rate. The play binary should use the WAV file's sample rate for `AudioOut`.

## Runtime Environment

- **Play command:** `cargo run -p audio --bin play -- <path-to-wav-file>`
- **Record command:** `cargo run -p audio --bin record -- <output-wav-path>`
- **Manual verification:** Play a known WAV file and listen. Record 5 seconds and play the output file back with `play`.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Create experiment crate and `play` binary
- [x] Task 2: Create `record` binary

**Total Tasks:** 2 | **Completed:** 2 | **Remaining:** 0

## Implementation Tasks

### Task 1: Create experiment crate and `play` binary

**Objective:** Create the `experiments/audio/` crate with Cargo.toml and the `play` binary that reads a WAV file and plays it through `AudioOut`.

**Dependencies:** None

**Files:**

- Create: `experiments/audio/Cargo.toml`
- Create: `experiments/audio/src/play.rs`

**Key Decisions / Notes:**

- **Cargo.toml:** Two `[[bin]]` sections — `play` at `src/play.rs` and `record` at `src/record.rs`. Dependencies: `deli-audio` (path), `deli-base` (path), `hound`, `tokio` (rt, rt-multi-thread, macros).
- **CLI:** `play <wav-file>`. Print usage and exit with code 1 if no argument given.
- **WAV reading:** Use `hound::WavReader::open(path)`. Get `WavSpec` for sample_rate and channels. Read all samples via `.samples::<i16>().collect::<Result<Vec<i16>, _>>()`. Log the WAV format: sample rate, channels, bits per sample. If non-16-bit input, log "Converting X-bit to 16-bit".
- **Stereo downmix:** If `channels == 2`, downmix to mono using `samples.chunks_exact(2).map(|pair| ((pair[0] as i32 + pair[1] as i32) / 2) as i16).collect()`. Log "Downmixing stereo to mono". If channels > 2, print error and exit with code 1. If channels == 1, use samples as-is.
- **Playback:** Log "Playing <file> (<sample_rate> Hz, <duration>s)...". Create `AudioOut::new(None, spec.sample_rate)`. Send the entire mono sample buffer in one `audio_out.send(&samples).await` call. No chunking needed — `AudioOut` handles the write internally.
- **End-of-playback wait:** After `send()` completes, the data is queued in the channel. The background task will write it to PA, which buffers and plays it. Calculate the audio duration in milliseconds from sample count and sample rate, then sleep for that duration plus a 500ms margin: `tokio::time::sleep(Duration::from_millis(duration_ms + 500))`. This allows PA to finish playing all buffered audio before dropping `AudioOut`. Log "Playback complete".
- **Error handling:** Use `?` operator throughout. Print meaningful error messages for file-not-found, invalid WAV format, etc.

**Definition of Done:**

- [ ] No diagnostics errors (linting, type checking)
- [ ] `cargo build -p audio --bin play` compiles without errors
- [ ] Running `play` with no arguments prints usage and exits with code 1
- [ ] Running `play nonexistent.wav` prints an error about missing file
- [ ] Running `play <valid.wav>` exits with code 0 and logs "Playback complete" (audible output verified by human tester)
- [ ] Stereo WAV files are downmixed to mono before playback (log message printed noting downmix)

**Verify:**

- `cargo check -p audio` — no compiler errors or warnings
- `cargo run -p audio --bin play 2>&1` — prints usage (no file argument)
- `cargo run -p audio --bin play -- test.wav` — plays audio (manual)

### Task 2: Create `record` binary

**Objective:** Create the `record` binary that captures 5 seconds of audio from `AudioIn` and saves it as a mono 16-bit WAV file.

**Dependencies:** Task 1 (Cargo.toml must exist)

**Files:**

- Create: `experiments/audio/src/record.rs`

**Key Decisions / Notes:**

- **CLI:** `record <output-wav-path>`. Print usage and exit with code 1 if no argument given.
- **Recording parameters:** Sample rate 16000 Hz, chunk_frames 1600 (100ms chunks). 5 seconds = 80000 samples = 50 chunks. These are reasonable defaults for speech-quality recording.
- **Capture loop:** Create `AudioIn::new(None, 16000, 1600)`. Loop calling `recv().await`, appending each chunk's samples to a `Vec<i16>`. Stop when total samples >= 80000 (16000 * 5). Truncate to exactly 80000 samples. Note: The recv loop (recv → Vec::extend) is microsecond-level with no disk I/O, so AudioIn's channel (capacity 4 = 400ms buffer) will not overflow. Chunks are only dropped if the consumer falls behind by 400ms, which won't happen in a simple extend loop.
- **WAV writing:** After the capture loop completes, use `hound::WavWriter::create(path, spec)` with `WavSpec { channels: 1, sample_rate: 16000, bits_per_sample: 16, sample_format: SampleFormat::Int }`. Write each sample via `writer.write_sample(sample)`. Call `writer.finalize()?` to ensure the WAV header is written correctly.
- **Progress output:** Print "Recording 5 seconds..." before starting capture. Print "Recorded {samples} samples" and "Saved to {path}" after writing.

**Definition of Done:**

- [ ] No diagnostics errors (linting, type checking)
- [ ] `cargo build -p audio --bin record` compiles without errors
- [ ] Running `record` with no arguments prints usage and exits with code 1
- [ ] Running `record output.wav` exits with code 0 after ~5 seconds, logs "Saved to output.wav"
- [ ] Output WAV file is mono, 16-bit, 16kHz, exactly 5 seconds (80000 samples) — verify with `ffprobe -v error -show_entries stream=codec_name,sample_rate,channels,duration output.wav`
- [ ] Output WAV file can be played back by the `play` binary

**Verify:**

- `cargo check -p audio` — no compiler errors or warnings
- `cargo run -p audio --bin record 2>&1` — prints usage (no file argument)
- `cargo run -p audio --bin record -- test_recording.wav` — records 5 seconds (manual)
- `cargo run -p audio --bin play -- test_recording.wav` — plays recorded audio back (manual)

## Testing Strategy

- **Unit tests:** None — experiment binaries are manual-run tools, matching the existing experiment pattern (camera-view, camera-pose, camera-viewer all have zero tests).
- **Build verification:** `cargo check -p audio` and `cargo build -p audio` must succeed with zero errors and zero warnings.
- **Manual verification:** Record a WAV, play it back, verify audio quality. Play an existing WAV file and verify output.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| WAV file has unsupported format (e.g., compressed) | Low | Low | `hound` returns a clear error for unsupported formats. The `play` binary propagates this error to the user via `?`. |
| Stereo WAV played as mono sounds wrong | Medium | Low | Downmix stereo to mono by averaging L+R channels. Print a log message noting the downmix. |
| AudioOut drops before audio finishes playing | Medium | Medium | After `send()`, sleep for the audio duration + 500ms margin before dropping `AudioOut`. This ensures PA has time to play all buffered audio. Log "Playback complete" before exit. |
| No PulseAudio server running | Medium | High | Both binaries print a status message (e.g., "Playing <file>..." or "Recording 5 seconds...") immediately after parsing CLI args and before calling `AudioIn`/`AudioOut::new()`. This confirms the program is running before any PA connection attempt. If PA is unavailable, the error recovery loops will retry — the user can Ctrl+C. |

## Open Questions

- None — requirements are clear.

### Deferred Ideas

- Device selection via CLI flags (--device)
- Configurable sample rate and duration for recording
- Real-time waveform visualization during playback/recording
- Support for other audio formats (FLAC, OGG, MP3)
- `drain()` method on AudioOut for clean end-of-playback
