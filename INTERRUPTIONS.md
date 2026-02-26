The Core Problem

When TTS is playing and the user speaks, you need to:

1. Detect the user is speaking (not just noise or echo from the speaker)
2. Cancel the in-flight pipeline (stop TTS, abort LLM generation)
3. Decide what to do with the new speech (append to previous query vs. new query)
4. Synchronize context (the LLM needs to know what the user actually heard before interruption)

Technique 1: VAD + Echo Cancellation (The Foundation)

The hardest sub-problem: distinguishing the user's voice from the TTS audio being picked up by the mic.

- Acoustic Echo Cancellation (AEC) — Subtracts the known TTS output signal from the mic input. Classical approach uses adaptive filters
  (NLMS). Modern approaches use hybrid DSP + neural networks. Without AEC, the system constantly interrupts itself.
- Voice Activity Detection (VAD) — Silero VAD is the go-to open source option. Runs on small audio frames (10-20ms), outputs speech
  probability. Key tuning: raise the threshold to 0.5-0.6 in production (default is too aggressive), require minimum segment length (e.g.,
  300ms) to avoid false triggers from keyboard clicks or breathing.

On a Jetson/local setup, the speaker-mic proximity problem is real — the speaker is physically closer to the mic than the user, so echo
ratio is terrible. Options: use a headset/earbuds (eliminates echo entirely), use a mic array with beamforming, or run software AEC
(SpeexDSP, WebRTC AEC, or noisereduce in Python).

Technique 2: Cancellable Pipeline Architecture

The pipeline must be designed so every stage can be cancelled instantly:

User speaks → VAD triggers → Cancel signal propagates: 1. Stop TTS audio playback (flush audio buffer) 2. Cancel LLM generation (close the stream) 3. Clear any queued TTS chunks not yet played 4. Resume ASR collection

Pipecat's approach: A dual-queue system where InterruptionFrame (system-level) bypasses the normal processing queue. It immediately cancels
pending tasks, clears buffered frames, and reconnects TTS websockets. This is the most battle-tested open source implementation.

DIY approach: Use asyncio.Task.cancel() or threading events. Each pipeline stage checks a shared cancellation token. The key insight is
that cancellation must be push-based (propagated immediately), not poll-based.

Technique 3: Context Synchronization

After interruption, the LLM generated text the user never heard. You need to track what was actually spoken:

- TTS word-level timestamps — Many TTS engines (Kokoro, ElevenLabs, Azure) provide per-word timing. Track which words were played before
  interruption. Only those words go into conversation history.
- Chunk-level tracking — If word timestamps aren't available, track which audio chunks were actually sent to the speaker. Cheaper but less
  precise.

This prevents the LLM from thinking it said something the user never heard.

Technique 4: Turn-Taking State Machine

Rather than ad-hoc event handling, model the conversation as states:

LISTENING → (VAD detects speech end) → PROCESSING → SPEAKING →
↑ |
└──── (barge-in detected) ──────────────────────────┘

States: LISTENING, PROCESSING (ASR→LLM), SPEAKING (TTS playing), INTERRUPTED. Key rule: when in SPEAKING state and barge-in fires,
transition immediately — don't process more VAD events until the state machine settles.

Technique 5: Endpointing / "Did They Really Interrupt?"

Not every sound during TTS playback is a real interruption. Strategies:

- Confidence gating — Require higher VAD confidence for short utterances (>0.9 for <0.5s) to filter "uh-huh" backchannels
- ASR confirmation — Wait for ASR to produce actual text before committing to interruption. ~200ms delay but dramatically reduces false
  positives
- Semantic classification — Classify the interruption: backchannel ("mm-hm", "yeah") vs. actual new input. Backchannels don't cancel TTS.

Open Source Projects Worth Studying

Project: Pipecat
What It Does: Full voice pipeline framework
Interruption Approach: Dual-queue priority frames, allow_interruptions, automatic context sync
────────────────────────────────────────
Project: LiveKit Agents
What It Does: Real-time voice agent framework
Interruption Approach: Built-in turn detection model, STT-LLM-TTS orchestration with interrupt handling
────────────────────────────────────────
Project: TEN Framework
What It Does: Multimodal conversational AI
Interruption Approach: VAD + turn detection extensions
────────────────────────────────────────
Project: NVIDIA PersonaPlex
What It Does: Full-duplex voice model
Interruption Approach: Sidesteps the problem — single model handles listening+speaking simultaneously (MIT license)

PersonaPlex is interesting because it's a speech-to-speech model that handles both directions simultaneously, eliminating the cascaded
pipeline interruption problem entirely. Still early but the architecture is fundamentally different.

For Your Setup (Local/Jetson)

Given your Parakeet ASR + local LLM + TTS setup, the practical path is probably:

1. Silero VAD running continuously on mic input (tiny model, runs fine on Jetson)
2. Software AEC if using speakers (WebRTC AEC3 or SpeexDSP) — or just use earbuds to sidestep it
3. asyncio cancellation — wrap LLM streaming and TTS in cancellable tasks, propagate a cancel event when VAD fires during SPEAKING state
4. Simple state machine — LISTENING/PROCESSING/SPEAKING/INTERRUPTED with clear transitions
5. Start simple — cancel everything on barge-in, collect new ASR text, submit as a fresh query. Context sync (tracking what was heard) can
   come later.

The "append vs. new query" decision is actually the easiest part — start with "always new query" and add append logic later if
conversational flow demands it.

Sources:

- Zoice – Interruption Handling in Conversational AI
- Sparkco – Optimizing Voice Agent Barge-in Detection
- Pipecat – Pipeline & Frame Processing
- Pipecat GitHub
- LiveKit Agents GitHub
- TEN Framework GitHub
- NVIDIA PersonaPlex
- Softcery – Real-Time vs Turn-Based Architecture
- VOCAL Technologies – AEC Barge-In
- AssemblyAI – Voice AI Stack 2026
- DEV Community – VAD and Turn-Taking

====

The Problem

Pocket TTS takes the full text, conditions on all of it at once (condition()), then generates audio frames sequentially until EOS. There's
no built-in mapping from "frame N" back to "word M in the input text."

Approaches, from Most to Least Practical

1. Sentence/Clause Chunking (Best for Your Setup)

Instead of sending the full LLM response as one TTS call, split it at sentence boundaries and generate each sentence separately. Track
which sentences finished playback.

LLM streams: "The weather is nice today. You should go for a walk. Maybe bring a jacket."
^
TTS chunk 1: "The weather is nice today." ← fully played
TTS chunk 2: "You should go for a walk." ← interrupted mid-playback
TTS chunk 3: "Maybe bring a jacket." ← never started

→ User heard: sentence 1 fully, sentence 2 partially

On interruption, you know with sentence-level precision what was heard. This is what Pipecat and LiveKit do internally. The trade-off is
that cross-sentence prosody may suffer slightly (the voice resets between chunks), but Pocket TTS re-conditions on each send() anyway so
this maps naturally to your current architecture.

Your Pocket struct already works this way — each send() triggers a fresh condition() + generation loop. You just need to split upstream.

2. Frame-to-Time Estimation

Each Mimi decoder frame produces a fixed-duration audio chunk. The Mimi codec runs at ~12.5 Hz frame rate (each frame ≈ 80ms of audio at
24kHz). So:

frames_played \* 80ms = approximate time spoken

If you track how many Vec<i16> chunks were actually sent to the audio output before interruption, you get elapsed time. Then estimate word
position proportionally:

fraction_spoken = frames_played / total_frames_generated
word_index ≈ fraction_spoken \* total_words

This is imprecise — speech rate isn't uniform (pauses between words, some words take longer) — but it's zero-cost and gives a rough
estimate. Good enough if you're just trying to tell the LLM "I got about halfway through the response."

3. ASR on the Generated Audio

You already have Parakeet running. Feed the audio chunks that were actually played back through ASR to get exact text. This gives you a
precise transcript of what was heard, not an estimate.

The downside is latency — you need to run ASR on the played-back audio at interruption time. But if you're accumulating the played chunks
in a buffer anyway, it's a single Parakeet inference call. On Jetson this might add 100-200ms to the interruption response, which is
acceptable since the user is already talking.

4. Hybrid: Chunk by Sentence + Frame Count Within

Combine approaches 1 and 2:

- Split text into sentences, send each to TTS separately
- For the interrupted sentence, use frame count to estimate how far through it you got
- Report to the LLM: "User heard: 'The weather is nice today.' and approximately the first half of 'You should go for a walk.'"

What I'd Recommend for Your Pipeline

Sentence chunking (approach 1) is the clear winner:

- Zero added latency — you're already splitting LLM streaming output somewhere
- Natural fit — your Pocket::send() already handles one text segment at a time
- Precise enough — for the LLM context, knowing which sentences were heard is sufficient. The LLM doesn't need to know the user heard "You
  should go for a wa—" vs. "You should go for a walk." — it just needs to know the user heard roughly that much.
- Simple cancellation — on barge-in, stop calling recv(), drop remaining sentences from the queue, done

The main implementation change would be in whatever sits between the LLM output stream and Pocket::send() — accumulate LLM tokens until you
hit sentence-ending punctuation (.!?), then send that sentence to TTS while continuing to buffer the next sentence. Track a Vec<String> of
sentences and an index of "last fully played sentence."
