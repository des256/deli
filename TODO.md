# TODO

- redesign to cancel directly from VAD
- redesign to run TTS as soon as the LLM outputs each sentence, and place the resulting audio in big queue to be released
- Add Llama 3.2 3B, test against Phi
- Add Qwen 2.5 3B, test as well
- figure out a prompt that makes the LLM respond like a person, rather than an "AI assistant"
- flash AEC firmware on respeaker, maybe figure out a way to turn off the LEDs

- test on jetson
- connect up diarization to get embeddings
- emotional voice cloning (Grainne)

# REFACTOR

- replace std::sync::mpsc with tokio::sync::mpsc
- replace |e| with |error|
