# TODO

- add ASR side to chat
- implement VAD-based turn detection
- add Qwen 2.5 3B for better complex prompts, A/B testing with Llama 3.2 3B
- look at latishab/turnsense, tiny model for turn detection
- make LLM and TTS cancelable
- when TTS canceled, figure out what was already said: start with sentence-level, then add linear estimation for slightly more precision
- maybe use ASR to figure out what was already said from a collected audio buffer
- use this partial utterance to augment chat history
- start entirely new query on interruption
- connect up diarization to get embeddings
- figure out a way to turn the ASR speech chunks into sentences

# REFACTOR

- replace std::sync::mpsc with tokio::sync::mpsc
- replace |e| with |error|
