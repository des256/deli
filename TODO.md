# TODO

- get window_size and chunk_shift from model, or window_size = 121, chunk_shift = 112
- CHUNK_SIZE must be multiple of ((num_frames - 1) \* HOP_SIZE + WINDOW_SIZE)
- encoder expects window_size features per call, advancing by chunk_shift; so, once you have window_size frames, call encoder, and remove the first chunk_shift frames; add chunk_shift more, call encoder, etc.
-

- make audio input chunk size variable from
- remove candle and everything depending on it
- connect up diarization

# REFACTOR

- replace std::sync::mpsc with tokio::sync::mpsc
- replace |e| with |error|

# VIBECODING FINDINGS

- very nice to find a quick solution to things
- you gotta treat it like an eager junior, set boundaries, explain exactly what you want
- it is very good at finding bugs
- it does test-driven development (Pilot), which is very useful
- it can make very stupid design mistakes, so still design your own structure
- changing stuff around tends to increase the mess up to a point where your code contains
- does not eliminate dead code
