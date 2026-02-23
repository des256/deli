- Parakeet STT: https://github.com/istupakov/onnx-asr

* use ONNX LLMs, get rid of Candle dependency (if Whisper is not needed anymore):

  SmolLm2-1.7b
  Gemma3-1b
  Llama3.2-1b

* organize STT, TTS and LLM with unified interfaces
* cut up the LLM output into sentences and pipe them to TTS as soon as they become available
* switch firmware on the respeaker
* kittenml TTS
