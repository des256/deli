- deli-camera: raspberry pi camera
- deli-codec/deli-codec-derive: add Dart encode/decode in separate file with new trait Dart, write dart file to env DELI_DART_EXPORT_PATH

* deli-infer: add candle as backend, maybe
* deli-infer: whisper API
* deli-infer: piper API
* deli-infer: phi 4 API
* deli-infer: llama 3.2 API
* deli-infer: gemma 3
* deli-camera: in camera trait, replace recv with recv_frame and recv_jpeg. same for any try_recv. recv_frame receives a frame (if internally this is a jpeg, unpack it there). recv_jpeg receives a jpeg.
* deli-camera: add .jpeg_available() to the trait, returns true if the camera can output jpegs.
