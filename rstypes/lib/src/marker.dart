import 'dart:typed_data';

class Marker {
  const Marker();

  factory Marker.fromBin(Uint8List bytes) {
    return const Marker();
  }

  static (Marker, int) decode(ByteData bd, Uint8List buf, int offset) {
    return (Marker(), offset);
  }

  Uint8List toBin() {
    return Uint8List(0);
  }

  void encode(BytesBuilder builder) {
    // Empty
  }
}
