import 'dart:typed_data';

sealed class Color {
  const Color();

  factory Color.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Color, int) decode(ByteData bd, Uint8List buf, int offset) {
    final variant = bd.getUint32(offset, Endian.little); offset += 4;
    switch (variant) {
      case 0: return (ColorRed(), offset);
      case 1: return (ColorGreen(), offset);
      case 2: return (ColorBlue(), offset);
      default:
        throw FormatException('Invalid variant: $variant');
    }
  }

  Uint8List toBin() {
    final builder = BytesBuilder();
    encode(builder);
    return builder.toBytes();
  }

  void encode(BytesBuilder builder);
}

class ColorRed extends Color {
  const ColorRed();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 0, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}

class ColorGreen extends Color {
  const ColorGreen();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 1, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}

class ColorBlue extends Color {
  const ColorBlue();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 2, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}
