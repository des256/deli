import 'dart:typed_data';
import 'dart:convert';

sealed class Value {
  const Value();

  factory Value.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Value, int) decode(ByteData bd, Uint8List buf, int offset) {
    final variant = bd.getUint32(offset, Endian.little); offset += 4;
    switch (variant) {
      case 0:
        final f0 = bd.getInt64(offset, Endian.little); offset += 8;
        return (ValueInt(f0: f0), offset);
      case 1:
        final f0 = () { final len = bd.getUint32(offset, Endian.little); offset += 4; final s = utf8.decode(buf.sublist(offset, offset + len)); offset += len; return s; }();
        return (ValueText(f0: f0), offset);
      case 2: return (ValueNothing(), offset);
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

class ValueInt extends Value {
  final int f0;

  const ValueInt({required this.f0});

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 0, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    _d.setInt64(0, f0, Endian.little); builder.add(_d.buffer.asUint8List(0, 8));
  }
}

class ValueText extends Value {
  final String f0;

  const ValueText({required this.f0});

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 1, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    { final encoded = utf8.encode(f0); _d.setUint32(0, encoded.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); builder.add(encoded); };
  }
}

class ValueNothing extends Value {
  const ValueNothing();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 2, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}
