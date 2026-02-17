import 'dart:typed_data';

class Data {
  final int value;
  final bool flag;

  Data({required this.value, required this.flag});

  factory Data.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Data, int) decode(ByteData bd, Uint8List buf, int offset) {
    final value = bd.getInt32(offset, Endian.little); offset += 4;
    final flag = buf[offset] != 0; offset += 1;
    return (Data(value: value, flag: flag), offset);
  }

  Uint8List toBin() {
    final builder = BytesBuilder();
    encode(builder);
    return builder.toBytes();
  }

  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setInt32(0, value, Endian.little); builder.add(_d.buffer.asUint8List(0, 4));
    builder.addByte(flag ? 1 : 0);
  }
}
