import 'dart:typed_data';

class Data {
  final int value;
  final bool flag;
  final List<int> frame;

  Data({required this.value, required this.flag, required this.frame});

  factory Data.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Data, int) decode(ByteData bd, Uint8List buf, int offset) {
    final value = bd.getInt32(offset, Endian.little); offset += 4;
    final flag = buf[offset] != 0; offset += 1;
    final frame = () { final _l0 = bd.getUint32(offset, Endian.little); offset += 4; final _v0 = List<int>.generate(_l0, (i) => bd.getUint8(offset + i)); offset += _l0 * 1; return _v0; }();
    return (Data(value: value, flag: flag, frame: frame), offset);
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
    _d.setUint32(0, frame.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final _e0 in frame) { builder.addByte(_e0); };
  }
}
