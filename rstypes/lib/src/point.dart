import 'dart:typed_data';

class Point {
  final double x;
  final double y;

  Point({required this.x, required this.y});

  factory Point.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Point, int) decode(ByteData bd, Uint8List buf, int offset) {
    final x = bd.getFloat32(offset, Endian.little); offset += 4;
    final y = bd.getFloat32(offset, Endian.little); offset += 4;
    return (Point(x: x, y: y), offset);
  }

  Uint8List toBin() {
    final builder = BytesBuilder();
    encode(builder);
    return builder.toBytes();
  }

  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setFloat32(0, x, Endian.little); builder.add(_d.buffer.asUint8List(0, 4));
    _d.setFloat32(0, y, Endian.little); builder.add(_d.buffer.asUint8List(0, 4));
  }
}
