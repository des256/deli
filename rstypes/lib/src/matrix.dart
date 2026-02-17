import 'dart:typed_data';

class Matrix {
  final List<List<double>> data;

  Matrix({required this.data});

  factory Matrix.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Matrix, int) decode(ByteData bd, Uint8List buf, int offset) {
    final data = () { final _l0 = bd.getUint32(offset, Endian.little); offset += 4; final _v0 = List<List<double>>.generate(_l0, (_) => () { final _l1 = bd.getUint32(offset, Endian.little); offset += 4; final _v1 = List<double>.generate(_l1, (i) => bd.getFloat32(offset + i * 4, Endian.little)); offset += _l1 * 4; return _v1; }()); return _v0; }();
    return (Matrix(data: data), offset);
  }

  Uint8List toBin() {
    final builder = BytesBuilder();
    encode(builder);
    return builder.toBytes();
  }

  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, data.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final _e0 in data) { _d.setUint32(0, _e0.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final _e1 in _e0) { _d.setFloat32(0, _e1, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); }; };
  }
}
