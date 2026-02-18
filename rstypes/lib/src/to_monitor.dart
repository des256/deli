import 'dart:typed_data';

sealed class ToMonitor {
  const ToMonitor();

  factory ToMonitor.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (ToMonitor, int) decode(ByteData bd, Uint8List buf, int offset) {
    final variant = bd.getUint32(offset, Endian.little); offset += 4;
    switch (variant) {
      case 0:
        final f0 = () { final _l0 = bd.getUint32(offset, Endian.little); offset += 4; final _v0 = List<int>.generate(_l0, (i) => bd.getUint8(offset + i)); offset += _l0 * 1; return _v0; }();
        return (ToMonitorVideoJpeg(f0: f0), offset);
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

class ToMonitorVideoJpeg extends ToMonitor {
  final List<int> f0;

  const ToMonitorVideoJpeg({required this.f0});

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 0, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    _d.setUint32(0, f0.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final _e0 in f0) { builder.addByte(_e0); };
  }
}
