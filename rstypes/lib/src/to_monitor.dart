import 'dart:typed_data';
import 'language.dart';

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
        final f0 = () { final _l0 = bd.getUint32(offset, Endian.little); offset += 4; final _v0 = buf.sublist(offset, offset + _l0); offset += _l0; return _v0; }();
        return (ToMonitorVideoJpeg(f0: f0), offset);
      case 1:
        final language = () { final r = Language.decode(bd, buf, offset); offset = r.$2; return r.$1; }();
        return (ToMonitorSettings(language: language), offset);
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
  final Uint8List f0;

  const ToMonitorVideoJpeg({required this.f0});

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 0, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    _d.setUint32(0, f0.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); builder.add(f0);
  }
}

class ToMonitorSettings extends ToMonitor {
  final Language language;

  const ToMonitorSettings({required this.language});

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 1, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    language.encode(builder);
  }
}
