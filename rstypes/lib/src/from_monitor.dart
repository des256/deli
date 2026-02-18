import 'dart:typed_data';
import 'language.dart';

sealed class FromMonitor {
  const FromMonitor();

  factory FromMonitor.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (FromMonitor, int) decode(ByteData bd, Uint8List buf, int offset) {
    final variant = bd.getUint32(offset, Endian.little); offset += 4;
    switch (variant) {
      case 0:
        final language = () { final r = Language.decode(bd, buf, offset); offset = r.$2; return r.$1; }();
        return (FromMonitorSettings(language: language), offset);
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

class FromMonitorSettings extends FromMonitor {
  final Language language;

  const FromMonitorSettings({required this.language});

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 0, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    language.encode(builder);
  }
}
