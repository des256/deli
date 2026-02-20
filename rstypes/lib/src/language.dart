import 'dart:typed_data';

sealed class Language {
  const Language();

  factory Language.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Language, int) decode(ByteData bd, Uint8List buf, int offset) {
    final variant = bd.getUint32(offset, Endian.little); offset += 4;
    switch (variant) {
      case 0: return (LanguageEnglishUs(), offset);
      case 1: return (LanguageChineseChina(), offset);
      case 2: return (LanguageKoreanKorea(), offset);
      case 3: return (LanguageDutchNetherlands(), offset);
      case 4: return (LanguageFrenchFrance(), offset);
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

class LanguageEnglishUs extends Language {
  const LanguageEnglishUs();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 0, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}

class LanguageChineseChina extends Language {
  const LanguageChineseChina();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 1, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}

class LanguageKoreanKorea extends Language {
  const LanguageKoreanKorea();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 2, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}

class LanguageDutchNetherlands extends Language {
  const LanguageDutchNetherlands();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 3, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}

class LanguageFrenchFrance extends Language {
  const LanguageFrenchFrance();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 4, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}
