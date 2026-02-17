import 'dart:typed_data';
import 'point.dart';

class Line {
  final Point start;
  final Point end;

  Line({required this.start, required this.end});

  factory Line.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Line, int) decode(ByteData bd, Uint8List buf, int offset) {
    final start = () { final r = Point.decode(bd, buf, offset); offset = r.$2; return r.$1; }();
    final end = () { final r = Point.decode(bd, buf, offset); offset = r.$2; return r.$1; }();
    return (Line(start: start, end: end), offset);
  }

  Uint8List toBin() {
    final builder = BytesBuilder();
    encode(builder);
    return builder.toBytes();
  }

  void encode(BytesBuilder builder) {
    start.encode(builder);
    end.encode(builder);
  }
}
