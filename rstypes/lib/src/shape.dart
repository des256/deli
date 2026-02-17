import 'dart:typed_data';

sealed class Shape {
  const Shape();

  factory Shape.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Shape, int) decode(ByteData bd, Uint8List buf, int offset) {
    final variant = bd.getUint32(offset, Endian.little); offset += 4;
    switch (variant) {
      case 0:
        final radius = bd.getFloat32(offset, Endian.little); offset += 4;
        return (ShapeCircle(radius: radius), offset);
      case 1:
        final width = bd.getFloat32(offset, Endian.little); offset += 4;
        final height = bd.getFloat32(offset, Endian.little); offset += 4;
        return (ShapeRectangle(width: width, height: height), offset);
      case 2: return (ShapePoint(), offset);
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

class ShapeCircle extends Shape {
  final double radius;

  const ShapeCircle({required this.radius});

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 0, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    _d.setFloat32(0, radius, Endian.little); builder.add(_d.buffer.asUint8List(0, 4));
  }
}

class ShapeRectangle extends Shape {
  final double width;
  final double height;

  const ShapeRectangle({required this.width, required this.height});

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 1, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    _d.setFloat32(0, width, Endian.little); builder.add(_d.buffer.asUint8List(0, 4));
    _d.setFloat32(0, height, Endian.little); builder.add(_d.buffer.asUint8List(0, 4));
  }
}

class ShapePoint extends Shape {
  const ShapePoint();

  @override
  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 2, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}
