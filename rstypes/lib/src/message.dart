import 'dart:typed_data';
import 'dart:convert';

class Message {
  final int id;
  final String text;
  final List<String> tags;

  Message({required this.id, required this.text, required this.tags});

  factory Message.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Message, int) decode(ByteData bd, Uint8List buf, int offset) {
    final id = bd.getUint32(offset, Endian.little); offset += 4;
    final text = () { final len = bd.getUint32(offset, Endian.little); offset += 4; final s = utf8.decode(buf.sublist(offset, offset + len)); offset += len; return s; }();
    final tags = () { final _l0 = bd.getUint32(offset, Endian.little); offset += 4; final _v0 = List<String>.generate(_l0, (_) => () { final len = bd.getUint32(offset, Endian.little); offset += 4; final s = utf8.decode(buf.sublist(offset, offset + len)); offset += len; return s; }()); return _v0; }();
    return (Message(id: id, text: text, tags: tags), offset);
  }

  Uint8List toBin() {
    final builder = BytesBuilder();
    encode(builder);
    return builder.toBytes();
  }

  void encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, id, Endian.little); builder.add(_d.buffer.asUint8List(0, 4));
    { final encoded = utf8.encode(text); _d.setUint32(0, encoded.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); builder.add(encoded); };
    _d.setUint32(0, tags.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final _e0 in tags) { final encoded = utf8.encode(_e0); _d.setUint32(0, encoded.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); builder.add(encoded); };
  }
}
