import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:rstypes/rstypes.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'config.dart';

class Server {
  WebSocketChannel? _channel;

  Server(Config config) {
    _connect(config);
  }

  Future<void> _connect(Config config) async {
    while (true) {
      try {
        final uri = Uri.parse('ws://${config.address}:${config.port}');
        _channel = WebSocketChannel.connect(uri);
        await _channel!.ready;

        final done = Completer<void>();

        _channel!.stream.listen(
          (event) {
            final data = Data.fromBin(Uint8List.fromList(event as List<int>));
            debugPrint('Data(value: ${data.value}, flag: ${data.flag})');
          },
          onDone: () {
            debugPrint('WebSocket connection closed');
            _channel?.sink.close();
            done.complete();
          },
          onError: (error) {
            debugPrint('WebSocket error: $error');
            _channel?.sink.close();
            done.complete();
          },
          cancelOnError: true,
        );

        await done.future;
      } catch (e) {
        debugPrint('WebSocket connection failed: $e');
      }
      await Future.delayed(const Duration(seconds: 3));
    }
  }
}
