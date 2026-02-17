import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:rstypes/rstypes.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'config.dart';

typedef OnDataUpdate = void Function(Data data);

class Server {
  WebSocketChannel? _channel;
  Data? _data;
  final List<OnDataUpdate> _onUpdates = [];

  Data? get data => _data;

  Server(Config config) {
    _connect(config);
  }

  void onUpdate(OnDataUpdate callback) {
    _onUpdates.add(callback);
  }

  void removeOnUpdate(OnDataUpdate callback) {
    _onUpdates.remove(callback);
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
            _data = Data.fromBin(Uint8List.fromList(event as List<int>));
            for (final callback in List.of(_onUpdates)) {
              callback(_data!);
            }
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
