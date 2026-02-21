import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'config.dart';
import 'rstypes/to_monitor.dart';

typedef OnUpdate = void Function();

class Server {
  WebSocketChannel? _channel;
  ToMonitor? _data;
  Uint8List? _jpeg;
  final Set<OnUpdate> _onUpdates = {};
  bool _isConnected = false;

  ToMonitor? get data => _data;
  Uint8List? get jpeg => _jpeg;
  bool get isConnected => _isConnected;

  Server(Config config) {
    _connect(config);
  }

  void onUpdate(OnUpdate callback) {
    _onUpdates.add(callback);
  }

  void removeOnUpdate(OnUpdate callback) {
    _onUpdates.remove(callback);
  }

  void _callOnUpdates() {
    for (final onUpdate in List.of(_onUpdates)) {
      onUpdate();
    }
  }

  Future<void> _connect(Config config) async {
    while (true) {
      try {
        final uri = Uri.parse('ws://${config.address}:${config.port}');
        _channel = WebSocketChannel.connect(uri);
        await _channel!.ready;

        _isConnected = true;
        _callOnUpdates();

        final done = Completer<void>();

        _channel!.stream.listen(
          (event) {
            final message = ToMonitor.fromBin(event as Uint8List);
            if (message is ToMonitorJpeg) {
              _jpeg = message.f0;
            }
            _callOnUpdates();
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

      _isConnected = false;
      _callOnUpdates();

      await Future.delayed(const Duration(seconds: 3));
    }
  }
}
