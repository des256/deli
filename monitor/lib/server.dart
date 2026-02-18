import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:rstypes/rstypes.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'config.dart';

typedef OnDataUpdate = void Function(ToMonitor data);

class Server {
  WebSocketChannel? _channel;
  ToMonitor? _data;
  Language? _language;  // Persists across VideoJpeg messages
  final List<OnDataUpdate> _onUpdates = [];

  ToMonitor? get data => _data;
  Language? get language => _language;

  Server(Config config) {
    _connect(config);
  }

  void onUpdate(OnDataUpdate callback) {
    _onUpdates.add(callback);
  }

  void removeOnUpdate(OnDataUpdate callback) {
    _onUpdates.remove(callback);
  }

  void setLanguage(Language language) {
    final message = FromMonitorSettings(language: language);
    final bytes = message.toBin();
    _channel?.sink.add(bytes);
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
            final message = ToMonitor.fromBin(Uint8List.fromList(event as List<int>));
            // Only update _data for VideoJpeg messages to avoid a single-frame
            // "waiting..." flash on the Audio/Video tab when Settings arrives.
            if (message is ToMonitorSettings) {
              _language = message.language;
            } else {
              _data = message;
            }
            for (final callback in List.of(_onUpdates)) {
              callback(message);
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
