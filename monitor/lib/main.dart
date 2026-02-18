import 'package:flutter/material.dart';

import 'config.dart';
import 'home.dart';
import 'server.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final config = await Config.load();
  final server = Server(config);
  runApp(MonitorApp(config: config, server: server));
}

class MonitorApp extends StatelessWidget {
  final Config config;
  final Server server;

  const MonitorApp({super.key, required this.config, required this.server});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Monitor',
      home: MonitorHome(config: config, server: server),
    );
  }
}

