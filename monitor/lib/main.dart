import 'package:flutter/material.dart';

import 'config.dart';
import 'server.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final config = await Config.load();
  Server(config);
  runApp(MonitorApp(config: config));
}

class MonitorApp extends StatelessWidget {
  final Config config;

  const MonitorApp({super.key, required this.config});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Monitor',
      home: const MonitorHome(),
    );
  }
}

class MonitorHome extends StatelessWidget {
  const MonitorHome({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold();
  }
}
