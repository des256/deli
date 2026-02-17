import 'package:flutter/material.dart';

import 'config.dart';
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

class MonitorHome extends StatefulWidget {
  final Config config;
  final Server server;

  const MonitorHome({super.key, required this.config, required this.server});

  @override
  State<MonitorHome> createState() => _MonitorHomeState();
}

class _MonitorHomeState extends State<MonitorHome> {
  @override
  void initState() {
    super.initState();
    widget.server.onUpdate(_onUpdate);
  }

  @override
  void dispose() {
    widget.server.removeOnUpdate(_onUpdate);
    super.dispose();
  }

  void _onUpdate(_) => setState(() {});

  @override
  Widget build(BuildContext context) {
    final data = widget.server.data;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Text(
          data != null ? 'value: ${data.value}, flag: ${data.flag}' : 'waiting...',
          style: const TextStyle(color: Colors.red, fontSize: 48),
        ),
      ),
    );
  }
}
