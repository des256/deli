import 'package:flutter/material.dart';

import 'config.dart';
import 'home.dart';
import 'server.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final config = await Config.load();
  final server = Server(config);
  runApp(Application(config: config, server: server));
}

class Application extends StatelessWidget {
  final Config config;
  final Server server;

  const Application({super.key, required this.config, required this.server});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WebUI',
      home: Home(config: config, server: server),
    );
  }
}
