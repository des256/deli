import 'package:flutter/services.dart';
import 'package:yaml/yaml.dart';

class Config {
  final String address;
  final int port;

  Config._({required this.address, required this.port});

  factory Config.fromYaml(String raw) {
    final doc = loadYaml(raw) as YamlMap;
    return Config._(
      address: doc['address'] as String,
      port: doc['port'] as int,
    );
  }

  static Future<Config> load() async {
    final raw = await rootBundle.loadString('assets/config.yaml');
    return Config.fromYaml(raw);
  }
}
