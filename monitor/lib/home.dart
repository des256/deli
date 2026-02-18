import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'package:rstypes/rstypes.dart';

import 'config.dart';
import 'server.dart';

class MonitorHome extends StatefulWidget {
  final Config config;
  final Server server;

  const MonitorHome({super.key, required this.config, required this.server});

  @override
  State<MonitorHome> createState() => _MonitorHomeState();
}

class _MonitorHomeState extends State<MonitorHome>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    widget.server.onUpdate(_onUpdate);
  }

  @override
  void dispose() {
    _tabController.dispose();
    widget.server.removeOnUpdate(_onUpdate);
    super.dispose();
  }

  void _onUpdate(_) => setState(() {});

  @override
  Widget build(BuildContext context) {
    final data = widget.server.data;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        elevation: 0,
        bottom: TabBar(
          controller: _tabController,
          labelColor: Colors.white,
          unselectedLabelColor: Colors.grey,
          indicatorColor: Colors.white,
          tabs: const [
            Tab(text: 'Overview'),
            Tab(text: 'Audio/Video'),
            Tab(text: 'Settings'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          // Overview tab
          const Center(
            child: Text(
              'Overview',
              style: TextStyle(color: Colors.white),
            ),
          ),
          // Audio/Video tab
          switch (data) {
            ToMonitorVideoJpeg(:final f0) => Image.memory(
                Uint8List.fromList(f0),
                gaplessPlayback: true,
                width: double.infinity,
                height: double.infinity,
                fit: BoxFit.contain,
              ),
            _ => const Center(
                child: Text(
                  'waiting...',
                  style: TextStyle(color: Colors.red, fontSize: 48),
                ),
              ),
          },
          // Settings tab
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text(
                  'Language',
                  style: TextStyle(color: Colors.white, fontSize: 18),
                ),
                const SizedBox(height: 16),
                DropdownButton<Language>(
                  value: widget.server.language,
                  hint: const Text(
                    'Waiting for server...',
                    style: TextStyle(color: Colors.white70),
                  ),
                  dropdownColor: Colors.grey[900],
                  style: const TextStyle(color: Colors.white),
                  items: const [
                    DropdownMenuItem(
                      value: LanguageEnglishUs(),
                      child: Text('English (US)'),
                    ),
                    DropdownMenuItem(
                      value: LanguageChineseChina(),
                      child: Text('Chinese (China)'),
                    ),
                    DropdownMenuItem(
                      value: LanguageKoreanKorea(),
                      child: Text('Korean (Korea)'),
                    ),
                    DropdownMenuItem(
                      value: LanguageDutchNetherlands(),
                      child: Text('Dutch (Netherlands)'),
                    ),
                    DropdownMenuItem(
                      value: LanguageFrenchFrance(),
                      child: Text('French (France)'),
                    ),
                  ],
                  onChanged: (Language? selectedLanguage) {
                    if (selectedLanguage != null) {
                      widget.server.setLanguage(selectedLanguage);
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
