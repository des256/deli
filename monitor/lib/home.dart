import 'dart:typed_data';

import 'package:flutter/material.dart';

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
          data != null
              ? Image.memory(
                  Uint8List.fromList(data.frame),
                  gaplessPlayback: true,
                  width: double.infinity,
                  height: double.infinity,
                  fit: BoxFit.contain,
                )
              : const Center(
                  child: Text(
                    'waiting...',
                    style: TextStyle(color: Colors.red, fontSize: 48),
                  ),
                ),
          // Settings tab
          const Center(
            child: Text(
              'Settings',
              style: TextStyle(color: Colors.white),
            ),
          ),
        ],
      ),
    );
  }
}
