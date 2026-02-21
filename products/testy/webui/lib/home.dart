import 'package:flutter/material.dart';

import 'config.dart';
import 'server.dart';

class Home extends StatefulWidget {
  final Config config;
  final Server server;

  const Home({super.key, required this.config, required this.server});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> with SingleTickerProviderStateMixin {
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

  void _onUpdate() {
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return widget.server.isConnected
        ? Scaffold(
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
                widget.server.jpeg != null
                    ? Image.memory(
                        widget.server.jpeg!,
                        gaplessPlayback: true,
                        width: double.infinity,
                        height: double.infinity,
                        fit: BoxFit.contain,
                      )
                    : const Center(
                        child: Text(
                          'waiting...',
                        ),
                      ),
                // Settings tab
                Center(
                  child: Text(
                    'coming soon...',
                  ),
                ),
              ],
            ),
          )
        : const Scaffold(
            backgroundColor: Colors.black,
            body: Center(
              child: Text(
                'Connecting...',
                style: TextStyle(color: Colors.white),
              ),
            ),
          );
  }
}
