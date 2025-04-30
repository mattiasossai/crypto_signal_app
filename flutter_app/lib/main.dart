import 'dart:convert';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:http/http.dart' as http;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await NotificationService().init();
  runApp(MyApp());
}

class NotificationService {
  final _plugin = FlutterLocalNotificationsPlugin();
  Future init() async {
    await _plugin.initialize(
      InitializationSettings(
        android: AndroidInitializationSettings('@mipmap/ic_launcher'),
        iOS: IOSInitializationSettings(),
      ),
    );
  }
  void show(Map rec) {
    _plugin.show(
      rec.hashCode,
      '${rec['symbol']} ${rec['signal']}',
      'Entry: ${rec['entry'].toStringAsFixed(2)}, TP: ${rec['tp'].toStringAsFixed(2)}, SL: ${rec['sl'].toStringAsFixed(2)}',
      NotificationDetails(
        android: AndroidNotificationDetails('crypto_chan','Crypto Signals',''),
      ),
    );
  }
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  List signals = [];
  Timer? timer;

  @override
  void initState() {
    super.initState();
    timer = Timer.periodic(Duration(seconds: 30), (_) => fetchSignals());
  }

  Future fetchSignals() async {
    final resp = await http.get(Uri.parse('http://YOUR_SERVER:8000/signals'));
    if (resp.statusCode == 200) {
      final list = json.decode(resp.body);
      setState(() => signals = list.reversed.toList());
      for (var rec in list) {
        NotificationService().show(rec);
      }
    }
  }

  @override
  void dispose() {
    timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('Crypto Entry-Signale')),
        body: ListView.builder(
          itemCount: signals.length,
          itemBuilder: (ctx, i) {
            final rec = signals[i];
            return ListTile(
              title: Text('${rec['symbol']} → ${rec['signal']} '
                  '(${(rec['probabilities'][2]*100).toStringAsFixed(1)}%)'),
              subtitle: Text(
                'Entry: ${rec['entry']}\n'
                'TP: ${rec['tp'].toStringAsFixed(2)}, SL: ${rec['sl'].toStringAsFixed(2)}'
              ),
            );
          },
        ),
      ),
    );
  }
}
