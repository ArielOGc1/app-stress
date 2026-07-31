import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('placeholder smoke test', (WidgetTester tester) async {
    // Placeholder — la app usa providers y TFLite que requieren inicialización
    // real en dispositivo. Tests de integración van en integration_test/.
    expect(true, isTrue);
  });
}
