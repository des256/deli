use deli_codec::{Codec, Dart};

// Test types - these will generate .dart files at compile time if DELI_RSTYPES_PATH is set

#[derive(Codec, Dart)]
struct Point {
    x: f32,
    y: f32,
}

#[derive(Codec, Dart)]
struct Message {
    id: u32,
    text: String,
    tags: Vec<String>,
}

#[derive(Codec, Dart)]
struct Matrix {
    data: Vec<Vec<f32>>,
}

#[derive(Codec, Dart)]
enum Color {
    Red,
    Green,
    Blue,
}

#[derive(Codec, Dart)]
enum Shape {
    Circle { radius: f32 },
    Rectangle { width: f32, height: f32 },
    Point,
}

#[derive(Codec, Dart)]
struct Marker;

#[derive(Codec, Dart)]
enum Value {
    Int(i64),
    Text(String),
    Nothing,
}

#[derive(Codec, Dart)]
struct Line {
    start: Point,
    end: Point,
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    fn get_dart_output_path() -> Option<PathBuf> {
        std::env::var("DELI_RSTYPES_PATH")
            .ok()
            .map(PathBuf::from)
    }

    #[test]
    fn test_point_dart_generated() {
        let Some(base_path) = get_dart_output_path() else {
            eprintln!("DELI_RSTYPES_PATH not set - skipping Dart codegen test");
            return;
        };

        let dart_file = base_path.join("point.dart");
        assert!(
            dart_file.exists(),
            "Expected point.dart to exist at {}",
            dart_file.display()
        );

        let content = fs::read_to_string(&dart_file)
            .expect("Failed to read point.dart");

        // Check for expected patterns
        assert!(content.contains("class Point"), "Should contain class Point");
        assert!(
            content.contains("final double x;"),
            "Should contain field x"
        );
        assert!(
            content.contains("final double y;"),
            "Should contain field y"
        );
        assert!(
            content.contains("factory Point.fromBin"),
            "Should contain fromBin factory"
        );
        assert!(
            content.contains("Uint8List toBin()"),
            "Should contain toBin method"
        );
        assert!(
            content.contains("static (Point, int) decode"),
            "Should contain decode method"
        );
    }

    #[test]
    fn test_message_dart_generated() {
        let Some(base_path) = get_dart_output_path() else {
            eprintln!("DELI_RSTYPES_PATH not set - skipping Dart codegen test");
            return;
        };

        let dart_file = base_path.join("message.dart");
        assert!(
            dart_file.exists(),
            "Expected message.dart to exist at {}",
            dart_file.display()
        );

        let content = fs::read_to_string(&dart_file)
            .expect("Failed to read message.dart");

        assert!(
            content.contains("class Message"),
            "Should contain class Message"
        );
        assert!(
            content.contains("final String text"),
            "Should contain text field"
        );
        assert!(
            content.contains("List<String> tags"),
            "Should contain tags field as List<String>"
        );
    }

    #[test]
    fn test_matrix_dart_generated() {
        let Some(base_path) = get_dart_output_path() else {
            eprintln!("DELI_RSTYPES_PATH not set - skipping Dart codegen test");
            return;
        };

        let dart_file = base_path.join("matrix.dart");
        assert!(
            dart_file.exists(),
            "Expected matrix.dart to exist at {}",
            dart_file.display()
        );

        let content = fs::read_to_string(&dart_file)
            .expect("Failed to read matrix.dart");

        assert!(
            content.contains("class Matrix"),
            "Should contain class Matrix"
        );
        assert!(
            content.contains("List<List<double>> data"),
            "Should contain nested Vec as List<List<double>>"
        );
    }

    #[test]
    fn test_color_enum_dart_generated() {
        let Some(base_path) = get_dart_output_path() else {
            eprintln!("DELI_RSTYPES_PATH not set - skipping Dart codegen test");
            return;
        };

        let dart_file = base_path.join("color.dart");
        assert!(
            dart_file.exists(),
            "Expected color.dart to exist at {}",
            dart_file.display()
        );

        let content = fs::read_to_string(&dart_file)
            .expect("Failed to read color.dart");

        assert!(
            content.contains("sealed class Color"),
            "Should contain sealed class Color"
        );
        assert!(
            content.contains("class ColorRed extends Color"),
            "Should contain ColorRed subclass"
        );
        assert!(
            content.contains("class ColorGreen extends Color"),
            "Should contain ColorGreen subclass"
        );
        assert!(
            content.contains("class ColorBlue extends Color"),
            "Should contain ColorBlue subclass"
        );
    }

    #[test]
    fn test_shape_enum_dart_generated() {
        let Some(base_path) = get_dart_output_path() else {
            eprintln!("DELI_RSTYPES_PATH not set - skipping Dart codegen test");
            return;
        };

        let dart_file = base_path.join("shape.dart");
        assert!(
            dart_file.exists(),
            "Expected shape.dart to exist at {}",
            dart_file.display()
        );

        let content = fs::read_to_string(&dart_file)
            .expect("Failed to read shape.dart");

        assert!(
            content.contains("sealed class Shape"),
            "Should contain sealed class Shape"
        );
        assert!(
            content.contains("class ShapeCircle extends Shape"),
            "Should contain ShapeCircle subclass"
        );
        assert!(
            content.contains("final double radius"),
            "ShapeCircle should have radius field"
        );
        assert!(
            content.contains("class ShapeRectangle extends Shape"),
            "Should contain ShapeRectangle subclass"
        );
        assert!(
            content.contains("final double width"),
            "ShapeRectangle should have width field"
        );
        assert!(
            content.contains("final double height"),
            "ShapeRectangle should have height field"
        );
        assert!(
            content.contains("class ShapePoint extends Shape"),
            "Should contain ShapePoint subclass"
        );
    }

    #[test]
    fn test_marker_unit_struct_dart_generated() {
        let Some(base_path) = get_dart_output_path() else {
            eprintln!("DELI_RSTYPES_PATH not set - skipping Dart codegen test");
            return;
        };

        let dart_file = base_path.join("marker.dart");
        assert!(
            dart_file.exists(),
            "Expected marker.dart to exist at {}",
            dart_file.display()
        );

        let content = fs::read_to_string(&dart_file).expect("Failed to read marker.dart");

        assert!(content.contains("class Marker"), "Should contain class Marker");
        assert!(
            content.contains("factory Marker.fromBin"),
            "Should contain fromBin factory"
        );
        assert!(
            content.contains("Uint8List toBin()"),
            "Should contain toBin method"
        );
        // Verify fromBin returns an instance, not the bare class name
        assert!(
            content.contains("Marker()"),
            "fromBin should return Marker() instance, not bare Marker"
        );
    }

    #[test]
    fn test_value_tuple_enum_dart_generated() {
        let Some(base_path) = get_dart_output_path() else {
            eprintln!("DELI_RSTYPES_PATH not set - skipping Dart codegen test");
            return;
        };

        let dart_file = base_path.join("value.dart");
        assert!(
            dart_file.exists(),
            "Expected value.dart to exist at {}",
            dart_file.display()
        );

        let content = fs::read_to_string(&dart_file).expect("Failed to read value.dart");

        assert!(
            content.contains("sealed class Value"),
            "Should contain sealed class Value"
        );
        assert!(
            content.contains("class ValueInt extends Value"),
            "Should contain ValueInt subclass"
        );
        assert!(
            content.contains("final int f0"),
            "ValueInt should have f0 field of type int"
        );
        assert!(
            content.contains("class ValueText extends Value"),
            "Should contain ValueText subclass"
        );
        assert!(
            content.contains("class ValueNothing extends Value"),
            "Should contain ValueNothing subclass"
        );
    }

    #[test]
    fn test_line_custom_type_import() {
        let Some(base_path) = get_dart_output_path() else {
            eprintln!("DELI_RSTYPES_PATH not set - skipping Dart codegen test");
            return;
        };

        let dart_file = base_path.join("line.dart");
        assert!(
            dart_file.exists(),
            "Expected line.dart to exist at {}",
            dart_file.display()
        );

        let content = fs::read_to_string(&dart_file).expect("Failed to read line.dart");

        assert!(content.contains("class Line"), "Should contain class Line");
        assert!(
            content.contains("import 'point.dart'"),
            "Should import point.dart for custom Point type"
        );
        assert!(
            content.contains("final Point start"),
            "Should have start field of type Point"
        );
        assert!(
            content.contains("final Point end"),
            "Should have end field of type Point"
        );
        // Verify no duplicate imports
        assert_eq!(
            content.matches("import 'point.dart'").count(),
            1,
            "Should have exactly one import for point.dart (no duplicates)"
        );
    }

    #[test]
    fn test_all_files_have_frombin_and_tobin() {
        let Some(base_path) = get_dart_output_path() else {
            eprintln!("DELI_RSTYPES_PATH not set - skipping Dart codegen test");
            return;
        };

        let files = vec![
            "point.dart", "message.dart", "matrix.dart", "color.dart",
            "shape.dart", "marker.dart", "value.dart", "line.dart",
        ];

        for file_name in files {
            let dart_file = base_path.join(file_name);
            assert!(
                dart_file.exists(),
                "Expected {} to exist at {}",
                file_name,
                dart_file.display()
            );

            let content = fs::read_to_string(&dart_file)
                .unwrap_or_else(|_| panic!("Failed to read {}", file_name));

            assert!(
                content.contains("factory") && content.contains(".fromBin(Uint8List bytes)"),
                "{} should contain fromBin factory constructor",
                file_name
            );
            assert!(
                content.contains("Uint8List toBin()"),
                "{} should contain toBin method",
                file_name
            );
        }
    }
}
