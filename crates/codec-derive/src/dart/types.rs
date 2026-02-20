// Dart type mapping and helper utilities

/// Convert PascalCase to snake_case
pub(crate) fn to_snake_case(name: &str) -> String {
    let mut result = String::new();
    let mut prev_was_uppercase = false;

    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !prev_was_uppercase {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
            prev_was_uppercase = true;
        } else {
            result.push(ch);
            prev_was_uppercase = false;
        }
    }

    result
}

/// Normalize type string by removing all whitespace
pub(super) fn normalize_type(ty: &str) -> String {
    ty.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Check if a normalized type is Vec<T> and extract T
/// Returns Some(inner_type) if it's a Vec, None otherwise
pub(super) fn extract_vec_inner(ty: &str) -> Option<String> {
    let normalized = normalize_type(ty);
    if normalized.starts_with("Vec<") && normalized.ends_with('>') {
        let inner = &normalized[4..normalized.len() - 1];
        Some(inner.to_string())
    } else {
        None
    }
}

/// Convert Rust type to Dart type recursively
/// Handles primitives, Vec<T>, and custom types
pub(super) fn rust_type_to_dart_type(ty: &str) -> String {
    let normalized = normalize_type(ty);

    if let Some(inner) = extract_vec_inner(&normalized) {
        if inner == "u8" {
            return "Uint8List".to_string();
        }
        let dart_inner = rust_type_to_dart_type(&inner);
        return format!("List<{}>", dart_inner);
    }

    match normalized.as_str() {
        "bool" => "bool".to_string(),
        "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" => "int".to_string(),
        "f32" | "f64" => "double".to_string(),
        "String" => "String".to_string(),
        _ => normalized,
    }
}

/// Collect all custom type names (for imports) from a type string
pub(super) fn collect_custom_types(ty: &str) -> Vec<String> {
    let normalized = normalize_type(ty);

    if let Some(inner) = extract_vec_inner(&normalized) {
        return collect_custom_types(&inner);
    }

    match normalized.as_str() {
        "bool" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "f32" | "f64"
        | "String" => vec![],
        custom => vec![custom.to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_type_removes_whitespace() {
        assert_eq!(normalize_type("Vec < u8 >"), "Vec<u8>");
        assert_eq!(normalize_type("Vec<String>"), "Vec<String>");
        assert_eq!(normalize_type("Vec < Vec < u8 > >"), "Vec<Vec<u8>>");
        assert_eq!(normalize_type("  f32  "), "f32");
    }

    #[test]
    fn test_to_snake_case_basic() {
        assert_eq!(to_snake_case("Point"), "point");
        assert_eq!(to_snake_case("MessageData"), "message_data");
        assert_eq!(to_snake_case("HTTPServer"), "httpserver");
        assert_eq!(to_snake_case("already_snake"), "already_snake");
    }

    #[test]
    fn test_rust_type_to_dart_type_with_whitespace() {
        assert_eq!(rust_type_to_dart_type("  u32  "), "int");
        assert_eq!(rust_type_to_dart_type(" f64 "), "double");
    }

    #[test]
    fn test_extract_vec_inner() {
        assert_eq!(extract_vec_inner("Vec<u8>"), Some("u8".to_string()));
        assert_eq!(extract_vec_inner("Vec<String>"), Some("String".to_string()));
        assert_eq!(
            extract_vec_inner("Vec < Vec < f32 > >"),
            Some("Vec<f32>".to_string())
        );
        assert_eq!(extract_vec_inner("u32"), None);
        assert_eq!(extract_vec_inner("Point"), None);
    }

    #[test]
    fn test_rust_type_to_dart_type() {
        assert_eq!(rust_type_to_dart_type("u32"), "int");
        assert_eq!(rust_type_to_dart_type("f64"), "double");
        assert_eq!(rust_type_to_dart_type("String"), "String");
        assert_eq!(rust_type_to_dart_type("Vec<u8>"), "Uint8List");
        assert_eq!(rust_type_to_dart_type("Vec<String>"), "List<String>");
        assert_eq!(rust_type_to_dart_type("Vec<Vec<f32>>"), "List<List<double>>");
        assert_eq!(rust_type_to_dart_type("Point"), "Point");
        assert_eq!(rust_type_to_dart_type("Vec<Point>"), "List<Point>");
    }

    #[test]
    fn test_collect_custom_types() {
        assert_eq!(collect_custom_types("u32"), Vec::<String>::new());
        assert_eq!(collect_custom_types("String"), Vec::<String>::new());
        assert_eq!(collect_custom_types("Point"), vec!["Point"]);
        assert_eq!(collect_custom_types("Vec<u8>"), Vec::<String>::new());
        assert_eq!(collect_custom_types("Vec<Point>"), vec!["Point"]);
        assert_eq!(
            collect_custom_types("Vec<Vec<CustomType>>"),
            vec!["CustomType"]
        );
    }
}
