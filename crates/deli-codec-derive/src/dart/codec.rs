// Dart decode/encode expression generation

use super::types::{extract_vec_inner, normalize_type, rust_type_to_dart_type};

// The decode method parameters are named "bd" (ByteData) and "buf" (Uint8List)
// to avoid shadowing conflicts with user field names like "data" or "bytes".

/// Generate decode expression for a type
/// Returns (decode_expr, offset_update)
pub(super) fn gen_decode_expr(ty_str: &str, var_name: &str) -> (String, String) {
    gen_decode_expr_depth(ty_str, var_name, 0)
}

fn gen_decode_expr_depth(ty_str: &str, var_name: &str, depth: usize) -> (String, String) {
    let normalized = normalize_type(ty_str);

    if let Some(inner) = extract_vec_inner(&normalized) {
        return gen_vec_decode_expr(&inner, var_name, depth);
    }

    match normalized.as_str() {
        "bool" => ("buf[offset] != 0".to_string(), "offset += 1".to_string()),
        "u8" => ("bd.getUint8(offset)".to_string(), "offset += 1".to_string()),
        "u16" => ("bd.getUint16(offset, Endian.little)".to_string(), "offset += 2".to_string()),
        "u32" => ("bd.getUint32(offset, Endian.little)".to_string(), "offset += 4".to_string()),
        "u64" => ("bd.getUint64(offset, Endian.little)".to_string(), "offset += 8".to_string()),
        "i8" => ("bd.getInt8(offset)".to_string(), "offset += 1".to_string()),
        "i16" => ("bd.getInt16(offset, Endian.little)".to_string(), "offset += 2".to_string()),
        "i32" => ("bd.getInt32(offset, Endian.little)".to_string(), "offset += 4".to_string()),
        "i64" => ("bd.getInt64(offset, Endian.little)".to_string(), "offset += 8".to_string()),
        "f32" => ("bd.getFloat32(offset, Endian.little)".to_string(), "offset += 4".to_string()),
        "f64" => ("bd.getFloat64(offset, Endian.little)".to_string(), "offset += 8".to_string()),
        "String" => (
            "() { final len = bd.getUint32(offset, Endian.little); offset += 4; final s = utf8.decode(buf.sublist(offset, offset + len)); offset += len; return s; }()".to_string(),
            "".to_string(),
        ),
        _custom => (
            format!(
                "() {{ final r = {}.decode(bd, buf, offset); offset = r.$2; return r.$1; }}()",
                _custom
            ),
            "".to_string(),
        ),
    }
}

fn gen_vec_decode_expr(inner: &str, _var_name: &str, depth: usize) -> (String, String) {
    let dart_inner_type = rust_type_to_dart_type(inner);
    let len_var = format!("_l{}", depth);
    let result_var = format!("_v{}", depth);
    let is_primitive_inner = matches!(
        inner,
        "bool" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "f32" | "f64"
    );

    if is_primitive_inner {
        let decode_stmt = match inner {
            "bool" => "buf[offset + i] != 0",
            "u8" => "bd.getUint8(offset + i)",
            "u16" => "bd.getUint16(offset + i * 2, Endian.little)",
            "u32" => "bd.getUint32(offset + i * 4, Endian.little)",
            "u64" => "bd.getUint64(offset + i * 8, Endian.little)",
            "i8" => "bd.getInt8(offset + i)",
            "i16" => "bd.getInt16(offset + i * 2, Endian.little)",
            "i32" => "bd.getInt32(offset + i * 4, Endian.little)",
            "i64" => "bd.getInt64(offset + i * 8, Endian.little)",
            "f32" => "bd.getFloat32(offset + i * 4, Endian.little)",
            "f64" => "bd.getFloat64(offset + i * 8, Endian.little)",
            _ => unreachable!(),
        };
        let element_size = match inner {
            "bool" | "u8" | "i8" => 1,
            "u16" | "i16" => 2,
            "u32" | "i32" | "f32" => 4,
            "u64" | "i64" | "f64" => 8,
            _ => unreachable!(),
        };
        (
            format!(
                "() {{ final {len} = bd.getUint32(offset, Endian.little); offset += 4; final {res} = List<{ty}>.generate({len}, (i) => {decode}); offset += {len} * {sz}; return {res}; }}()",
                len = len_var, res = result_var, ty = dart_inner_type, decode = decode_stmt, sz = element_size
            ),
            "".to_string(),
        )
    } else {
        let decode_call = if inner == "String" {
            "() { final len = bd.getUint32(offset, Endian.little); offset += 4; final s = utf8.decode(buf.sublist(offset, offset + len)); offset += len; return s; }()".to_string()
        } else if extract_vec_inner(inner).is_some() {
            let (inner_decode, _) = gen_decode_expr_depth(inner, "_unused", depth + 1);
            inner_decode
        } else {
            format!("() {{ final r = {}.decode(bd, buf, offset); offset = r.$2; return r.$1; }}()", inner)
        };
        (
            format!(
                "() {{ final {len} = bd.getUint32(offset, Endian.little); offset += 4; final {res} = List<{ty}>.generate({len}, (_) => {decode}); return {res}; }}()",
                len = len_var, res = result_var, ty = dart_inner_type, decode = decode_call
            ),
            "".to_string(),
        )
    }
}

/// Generate encode expression for a type
/// var_name is the variable being encoded
pub(super) fn gen_encode_expr(ty_str: &str, var_name: &str) -> String {
    gen_encode_expr_depth(ty_str, var_name, 0)
}

fn gen_encode_expr_depth(ty_str: &str, var_name: &str, depth: usize) -> String {
    let normalized = normalize_type(ty_str);

    if let Some(inner) = extract_vec_inner(&normalized) {
        return gen_vec_encode_expr(&inner, var_name, depth);
    }

    match normalized.as_str() {
        "bool" => format!("builder.addByte({} ? 1 : 0)", var_name),
        "u8" => format!("builder.addByte({})", var_name),
        "u16" => format!("_d.setUint16(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 2))", var_name),
        "u32" => format!("_d.setUint32(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 4))", var_name),
        "u64" => format!("_d.setUint64(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 8))", var_name),
        "i8" => format!("_d.setInt8(0, {}); builder.addByte(_d.getUint8(0))", var_name),
        "i16" => format!("_d.setInt16(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 2))", var_name),
        "i32" => format!("_d.setInt32(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 4))", var_name),
        "i64" => format!("_d.setInt64(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 8))", var_name),
        "f32" => format!("_d.setFloat32(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 4))", var_name),
        "f64" => format!("_d.setFloat64(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 8))", var_name),
        "String" => format!(
            "{{ final encoded = utf8.encode({}); _d.setUint32(0, encoded.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); builder.add(encoded); }}",
            var_name
        ),
        _custom => format!("{}.encode(builder)", var_name),
    }
}

fn gen_vec_encode_expr(inner: &str, var_name: &str, depth: usize) -> String {
    let loop_var = format!("_e{}", depth);
    let is_primitive_inner = matches!(
        inner,
        "bool" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "f32" | "f64"
    );

    if is_primitive_inner {
        let encode_stmt = match inner {
            "bool" => format!("builder.addByte({} ? 1 : 0)", loop_var),
            "u8" => format!("builder.addByte({})", loop_var),
            "u16" => format!("_d.setUint16(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 2))", loop_var),
            "u32" => format!("_d.setUint32(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 4))", loop_var),
            "u64" => format!("_d.setUint64(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 8))", loop_var),
            "i8" => format!("_d.setInt8(0, {}); builder.addByte(_d.getUint8(0))", loop_var),
            "i16" => format!("_d.setInt16(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 2))", loop_var),
            "i32" => format!("_d.setInt32(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 4))", loop_var),
            "i64" => format!("_d.setInt64(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 8))", loop_var),
            "f32" => format!("_d.setFloat32(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 4))", loop_var),
            "f64" => format!("_d.setFloat64(0, {}, Endian.little); builder.add(_d.buffer.asUint8List(0, 8))", loop_var),
            _ => unreachable!(),
        };
        format!(
            "_d.setUint32(0, {var}.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final {lv} in {var}) {{ {enc}; }}",
            var = var_name, lv = loop_var, enc = encode_stmt
        )
    } else if inner == "String" {
        format!(
            "_d.setUint32(0, {var}.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final {lv} in {var}) {{ final encoded = utf8.encode({lv}); _d.setUint32(0, encoded.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); builder.add(encoded); }}",
            var = var_name, lv = loop_var
        )
    } else if extract_vec_inner(inner).is_some() {
        let inner_encode = gen_encode_expr_depth(inner, &loop_var, depth + 1);
        format!(
            "_d.setUint32(0, {var}.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final {lv} in {var}) {{ {enc}; }}",
            var = var_name, lv = loop_var, enc = inner_encode
        )
    } else {
        format!(
            "_d.setUint32(0, {var}.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final {lv} in {var}) {{ {lv}.encode(builder); }}",
            var = var_name, lv = loop_var
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nested_vec_decode_uses_unique_variable_names() {
        let (expr, _) = gen_decode_expr("Vec<Vec<f32>>", "data");
        assert!(
            !expr.contains("final data"),
            "Should not shadow 'data' parameter: {expr}"
        );
        assert!(
            expr.matches("final len").count() <= 1,
            "Should use unique len variable names at each depth: {expr}"
        );
    }

    #[test]
    fn test_decode_exprs_use_bd_not_data() {
        // All decode expressions must reference "bd" (the ByteData parameter),
        // not "data" which could collide with a field named "data".
        let primitives = ["u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64"];
        for ty in &primitives {
            let (expr, _) = gen_decode_expr(ty, "x");
            assert!(
                !expr.contains("data."),
                "{ty} decode should use 'bd' not 'data': {expr}"
            );
        }
        // Vec decode
        let (expr, _) = gen_decode_expr("Vec<f32>", "x");
        assert!(
            !expr.contains("data."),
            "Vec<f32> decode should use 'bd' not 'data': {expr}"
        );
        // String decode
        let (expr, _) = gen_decode_expr("String", "x");
        assert!(
            !expr.contains("data."),
            "String decode should use 'bd' not 'data': {expr}"
        );
        // Nested Vec
        let (expr, _) = gen_decode_expr("Vec<Vec<f32>>", "x");
        assert!(
            !expr.contains("data."),
            "Vec<Vec<f32>> decode should use 'bd' not 'data': {expr}"
        );
        // Custom type decode passes bd/buf
        let (expr, _) = gen_decode_expr("Point", "x");
        assert!(
            expr.contains("decode(bd, buf, offset)"),
            "Custom type decode should pass bd/buf: {expr}"
        );
    }

    #[test]
    fn test_nested_vec_encode_uses_unique_loop_variables() {
        let expr = gen_encode_expr("Vec<Vec<Vec<f32>>>", "data");
        assert!(
            expr.matches("for (final item in").count() <= 1,
            "Should use unique loop variable names at each depth: {expr}"
        );
    }
}
