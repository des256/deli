// Dart decode/encode expression generation

use super::types::{extract_vec_inner, normalize_type, rust_type_to_dart_type};

/// Generate decode expression for a type
/// Returns (decode_expr, offset_update)
/// var_name is used for the variable being decoded
pub(super) fn gen_decode_expr(ty_str: &str, var_name: &str) -> (String, String) {
    let normalized = normalize_type(ty_str);

    if let Some(inner) = extract_vec_inner(&normalized) {
        return gen_vec_decode_expr(&inner, var_name);
    }

    match normalized.as_str() {
        "bool" => ("bytes[offset] != 0".to_string(), "offset += 1".to_string()),
        "u8" => ("data.getUint8(offset)".to_string(), "offset += 1".to_string()),
        "u16" => ("data.getUint16(offset, Endian.little)".to_string(), "offset += 2".to_string()),
        "u32" => ("data.getUint32(offset, Endian.little)".to_string(), "offset += 4".to_string()),
        "u64" => ("data.getUint64(offset, Endian.little)".to_string(), "offset += 8".to_string()),
        "i8" => ("data.getInt8(offset)".to_string(), "offset += 1".to_string()),
        "i16" => ("data.getInt16(offset, Endian.little)".to_string(), "offset += 2".to_string()),
        "i32" => ("data.getInt32(offset, Endian.little)".to_string(), "offset += 4".to_string()),
        "i64" => ("data.getInt64(offset, Endian.little)".to_string(), "offset += 8".to_string()),
        "f32" => ("data.getFloat32(offset, Endian.little)".to_string(), "offset += 4".to_string()),
        "f64" => ("data.getFloat64(offset, Endian.little)".to_string(), "offset += 8".to_string()),
        "String" => (
            "() { final len = data.getUint32(offset, Endian.little); offset += 4; final s = utf8.decode(bytes.sublist(offset, offset + len)); offset += len; return s; }()".to_string(),
            "".to_string(),
        ),
        _custom => (
            format!(
                "() {{ final r = {}.decode(data, bytes, offset); offset = r.$2; return r.$1; }}()",
                _custom
            ),
            "".to_string(),
        ),
    }
}

fn gen_vec_decode_expr(inner: &str, var_name: &str) -> (String, String) {
    let dart_inner_type = rust_type_to_dart_type(inner);
    let is_primitive_inner = matches!(
        inner,
        "bool" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "f32" | "f64"
    );

    if is_primitive_inner {
        let decode_stmt = match inner {
            "bool" => "bytes[offset + i] != 0",
            "u8" => "data.getUint8(offset + i)",
            "u16" => "data.getUint16(offset + i * 2, Endian.little)",
            "u32" => "data.getUint32(offset + i * 4, Endian.little)",
            "u64" => "data.getUint64(offset + i * 8, Endian.little)",
            "i8" => "data.getInt8(offset + i)",
            "i16" => "data.getInt16(offset + i * 2, Endian.little)",
            "i32" => "data.getInt32(offset + i * 4, Endian.little)",
            "i64" => "data.getInt64(offset + i * 8, Endian.little)",
            "f32" => "data.getFloat32(offset + i * 4, Endian.little)",
            "f64" => "data.getFloat64(offset + i * 8, Endian.little)",
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
                "() {{ final len = data.getUint32(offset, Endian.little); offset += 4; final {} = List<{}>.generate(len, (i) => {}); offset += len * {}; return {}; }}()",
                var_name, dart_inner_type, decode_stmt, element_size, var_name
            ),
            "".to_string(),
        )
    } else {
        let decode_call = if inner == "String" {
            "() { final len = data.getUint32(offset, Endian.little); offset += 4; final s = utf8.decode(bytes.sublist(offset, offset + len)); offset += len; return s; }()".to_string()
        } else if extract_vec_inner(inner).is_some() {
            let (inner_decode, inner_update) = gen_decode_expr(inner, "_item");
            if inner_update.is_empty() {
                format!("() {{ final _item = {}; return _item; }}()", inner_decode)
            } else {
                format!("() {{ final _item = {}; {}; return _item; }}()", inner_decode, inner_update)
            }
        } else {
            format!("() {{ final r = {}.decode(data, bytes, offset); offset = r.$2; return r.$1; }}()", inner)
        };
        (
            format!(
                "() {{ final len = data.getUint32(offset, Endian.little); offset += 4; final {} = List<{}>.generate(len, (_) => {}); return {}; }}()",
                var_name, dart_inner_type, decode_call, var_name
            ),
            "".to_string(),
        )
    }
}

/// Generate encode expression for a type
/// var_name is the variable being encoded
pub(super) fn gen_encode_expr(ty_str: &str, var_name: &str) -> String {
    let normalized = normalize_type(ty_str);

    if let Some(inner) = extract_vec_inner(&normalized) {
        return gen_vec_encode_expr(&inner, var_name);
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
        _custom => format!("{}._encode(builder)", var_name),
    }
}

fn gen_vec_encode_expr(inner: &str, var_name: &str) -> String {
    let is_primitive_inner = matches!(
        inner,
        "bool" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "f32" | "f64"
    );

    if is_primitive_inner {
        let encode_stmt = match inner {
            "bool" => "builder.addByte(v ? 1 : 0)",
            "u8" => "builder.addByte(v)",
            "u16" => "_d.setUint16(0, v, Endian.little); builder.add(_d.buffer.asUint8List(0, 2))",
            "u32" => "_d.setUint32(0, v, Endian.little); builder.add(_d.buffer.asUint8List(0, 4))",
            "u64" => "_d.setUint64(0, v, Endian.little); builder.add(_d.buffer.asUint8List(0, 8))",
            "i8" => "_d.setInt8(0, v); builder.addByte(_d.getUint8(0))",
            "i16" => "_d.setInt16(0, v, Endian.little); builder.add(_d.buffer.asUint8List(0, 2))",
            "i32" => "_d.setInt32(0, v, Endian.little); builder.add(_d.buffer.asUint8List(0, 4))",
            "i64" => "_d.setInt64(0, v, Endian.little); builder.add(_d.buffer.asUint8List(0, 8))",
            "f32" => "_d.setFloat32(0, v, Endian.little); builder.add(_d.buffer.asUint8List(0, 4))",
            "f64" => "_d.setFloat64(0, v, Endian.little); builder.add(_d.buffer.asUint8List(0, 8))",
            _ => unreachable!(),
        };
        format!(
            "_d.setUint32(0, {}.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final v in {}) {{ {} }}",
            var_name, var_name, encode_stmt
        )
    } else if inner == "String" {
        format!(
            "_d.setUint32(0, {}.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final s in {}) {{ final encoded = utf8.encode(s); _d.setUint32(0, encoded.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); builder.add(encoded); }}",
            var_name, var_name
        )
    } else if extract_vec_inner(inner).is_some() {
        let inner_encode = gen_encode_expr(inner, "item");
        format!(
            "_d.setUint32(0, {}.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final item in {}) {{ {}; }}",
            var_name, var_name, inner_encode
        )
    } else {
        format!(
            "_d.setUint32(0, {}.length, Endian.little); builder.add(_d.buffer.asUint8List(0, 4)); for (final item in {}) {{ item._encode(builder); }}",
            var_name, var_name
        )
    }
}
