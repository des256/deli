// Dart enum code generation

use super::codec::{gen_decode_expr, gen_encode_expr};
use super::types::{collect_custom_types, rust_type_to_dart_type, to_snake_case};

pub(super) fn generate_enum_dart(name: &str, variants: &[crate::Variant]) -> String {
    let mut imports = vec!["import 'dart:typed_data';".to_string()];
    let mut custom_types = Vec::new();

    // Check if any variant uses String (needs dart:convert)
    let needs_utf8 = variants.iter().any(|v| variant_uses_string(&v.fields));
    if needs_utf8 {
        imports.push("import 'dart:convert';".to_string());
    }

    for variant in variants {
        collect_variant_custom_types(&variant.fields, &mut custom_types);
    }
    custom_types.sort();
    custom_types.dedup();

    for custom in &custom_types {
        imports.push(format!("import '{}.dart';", to_snake_case(custom)));
    }

    let imports_str = imports.join("\n");
    let decode_cases = gen_decode_cases(name, variants);
    let variant_classes = gen_variant_classes(name, variants);

    format!(
        r#"{}

sealed class {} {{
  const {}();

  factory {}.fromBin(Uint8List bytes) {{
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }}

  static ({}, int) decode(ByteData data, Uint8List bytes, int offset) {{
    final variant = data.getUint32(offset, Endian.little); offset += 4;
    switch (variant) {{
{}
      default:
        throw FormatException('Invalid variant: $variant');
    }}
  }}

  Uint8List toBin() {{
    final builder = BytesBuilder();
    _encode(builder);
    return builder.toBytes();
  }}

  void _encode(BytesBuilder builder);
}}

{}
"#,
        imports_str,
        name,
        name,
        name,
        name,
        decode_cases.join("\n"),
        variant_classes.join("\n\n")
    )
}

fn variant_uses_string(fields: &crate::FieldsKind) -> bool {
    match fields {
        crate::FieldsKind::Named(fields) => {
            fields.iter().any(|f| f.ty.to_string().contains("String"))
        }
        crate::FieldsKind::Tuple(fields) => {
            fields.iter().any(|f| f.ty.to_string().contains("String"))
        }
        crate::FieldsKind::Unit => false,
    }
}

fn collect_variant_custom_types(fields: &crate::FieldsKind, custom_types: &mut Vec<String>) {
    match fields {
        crate::FieldsKind::Named(fields) => {
            for field in fields {
                custom_types.extend(collect_custom_types(&field.ty.to_string()));
            }
        }
        crate::FieldsKind::Tuple(fields) => {
            for field in fields {
                custom_types.extend(collect_custom_types(&field.ty.to_string()));
            }
        }
        crate::FieldsKind::Unit => {}
    }
}

fn gen_decode_cases(name: &str, variants: &[crate::Variant]) -> Vec<String> {
    variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let variant_name = v.name.to_string();
            let class_name = format!("{}{}", name, variant_name);

            match &v.fields {
                crate::FieldsKind::Named(fields) => {
                    let decode_stmts = gen_named_decode_stmts(fields, 8);
                    let field_names: Vec<_> = fields
                        .iter()
                        .map(|f| format!("{}: {}", f.name, f.name))
                        .collect();
                    let ctor_call = format!("{}({})", class_name, field_names.join(", "));
                    format!(
                        "      case {}:\n{}\n        return ({}, offset);",
                        i,
                        decode_stmts.join("\n"),
                        ctor_call
                    )
                }
                crate::FieldsKind::Tuple(fields) => {
                    let decode_stmts = gen_tuple_decode_stmts(fields, 8);
                    let field_names: Vec<_> = (0..fields.len())
                        .map(|idx| format!("f{}: f{}", idx, idx))
                        .collect();
                    let ctor_call = format!("{}({})", class_name, field_names.join(", "));
                    format!(
                        "      case {}:\n{}\n        return ({}, offset);",
                        i,
                        decode_stmts.join("\n"),
                        ctor_call
                    )
                }
                crate::FieldsKind::Unit => {
                    format!("      case {}: return ({}(), offset);", i, class_name)
                }
            }
        })
        .collect()
}

fn gen_named_decode_stmts(fields: &[crate::NamedField], indent: usize) -> Vec<String> {
    let pad = " ".repeat(indent);
    fields
        .iter()
        .map(|f| {
            let field_name = f.name.to_string();
            let ty_str = f.ty.to_string();
            let (decode_expr, offset_update) = gen_decode_expr(&ty_str, &field_name);
            if offset_update.is_empty() {
                format!("{}final {} = {};", pad, field_name, decode_expr)
            } else {
                format!("{}final {} = {}; {};", pad, field_name, decode_expr, offset_update)
            }
        })
        .collect()
}

fn gen_tuple_decode_stmts(fields: &[crate::TupleField], indent: usize) -> Vec<String> {
    let pad = " ".repeat(indent);
    fields
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let field_name = format!("f{}", idx);
            let ty_str = f.ty.to_string();
            let (decode_expr, offset_update) = gen_decode_expr(&ty_str, &field_name);
            if offset_update.is_empty() {
                format!("{}final {} = {};", pad, field_name, decode_expr)
            } else {
                format!("{}final {} = {}; {};", pad, field_name, decode_expr, offset_update)
            }
        })
        .collect()
}

fn gen_variant_classes(name: &str, variants: &[crate::Variant]) -> Vec<String> {
    variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let variant_name = v.name.to_string();
            let class_name = format!("{}{}", name, variant_name);

            match &v.fields {
                crate::FieldsKind::Named(fields) => gen_named_variant_class(name, &class_name, i, fields),
                crate::FieldsKind::Tuple(fields) => gen_tuple_variant_class(name, &class_name, i, fields),
                crate::FieldsKind::Unit => gen_unit_variant_class(name, &class_name, i),
            }
        })
        .collect()
}

fn gen_named_variant_class(
    parent_name: &str,
    class_name: &str,
    discriminant: usize,
    fields: &[crate::NamedField],
) -> String {
    let field_decls: Vec<_> = fields
        .iter()
        .map(|f| {
            let field_name = f.name.to_string();
            let dart_type = rust_type_to_dart_type(&f.ty.to_string());
            format!("  final {} {};", dart_type, field_name)
        })
        .collect();

    let ctor_params: Vec<_> = fields
        .iter()
        .map(|f| format!("required this.{}", f.name))
        .collect();

    let encode_stmts: Vec<_> = fields
        .iter()
        .map(|f| {
            let field_name = f.name.to_string();
            format!("    {};", gen_encode_expr(&f.ty.to_string(), &field_name))
        })
        .collect();

    format!(
        r#"class {} extends {} {{
{}

  const {}({{{}}});

  @override
  void _encode(BytesBuilder builder) {{
    final _d = ByteData(8);
    _d.setUint32(0, {}, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
{}
  }}
}}"#,
        class_name,
        parent_name,
        field_decls.join("\n"),
        class_name,
        ctor_params.join(", "),
        discriminant,
        encode_stmts.join("\n")
    )
}

fn gen_tuple_variant_class(
    parent_name: &str,
    class_name: &str,
    discriminant: usize,
    fields: &[crate::TupleField],
) -> String {
    let field_decls: Vec<_> = fields
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let dart_type = rust_type_to_dart_type(&f.ty.to_string());
            format!("  final {} f{};", dart_type, idx)
        })
        .collect();

    let ctor_params: Vec<_> = (0..fields.len())
        .map(|idx| format!("required this.f{}", idx))
        .collect();

    let encode_stmts: Vec<_> = fields
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let field_name = format!("f{}", idx);
            format!("    {};", gen_encode_expr(&f.ty.to_string(), &field_name))
        })
        .collect();

    format!(
        r#"class {} extends {} {{
{}

  const {}({{{}}});

  @override
  void _encode(BytesBuilder builder) {{
    final _d = ByteData(8);
    _d.setUint32(0, {}, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
{}
  }}
}}"#,
        class_name,
        parent_name,
        field_decls.join("\n"),
        class_name,
        ctor_params.join(", "),
        discriminant,
        encode_stmts.join("\n")
    )
}

fn gen_unit_variant_class(parent_name: &str, class_name: &str, discriminant: usize) -> String {
    format!(
        r#"class {} extends {} {{
  const {}();

  @override
  void _encode(BytesBuilder builder) {{
    final _d = ByteData(8);
    _d.setUint32(0, {}, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }}
}}"#,
        class_name, parent_name, class_name, discriminant
    )
}
