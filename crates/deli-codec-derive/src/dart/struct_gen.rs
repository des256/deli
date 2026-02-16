// Dart struct code generation

use super::codec::{gen_decode_expr, gen_encode_expr};
use super::types::{collect_custom_types, rust_type_to_dart_type, to_snake_case};

pub(super) fn generate_struct_dart(name: &str, fields: &crate::FieldsKind) -> String {
    let mut imports = vec!["import 'dart:typed_data';".to_string()];
    let mut custom_types = Vec::new();

    collect_field_custom_types(fields, &mut custom_types);
    custom_types.sort();
    custom_types.dedup();

    for custom in &custom_types {
        imports.push(format!("import '{}.dart';", to_snake_case(custom)));
    }

    if needs_utf8_import(fields) {
        imports.push("import 'dart:convert';".to_string());
    }

    let imports_str = imports.join("\n");

    match fields {
        crate::FieldsKind::Named(fields) => generate_named_struct(name, fields, &imports_str),
        crate::FieldsKind::Tuple(_) => format!("// Tuple struct {} - not yet supported\n", name),
        crate::FieldsKind::Unit => generate_unit_struct(name, &imports_str),
    }
}

fn collect_field_custom_types(fields: &crate::FieldsKind, custom_types: &mut Vec<String>) {
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

fn needs_utf8_import(fields: &crate::FieldsKind) -> bool {
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

fn generate_named_struct(name: &str, fields: &[crate::NamedField], imports_str: &str) -> String {
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

    let decode_stmts: Vec<_> = fields
        .iter()
        .map(|f| {
            let field_name = f.name.to_string();
            let ty_str = f.ty.to_string();
            let (decode_expr, offset_update) = gen_decode_expr(&ty_str, &field_name);
            if offset_update.is_empty() {
                format!("    final {} = {};", field_name, decode_expr)
            } else {
                format!("    final {} = {}; {};", field_name, decode_expr, offset_update)
            }
        })
        .collect();

    let ctor_call = if fields.is_empty() {
        format!("{}()", name)
    } else {
        let field_names: Vec<_> = fields
            .iter()
            .map(|f| format!("{}: {}", f.name, f.name))
            .collect();
        format!("{}({})", name, field_names.join(", "))
    };

    let encode_stmts: Vec<_> = fields
        .iter()
        .map(|f| {
            let field_name = f.name.to_string();
            let ty_str = f.ty.to_string();
            format!("    {};", gen_encode_expr(&ty_str, &field_name))
        })
        .collect();

    let encode_body = encode_stmts.join("\n");
    let needs_bytedata = encode_body.contains("_d.");
    let encode_block = if needs_bytedata {
        format!("    final _d = ByteData(8);\n{}", encode_body)
    } else {
        encode_body
    };

    format!(
        r#"{}

class {} {{
{}

  {}({{{}}});

  factory {}.fromBin(Uint8List bytes) {{
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }}

  static ({}, int) decode(ByteData bd, Uint8List buf, int offset) {{
{}
    return ({}, offset);
  }}

  Uint8List toBin() {{
    final builder = BytesBuilder();
    encode(builder);
    return builder.toBytes();
  }}

  void encode(BytesBuilder builder) {{
{}
  }}
}}
"#,
        imports_str,
        name,
        field_decls.join("\n"),
        name,
        ctor_params.join(", "),
        name,
        name,
        decode_stmts.join("\n"),
        ctor_call,
        encode_block,
    )
}

fn generate_unit_struct(name: &str, imports_str: &str) -> String {
    format!(
        r#"{}

class {} {{
  const {}();

  factory {}.fromBin(Uint8List bytes) {{
    return const {}();
  }}

  static ({}, int) decode(ByteData bd, Uint8List buf, int offset) {{
    return ({}(), offset);
  }}

  Uint8List toBin() {{
    return Uint8List(0);
  }}

  void encode(BytesBuilder builder) {{
    // Empty
  }}
}}
"#,
        imports_str, name, name, name, name, name, name
    )
}
