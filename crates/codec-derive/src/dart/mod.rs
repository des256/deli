// Dart code generation module
//
// Generates Dart source files with binary-compatible serialization (fromBin/toBin)
// matching the Rust Codec wire format.

mod codec;
mod enum_gen;
mod struct_gen;
mod types;

pub(crate) use types::to_snake_case;

/// Main entry point: generate Dart source code from a ParsedItem
pub fn generate_dart(item: &crate::ParsedItem) -> String {
    let class_name = item.name.to_string();

    match &item.data {
        crate::ItemData::Struct(fields) => struct_gen::generate_struct_dart(&class_name, fields),
        crate::ItemData::Enum(variants) => enum_gen::generate_enum_dart(&class_name, variants),
    }
}
