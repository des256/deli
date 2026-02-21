use proc_macro::TokenStream;
use proc_macro2::{Delimiter, Ident, Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;

mod dart;

// --- Minimal manual parser (no syn) ---

pub(crate) struct NamedField {
    pub(crate) name: Ident,
    pub(crate) ty: TokenStream2,
}

pub(crate) struct TupleField {
    pub(crate) ty: TokenStream2,
}

pub(crate) enum FieldsKind {
    Named(Vec<NamedField>),
    Tuple(Vec<TupleField>),
    Unit,
}

pub(crate) struct Variant {
    pub(crate) name: Ident,
    pub(crate) fields: FieldsKind,
}

pub(crate) enum ItemData {
    Struct(FieldsKind),
    Enum(Vec<Variant>),
}

pub(crate) struct DartAttrs {
    pub(crate) target: Option<String>,
}

pub(crate) struct ParsedItem {
    pub(crate) name: Ident,
    pub(crate) data: ItemData,
    pub(crate) dart_attrs: DartAttrs,
}

/// Collect tokens for a type, stopping at a `,` or end of iterator.
/// Handles nested `<>` so generic types like `Vec<String>` are captured whole.
fn collect_type(tokens: &[TokenTree], start: usize) -> (TokenStream2, usize) {
    let mut depth = 0usize;
    let mut i = start;
    let mut out = Vec::new();

    while i < tokens.len() {
        match &tokens[i] {
            TokenTree::Punct(p) if p.as_char() == ',' && depth == 0 => break,
            TokenTree::Punct(p) if p.as_char() == '<' => {
                depth += 1;
                out.push(tokens[i].clone());
            }
            TokenTree::Punct(p) if p.as_char() == '>' => {
                depth = depth.saturating_sub(1);
                out.push(tokens[i].clone());
            }
            _ => {
                out.push(tokens[i].clone());
            }
        }
        i += 1;
    }

    (out.into_iter().collect(), i)
}

fn parse_named_fields(group: &proc_macro2::Group) -> Vec<NamedField> {
    let tokens: Vec<_> = group.stream().into_iter().collect();
    let mut fields = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        // field_name : Type ,
        let name = match &tokens[i] {
            TokenTree::Ident(id) => id.clone(),
            _ => {
                i += 1;
                continue;
            }
        };
        i += 1;

        // skip ':'
        if i < tokens.len() {
            if let TokenTree::Punct(p) = &tokens[i] {
                if p.as_char() == ':' {
                    i += 1;
                }
            }
        }

        let (ty, end) = collect_type(&tokens, i);
        i = end;

        // skip ','
        if i < tokens.len() {
            if let TokenTree::Punct(p) = &tokens[i] {
                if p.as_char() == ',' {
                    i += 1;
                }
            }
        }

        fields.push(NamedField { name, ty });
    }

    fields
}

fn parse_tuple_fields(group: &proc_macro2::Group) -> Vec<TupleField> {
    let tokens: Vec<_> = group.stream().into_iter().collect();
    let mut fields = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        let (ty, end) = collect_type(&tokens, i);
        i = end;

        // skip ','
        if i < tokens.len() {
            if let TokenTree::Punct(p) = &tokens[i] {
                if p.as_char() == ',' {
                    i += 1;
                }
            }
        }

        if !ty.is_empty() {
            fields.push(TupleField { ty });
        }
    }

    fields
}

fn parse_variants(group: &proc_macro2::Group) -> Vec<Variant> {
    let tokens: Vec<_> = group.stream().into_iter().collect();
    let mut variants = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        // Variant name
        let name = match &tokens[i] {
            TokenTree::Ident(id) => id.clone(),
            _ => {
                i += 1;
                continue;
            }
        };
        i += 1;

        // Check what follows: { named }, ( tuple ), or , / end (unit)
        let fields = if i < tokens.len() {
            match &tokens[i] {
                TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => {
                    i += 1;
                    FieldsKind::Named(parse_named_fields(g))
                }
                TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => {
                    i += 1;
                    FieldsKind::Tuple(parse_tuple_fields(g))
                }
                _ => FieldsKind::Unit,
            }
        } else {
            FieldsKind::Unit
        };

        // skip ','
        if i < tokens.len() {
            if let TokenTree::Punct(p) = &tokens[i] {
                if p.as_char() == ',' {
                    i += 1;
                }
            }
        }

        variants.push(Variant { name, fields });
    }

    variants
}

fn parse_dart_attr(group: &proc_macro2::Group) -> DartAttrs {
    let tokens: Vec<_> = group.stream().into_iter().collect();
    let mut attrs = DartAttrs { target: None };
    let mut i = 0;
    while i < tokens.len() {
        if let TokenTree::Ident(id) = &tokens[i] {
            if id.to_string() == "target" {
                // expect `=` then a string literal
                i += 1;
                if i < tokens.len() {
                    if let TokenTree::Punct(p) = &tokens[i] {
                        if p.as_char() == '=' {
                            i += 1;
                        }
                    }
                }
                if i < tokens.len() {
                    if let TokenTree::Literal(lit) = &tokens[i] {
                        let s = lit.to_string();
                        // Strip surrounding quotes
                        if s.starts_with('"') && s.ends_with('"') {
                            attrs.target = Some(s[1..s.len() - 1].to_string());
                        }
                    }
                }
            }
        }
        i += 1;
    }
    attrs
}

fn parse_input(input: TokenStream2) -> ParsedItem {
    let tokens: Vec<_> = input.into_iter().collect();

    // Extract #[dart(...)] attributes and find 'struct' or 'enum' keyword
    let mut dart_attrs = DartAttrs { target: None };
    let mut i = 0;
    let mut kind = None;
    while i < tokens.len() {
        // Check for #[dart(...)]
        if let TokenTree::Punct(p) = &tokens[i] {
            if p.as_char() == '#' && i + 1 < tokens.len() {
                if let TokenTree::Group(bracket) = &tokens[i + 1] {
                    if bracket.delimiter() == Delimiter::Bracket {
                        let inner: Vec<_> = bracket.stream().into_iter().collect();
                        if let Some(TokenTree::Ident(id)) = inner.first() {
                            if id.to_string() == "dart" {
                                if let Some(TokenTree::Group(args)) = inner.get(1) {
                                    if args.delimiter() == Delimiter::Parenthesis {
                                        dart_attrs = parse_dart_attr(args);
                                    }
                                }
                                i += 2;
                                continue;
                            }
                        }
                    }
                }
            }
        }
        if let TokenTree::Ident(id) = &tokens[i] {
            let s = id.to_string();
            if s == "struct" || s == "enum" {
                kind = Some(s);
                i += 1;
                break;
            }
        }
        i += 1;
    }

    let kind = kind.expect("expected struct or enum");

    // Next token is the name
    let name = match &tokens[i] {
        TokenTree::Ident(id) => id.clone(),
        _ => panic!("expected item name"),
    };
    i += 1;

    // Find the body (brace group), skipping generics/where clauses/semicolons
    let data = if kind == "struct" {
        // Look for brace group or semicolon (unit struct)
        loop {
            if i >= tokens.len() {
                break ItemData::Struct(FieldsKind::Unit);
            }
            match &tokens[i] {
                TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => {
                    break ItemData::Struct(FieldsKind::Named(parse_named_fields(g)));
                }
                TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => {
                    break ItemData::Struct(FieldsKind::Tuple(parse_tuple_fields(g)));
                }
                TokenTree::Punct(p) if p.as_char() == ';' => {
                    break ItemData::Struct(FieldsKind::Unit);
                }
                _ => i += 1,
            }
        }
    } else {
        // enum — find brace group
        while i < tokens.len() {
            if let TokenTree::Group(g) = &tokens[i] {
                if g.delimiter() == Delimiter::Brace {
                    return ParsedItem {
                        name,
                        data: ItemData::Enum(parse_variants(g)),
                        dart_attrs,
                    };
                }
            }
            i += 1;
        }
        panic!("expected enum body");
    };

    ParsedItem { name, data, dart_attrs }
}

// --- Code generation ---

fn gen_named_encode(fields: &[NamedField], self_prefix: TokenStream2) -> TokenStream2 {
    let stmts = fields.iter().map(|f| {
        let name = &f.name;
        quote! { #self_prefix #name.encode(buf); }
    });
    quote! { #(#stmts)* }
}

fn gen_named_decode(fields: &[NamedField]) -> TokenStream2 {
    let field_decodes = fields.iter().map(|f| {
        let name = &f.name;
        let ty = &f.ty;
        quote! { #name: <#ty as codec::Codec>::decode(buf, pos)?, }
    });
    quote! { { #(#field_decodes)* } }
}

fn gen_tuple_decode(fields: &[TupleField]) -> TokenStream2 {
    let field_decodes = fields.iter().map(|f| {
        let ty = &f.ty;
        quote! { <#ty as codec::Codec>::decode(buf, pos)?, }
    });
    quote! { ( #(#field_decodes)* ) }
}

#[proc_macro_derive(Codec)]
pub fn derive_codec(input: TokenStream) -> TokenStream {
    let parsed = parse_input(input.into());
    let name = &parsed.name;

    let (encode_body, decode_body) = match &parsed.data {
        ItemData::Struct(fields) => match fields {
            FieldsKind::Named(fields) => {
                let enc = gen_named_encode(fields, quote! { self. });
                let dec = gen_named_decode(fields);
                (enc, quote! { Ok(Self #dec) })
            }
            FieldsKind::Tuple(fields) => {
                let enc_stmts = fields.iter().enumerate().map(|(i, _)| {
                    let idx = proc_macro2::Literal::usize_unsuffixed(i);
                    quote! { self.#idx.encode(buf); }
                });
                let dec = gen_tuple_decode(fields);
                (quote! { #(#enc_stmts)* }, quote! { Ok(Self #dec) })
            }
            FieldsKind::Unit => (quote! {}, quote! { Ok(Self) }),
        },
        ItemData::Enum(variants) => {
            let encode_arms = variants.iter().enumerate().map(|(i, v)| {
                let vname = &v.name;
                let disc = i as u32;
                match &v.fields {
                    FieldsKind::Named(fields) => {
                        let field_names: Vec<_> = fields.iter().map(|f| &f.name).collect();
                        let encode_fields = field_names.iter().map(|n| {
                            quote! { #n.encode(buf); }
                        });
                        quote! {
                            #name::#vname { #(#field_names),* } => {
                                (#disc as u32).encode(buf);
                                #(#encode_fields)*
                            }
                        }
                    }
                    FieldsKind::Tuple(fields) => {
                        let bindings: Vec<_> = (0..fields.len())
                            .map(|i| Ident::new(&format!("f{i}"), Span::call_site()))
                            .collect();
                        let encode_fields = bindings.iter().map(|b| {
                            quote! { #b.encode(buf); }
                        });
                        quote! {
                            #name::#vname(#(#bindings),*) => {
                                (#disc as u32).encode(buf);
                                #(#encode_fields)*
                            }
                        }
                    }
                    FieldsKind::Unit => {
                        quote! {
                            #name::#vname => {
                                (#disc as u32).encode(buf);
                            }
                        }
                    }
                }
            });

            let decode_arms = variants.iter().enumerate().map(|(i, v)| {
                let vname = &v.name;
                let disc = i as u32;
                match &v.fields {
                    FieldsKind::Named(fields) => {
                        let dec = gen_named_decode(fields);
                        quote! { #disc => Ok(#name::#vname #dec), }
                    }
                    FieldsKind::Tuple(fields) => {
                        let dec = gen_tuple_decode(fields);
                        quote! { #disc => Ok(#name::#vname #dec), }
                    }
                    FieldsKind::Unit => {
                        quote! { #disc => Ok(#name::#vname), }
                    }
                }
            });

            let encode = quote! {
                match self {
                    #(#encode_arms)*
                }
            };
            let decode = quote! {
                let variant = <u32 as codec::Codec>::decode(buf, pos)?;
                match variant {
                    #(#decode_arms)*
                    v => Err(codec::DecodeError::InvalidVariant(v)),
                }
            };
            (encode, decode)
        }
    };

    let expanded = quote! {
        impl codec::Codec for #name {
            fn encode(&self, buf: &mut Vec<u8>) {
                #encode_body
            }

            fn decode(buf: &[u8], pos: &mut usize) -> Result<Self, codec::DecodeError> {
                #decode_body
            }
        }
    };

    TokenStream::from(expanded)
}

fn find_workspace_root(manifest_dir: &str) -> std::path::PathBuf {
    let mut dir = std::path::PathBuf::from(manifest_dir);
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(contents) = std::fs::read_to_string(&cargo_toml) {
                if contents.contains("[workspace]") {
                    return dir;
                }
            }
        }
        if !dir.pop() {
            return std::path::PathBuf::from(manifest_dir);
        }
    }
}

#[proc_macro_derive(Dart, attributes(dart))]
pub fn derive_dart(input: TokenStream) -> TokenStream {
    let parsed = parse_input(input.into());

    // Generate Dart source code
    let dart_source = dart::generate_dart(&parsed);

    // Resolve output directory: #[dart(target = "...")] > DELI_RSTYPES_PATH > <workspace>/rstypes/lib/src
    let path = if let Some(target) = &parsed.dart_attrs.target {
        // Resolve relative paths against workspace root
        let p = std::path::PathBuf::from(target);
        if p.is_absolute() {
            target.clone()
        } else {
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
            let workspace_root = find_workspace_root(&manifest_dir);
            workspace_root.join(target).to_string_lossy().into_owned()
        }
    } else {
        std::env::var("DELI_RSTYPES_PATH").unwrap_or_else(|_| {
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
            find_workspace_root(&manifest_dir).join("rstypes/lib/src").to_string_lossy().into_owned()
        })
    };

    // Convert type name to snake_case for filename
    let file_name = dart::to_snake_case(&parsed.name.to_string());
    let file_path = format!("{}/{}.dart", path, file_name);

    // Create directory if it doesn't exist
    if let Some(parent) = std::path::Path::new(&file_path).parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            panic!(
                "Failed to create directory for Dart file at {}: {}",
                parent.display(),
                e
            );
        }
    }

    // Write Dart file
    if let Err(e) = std::fs::write(&file_path, dart_source) {
        panic!("Failed to write Dart file at {}: {}", file_path, e);
    }

    // Return empty token stream - we only have the side effect of writing the file
    TokenStream::new()
}
