# Dart Code Generation Derive Macro Implementation Plan

Created: 2026-02-16
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No

> **Status Lifecycle:** PENDING → COMPLETE → VERIFIED
> **Iterations:** Tracks implement→verify cycles (incremented by verify phase)
>
> - PENDING: Initial state, awaiting implementation
> - COMPLETE: All tasks implemented
> - VERIFIED: All checks passed
>
> **Approval Gate:** Implementation CANNOT proceed until `Approved: Yes`
> **Worktree:** Set at plan creation (from dispatcher). `Yes` uses git worktree isolation; `No` works directly on current branch (default)

## Summary

**Goal:** Add a `#[derive(Dart)]` proc macro to `deli-codec-derive` that generates `.dart` files with binary-compatible serialization code (`fromBin`/`toBin`) matching the Rust `Codec` wire format, enabling Rust↔Dart communication over websockets.

**Architecture:** Add a `dart.rs` module to `deli-codec-derive` containing Dart code generation logic. The existing hand-rolled parser (`ParsedItem`, `FieldsKind`, `Variant`) is reused — no duplication. The new `#[proc_macro_derive(Dart)]` entry point parses the Rust type, generates a Dart class as a string, and writes it to `$DELI_RSTYPES_PATH/<snake_case_name>.dart` at compile time. Each generated Dart class has a `fromBin` factory constructor and a `toBin` method. Internal `decode`/`_encode` helpers handle offset tracking for nested types.

**Tech Stack:** Rust proc-macro (`proc_macro2`, `quote`), Dart (`dart:typed_data` for binary I/O).

## Scope

### In Scope

- `#[proc_macro_derive(Dart)]` macro in `deli-codec-derive`
- Dart code generation for structs (named fields, tuple fields, unit)
- Dart code generation for enums (named field variants, tuple variants, unit variants)
- Type mapping: Rust primitives (bool, u8–u64, i8–i64, f32, f64), String, `Vec<T>`, and nested custom types
- File writing to `$DELI_RSTYPES_PATH` env var path
- `fromBin(Uint8List bytes)` factory constructor (top-level decode entry point)
- `toBin()` method returning `Uint8List` (top-level encode entry point)
- Static `decode(ByteData data, Uint8List bytes, int offset)` returning `(T, int)` for nested type decode
- `_encode(BytesBuilder builder)` instance method for nested type encode
- Dart imports for `dart:typed_data`, `dart:convert` (for UTF-8), and cross-type imports
- Integration tests verifying generated Dart matches expected output
- Silent skip (no file writing, no panic) when `DELI_RSTYPES_PATH` env var is not set — macro returns empty `TokenStream`

### Out of Scope

- Dart code execution or compilation verification (no Dart SDK in CI)
- `Option<T>` / nullable field support (can add later)
- Generics in Rust types (the parser already doesn't handle generics beyond Vec)
- Barrel file (`rstypes.dart`) auto-update — generated files are individual
- Dart unit tests auto-generation
- Tuple structs (tuple struct Dart representation is ambiguous — can add later with positional constructors)

## Prerequisites

- `DELI_RSTYPES_PATH` environment variable must be set to the target directory when compiling types that derive `Dart` (e.g., `DELI_RSTYPES_PATH=rstypes/lib/src`)
- Types that derive `Dart` should also derive `Codec` (the binary format is defined by `Codec`)

## Context for Implementer

- **Patterns to follow:** The existing `derive_codec` function at `crates/deli-codec-derive/src/lib.rs:275` is the template. It calls `parse_input()` to get a `ParsedItem`, then generates code from it. The new `derive_dart` will call the same `parse_input()` but generate a Dart source string instead of Rust tokens.
- **Conventions:** The crate uses no `syn` — it has a hand-rolled parser. All parsing types (`ParsedItem`, `ItemData`, `FieldsKind`, `NamedField`, `TupleField`, `Variant`) are in `lib.rs`. The Dart generation should be in a separate `dart.rs` module for file length management.
- **Key files:**
  - `crates/deli-codec-derive/src/lib.rs` — Parser types + `derive_codec`. Add `derive_dart` entry point here, import generation from `dart.rs`
  - `crates/deli-codec-derive/src/dart.rs` — New file with all Dart code generation logic
  - `crates/deli-codec/src/primitives.rs` — Defines the binary format for each primitive type (reference for Dart decode/encode)
  - `crates/deli-codec/src/lib.rs` — Re-exports. Will also re-export `Dart` derive macro
  - `rstypes/lib/src/` — Target directory for generated `.dart` files
- **Gotchas:**
  - Proc macros run at compile time. File I/O (`std::fs::write`) works but the output directory must exist. If `DELI_RSTYPES_PATH` is not set, silently return empty `TokenStream` (no panic, no file write). This allows `#[derive(Dart)]` to coexist with `#[derive(Codec)]` without requiring the env var in all build contexts.
  - The parser stores types as `TokenStream2`. To map to Dart types, convert to string and normalize: remove all spaces, then pattern match (e.g., `"f32"` → `"double"`, `"Vec<u8>"` → `"List<int>"`). Normalization function: strip whitespace from token string, then match with exact prefix `"Vec<"` and closing `">"` for generics. This handles any token spacing (`Vec < u8 >`, `Vec<u8>`, `Vec< u8>` all normalize to `Vec<u8>`). Nested generics like `Vec<Vec<u8>>` must also normalize correctly to `List<List<int>>`.
  - Nested types (a struct field of type `Point`) require the Dart file to `import 'point.dart';`. The generator must detect non-primitive types and add imports.
  - Enum discriminants are `u32` LE in the `Codec` format, indexed 0, 1, 2... in declaration order.
  - `String` encoding: u32 LE length prefix + UTF-8 bytes. Dart decode uses `utf8.decode(bytes.sublist(offset, offset + len))`.
  - `Vec<T>` encoding: u32 LE length prefix + N encoded elements.
  - For the `decode` static method signature, Dart 3 records `(Type, int)` are the cleanest way to return both the decoded value and the new offset.
- **Binary format reference (from `primitives.rs`):**
  - `bool`: 1 byte (0=false, 1=true)
  - `u8`: 1 byte LE
  - `u16`: 2 bytes LE
  - `u32`: 4 bytes LE
  - `u64`: 8 bytes LE
  - `i8`: 1 byte LE (signed)
  - `i16`: 2 bytes LE
  - `i32`: 4 bytes LE
  - `i64`: 8 bytes LE
  - `f32`: 4 bytes LE
  - `f64`: 8 bytes LE
  - `String`: u32 length + UTF-8 bytes
  - `Vec<T>`: u32 count + N × T
  - Struct: fields concatenated in order
  - Enum: u32 discriminant + variant fields

## Dart Type Mapping Reference

| Rust Type | Dart Type | ByteData Decode | Size |
|-----------|-----------|-----------------|------|
| `bool` | `bool` | `bytes[offset] != 0` | 1 |
| `u8` | `int` | `data.getUint8(offset)` | 1 |
| `u16` | `int` | `data.getUint16(offset, Endian.little)` | 2 |
| `u32` | `int` | `data.getUint32(offset, Endian.little)` | 4 |
| `u64` | `int` | `data.getUint64(offset, Endian.little)` | 8 |
| `i8` | `int` | `data.getInt8(offset)` | 1 |
| `i16` | `int` | `data.getInt16(offset, Endian.little)` | 2 |
| `i32` | `int` | `data.getInt32(offset, Endian.little)` | 4 |
| `i64` | `int` | `data.getInt64(offset, Endian.little)` | 8 |
| `f32` | `double` | `data.getFloat32(offset, Endian.little)` | 4 |
| `f64` | `double` | `data.getFloat64(offset, Endian.little)` | 8 |
| `String` | `String` | u32 len + `utf8.decode(bytes.sublist(...))` | 4+len |
| `Vec<T>` | `List<DartT>` | u32 count + decode each | 4+N×T |
| Custom | ClassName | `ClassName.decode(data, bytes, offset)` | variable |

## Dart Generated Code Shape

### Named Struct Example

For `struct Point { x: f32, y: f32 }`:

```dart
import 'dart:typed_data';

class Point {
  final double x;
  final double y;

  Point({required this.x, required this.y});

  factory Point.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Point, int) decode(ByteData data, Uint8List bytes, int offset) {
    final x = data.getFloat32(offset, Endian.little); offset += 4;
    final y = data.getFloat32(offset, Endian.little); offset += 4;
    return (Point(x: x, y: y), offset);
  }

  Uint8List toBin() {
    final builder = BytesBuilder();
    _encode(builder);
    return builder.toBytes();
  }

  void _encode(BytesBuilder builder) {
    final _d = ByteData(8); // max needed for f64/i64/u64
    _d.setFloat32(0, x, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    _d.setFloat32(0, y, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
  }
}
```

### Enum Example

For `enum Value { Int(i64), Text(String), Nothing }`:

```dart
import 'dart:typed_data';
import 'dart:convert';

sealed class Value {
  const Value();

  factory Value.fromBin(Uint8List bytes) {
    final r = decode(ByteData.sublistView(bytes), bytes, 0);
    return r.$1;
  }

  static (Value, int) decode(ByteData data, Uint8List bytes, int offset) {
    final variant = data.getUint32(offset, Endian.little); offset += 4;
    switch (variant) {
      case 0:
        final f0 = data.getInt64(offset, Endian.little); offset += 8;
        return (ValueInt(f0: f0), offset);
      case 1:
        final len = data.getUint32(offset, Endian.little); offset += 4;
        final f0 = utf8.decode(bytes.sublist(offset, offset + len)); offset += len;
        return (ValueText(f0: f0), offset);
      case 2:
        return (ValueNothing(), offset);
      default:
        throw FormatException('Invalid variant: $variant');
    }
  }

  Uint8List toBin() {
    final builder = BytesBuilder();
    _encode(builder);
    return builder.toBytes();
  }

  void _encode(BytesBuilder builder);
}

class ValueInt extends Value {
  final int f0;
  const ValueInt({required this.f0});

  @override
  void _encode(BytesBuilder builder) {
    final _d = ByteData(8);
    _d.setUint32(0, 0, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 4));
    _d.setInt64(0, f0, Endian.little);
    builder.add(_d.buffer.asUint8List(0, 8));
  }
}
// ... etc for each variant
```

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Implement Dart code generation module (`dart.rs`)
- [x] Task 2: Wire up `#[proc_macro_derive(Dart)]` and re-export
- [x] Task 3: Add integration tests for struct and enum codegen

**Total Tasks:** 3 | **Completed:** 3 | **Remaining:** 0

## Implementation Tasks

### Task 1: Implement Dart code generation module (`dart.rs`)

**Objective:** Create the core Dart code generation logic as a separate module in `deli-codec-derive`. This module takes a `ParsedItem` and produces a Dart source code string.

**Dependencies:** None

**Files:**

- Create: `crates/deli-codec-derive/src/dart.rs`

**Key Decisions / Notes:**

- The module receives a `ParsedItem` (from the existing parser) and returns a `String` containing valid Dart source code.
- Entry point: `pub fn generate_dart(item: &ParsedItem) -> String`
- Internal helpers:
  - `rust_type_to_dart(ty: &str) -> &str` — maps Rust type strings to Dart types
  - `gen_decode_expr(ty: &str, field_name: &str) -> String` — generates Dart decode code for a type
  - `gen_encode_expr(ty: &str, field_name: &str) -> String` — generates Dart encode code for a type
  - `rust_type_to_imports(ty: &str) -> Vec<String>` — returns any Dart import paths needed for custom types
  - `to_snake_case(name: &str) -> String` — converts PascalCase to snake_case for file names
- For `Vec<T>` detection: normalize the type token stream to string, check if it starts with `Vec <` and extract the inner type
- For custom types (not in the primitive map and not Vec): assume it's another Dart-generated type, add `import '<snake_case>.dart';` and use its `decode`/`_encode` methods
- The parser types (`ParsedItem`, `NamedField`, etc.) must be made `pub(crate)` so `dart.rs` can use them. Currently they're private in `lib.rs`.
- Structs: generate a single Dart class with named constructor, `fromBin`, `decode`, `toBin`, `_encode`
- Enums: generate a Dart `sealed class` with subclasses per variant. Each variant is `EnumNameVariantName extends EnumName`. The sealed base class has the `fromBin`, `decode`, and `toBin` methods. Each subclass implements `_encode`.

**Definition of Done:**

- [ ] `dart.rs` exists with `generate_dart(item: &ParsedItem) -> String`
- [ ] Generates valid Dart for named-field structs (fields, constructor, fromBin, decode, toBin, _encode)
- [ ] Generates valid Dart for enums with unit, tuple, and named-field variants (sealed class + subclasses)
- [ ] Handles all primitive types from the mapping table (bool, u8–u64, i8–i64, f32, f64, String)
- [ ] Handles `Vec<T>` for any T (primitive, String, or custom), including nested `Vec<Vec<T>>`
- [ ] Type string normalization strips whitespace before matching (handles `Vec < u8 >` → `Vec<u8>`)
- [ ] Handles nested custom types with correct imports
- [ ] Tuple variant fields use named parameters with `f0`, `f1`, ... naming convention (consistent with Rust codegen patterns)
- [ ] `to_snake_case` converts PascalCase to snake_case correctly

**Verify:**

- `cargo check -p deli-codec-derive` — compiles without errors

### Task 2: Wire up `#[proc_macro_derive(Dart)]` and re-export

**Objective:** Add the `derive_dart` proc macro entry point that parses the input, calls `generate_dart`, and writes the result to `$DELI_RSTYPES_PATH`. Re-export from `deli-codec`.

**Dependencies:** Task 1

**Files:**

- Modify: `crates/deli-codec-derive/src/lib.rs` — add `#[proc_macro_derive(Dart)]` function, make parser types `pub(crate)`
- Modify: `crates/deli-codec/src/lib.rs` — add `pub use deli_codec_derive::Dart;` re-export

**Key Decisions / Notes:**

- Make parser types (`ParsedItem`, `ItemData`, `FieldsKind`, `NamedField`, `TupleField`, `Variant`) `pub(crate)` so `dart.rs` can access them
- The `derive_dart` function:
  1. Calls `parse_input(input.into())` (same as `derive_codec`)
  2. Calls `dart::generate_dart(&parsed)` to get Dart source
  3. Reads `std::env::var("DELI_RSTYPES_PATH")` — if not set, do nothing (silent skip, since the macro might be used in contexts where Dart gen isn't needed)
  4. Creates the full `DELI_RSTYPES_PATH` directory and any parents using `std::fs::create_dir_all(path)` before writing (handles first-time builds where target dir doesn't exist)
  5. Writes to `{DELI_RSTYPES_PATH}/{snake_case_name}.dart`
  6. Returns an empty `TokenStream` (no Rust code generated — the macro only has a side effect of writing the Dart file)
- Re-export from `deli-codec` so users can `use deli_codec::Dart;` or `#[derive(deli_codec::Dart)]`

**Definition of Done:**

- [ ] `#[proc_macro_derive(Dart)]` exists in `lib.rs`
- [ ] Re-exported from `deli-codec` as `pub use deli_codec_derive::Dart;`
- [ ] Reads `DELI_RSTYPES_PATH` and writes `.dart` file when the env var is set
- [ ] Silently does nothing when `DELI_RSTYPES_PATH` is not set
- [ ] Creates full `DELI_RSTYPES_PATH` directory tree if it doesn't exist (using `create_dir_all`)
- [ ] `cargo check -p deli-codec` succeeds
- [ ] `derive(Dart)` alongside `derive(Codec)` works without `DELI_RSTYPES_PATH` set (no compile errors)

**Verify:**

- `cargo check -p deli-codec` — re-export compiles
- `cargo check -p deli-codec-derive` — proc macro compiles

### Task 3: Add integration tests for struct and enum codegen

**Objective:** Add tests that use `#[derive(Dart)]` on structs and enums, then verify the generated `.dart` files have correct content.

**Dependencies:** Task 2

**Files:**

- Create: `crates/deli-codec/tests/dart_codegen_tests.rs`

**Key Decisions / Notes:**

- Tests set `DELI_RSTYPES_PATH` via a `build.rs` or by relying on the env var being set in the test environment. Since proc macros run at compile time, the env var must be set BEFORE compilation. Use `cargo test` with `DELI_RSTYPES_PATH` env var set.
- Actually, proc macro file writing happens at compile time, not test runtime. So the test file will:
  1. Define types with `#[derive(Codec, Dart)]`
  2. At test runtime, read the generated `.dart` files from the `DELI_RSTYPES_PATH` directory
  3. Assert the files contain expected Dart code patterns (class name, fromBin, toBin, correct field types)
- Test types:
  - `Point { x: f32, y: f32 }` — simple struct with primitives
  - `Message { id: u32, text: String, tags: Vec<String> }` — struct with String and Vec
  - `Matrix { data: Vec<Vec<f32>> }` — nested Vec generic (verifies normalization handles `Vec<Vec<T>>`)
  - `Color { Red, Green, Blue }` — unit enum
  - `Shape { Circle { radius: f32 }, Rectangle { width: f32, height: f32 }, Point }` — mixed enum
- Assertions check:
  - File exists at expected path
  - Contains `class PointName`
  - Contains `factory ClassName.fromBin`
  - Contains `Uint8List toBin()`
  - Contains correct Dart types for fields
  - Enum files contain `sealed class` and subclasses
- The env var for tests: set `DELI_RSTYPES_PATH` to a temp dir or a known test output dir. Can use `env!("OUT_DIR")` via build.rs to set it, or use a `.cargo/config.toml` for testing. Simplest: add a `build.rs` to `deli-codec` that sets `DELI_RSTYPES_PATH` to `OUT_DIR/dart_test_output` during testing.

**Definition of Done:**

- [ ] `dart_codegen_tests.rs` exists with tests for struct and enum code generation
- [ ] Tests verify generated `.dart` files exist and contain correct Dart code:
  - `point.dart` contains `class Point` with `final double x;` and `final double y;`
  - `message.dart` contains `final String text` and `List<String> tags`
  - `matrix.dart` contains `List<List<double>> data` (verifies nested Vec generic handling)
  - `color.dart` contains `sealed class Color` and subclasses `ColorRed`, `ColorGreen`, `ColorBlue`
  - `shape.dart` contains `sealed class Shape`, `class ShapeCircle extends Shape` with `final double radius;`
  - All files contain `factory ClassName.fromBin(Uint8List bytes)` and `Uint8List toBin()`
- [ ] Tests for: simple struct, struct with String/Vec, nested Vec<Vec<T>>, unit enum, mixed variant enum
- [ ] `DELI_RSTYPES_PATH=<dir> cargo test -p deli-codec` passes all tests including Dart codegen tests

**Verify:**

- `DELI_RSTYPES_PATH=/tmp/deli-dart-test cargo test -p deli-codec` — all tests pass
- Inspect generated files in `/tmp/deli-dart-test/` for correctness

## Testing Strategy

- **Unit tests:** `dart.rs` functions are tested indirectly via the integration tests (they're not independently callable from outside the proc-macro crate). Internal `#[cfg(test)]` tests can verify `to_snake_case`, type string normalization (e.g., `"Vec < Vec < u8 > >"` → `"Vec<Vec<u8>>"`), and type mapping helpers (`rust_type_to_dart`).
- **Integration tests:** `deli-codec/tests/dart_codegen_tests.rs` uses `#[derive(Dart)]` on various types, then reads the generated `.dart` files to verify content.
- **Manual verification (post-implementation, requires Dart SDK):** Generate Dart files, then test roundtrip binary compatibility in a Dart project: (1) Rust struct encodes via `Codec::to_bytes()`, (2) Dart `fromBin()` decodes the same bytes and produces matching values, (3) Dart `toBin()` re-encodes and Rust `Codec::from_bytes()` decodes successfully. This verifies the wire format is truly compatible. Not part of CI — performed manually when Dart SDK is available.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Token stream to string conversion loses type info (spacing, generics) | Med | High | Normalize whitespace in type strings before matching. Test with `Vec<String>`, `Vec<Vec<u8>>` and other generic types. |
| Proc macro file I/O fails (permissions, missing dir) | Low | Med | Use `create_dir_all` before writing. If write fails, `panic!` with the path and error so the user gets a clear compile error. |
| Dart sealed class pattern may not work on older Dart SDKs | Low | Med | `pubspec.yaml` already requires `sdk: ">=3.0.0"` which supports sealed classes and records. |
| Nested type imports may have circular dependencies | Low | Med | Each file imports only what it needs. Dart handles circular imports at the package level. |

## Open Questions

- None — design decisions resolved.

### Deferred Ideas

- `Option<T>` / nullable field support (Dart nullable types)
- Tuple struct Dart representation (positional constructors)
- Auto-update barrel file (`rstypes.dart`) with exports
- Generate Dart unit tests alongside the Dart classes
- Support for `HashMap<K, V>` / `Map<K, V>` types
