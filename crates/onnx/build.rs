use std::{env, path::PathBuf};

fn main() {
    // Always link against onnxruntime
    println!("cargo:rustc-link-lib=onnxruntime");

    // Try to find the library in this order:
    // 1. ONNXRUNTIME_DIR/lib
    // 2. ONNXRUNTIME_LIB_DIR
    // 3. pkg-config
    // 4. Common system paths

    if let Ok(dir) = env::var("ONNXRUNTIME_DIR") {
        let lib_path = PathBuf::from(dir).join("lib");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:warning=Using ONNXRUNTIME_DIR: {}", lib_path.display());
        return;
    }

    if let Ok(lib_dir) = env::var("ONNXRUNTIME_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:warning=Using ONNXRUNTIME_LIB_DIR: {}", lib_dir);
        return;
    }

    // Try pkg-config
    if let Ok(output) = std::process::Command::new("pkg-config")
        .args(&["--libs-only-L", "libonnxruntime"])
        .output()
    {
        if output.status.success() {
            if let Ok(libs) = String::from_utf8(output.stdout) {
                for lib in libs.trim().split_whitespace() {
                    if let Some(path) = lib.strip_prefix("-L") {
                        println!("cargo:rustc-link-search=native={}", path);
                        println!("cargo:warning=Using pkg-config path: {}", path);
                        return;
                    }
                }
            }
        }
    }

    // Try common system paths
    let system_paths = [
        "/usr/local/lib",
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ];

    for path in &system_paths {
        let lib_file = PathBuf::from(path).join("libonnxruntime.so");
        if lib_file.exists() {
            println!("cargo:rustc-link-search=native={}", path);
            println!("cargo:warning=Found libonnxruntime.so in system path: {}", path);
            return;
        }
    }

    println!("cargo:warning=Could not find libonnxruntime.so. Set ONNXRUNTIME_DIR or ONNXRUNTIME_LIB_DIR environment variable.");
}
