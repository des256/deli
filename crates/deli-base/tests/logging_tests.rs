use deli_base::logging::{StdoutLogger, FileLogger, init_stdout_logger, init_file_logger};
use log::Log;
use std::fs;

#[test]
fn test_stdout_logger_implements_log_trait() {
    let logger = StdoutLogger;

    // Verify Log trait methods can be called
    let metadata = log::MetadataBuilder::new()
        .level(log::Level::Info)
        .target("test")
        .build();

    assert!(logger.enabled(&metadata));

    // Create a test record
    let record = log::RecordBuilder::new()
        .level(log::Level::Info)
        .target("test")
        .file(Some("test.rs"))
        .line(Some(42))
        .args(format_args!("test message"))
        .build();

    // This should not panic
    logger.log(&record);
    logger.flush();
}

#[test]
fn test_file_logger_creates_directory() {
    let test_dir = std::env::temp_dir().join(format!("deli-log-test-{}-dir", std::process::id()));

    // Clean up if exists from previous run
    let _ = fs::remove_dir_all(&test_dir);

    let _logger = FileLogger::new(&test_dir).expect("Failed to create FileLogger");

    // Directory should exist
    assert!(test_dir.exists());
    assert!(test_dir.is_dir());

    // Clean up
    fs::remove_dir_all(&test_dir).ok();
}

#[test]
fn test_file_logger_writes_to_file() {
    let test_dir = std::env::temp_dir().join(format!("deli-log-test-{}-write", std::process::id()));

    // Clean up if exists
    let _ = fs::remove_dir_all(&test_dir);

    let logger = FileLogger::new(&test_dir).expect("Failed to create FileLogger");

    // Create a test record
    let record = log::RecordBuilder::new()
        .level(log::Level::Error)
        .target("test")
        .file(Some("test.rs"))
        .line(Some(100))
        .args(format_args!("test error message"))
        .build();

    logger.log(&record);
    logger.flush();

    // Find the log file (should be named YYYY-MM-DD.log)
    let entries: Vec<_> = fs::read_dir(&test_dir)
        .expect("Failed to read test directory")
        .filter_map(|e| e.ok())
        .collect();

    assert_eq!(entries.len(), 1, "Should have exactly one log file");

    let log_file = &entries[0].path();
    let content = fs::read_to_string(log_file).expect("Failed to read log file");

    // Verify log content contains expected parts
    assert!(content.contains("[ERROR]"), "Should contain log level");
    assert!(content.contains("thread:"), "Should contain thread ID");
    assert!(content.contains("test.rs:100"), "Should contain file and line");
    assert!(content.contains("test error message"), "Should contain message");

    // Clean up
    fs::remove_dir_all(&test_dir).ok();
}

#[test]
fn test_file_logger_implements_log_trait() {
    let test_dir = std::env::temp_dir().join(format!("deli-log-test-{}-trait", std::process::id()));

    let _ = fs::remove_dir_all(&test_dir);

    let logger = FileLogger::new(&test_dir).expect("Failed to create FileLogger");

    let metadata = log::MetadataBuilder::new()
        .level(log::Level::Info)
        .target("test")
        .build();

    assert!(logger.enabled(&metadata));

    // Clean up
    fs::remove_dir_all(&test_dir).ok();
}

#[test]
fn test_init_stdout_logger_sets_global_logger() {
    // This test can only run once per process since log::set_logger can only be called once
    // If it's already initialized, this is a no-op
    init_stdout_logger();

    // Verify the global logger is set by checking that log::logger() returns a valid logger
    let logger = log::logger();
    assert!(logger.enabled(&log::MetadataBuilder::new()
        .level(log::Level::Info)
        .target("test")
        .build()));

    // Verify we can log through the global interface
    log::info!("Test message from global logger");
}

#[test]
fn test_init_file_logger_invalid_dir_returns_error() {
    let result = init_file_logger("/proc/nonexistent/path");
    assert!(result.is_err(), "init_file_logger with invalid path should return Err");
}
