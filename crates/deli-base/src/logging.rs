use log::{Log, LevelFilter, Metadata, Record};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// A logger that writes to stdout using println!
pub struct StdoutLogger;

/// A logger that writes to date-named files with automatic day rollover
pub struct FileLogger {
    state: Mutex<FileLoggerState>,
}

struct FileLoggerState {
    dir: PathBuf,
    current_date: String,
    file: File,
}

impl FileLogger {
    /// Create a new FileLogger that writes to the specified directory
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;

        let current_date = format_today();
        let file_path = dir.join(format!("{}.log", current_date));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)?;

        Ok(FileLogger {
            state: Mutex::new(FileLoggerState {
                dir,
                current_date,
                file,
            }),
        })
    }
}

impl Log for StdoutLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        let timestamp = format_timestamp();
        let level = record.level();
        let thread_id = std::thread::current().id();
        let file = record.file().unwrap_or("unknown");
        let line = record.line().unwrap_or(0);
        let message = record.args();

        println!("{} [{}] [thread:{:?}] {}:{} - {}", timestamp, level, thread_id, file, line, message);
    }

    fn flush(&self) {
        std::io::stdout().flush().ok();
    }
}

impl Log for FileLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        // Acquire mutex with poisoning recovery
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        // Check for day rollover
        let today = format_today();
        if today != state.current_date {
            // Day has changed, need to roll over to new file
            let new_file_path = state.dir.join(format!("{}.log", today));
            match OpenOptions::new()
                .create(true)
                .append(true)
                .open(&new_file_path)
            {
                Ok(new_file) => {
                    state.file = new_file;
                    state.current_date = today;
                }
                Err(e) => {
                    eprintln!("Failed to open new log file {:?}: {}", new_file_path, e);
                    // Continue using old file
                }
            }
        }

        // Format log message (same format as StdoutLogger)
        let timestamp = format_timestamp();
        let level = record.level();
        let thread_id = std::thread::current().id();
        let file = record.file().unwrap_or("unknown");
        let line = record.line().unwrap_or(0);
        let message = record.args();

        let log_line = format!(
            "{} [{}] [thread:{:?}] {}:{} - {}\n",
            timestamp, level, thread_id, file, line, message
        );

        // Write to file, fall back to eprintln if it fails
        if let Err(e) = state.file.write_all(log_line.as_bytes()) {
            eprintln!("Failed to write to log file: {}", e);
            eprintln!("{}", log_line.trim_end());
        }
    }

    fn flush(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.file.flush().ok();
    }
}

/// Format current time as YYYY-MM-DDTHH:MM:SS (UTC)
pub fn format_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let days = secs / 86400;
    let time_of_day = secs % 86400;

    let (year, month, day) = civil_from_days(days as i64);

    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", year, month, day, hours, minutes, seconds)
}

/// Format current date as YYYY-MM-DD (UTC)
pub fn format_today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let days = secs / 86400;
    let (year, month, day) = civil_from_days(days as i64);

    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// Convert days since Unix epoch to civil date (year, month, day)
/// Uses Howard Hinnant's algorithm (public domain)
/// http://howardhinnant.github.io/date_algorithms.html
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Initialize the global logger with StdoutLogger
///
/// Sets the max level based on build mode:
/// - Debug builds: LevelFilter::Debug (all levels active)
/// - Release builds: LevelFilter::Info (Debug suppressed)
///
/// This can only be called once per process. Subsequent calls are silently ignored.
pub fn init_stdout_logger() {
    static LOGGER: StdoutLogger = StdoutLogger;

    let max_level = if cfg!(debug_assertions) {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(max_level);
    }
}

/// Initialize the global logger with FileLogger
///
/// Sets the max level based on build mode:
/// - Debug builds: LevelFilter::Debug (all levels active)
/// - Release builds: LevelFilter::Info (Debug suppressed)
///
/// This can only be called once per process. Subsequent calls are silently ignored.
///
/// Returns an error if the FileLogger cannot be created (e.g., invalid directory).
pub fn init_file_logger(dir: impl Into<PathBuf>) -> std::io::Result<()> {
    let logger = FileLogger::new(dir)?;

    let max_level = if cfg!(debug_assertions) {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    // Box::leak is required for the &'static reference that set_logger needs.
    // If set_logger fails (logger already set), the leaked FileLogger cannot be
    // reclaimed, but this is a one-time init that should only be called once.
    if log::set_logger(Box::leak(Box::new(logger))).is_ok() {
        log::set_max_level(max_level);
    }

    Ok(())
}

/// Log a fatal error and exit the process
///
/// Logs at Error level (since the log crate has no Fatal level),
/// flushes stdout, and calls std::process::exit(1).
#[macro_export]
macro_rules! log_fatal {
    ($($arg:tt)*) => {{
        log::error!($($arg)*);
        // Flush stdout to ensure message is visible
        {
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        std::process::exit(1);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_civil_from_days_epoch() {
        // 0 days since epoch = 1970-01-01
        let (y, m, d) = civil_from_days(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn test_civil_from_days_leap_year() {
        // 2000-02-29 (leap year)
        let days = 11016; // Days from 1970-01-01 to 2000-02-29
        let (y, m, d) = civil_from_days(days);
        assert_eq!((y, m, d), (2000, 2, 29));
    }

    #[test]
    fn test_civil_from_days_year_boundary() {
        // 2024-12-31
        let days = 20088; // Days from 1970-01-01 to 2024-12-31
        let (y, m, d) = civil_from_days(days);
        assert_eq!((y, m, d), (2024, 12, 31));
    }

    #[test]
    fn test_format_timestamp_structure() {
        let ts = format_timestamp();
        // Should be in format YYYY-MM-DDTHH:MM:SS
        assert_eq!(ts.len(), 19);
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
    }

    #[test]
    fn test_file_logger_day_rollover() {
        let test_dir = std::env::temp_dir()
            .join(format!("deli-log-test-{}-rollover", std::process::id()));

        let _ = fs::remove_dir_all(&test_dir);

        let logger = FileLogger::new(&test_dir).expect("Failed to create FileLogger");

        // Simulate the logger was created "yesterday" by replacing the state
        // with a fake past date and a file handle pointing to that date's file
        let fake_date = "1999-01-01".to_string();
        let fake_file_path = test_dir.join("1999-01-01.log");
        let fake_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&fake_file_path)
            .expect("Failed to create fake old file");
        {
            let mut state = logger.state.lock().unwrap();
            state.current_date = fake_date;
            state.file = fake_file;
        }

        // Write a message to the "old" file (1999-01-01.log)
        let record = log::RecordBuilder::new()
            .level(log::Level::Info)
            .target("test")
            .file(Some("test.rs"))
            .line(Some(1))
            .args(format_args!("before rollover"))
            .build();
        logger.log(&record);

        // Verify message went to the old date file (no rollover yet,
        // because format_today() != "1999-01-01" triggers rollover on this call)
        // Actually, the first log() call already sees the date mismatch and rolls over.
        // So "before rollover" goes to today's file, and the old file stays empty or
        // has just the initial open.

        // Let's verify: two files should exist now
        let today = format_today();
        let today_file = test_dir.join(format!("{}.log", today));

        assert!(fake_file_path.exists(), "Old date file should exist");
        assert!(today_file.exists(), "Today's date file should exist after rollover");

        // Today's file should have the message (rollover happened before writing)
        let today_content = fs::read_to_string(&today_file).expect("Failed to read today file");
        assert!(today_content.contains("before rollover"), "Today file should contain the message");

        // Verify the state was updated to today's date
        {
            let state = logger.state.lock().unwrap();
            assert_eq!(state.current_date, today, "current_date should be updated to today");
        }

        // Clean up
        fs::remove_dir_all(&test_dir).ok();
    }
}
