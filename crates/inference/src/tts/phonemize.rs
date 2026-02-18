use std::ffi::{c_char, CStr, CString};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

// Thread safety for espeak-ng global state
static ESPEAK_MUTEX: Mutex<()> = Mutex::new(());
static ESPEAK_INITIALIZED: AtomicBool = AtomicBool::new(false);

// Clause terminator constants from espeak-ng with proper bit masks
const CLAUSE_INTONATION_FULL_STOP: i32 = 0x00000000;
const CLAUSE_INTONATION_COMMA: i32 = 0x00001000;
const CLAUSE_INTONATION_QUESTION: i32 = 0x00002000;
const CLAUSE_INTONATION_EXCLAMATION: i32 = 0x00003000;
const CLAUSE_TYPE_CLAUSE: i32 = 0x00040000;
const CLAUSE_TYPE_SENTENCE: i32 = 0x00080000;

const CLAUSE_PERIOD: i32 = 40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE;
const CLAUSE_COMMA: i32 = 20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE;
const CLAUSE_QUESTION: i32 = 40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE;
const CLAUSE_EXCLAMATION: i32 = 45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE;
const CLAUSE_COLON: i32 = 30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE;
const CLAUSE_SEMICOLON: i32 = 30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE;

// FFI declarations for espeak-ng
unsafe extern "C" {
    fn espeak_Initialize(output: i32, buflength: i32, path: *const c_char, options: i32) -> i32;
    fn espeak_SetVoiceByName(name: *const c_char) -> i32;
    fn espeak_TextToPhonemesWithTerminator(
        textptr: *mut *const c_char,
        textmode: i32,
        phonememode: i32,
        terminator: *mut i32,
    ) -> *const c_char;
    #[allow(dead_code)]
    fn espeak_Terminate() -> i32;
}

/// Initialize espeak-ng library
///
/// # Arguments
/// * `data_path` - Path to espeak-ng data directory (None uses default)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(String)` if initialization fails
pub fn espeak_init(data_path: Option<&str>) -> Result<(), String> {
    let _guard = ESPEAK_MUTEX.lock().unwrap();

    if ESPEAK_INITIALIZED.load(Ordering::Acquire) {
        return Ok(());
    }

    unsafe {
        let path_cstring;
        let path_ptr = if let Some(path) = data_path {
            path_cstring = CString::new(path).map_err(|e| format!("Invalid data path: {}", e))?;
            path_cstring.as_ptr()
        } else {
            std::ptr::null()
        };

        // AUDIO_OUTPUT_SYNCHRONOUS = 2, buflength = 0 (not used), options = 0
        let result = espeak_Initialize(2, 0, path_ptr, 0);
        if result < 0 {
            return Err(format!("espeak_Initialize failed with code {}", result));
        }

        // Set voice to en-us
        let voice = CString::new("en-us").unwrap();
        let voice_result = espeak_SetVoiceByName(voice.as_ptr());
        if voice_result != 0 {
            return Err(format!("espeak_SetVoiceByName failed with code {}", voice_result));
        }
    }

    ESPEAK_INITIALIZED.store(true, Ordering::Release);
    Ok(())
}

/// Convert text to IPA phonemes using espeak-ng
///
/// # Arguments
/// * `text` - Input text to phonemize
///
/// # Returns
/// * `Ok(String)` containing IPA phonemes
/// * `Err(String)` if phonemization fails
pub fn phonemize(text: &str) -> Result<String, String> {
    let _guard = ESPEAK_MUTEX.lock().unwrap();

    if !ESPEAK_INITIALIZED.load(Ordering::Acquire) {
        return Err("espeak-ng not initialized. Call espeak_init first.".to_string());
    }

    unsafe {
        let text_cstr = CString::new(text).map_err(|e| format!("Invalid text: {}", e))?;
        let mut text_ptr: *const c_char = text_cstr.as_ptr();
        let mut result = String::new();

        // Loop until text_ptr is exhausted
        loop {
            let mut terminator: i32 = 0;

            // textmode = 0 (normal text), phonememode = 2 (IPA with ties)
            let phonemes_ptr = espeak_TextToPhonemesWithTerminator(
                &mut text_ptr,
                0,
                2,
                &mut terminator
            );

            if phonemes_ptr.is_null() {
                break;
            }

            let phonemes_cstr = CStr::from_ptr(phonemes_ptr);
            let phonemes_str = phonemes_cstr.to_str()
                .map_err(|e| format!("Invalid UTF-8 in phonemes: {}", e))?;

            result.push_str(phonemes_str);

            // Append punctuation based on terminator value (mask to get clause type)
            match terminator & 0x000FFFFF {
                CLAUSE_PERIOD => result.push('.'),
                CLAUSE_QUESTION => result.push('?'),
                CLAUSE_EXCLAMATION => result.push('!'),
                CLAUSE_COMMA => result.push_str(", "),
                CLAUSE_COLON => result.push_str(": "),
                CLAUSE_SEMICOLON => result.push_str("; "),
                _ => {}
            }

            // Check if we've consumed all text
            if text_ptr.is_null() || *text_ptr == 0 {
                break;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ensure_init() {
        espeak_init(Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data")).unwrap();
    }

    #[test]
    fn test_espeak_init_success() {
        let result = espeak_init(Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_espeak_init_multiple_calls() {
        let result1 = espeak_init(Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data"));
        let result2 = espeak_init(Some("/usr/lib/x86_64-linux-gnu/espeak-ng-data"));
        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[test]
    fn test_phonemize_simple_text() {
        ensure_init();
        let result = phonemize("hello");
        assert!(result.is_ok(), "phonemize should succeed");
        let phonemes = result.unwrap();
        assert!(!phonemes.is_empty(), "phonemes should not be empty");
    }

    #[test]
    fn test_phonemize_with_punctuation() {
        ensure_init();
        let result = phonemize("Hello. How are you?");
        assert!(result.is_ok());
        let phonemes = result.unwrap();
        assert!(
            phonemes.contains('.') || phonemes.contains('?'),
            "phonemes should contain punctuation: {}",
            phonemes
        );
    }

    #[test]
    fn test_phonemize_empty_text() {
        ensure_init();
        let result = phonemize("");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn test_phonemize_multiple_clauses() {
        ensure_init();
        let result = phonemize("Hello! How are you? I am fine.");
        assert!(result.is_ok());
        let phonemes = result.unwrap();
        assert!(!phonemes.is_empty());
    }
}
