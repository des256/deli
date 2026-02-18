use crate::AudioError;
use libpulse_binding as pulse;
use libpulse_binding::callbacks::ListResult;
use libpulse_binding::context::{Context, State};
use libpulse_binding::mainloop::standard::{IterateResult, Mainloop};
use std::cell::RefCell;
use std::rc::Rc;

/// Represents an audio input device.
#[derive(Clone, Debug)]
pub struct AudioDevice {
    pub name: String,
    pub description: String,
}

/// List available PulseAudio input devices.
///
/// Creates a temporary PulseAudio mainloop and context to enumerate audio sources.
/// Filters out monitor sources (which capture output audio).
///
/// # Errors
///
/// Returns `AudioError::Device` if:
/// - PulseAudio server is unavailable
/// - Connection times out
/// - Context enters Failed or Terminated state
pub fn list_devices() -> Result<Vec<AudioDevice>, AudioError> {
    let mut mainloop = Mainloop::new()
        .ok_or_else(|| AudioError::Device("Failed to create PulseAudio mainloop".to_string()))?;

    let mut context = Context::new(&mainloop, "deli-audio")
        .ok_or_else(|| AudioError::Device("Failed to create PulseAudio context".to_string()))?;

    context
        .connect(None, pulse::context::FlagSet::NOFLAGS, None)
        .map_err(|e| AudioError::Device(format!("Failed to connect to PulseAudio server: {}", e)))?;

    // Wait for connection to be ready (with timeout)
    let max_iterations = 100;
    let mut iterations = 0;

    loop {
        match mainloop.iterate(true) {
            IterateResult::Quit(_) | IterateResult::Err(_) => {
                return Err(AudioError::Device(
                    "PulseAudio mainloop error during connection".to_string(),
                ));
            }
            IterateResult::Success(_) => {}
        }
        iterations += 1;

        match context.get_state() {
            State::Ready => break,
            State::Failed | State::Terminated => {
                return Err(AudioError::Device(
                    "PulseAudio connection failed or terminated".to_string(),
                ));
            }
            _ => {
                if iterations >= max_iterations {
                    return Err(AudioError::Device(
                        "PulseAudio server unavailable or connection timed out".to_string(),
                    ));
                }
            }
        }
    }

    // Collect devices using introspect API
    let devices = Rc::new(RefCell::new(Vec::new()));
    let devices_clone = Rc::clone(&devices);

    let introspect = context.introspect();
    let op = introspect.get_source_info_list(move |list_result| {
        if let ListResult::Item(source_info) = list_result {
            // Filter out monitor sources
            if source_info.monitor_of_sink.is_none() {
                if let (Some(name), Some(desc)) = (&source_info.name, &source_info.description) {
                    devices_clone.borrow_mut().push(AudioDevice {
                        name: name.to_string(),
                        description: desc.to_string(),
                    });
                }
            }
        }
    });

    // Iterate mainloop until operation completes
    loop {
        match mainloop.iterate(true) {
            IterateResult::Quit(_) | IterateResult::Err(_) => {
                return Err(AudioError::Device(
                    "Mainloop error during device enumeration".to_string(),
                ));
            }
            IterateResult::Success(_) => {}
        }
        match op.get_state() {
            pulse::operation::State::Done => break,
            pulse::operation::State::Cancelled => {
                return Err(AudioError::Device(
                    "Device enumeration cancelled".to_string(),
                ));
            }
            pulse::operation::State::Running => {}
        }
    }

    let result = devices.borrow().clone();
    Ok(result)
}
