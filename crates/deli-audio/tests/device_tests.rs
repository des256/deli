use deli_audio::{list_devices, AudioDevice, AudioError};

#[test]
fn test_audio_device_construction() {
    let device = AudioDevice {
        name: "test_device".to_string(),
        description: "Test Audio Device".to_string(),
    };

    assert_eq!(device.name, "test_device");
    assert_eq!(device.description, "Test Audio Device");
}

#[test]
fn test_audio_device_fields() {
    let device = AudioDevice {
        name: "alsa_input.pci-0000_00_1f.3.analog-stereo".to_string(),
        description: "Built-in Audio Analog Stereo".to_string(),
    };

    assert!(!device.name.is_empty());
    assert!(!device.description.is_empty());
}

#[test]
fn test_list_devices_signature() {
    // This test verifies the function signature returns Result<Vec<AudioDevice>, AudioError>
    // Actual device list depends on hardware, so we only check the return type
    let result = list_devices();

    // Should return Ok or Err, both are valid (depends on PulseAudio availability)
    match result {
        Ok(_devices) => {
            // If PulseAudio is available, we should get a Vec (possibly empty)
            // Test passes - we got a valid result
        }
        Err(e) => {
            // If PulseAudio is unavailable, we should get an AudioError::Device
            match e {
                AudioError::Device(_) => {}
                _ => panic!("Expected AudioError::Device for connection failure"),
            }
        }
    }
}
