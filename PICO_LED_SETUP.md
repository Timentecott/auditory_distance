# Pico LED Control Setup for Experiment

## Overview
The `experiment_with_arduino.py` script has been updated to work with your Raspberry Pi Pico LED control system.

## Pin Configuration
- **Pin 15 (GPIO 15)**: Near LED
- **Pin 16 (GPIO 16)**: Far LED

## Serial Command Protocol
The Pico listens for single-character commands:

| Command | Action |
|---------|--------|
| '1' | Pin 15 (Near) ON |
| '0' | Pin 15 (Near) OFF |
| '3' | Pin 16 (Far) ON |
| '2' | Pin 16 (Far) OFF |

## Files Updated

### `experiment_2/experiment_with_arduino.py`
Modified the following functions:
- `send_led_command(location)` - Sends LED ON command based on location (near/far)
- `send_led_off_command(location)` - New function to send LED OFF command
- `wait_for_led_complete()` - Updated to wait for LED_FLASH_DURATION instead of waiting for serial response
- `trigger_led_flash(location)` - Updated to turn off LED after flash duration

### `pico_led_control.py`
Running on the Pico, handles the serial commands and controls GPIO pins.

## Setup Instructions

### 1. Prepare the Pico
```bash
# In Thonny IDE:
# 1. Open pico_led_control.py
# 2. File > Save As > Save to Pico as main.py
# 3. Close Thonny
```

### 2. Run the Experiment
```bash
python experiment_2/experiment_with_arduino.py
```

The script will:
1. Auto-detect the Pico on COM3 (or find it in available ports)
2. When a trial triggers an LED flash for a location:
   - Send the ON command ('1' for near, '3' for far)
   - Wait for LED_FLASH_DURATION seconds
   - Send the OFF command ('0' for near, '2' for far)

## Configuration
Key parameters in `experiment_with_arduino.py`:
- `PICO_BAUD_RATE = 115200` - Serial communication speed
- `LED_FLASH_DURATION = 0.2` - Duration of LED flash in seconds

## Troubleshooting
- **LED doesn't flash**: Ensure `main.py` is saved on the Pico and Thonny is closed
- **Serial connection fails**: Check Device Manager for COM port, or run `python -m serial.tools.list_ports`
- **Commands not received**: Verify the Pico code shows "LED XX ON/OFF" messages in Thonny while connected

## Testing
Use the provided `test_pico_debug.py` script to test individual LED commands:
```bash
# Make sure Thonny is CLOSED
python test_pico_debug.py
```
