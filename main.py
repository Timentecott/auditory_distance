"""
Raspberry Pi Pico Firmware for LED Control
Listens for serial commands and controls LEDs on GPIO 15 and 16.

Command format:
  LED_1 (near location)
  LED_2 (far location)

Sends back: DONE when flash completes
"""

from machine import Pin, UART
import sys
import time

# Configuration
LED_FLASH_DURATION = 0.2  # seconds (200ms)

# Use UART(0) on GP0/GP1 for serial communication with PC
uart = UART(0, 115200)

# Initialize LED pins
led_1 = Pin(15, Pin.OUT)  # Near location
led_2 = Pin(16, Pin.OUT)  # Far location

# Turn off LEDs on startup
led_1.off()
led_2.off()

print("Pico LED Controller started", flush=True)
print("Commands: LED_1 (near), LED_2 (far)", flush=True)

def flash_led(led_pin, duration=LED_FLASH_DURATION):
    """Flash an LED for the specified duration."""
    led_pin.on()
    time.sleep(duration)
    led_pin.off()
    uart.write(b"DONE\n")
    print(f"LED flash complete", flush=True)

# Main loop
while True:
    try:
        # Check if data is available on serial
        if uart.any():
            # Read the command
            command = uart.readline()

            # Decode and strip whitespace
            cmd_str = command.decode().strip()
            print(f"Received: {cmd_str}", flush=True)

            if cmd_str == "LED_1":
                print("Flashing near LED (GP15)", flush=True)
                flash_led(led_1)
            elif cmd_str == "LED_2":
                print("Flashing far LED (GP16)", flush=True)
                flash_led(led_2)
            else:
                print(f"Unknown command: {cmd_str}", flush=True)
                uart.write(b"ERROR\n")
    except Exception as e:
        print(f"Error: {e}", flush=True)

    time.sleep(0.01)  # Small delay to prevent CPU spinning
