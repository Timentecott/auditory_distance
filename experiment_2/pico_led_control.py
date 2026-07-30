"""
MicroPython script for Raspberry Pi Pico.
Receives serial commands via USB to control LEDs on pins 15 and 16.
"""

from machine import Pin
import sys
import time

# Setup
LED_15 = Pin(15, Pin.OUT)
LED_16 = Pin(16, Pin.OUT)

def led_15_on():
    """Turn LED on pin 15 on"""
    LED_15.on()
    print("LED 15 ON")

def led_15_off():
    """Turn LED on pin 15 off"""
    LED_15.off()
    print("LED 15 OFF")

def led_16_on():
    """Turn LED on pin 16 on"""
    LED_16.on()
    print("LED 16 ON")

def led_16_off():
    """Turn LED on pin 16 off"""
    LED_16.off()
    print("LED 16 OFF")

def startup_blink():
    """Blink both LEDs to show code is running"""
    for _ in range(3):
        LED_15.on()
        LED_16.on()
        time.sleep(0.2)
        LED_15.off()
        LED_16.off()
        time.sleep(0.2)

def main():
    """
    Main loop: listen for serial commands on USB and control LEDs.
    Commands:
        '1' = Pin 15 LED ON
        '0' = Pin 15 LED OFF
        '3' = Pin 16 LED ON
        '2' = Pin 16 LED OFF
    """
    print("Pico starting up...")
    startup_blink()  # Let user know code is running
    print("Pico ready, waiting for commands...")
    print("Commands: 1=LED15 ON, 0=LED15 OFF, 3=LED16 ON, 2=LED16 OFF")

    while True:
        try:
            # This will block and wait for input from serial
            data = sys.stdin.read(1)

            if data == '1':
                led_15_on()
            elif data == '0':
                led_15_off()
            elif data == '3':
                led_16_on()
            elif data == '2':
                led_16_off()

        except KeyboardInterrupt:
            LED_15.off()
            LED_16.off()
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
