from machine import Pin, UART
import time

# Initialize GPIO pin 15 as output
led = Pin(15, Pin.OUT)

# Initialize UART0 for serial communication (TX=GP0, RX=GP1)
uart = UART(0, baudrate=115200, timeout=100)

LED_FLASH_DURATION = 0.2  # Match the Windows experiment setting

def flash_led():
    """Flash the LED for LED_FLASH_DURATION"""
    led.on()
    time.sleep(LED_FLASH_DURATION)
    led.off()

def process_command(command):
    """Process LED commands from Windows"""
    command = command.strip()
    print(f"Received: {command}")

    if command == "LED_1" or command == "LED_2":
        print(f"Flashing LED for {LED_FLASH_DURATION}s")
        flash_led()
        uart.write(b"DONE\n")  # Send confirmation back to Windows
    else:
        print(f"Unknown command: {command}")

# Main loop: listen for serial commands
print("Pico ready - waiting for commands...")
uart.write(b"PICO_READY\n")

while True:
    if uart.any():  # Check if data is available
        try:
            data = uart.readline()
            if data:
                command = data.decode('utf-8', errors='ignore')
                process_command(command)
        except Exception as e:
            print(f"Error: {e}")

    time.sleep(0.01)  # Small delay to prevent blocking