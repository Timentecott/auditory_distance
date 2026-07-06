"""
Debugging script to test Pico LED control.
Helps identify connection and communication issues.
"""

import serial
import time

def test_pico_connection(port='COM3', baud_rate=115200):
    """
    Test serial connection to Pico and LED control.
    """
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)
        print(f"✓ Connected to Pico on {port}")

        # Test LED 15 ON
        print("\nSending LED 15 ON command...")
        ser.write(b'1')
        time.sleep(1)

        # Test LED 15 OFF
        print("Sending LED 15 OFF command...")
        ser.write(b'0')
        time.sleep(1)

        # Test LED 16 ON
        print("Sending LED 16 ON command...")
        ser.write(b'3')
        time.sleep(1)

        # Test LED 16 OFF
        print("Sending LED 16 OFF command...")
        ser.write(b'2')
        time.sleep(1)

        print("\n✓ Test complete. Did you see the LED turn on then off?")

        ser.close()

    except serial.SerialException as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check the Pico is connected via USB")
        print("2. Find your port with: python -m serial.tools.list_ports")
        print("3. Update the 'port' variable with the correct COM port")

if __name__ == '__main__':
    # Find available ports
    print("Available serial ports:")
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"  {port.device} - {port.description}")

    print("\n" + "="*50)
    print("Testing connection...")
    print("="*50)

    test_pico_connection(port='COM3')  # Change to your port
