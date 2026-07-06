import serial
import serial.tools.list_ports
import time

ports = serial.tools.list_ports.comports()
pico_port = None

for port in ports:
    print(f"Found: {port.device} - {port.description}")
    if 'Pico' in port.description or 'RP2040' in port.description:
        pico_port = port.device
        break

if not pico_port:
    print("Pico not found!")
else:
    print(f"Connecting to {pico_port}...")
    ser = serial.Serial(pico_port, 115200, timeout=2)
    time.sleep(2)
    
    if ser.in_waiting:
        msg = ser.readline().decode()
        print(f"Pico says: {msg}")
    
    print("Sending LED_1...")
    ser.write(b'LED_1\n')
    time.sleep(1)
    
    if ser.in_waiting:
        print(f"Response: {ser.readline().decode()}")
    
    ser.close()
    print("Done!")
