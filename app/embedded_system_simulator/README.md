# Embedded System Simulator

This app simulates actuators, sensors, and serial communication.

## Components

### Actuator

The actuator can connect to pins and its output is computed by a user-specified function.

### Sensor

The sensor can connect to pins and its output is computed by a user-specified function.
The update function is called every iteration.

### Serial Communication

- Controller Area Network (CAN)
- Inter-Integrated Communication (I2C)
- Serial Peripheral Interface (SPI)
- Universal Asynchronous Receiver/Transceiver (UART)

## Simulation

### Main Loop

The main loop calls the user-specified update function of each component.

    def main():
        while (simulating):
            for sensor in sensors:
                sensor.update()
            for actuator in actuators:
                actuator.update()
            for serial in serials:
                serial.update()