class Actuator:
    def __init__(self, transfer_function):
        self.transfer_function = transfer_function
        self.voltage = 0

    def apply_voltage(self, voltage):
        self.voltage = voltage

    def output(self):
        self.transfer_function(self.voltage)


class Communicator:
    def __init__(self):
        self.messageSend = ''
        self.messageReceive = ''
        self.baud = 9600

    def transmit(self):
        pass

    def receive(self):
        pass


class Sensor:
    def __init__(self, initialValue: float, operatingRange: list, update_function):
        self.operatingRange = operatingRange
        self.update_function = update_function
        self.value = initialValue

    def read(self):
        return self.value

    def update(self):
        self.value = self.update_function(self.value)
