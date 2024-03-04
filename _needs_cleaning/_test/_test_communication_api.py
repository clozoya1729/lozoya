import threading

import client
import configuration
import serial

place = 55

cubeSat = serial.Serial('COM5', 115200)


def cubeRCallback(*args):
    msg, timeMS = args
    if msg:
        try:
            cubeSat.write(msg)
        except Exception as e:
            return str(e)
        try:
            x = cubeSat.readline()
            y = cubeSat.readline()
            z = cubeSat.readline()
            return '{},{},{}'.format(x, y, z)
        except Exception as e:
            return str(e)
    return 'ERROR'


username, clientSocket = client.initialize(ip=configuration.ip, port=1234)
x = threading.Thread(target=client.main_loop,
                     args=(username, clientSocket),
                     kwargs={
                         'rCallback': cubeRCallback,
                     }
                     )
x.start()
cubeSat.close()
ip = "129.108.152.32"

# from satellite2.thruster import configurations
# from communication.internet import client
#
#
# callback = None
# IP = "129.108.152.70"
# username, clientSocket = client.initialize(ip=variables.ip, port=1234)
# ## client.main_loop(username, clientSocket, rCallback=plotter)
#
# x = threading.Thread(target=client.main_loop, args=(username, clientSocket), kwargs={'rCallback': plotter})
# x.start()
#
