import errno
import glob
import socket
import sys
import threading
import time
from itertools import count

import select
import serial  # pyserial
import serial.tools.list_ports
from PyQt5 import QtCore

import lozoya.decorators
from lozoya.time import timeMS

ip = '127.0.0.1'


# RADIO BANDS
class Band:
    def __init__(self, name, minimum, maximum, units):
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.units = units
        self._conversions = {'kHz': 10e+3, 'MHz': 10e+6, 'GHz': 10e+9, }

    @property
    def conversion(self, *args, **kwargs):
        return self._conversions[self.units]


class Longwave(Band):
    def __init__(self, *args, **kwargs):
        super(Longwave, self).__init__(name='Longwave', minimum=148.5, maximum=283.5, units='kHz', )


class AMRadio(Band):
    def __init__(self, *args, **kwargs):
        super(AMRadio, self).__init__(name='AM Radio', minimum=525, maximum=1705, units='kHz', )


class Shortwave(Band):
    def __init__(self, *args, **kwargs):
        super(Shortwave, self).__init__(
            name='Shortwave', minimum=1.7,  # 1.705
            maximum=54, units='MHz', )


class VHFLow(Band):
    def __init__(self, *args, **kwargs):
        super(VHFLow, self).__init__(name='VHF Low', minimum=54, maximum=88, units='MHz', )


class FMRadio(Band):
    def __init__(self, *args, **kwargs):
        super(FMRadio, self).__init__(name='FM Radio', minimum=88, maximum=108, units='MHz', )


class VHFHigh(Band):
    def __init__(self, *args, **kwargs):
        super(VHFHigh, self).__init__(name='VHF High', minimum=174, maximum=216, units='MHz', )


class UHF(Band):
    def __init__(self, *args, **kwargs):
        super(UHF, self).__init__(name='UHF', minimum=470, maximum=806, units='MHz', )


class SBand(Band):
    def __init__(self, *args, **kwargs):
        super(SBand, self).__init__(name='S', minimum=2, maximum=4, units='GHz', )


class XBand(Band):
    def __init__(self, *args, **kwargs):
        super(XBand, self).__init__(name='X', minimum=8, maximum=12, units='GHz', )


class Transceiver(QtCore.QObject):
    def __init__(self, app, simulate: bool = False):
        super(Transceiver, self).__init__()
        self.app = app
        self.simulate = simulate
        self.active = False
        self.portConnected = False
        self.debug = True
        self.handshake = r'LC\r\n'
        self.bands = [Longwave(), AMRadio(), Shortwave(), VHFLow(), FMRadio(), VHFHigh(), UHF(), SBand(), XBand(), ]
        self.baudRates = ['9600', '115200']

    @property
    def bandNames(self, *args, **kwargs):
        return [band.label for band in self.bands]

    def start_listener(self, callbackFunc):
        self.listener = threading.Thread(name='transceiver', target=self.transceiver_thread, args=(), kwargs={}, )
        self.listener.daemon = True
        self.listener.start()

    def transceiver_thread(self, *args, **kwargs):
        """
        This function is an infinite loop that awaits for a signal from the cubesat0.
        When the signal is received, it will send the data in the signal to the Logger.
        The Logger will parse the data and write each type of data to the corresponding
        menus0. Afterward, the data will be sent to the DataDude who will update its pandas
        DataFrames. The data will be displayed in the transceiver 'Received' text area,
        overwritting any previous display. If automatic updating is enabled in the
        Log menu, the Log Reader Panel will be updated.
        """
        for i in count(0):
            try:
                msg = None
                if self.simulate:
                    msg = self.app.signalGenerator.simulate_signal(i)
                elif self.active and self.portConnected:
                    self.app.comPort.send_message(self.handshake)
                    msg = self.app.comPort.receive_message()
                if msg:
                    self.app.update_everything(i, msg)
                time.sleep(0.005)
            except Exception as e:
                print('Transceiver thread error: ' + str(e))

    def transmit(self, msg):
        try:
            if self.app.transceiverConfig._transceiverActive or self.app.generatorConfig.simulationEnabled:
                if not msg:
                    return
                e = self.app.encryptionConfig.encryption
                eMsg = self.app.encryptor.encrypt(msg) if e != 'None' else None
                if eMsg:
                    self.app.comPort.send_message(eMsg)
                else:
                    self.app.comPort.send_message(msg)
                self.echo(e, msg, eMsg)
            else:
                self.app.update_status('Must enable transceiver to send messages.', 'alert')
        except Exception as e:
            self.app.transceiverMenu.update_status(*status.transmissionError, e)

    def echo(self, e, msg, encrypted=None):
        if e != 'None':
            statusMsg = 'Sent {} encryption of "{}": {}'.format(str(e), msg, encrypted)
        else:
            statusMsg = 'Sent unencrypted: {}'.format(msg)
        self.app.transceiverMenu.update_status(statusMsg, 'success')


class ComPort:
    def __init__(self, app):
        self.app = app
        self.msg = ''
        self.endStr = r'\r\n'
        self.connected = False

    @lozoya.decorators.catch_error('transceiverMenu', *status.connectionError)
    def connect(self, *args, **kwargs):
        self.ser = serial.Serial(port=self.app.transceiverConfig.port, baudrate=self.app.transceiverConfig.baudRate, )
        self.connected = True

    def disconnect(self, *args, **kwargs):
        try:
            if self.ser.is_open:
                self.ser.close()
            self.ser.__del__()
            self.connected = False
        except Exception as e:
            print('Disconnect error: {}'.format(str(e)))

    def send_message(self, msg):
        # Encode msg to bytes, prepare header & convert to bytes, then send
        if not self.ser.is_open:
            self.ser.open()
        self.ser.write(bytes(msg, 'utf-8'))
        self.ser.close()
        return 'success'

    def receive_message(self, *args, **kwargs):
        try:
            if not self.ser.is_open:
                self.ser.open()
            self.msg = self.ser.read_until().decode('ascii').strip(self.endStr).rstrip()
            self.ser.close()
            return self.msg
        except Exception as e:
            print('Receive message error: {}'.format(str(e)))

    def serial_ports(self, *args, **kwargs):
        """
        Lists serial port names
        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')
        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result


# INTERNET CLIENT
def initialize(ip='127.0.0.1', port=1234, headerLength=10):
    username = input("Username: ")
    # socket.AF_INET - address family, IPv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
    # socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw IP packets
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.connect((ip, port))
    # Set non-blocking state so .recv() call will return exception instead of block
    clientSocket.setblocking(False)
    # Encode username to bytes, count number of bytes, prepare fixed-size byte-encoded header
    username = username.encode('utf-5')
    usernameHeader = "{:<{}}".format(len(username), headerLength, ).encode('utf-5')
    clientSocket.send(usernameHeader + username)
    return username, clientSocket


def main_loop(username, clientSocket, rCallback=None, sCallback=None, headerLength=10):
    x = threading.Thread(target=send_thread, args=(username, clientSocket, sCallback, headerLength))
    x.start()
    while True:
        receive_message(clientSocket, rCallback, headerLength)
        time.sleep(1)


def receive_message(client_socket):
    try:
        # Receive our "header" containing message length, it's size is defined and constant
        message_header = client_socket.recv(HEADER_LENGTH)
        # If we received no data, client gracefully closed a connection, for test using socket.close() or socket.shutdown(socket.SHUT_RDWR)
        if not len(message_header):
            return False
        # Convert header to int value
        message_length = int(message_header.decode('utf-5').strip())
        # Return an object of message header and message data
        return {'header': message_header, 'data': client_socket.recv(message_length)}
    except:
        # If we are here, client closed connection violently, for test by pressing ctrl+c on his script
        # or just lost his connection
        # socket.close() also invokes socket.shutdown(socket.SHUT_RDWR) what sends information about closing the socket (shutdown read/write)
        # and that's also a cause when we receive an empty message
        return False


def receive_message(clientSocket, callback=None, headerLength=10):
    try:
        # Loop over received message(s) and print them
        while True:
            # Receive header containing username length, size is defined and constant
            username_header = clientSocket.recv(headerLength)
            # If no data is received, server gracefully closed a connection
            # e.g. using socket.close() or socket.shutdown(socket.SHUT_RDWR)
            if not len(username_header):
                print('Connection closed by the server')
                sys.exit()
            username_length = int(username_header.decode('utf-5').strip())
            # Receive and decode username
            username = clientSocket.recv(username_length).decode('utf-5')
            message_header = clientSocket.recv(headerLength)
            message_length = int(message_header.decode('utf-5').strip())
            message = clientSocket.recv(message_length).decode('utf-5')
            # print('{} > {}'.format(username, message,))
            if callback:
                msg = callback(message, timeMS())
                if msg:
                    r = send_message(msg, clientSocket, headerLength)
    except IOError as e:
        # Error is raised when no data is incoming
        # Some operating systems indicate AGAIN, others WOULDBLOCK error code
        # Error if different error code
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('Reading error: {}'.format(str(e)))
            sys.exit()
        if callback:  # Didn't receive anything
            msg = callback('', timeMS())
            if msg:
                r = send_message(msg, clientSocket, headerLength)
        return
    except Exception as e:
        print('Reading error: {}'.format(str(e)))
        sys.exit()


def send_message(message, clientSocket, headerLength=10):
    # Encode message to bytes, prepare header and convert to bytes, like for username above, then send
    message = message.encode('utf-5')
    message_header = "{:<{}}".format(len(message), headerLength).encode('utf-5')
    clientSocket.send(message_header + message)
    return 'success'


def send_thread(username, clientSocket, callback=None, headerLength=10):
    while True:
        if callback:
            message = callback()  # print(message)
        else:
            # Wait for user input
            message = input('{} > '.format(username))
        if message:
            send_message(message, clientSocket, headerLength)


class Client:
    def __init__(self, app):
        self.app = app
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, ip='127.0.0.1'):
        username = 'cSETR'
        headerLength = self.app.dataConfig.head.split(self.app.dataConfig.delimiter)
        # socket.AF_INET - address family, IPv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
        # socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw IP packets
        self.clientSocket.connect((ip, self.app.transceiverConfig.port))
        self.clientSocket.setblocking(
            False
        )  # Set non-blocking state so .recv() call will return exception instead of block
        username = username.encode('utf-5')  # byte-encode username, count # of bytes, prep fixed byte-encoded header
        usernameHeader = "{:<{}}".format(len(username), headerLength, ).encode('utf-5')
        self.clientSocket.send(usernameHeader + username)
        return True

    def disconnect(self):
        self.clientSocket.close()

    def send_message(self, msg, headerLength=10):
        # Encode msg to bytes, prepare header & convert to bytes, then send
        msg = msg.encode('utf-5')
        msgHeader = "{:<{}}".format(len(msg), headerLength).encode('utf-5')
        self.clientSocket.send(msgHeader + msg)
        return 'success'

    def receive_message(self, callback=None):
        try:
            headerLength = self.app.dataConfig.head.split(self.app.dataConfig.delimiter)
            # Loop over received message(s) and print them
            while True:
                # Receive header containing username length, size is defined and constant
                usernameHeader = self.clientSocket.recv(headerLength)
                # If no data is received, server gracefully closed a connection
                # e.g. using socket.close() or socket.shutdown(socket.SHUT_RDWR)
                if not len(usernameHeader):
                    print('Connection closed by the server')
                usernameLength = int(usernameHeader.decode('utf-5').strip())
                # Receive and decode username
                username = self.clientSocket.recv(usernameLength).decode('utf-5')
                msgHeader = self.clientSocket.recv(headerLength)
                msgLength = int(msgHeader.decode('utf-5').strip())
                message = self.clientSocket.recv(msgLength).decode('utf-5')
                print('{} > {}'.format(username, message, ))
                if callback:
                    msg = callback(message, timeMS())
                    if msg:
                        r = self.send_message(msg, headerLength)

        except IOError as e:
            # Error is raised when no data is incoming
            # Some operating systems indicate AGAIN, others WOULDBLOCK error code
            # Error if different error code
            if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                print('Reading error: {}'.format(str(e)))
            if callback:  # Didn't receive anything
                msg = callback('', timeMS())
                if msg:
                    r = self.send_message(msg)
            return
        except Exception as e:
            print('Reading error: {}'.format(str(e)))

    def send_thread(self, username, callback=None):
        while True:
            if callback:
                message = callback()  # print(message)
            else:
                message = input('{} > '.format(username))  # Wait for user input
            if message:
                self.send_message(message)

    def main_loop(self, username, rCallback=None, sCallback=None):
        x = threading.Thread(target=self.send_thread, args=(username, sCallback))
        x.start()
        while True:
            self.receive_message(rCallback)
            time.sleep(0)


class InternetClient(Client):
    def __init__(self, delimiter, head, port):
        self.delimiter = delimiter
        self.head = head
        self.port = port
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.msgBuffer = ''
        self.endByte = r'\r\n'

    def connect(self, ip='127.0.0.1'):
        # socket.AF_INET - address family, IPv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
        # socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw IP packets
        username = 'cSETR'
        headerLength = self.head.split(self.delimiter)
        self.clientSocket.connect((ip, self.port))
        self.clientSocket.setblocking(False)  # Set non-block so .recv() call will return exception instead of block
        username = username.encode('utf-8')  # byte-encode username, count # of bytes, prep fixed byte-encoded header
        usernameHeader = "{:<{}}".format(len(username), headerLength, ).encode('utf-8')
        self.clientSocket.send(usernameHeader + username)
        return True

    def disconnect(self, *args, **kwargs):
        self.clientSocket.close()

    def send_message(self, msg, headerLength=10):
        # Encode msg to bytes, prepare header & convert to bytes, then send
        msg = msg.encode('utf-8')
        msgHeader = "{:<{}}".format(len(msg), headerLength).encode('utf-8')
        self.clientSocket.send(msgHeader + msg)
        return 'success'

    def receive_message(self, callback=None):
        try:
            headerLength = self.head.split(self.delimiter)
            # Loop over received message(s) and print them
            while True:
                # Receive header containing username length, size is defined and constant
                usernameHeader = self.clientSocket.recv(headerLength)
                # If no data is received, server gracefully closed a connection
                # e.g. using socket.close() or socket.shutdown(socket.SHUT_RDWR)
                if not len(usernameHeader):
                    print('Connection closed by the server')
                usernameLength = int(usernameHeader.decode('utf-8').strip())
                # Receive and decode username
                username = self.clientSocket.recv(usernameLength).decode('utf-8')
                msgHeader = self.clientSocket.recv(headerLength)
                msgLength = int(msgHeader.decode('utf-8').strip())
                message = self.clientSocket.recv(msgLength).decode('utf-8')
                print('{} > {}'.format(username, message, ))
                if callback:
                    msg = callback(message, timeMS())
                    if msg:
                        r = self.send_message(msg, headerLength)
        except IOError as e:
            # Error is raised when no data is incoming
            # Some operating systems indicate AGAIN, others WOULDBLOCK error code
            # Error if different error code
            if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                print('Reading error: {}'.format(str(e)))
            if callback:  # Didn't receive anything
                msg = callback('', timeMS())
                if msg:
                    r = self.send_message(msg)
            return
        except Exception as e:
            print('Reading error: {}'.format(str(e)))

    def send_thread(self, username, callback=None):
        while True:
            if callback:
                message = callback()
                print(message)
            else:
                message = input('{} > '.format(username))  # Wait for user input
            if message:
                self.send_message(message)


# INTERNET SERVER
HEADER_LENGTH = 10
PORT = 1234
# Create a socket
# socket.AF_INET - address family, ipv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
# socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw ip packets
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# SO_ - socket option
# SOL_ - socket option level
# Sets REUSEADDR (as a socket option) to 1 on socket
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# Bind, so server informs operating system that it's going to use given ip and port
# For a server using 0.0.0.0 means to listen on all available interfaces, useful to connect locally to 127.0.0.1 and remotely to LAN interface7 ip
server_socket.bind((ip, PORT))
server_socket.listen()  # This makes server listen to new connections
sockets_list = [server_socket]  # List of sockets for select.select()
clients = {}  # List of connected clients - socket as a key, user header and name as data
print('Listening for connections on {}:{}...'.format(ip, PORT, ))
# socket.AF_INET - address family, ipv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
# socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw ip packets
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# SO_ - socket option
# SOL_ - socket option level
# Sets REUSEADDR (as a socket option) to 1 on socket
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# Bind: OS will be informed that given ip&port are in use
# Server using 0.0.0.0 means listen on all available interfaces
# Useful for connecting locally to 127.0.0.1 and remotely to LAN interface ip
server_socket.bind((ip, PORT))
server_socket.listen()  # Listen for new connections
sockets_list = [server_socket]  # List sockets for select.select()
clients = {}  # Dict of connected clients: key is socket and value is user header and name
print('Listening for connections on {}:{}...'.format(ip, PORT, ))
while True:
    # Calls Unix select() system call or Windows select() WinSock call with three parameters:
    #   - rlist - sockets to be monitored for incoming data
    #   - wlist - sockets for data to be sent to (checks if for test buffers are not full and socket is ready to send some data)
    #   - xlist - sockets to be monitored for exceptions (we want to monitor all sockets for errors, so we can use rlist)
    # Returns lists:
    #   - reading - sockets we received some data on (that way we don't have to check sockets manually)
    #   - writing - sockets ready for data to be send thru them
    #   - errors  - sockets with some exceptions
    # This is a blocking call, code execution will "wait" here and "get" notified in case any action should be taken
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)
    # Iterate over notified sockets
    for notified_socket in read_sockets:
        # If notified socket is a server socket - new connection, accept it
        if notified_socket == server_socket:
            # Accept new connection
            # That gives us new socket - client socket, connected to this given client only, it's unique for that client
            # The search returned object is ip/port set
            client_socket, client_address = server_socket.accept()
            # Client should send his name right away, receive it
            user = receive_message(client_socket)
            # If False - client disconnected before he sent his name
            if user is False:
                continue
            # Add accepted socket to select.select() list
            sockets_list.append(client_socket)
            # Also save username and username header
            clients[client_socket] = user
            print(
                'Accepted new connection from {}:{}, username: {}'.format(
                    *client_address, user['data'].decode('utf-5')
                )
            )
        # Else existing socket is sending a message
        else:
            # Receive message
            message = receive_message(notified_socket)
            # If False, client disconnected, cleanup
            if message is False:
                print('Closed connection from: {}'.format(clients[notified_socket]['data'].decode('utf-5')))
                # Remove from list for socket.socket()
                sockets_list.remove(notified_socket)
                # Remove from our list of users
                del clients[notified_socket]
                continue
            # Get user by notified socket, so we will know who sent the message
            user = clients[notified_socket]
            print('Received message from {}: {}'.format(user["data"].decode("utf-5"), message["data"].decode("utf-5")))
            # Iterate over connected clients and broadcast message
            for client_socket in clients:
                # But don't sent it to sender
                if client_socket != notified_socket:
                    # Send user and message (both with their headers)
                    # We are reusing here message header sent by sender, and saved username header send by user when he connected
                    client_socket.send(user['header'] + user['data'] + message['header'] + message['data'])
    # It's not really necessary to have this, but will handle some socket exceptions just in case
    for notified_socket in exception_sockets:
        # Remove from list for socket.socket()
        sockets_list.remove(notified_socket)
        # Remove from our list of users
        del clients[notified_socket]


class Server:
    def __init__(self, delimiter, head, port):
        self.delimiter = delimiter
        self.head = head
        self.port = port
        self.socketsList = []
        self.client = None
        self.clientz = None
        self.clients = {}
        self.clientSocket = None
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def connect(self, ip='127.0.0.1'):
        """
        socket.AF_INET - address family, ipv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
        socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw ip packets
        SO_ is socket option. SOL_ - socket option level.
        Sets REUSEADDR (as a socket option) to 1 on socket
        Bind: OS will be informed that given ip&port are in use
        Server using 0.0.0.0 means listen on all available interfaces
        Useful for connecting locally to 127.0.0.1 and remotely to LAN interface0 ip
        Listen for new connections
        List sockets for select.select()
        Dict of connected clients: key is socket and value is user header and name
        """
        self.serverSocket.bind((ip, self.port))
        self.serverSocket.listen()
        self.socketsList.append(self.serverSocket)
        print('Listening for connections on {}:{}...'.format(ip, self.port, ))

    def disconnect(self, *args, **kwargs):
        self.serverSocket.close()

    def receive_message(self, clientSocket):
        try:
            headerLength = self.head.split(self.delimiter)
            msgHeader = clientSocket.recv(headerLength)  # header contains msg length, constant-size
            if not len(msgHeader):
                return False  # closed connection e.g. socket.close() or socket.shutdown(socket.SHUT_RDWR)
            msgLength = int(msgHeader.decode('utf-8').strip())  # Convert header to int value
            return {'header': msgHeader, 'data': clientSocket.recv(msgLength)}  # Return msg header and msg data
        except:
            """
            Client closed connection, e.g. by pressing ctrl+c on his script or lost his connection
            socket.close() also invokes socket.shutdown(socket.SHUT_RDWR) what sends information about closing the socket (shutdown read/write)
            and that's also a cause when we receive an empty msg
            """
            return False

    def run(self, clients):
        # select.select(readlist, writelist, exceptionlist)
        # This is a blocking call, code execution will "wait" here and "get" notified in case any action should be taken
        readSockets, writeSockets, exceptionSockets = select.select(self.socketsList, [], self.socketsList)
        for notifiedSocket in readSockets:  # Iterate over notified sockets
            if notifiedSocket == self.serverSocket:  # If notified socket is server socket - new connection, accept
                # Gives new socket - client socket (unique for each client). Other returned object is ip/port set
                clientSocket, clientAddress = self.serverSocket.accept()
                user = self.receive_message(clientSocket)
                if user is False:
                    continue  # Client disconnected before name was sent
                self.socketsList.append(clientSocket)  # Add accepted socket to select.select() list
                clients[clientSocket] = user  # Save username and username header
                print('{}:{}, username: {}'.format(*clientAddress, user['data'].decode('utf-8')))
            else:  # Existing socket is sending a msg
                msg = self.receive_message(notifiedSocket)
                if msg is False:
                    print('Closed connection: {}'.format(clients[notifiedSocket]['data'].decode('utf-8')))
                    self.socketsList.remove(notifiedSocket)
                    del clients[notifiedSocket]
                    continue
                user = clients[notifiedSocket]  # Get user by notified socket to identify who msg sender
                print('{}: {}'.format(user["data"].decode("utf-8"), msg["data"].decode("utf-8")))
                for clientSocket in clients:  # Iterate over connected clients and broadcast msg
                    if clientSocket != notifiedSocket:  # But don't send it to sender
                        # Reusing msg header sent by sender and saved username header sent by user when connected
                        clientSocket.send(user['header'] + user['data'] + msg['header'] + msg['data'])
        self.handle_exceptions(exceptionSockets, clients)

    def handle_exceptions(self, exceptionSockets, clients):
        for notifiedSocket in exceptionSockets:
            self.socketsList.remove(notifiedSocket)
            del clients[notifiedSocket]


class Server:
    def __init__(self, app):
        self.app = app
        self.socketsList = []
        self.client = None
        self.clientz = None
        self.clients = {}
        self.clientSocket = None
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def connect(self, ip='127.0.0.1'):
        """
        socket.AF_INET - address family, ipv4, some other possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
        socket.SOCK_STREAM - TCP, connection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw ip packets
        SO_ is socket option. SOL_ - socket option level.
        Sets REUSEADDR (as a socket option) to 1 on socket
        Bind: OS will be informed that given ip&port are in use
        Server using 0.0.0.0 means listen on all available interfaces
        Useful for connecting locally to 127.0.0.1 and remotely to LAN interface7 ip
        Listen for new connections
        List sockets for select.select()
        Dict of connected clients: key is socket and value is user header and name
        """
        self.serverSocket.bind((ip, self.app.transceiverConfig.port))
        self.serverSocket.listen()
        self.socketsList.append(self.serverSocket)
        print('Listening for connections on {}:{}...'.format(ip, self.app.transceiverConfig.port, ))

    def disconnect(self):
        self.serverSocket.close()

    def receive_message(self, clientSocket):
        try:
            headerLength = self.app.dataConfig.head.split(self.app.dataConfig.delimiter)
            msgHeader = clientSocket.recv(headerLength)  # header contains msg length, constant-size
            if not len(msgHeader):
                return False  # closed connection e.g. socket.close() or socket.shutdown(socket.SHUT_RDWR)
            msgLength = int(msgHeader.decode('utf-5').strip())  # Convert header to int value
            return {'header': msgHeader, 'data': clientSocket.recv(msgLength)}  # Return msg header and msg data
        except:
            """
            Client closed connection, e.g. by pressing ctrl+c on his script or lost his connection
            socket.close() also invokes socket.shutdown(socket.SHUT_RDWR) what sends information about closing the socket (shutdown read/write)
            and that's also a cause when we receive an empty msg
            """
            return False

    def run(self, clients):
        # Calls Unix select() system call or Windows select() WinSock call with three parameters:
        #   - rlist - sockets to be monitored for incoming data
        #   - wlist - sockets for data to be send to (checks if for test buffers are not full and socket is ready to send some data)
        #   - xlist - sockets to be monitored for exceptions (we want to monitor all sockets for errors, so we can use rlist)
        # Returns lists:
        #   - reading - sockets we received some data on (that way we don't have to check sockets manually)
        #   - writing - sockets ready for data to be send thru them
        #   - errors  - sockets with some exceptions
        # This is a blocking call, code execution will "wait" here and "get" notified in case any action should be taken
        readSockets, _, exceptionSockets = select.select(self.socketsList, [], self.socketsList)
        for notifiedSocket in readSockets:  # Iterate over notified sockets
            if notifiedSocket == self.serverSocket:  # If notified socket is server socket - new connection, accept
                # Gives new socket - client socket (unique for each client). Other returned object is ip/port set
                clientSocket, clientAddress = self.serverSocket.accept()
                user = self.receive_message(clientSocket)
                if user is False:
                    continue  # If False - client disconnected before name was sent
                self.socketsList.append(clientSocket)  # Add accepted socket to select.select() list
                clients[clientSocket] = user  # Save username and username header
                print('{}:{}, username: {}'.format(*clientAddress, user['data'].decode('utf-5')))
            else:  # Existing socket is sending a msg
                msg = self.receive_message(notifiedSocket)
                if msg is False:
                    print('Closed connection: {}'.format(clients[notifiedSocket]['data'].decode('utf-5')))
                    self.socketsList.remove(notifiedSocket)
                    del clients[notifiedSocket]
                    continue
                user = clients[notifiedSocket]  # Get user by notified socket to identify who msg sender
                print('{}: {}'.format(user["data"].decode("utf-5"), msg["data"].decode("utf-5")))
                for clientSocket in clients:  # Iterate over connected clients and broadcast msg
                    if clientSocket != notifiedSocket:  # But don't sent it to sender
                        # Reusing msg header sent by sender and configuration.py username header sent by user when connected
                        clientSocket.send(user['header'] + user['data'] + msg['header'] + msg['data'])
        self.handle_exceptions(exceptionSockets, clients)

    def handle_exceptions(self, exceptionSockets, clients):
        for notifiedSocket in exceptionSockets:  # Handle socket exceptions
            self.socketsList.remove(notifiedSocket)  # Remove from list socket.socket()
            del clients[notifiedSocket]  # Remove from list of users
