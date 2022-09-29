#
# Last value cache
# Uses XPUB subscription messages to re-send data
#

import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b'')
socket.setsockopt(zmq.CONFLATE, True)
socket.connect("tcp://127.0.0.1:5600")


while True:
    msg = socket.recv_string()
    print(msg)
    time.sleep(1.5)