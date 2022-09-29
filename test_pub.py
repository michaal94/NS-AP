#
# Last value cache
# Uses XPUB subscription messages to re-send data
#

import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5600")

i = 0

while True:
    i += 1
    socket.send_string(str(i))
    print(i)
    time.sleep(0.5)
